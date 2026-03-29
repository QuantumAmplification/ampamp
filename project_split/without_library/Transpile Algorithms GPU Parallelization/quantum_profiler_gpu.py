import numpy as np
from typing import Dict, Any, Optional, Iterable, List
from concurrent.futures import ThreadPoolExecutor

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, InstructionDurations
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class HardwareProfiler:
    def __init__(self, 
                 coupling_map_edges: Optional[list] = None, 
                 basis_gates: Optional[list] = None, 
                 single_qubit_ns: int = 20, 
                 two_qubit_ns: int = 100,
                 w_time: float = 1.0, 
                 w_cnot: float = 10.0, 
                 w_distance: float = 5.0):
        
        self.coupling_map = CouplingMap(coupling_map_edges) if coupling_map_edges else None
        self.basis_gates = basis_gates or ['cx', 'id', 'rz', 'sx', 'x']
        self.single_qubit_ns = single_qubit_ns
        self.two_qubit_ns = two_qubit_ns
        
        self.w_time = w_time
        self.w_cnot = w_cnot
        self.w_distance = w_distance
        
        # Build instruction durations for all basis gates
        durations_list = []
        for gate in self.basis_gates:
            if gate in ['cx', 'ecr', 'cz', 'swap']:
                durations_list.append((gate, None, self.two_qubit_ns))
            else:
                durations_list.append((gate, None, self.single_qubit_ns))
        # Add measurement just in case
        durations_list.append(('measure', None, 500))
        self.durations = InstructionDurations(durations_list)

    def profile_circuit(self, logical_circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Pushes a logical circuit through 5 distinct compiler stages to extract physical metrics.
        Returns a dictionary of the profiling results.
        """
        metrics = {
            'logical_depth': logical_circuit.depth(),
            'logical_gates': sum(logical_circuit.count_ops().values())
        }
        
        # We will use preset pass managers to break down the compilation.
        pm = generate_preset_pass_manager(
            optimization_level=3,
            basis_gates=self.basis_gates,
            coupling_map=self.coupling_map
        )
        
        # For initial layout, we need to handle circuits larger than coupling map gracefully
        if self.coupling_map and logical_circuit.num_qubits > self.coupling_map.size():
            raise ValueError(f"Circuit requires {logical_circuit.num_qubits} qubits, but hardware map only has {self.coupling_map.size()}.")

        # Stage 0: Init
        # Unroll massive multi-qubit gates (like MCX-based reflections) into the target basis
        # before extracting profiling metrics. We then use stable end-to-end transpilation runs
        # for routed and optimized views, since directly chaining staged pass managers is brittle
        # across Qiskit versions for large already-decomposed circuits.
        pm_unroll = generate_preset_pass_manager(optimization_level=1, basis_gates=self.basis_gates)
        qc_init = pm_unroll.run(logical_circuit)
        
        # Stage 1: Layout
        qc_layout = pm.layout.run(qc_init)
        
        # Extract Initial Distance (Topological Strain)
        initial_distance = 0
        if self.coupling_map and qc_layout.layout is not None:
            layout = qc_layout.layout
            for inst in logical_circuit.data:
                # We only care about 2-qubit interactions
                if len(inst.qubits) == 2:
                    q0, q1 = inst.qubits
                    if q0 in layout.initial_layout.get_virtual_bits() and q1 in layout.initial_layout.get_virtual_bits():
                        p0 = layout.initial_layout[q0]
                        p1 = layout.initial_layout[q1]
                        dist = self.coupling_map.distance(p0, p1)
                        if dist > 1:
                            initial_distance += (dist - 1)
                            
        metrics['initial_distance_penalty'] = initial_distance
        
        # Stage 2 + 3: Routing and Translation
        # Use a non-optimizing physical transpilation pass to obtain a routed native-gate circuit.
        qc_translation = transpile(
            qc_init,
            coupling_map=self.coupling_map,
            basis_gates=self.basis_gates,
            optimization_level=0,
        )
        routing_ops = qc_translation.count_ops()
        metrics['post_routing_depth'] = qc_translation.depth()
        metrics['routing_swaps'] = routing_ops.get('swap', 0)
        metrics['post_translation_depth'] = qc_translation.depth()
        metrics['translation_gates'] = sum(routing_ops.values())
        metrics['translation_cnots'] = routing_ops.get('cx', 0)
        
        # Stage 4: Optimization
        qc_opt = transpile(
            qc_init,
            coupling_map=self.coupling_map,
            basis_gates=self.basis_gates,
            optimization_level=3,
        )
        opt_ops = qc_opt.count_ops()
        metrics['post_optimization_depth'] = qc_opt.depth()
        metrics['final_gates'] = sum(opt_ops.values())
        metrics['final_cnots'] = opt_ops.get('cx', 0)
        
        # Stage 5: Scheduling
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import ALAPScheduleAnalysis
        
        try:
            pm_sched = PassManager([ALAPScheduleAnalysis(self.durations)])
            pm_sched.run(qc_opt)
            node_start = pm_sched.property_set['node_start_time']
            duration_ns = 0
            if node_start:
                for node, start in node_start.items():
                    dur_ns = self.durations.get(node.op, node.qargs)
                    if dur_ns is None: dur_ns = 0
                    if start + dur_ns > duration_ns:
                        duration_ns = start + dur_ns
        except Exception:
            # Fallback naive time calculation if schedule pass skips or fails
            duration_ns = metrics['post_optimization_depth'] * self.two_qubit_ns
            
        metrics['total_time_ns'] = float(duration_ns)
        
        # Phase 3: The Unified Cost Function
        metrics['hardware_penalty_score'] = self.calculate_cost_function(
            time_ns=metrics['total_time_ns'],
            cnots=metrics['final_cnots'],
            distance=metrics['initial_distance_penalty']
        )
        
        return metrics

    def calculate_cost_function(self, time_ns: float, cnots: int, distance: int) -> float:
        """
        Unified Cost Function combining Time, Entanglement (CNOTs), and Topological Strain.
        Cost = (w1 * Total Time) + (w2 * Total CNOTs) + (w3 * Routing Distance)
        """
        cost = (self.w_time * time_ns) + (self.w_cnot * cnots) + (self.w_distance * distance)
        return cost

    def profile_circuits_parallel(
        self,
        circuits: Iterable[QuantumCircuit],
        *,
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Parallelize independent profiling jobs over a thread pool."""
        circuit_list = list(circuits)
        if not circuit_list:
            return []
        if max_workers is not None and int(max_workers) <= 1:
            return [self.profile_circuit(circ) for circ in circuit_list]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.profile_circuit, circuit_list))

if __name__ == "__main__":
    # Test script locally
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    profiler = HardwareProfiler(coupling_map_edges=[[0, 1], [1, 2]])
    metrics = profiler.profile_circuit(qc)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
