import numpy as np
from dataclasses import dataclass, field
from qiskit import QuantumCircuit, transpile
from typing import Optional, Dict, Any, List

@dataclass(frozen=True)
class CircuitMetrics:
    """Quantitative evaluation of a transpiled quantum circuit."""
    depth: int
    gate_counts: Dict[str, int]
    qubits: int
    clbits: int
    blowup_factor: Optional[float] = None
    
    @property
    def cx_count(self) -> int:
        return self.gate_counts.get('cx', 0)
    
    @property
    def t_count(self) -> int:
        return self.gate_counts.get('t', 0) + self.gate_counts.get('tdg', 0)

def transpile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: Optional[List[List[int]]] = None,
    basis_gates: Optional[List[str]] = None,
    optimization_level: int = 3,
    **kwargs: Any
) -> tuple[QuantumCircuit, CircuitMetrics]:
    """Wraps Qiskit's transpile and extracts detailed hardware metrics.
    
    Args:
        qc (QuantumCircuit): The logical circuit to transpile.
        coupling_map (list): Physical qubit connectivity map.
        basis_gates (list): Supported physical hardware gates.
        optimization_level (int): Qiskit transpilation optimization (0-3).
        **kwargs: Additional arguments for qiskit.transpile.
        
    Returns:
        tuple: (Transpiled QuantumCircuit, CircuitMetrics)
    """
    ideal_depth = qc.depth()
    
    t_qc = transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        **kwargs
    )
    
    depth = t_qc.depth()
    ops = dict(t_qc.count_ops())
    
    blowup = depth / ideal_depth if ideal_depth > 0 else 1.0
    
    metrics = CircuitMetrics(
        depth=depth,
        gate_counts=ops,
        qubits=t_qc.num_qubits,
        clbits=t_qc.num_clbits,
        blowup_factor=float(blowup)
    )
    
    return t_qc, metrics

# Predefined Hardware Profiles
class HardwareProfiles:
    """Common hardware connectivity and basis gate profiles."""
    
    @staticmethod
    def ibm_basis_gates() -> List[str]:
        return ['cx', 'id', 'rz', 'sx', 'x']
    
    @staticmethod
    def linear_map(n_qubits: int) -> List[List[int]]:
        cmap = []
        for i in range(n_qubits - 1):
            cmap.append([i, i + 1])
            cmap.append([i + 1, i])
        return cmap

    @staticmethod
    def heavy_hex_subgraph_6() -> List[List[int]]:
        """Approximate 6-qubit heavy-hex sub-graph."""
        return [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3], [4, 5], [5, 4]]
