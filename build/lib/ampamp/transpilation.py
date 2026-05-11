"""Transpilation and hardware-profiling utilities for ampamp.

This module provides reusable, library-grade classes for staged circuit
transpilation analysis, routing diagnostics, and unified hardware scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, InstructionDurations, PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


@dataclass(frozen=True)
class HardwareCostWeights:
    """Weights used by the unified hardware cost function."""

    time: float = 1.0
    cnot: float = 10.0
    distance: float = 5.0


@dataclass(frozen=True)
class TranspilationProfileConfig:
    """Configuration for staged transpilation profiling."""

    coupling_map_edges: Optional[list[list[int]]] = None
    basis_gates: tuple[str, ...] = ("cx", "id", "rz", "sx", "x")
    single_qubit_ns: int = 20
    two_qubit_ns: int = 100
    measurement_ns: int = 500
    unroll_optimization_level: int = 1
    route_optimization_level: int = 0
    optimize_optimization_level: int = 3
    cost_weights: HardwareCostWeights = field(default_factory=HardwareCostWeights)


class TranspilationProfiler:
    """Profiles a circuit through layout, routing, optimization, and scheduling."""

    def __init__(self, config: Optional[TranspilationProfileConfig] = None):
        self.config = config or TranspilationProfileConfig()
        self.coupling_map = (
            CouplingMap(self.config.coupling_map_edges)
            if self.config.coupling_map_edges
            else None
        )
        self.durations = self._build_instruction_durations()

    def _build_instruction_durations(self) -> InstructionDurations:
        duration_entries: list[tuple[str, Optional[Iterable[int]], int]] = []
        for gate in self.config.basis_gates:
            if gate in {"cx", "cz", "ecr", "swap"}:
                duration_entries.append((gate, None, self.config.two_qubit_ns))
            else:
                duration_entries.append((gate, None, self.config.single_qubit_ns))
        duration_entries.append(("measure", None, self.config.measurement_ns))
        return InstructionDurations(duration_entries)

    def calculate_cost_function(self, *, time_ns: float, cnots: int, distance: int) -> float:
        """Compute unified hardware penalty score.

        score = w_time*time + w_cnot*cnots + w_distance*distance
        """
        w = self.config.cost_weights
        return (w.time * float(time_ns)) + (w.cnot * int(cnots)) + (w.distance * int(distance))

    @staticmethod
    def _gate_count(ops: Mapping[str, int]) -> int:
        return int(sum(int(v) for v in ops.values()))

    def _initial_distance_penalty(self, qc_layout: QuantumCircuit, logical_circuit: QuantumCircuit) -> int:
        if self.coupling_map is None or qc_layout.layout is None:
            return 0

        distance_penalty = 0
        layout = qc_layout.layout
        virtual_bits = layout.initial_layout.get_virtual_bits()

        for inst in logical_circuit.data:
            if len(inst.qubits) != 2:
                continue
            q0, q1 = inst.qubits
            if q0 not in virtual_bits or q1 not in virtual_bits:
                continue
            p0 = layout.initial_layout[q0]
            p1 = layout.initial_layout[q1]
            dist = int(self.coupling_map.distance(p0, p1))
            if dist > 1:
                distance_penalty += (dist - 1)
        return int(distance_penalty)

    def _scheduled_duration_ns(self, qc_opt: QuantumCircuit) -> float:
        try:
            pm_sched = PassManager([ALAPScheduleAnalysis(self.durations)])
            pm_sched.run(qc_opt)
            node_start = pm_sched.property_set.get("node_start_time", {})
            duration_ns = 0
            for node, start in node_start.items():
                gate_duration = self.durations.get(node.op, node.qargs)
                if gate_duration is None:
                    gate_duration = 0
                duration_ns = max(duration_ns, int(start) + int(gate_duration))
            return float(duration_ns)
        except Exception:
            return float(qc_opt.depth() * self.config.two_qubit_ns)

    def profile_circuit(self, logical_circuit: QuantumCircuit) -> dict[str, Any]:
        """Run a multi-stage transpilation profile and return hardware metrics."""
        if self.coupling_map and logical_circuit.num_qubits > self.coupling_map.size():
            raise ValueError(
                f"Circuit requires {logical_circuit.num_qubits} qubits, "
                f"but coupling map has {self.coupling_map.size()}."
            )

        metrics: dict[str, Any] = {
            "logical_depth": int(logical_circuit.depth()),
            "logical_gates": self._gate_count(logical_circuit.count_ops()),
            "num_qubits": int(logical_circuit.num_qubits),
            "basis_gates": list(self.config.basis_gates),
        }

        pm_unroll = generate_preset_pass_manager(
            optimization_level=self.config.unroll_optimization_level,
            basis_gates=list(self.config.basis_gates),
        )
        qc_init = pm_unroll.run(logical_circuit)

        metrics["initial_distance_penalty"] = 0
        if self.coupling_map is not None:
            pm_layout = generate_preset_pass_manager(
                optimization_level=self.config.optimize_optimization_level,
                basis_gates=list(self.config.basis_gates),
                coupling_map=self.coupling_map,
            )
            if pm_layout.layout is not None:
                qc_layout = pm_layout.layout.run(qc_init)
                metrics["initial_distance_penalty"] = self._initial_distance_penalty(
                    qc_layout, logical_circuit
                )

        qc_routed = transpile(
            qc_init,
            coupling_map=self.coupling_map,
            basis_gates=list(self.config.basis_gates),
            optimization_level=self.config.route_optimization_level,
        )
        routed_ops = qc_routed.count_ops()
        metrics["post_routing_depth"] = int(qc_routed.depth())
        metrics["routing_swaps"] = int(routed_ops.get("swap", 0))
        metrics["translation_gates"] = self._gate_count(routed_ops)
        metrics["translation_cnots"] = int(routed_ops.get("cx", 0))

        qc_opt = transpile(
            qc_init,
            coupling_map=self.coupling_map,
            basis_gates=list(self.config.basis_gates),
            optimization_level=self.config.optimize_optimization_level,
        )
        opt_ops = qc_opt.count_ops()
        metrics["post_optimization_depth"] = int(qc_opt.depth())
        metrics["final_gates"] = self._gate_count(opt_ops)
        metrics["final_cnots"] = int(opt_ops.get("cx", 0))

        metrics["total_time_ns"] = self._scheduled_duration_ns(qc_opt)
        metrics["hardware_penalty_score"] = self.calculate_cost_function(
            time_ns=metrics["total_time_ns"],
            cnots=metrics["final_cnots"],
            distance=metrics["initial_distance_penalty"],
        )

        return metrics


class TranspilationBatchProfiler:
    """Profiles multiple circuits with one shared transpilation configuration."""

    def __init__(self, profiler: Optional[TranspilationProfiler] = None):
        self.profiler = profiler or TranspilationProfiler()

    def profile_many(self, circuits: Mapping[str, QuantumCircuit]) -> dict[str, dict[str, Any]]:
        """Profile a dictionary of named circuits."""
        return {name: self.profiler.profile_circuit(qc) for name, qc in circuits.items()}
