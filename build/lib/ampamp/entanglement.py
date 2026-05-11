from __future__ import annotations

"""Library-native entanglement-count profiling utilities."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, Statevector, entropy, partial_trace


_TERMINAL_ONLY_OPS = {"barrier", "measure", "reset"}
_MEASUREMENT_OPS = {"measure"}
_RESET_OPS = {"reset"}
_COUNT_MODE_ALIASES = {
    "light": "light",
    "sample": "light",
    "sampled": "light",
    "cheap": "light",
    "hard": "hard",
    "full": "hard",
    "heavy": "hard",
    "exact": "hard",
    "every_step": "hard",
}
_MEASURE_CHANNEL = Kraus(
    [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    ]
)
_RESET_CHANNEL = Kraus(
    [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
    ]
)


@dataclass(frozen=True)
class EntanglementCountConfig:
    """Controls active-entanglement counting cost and hardware limits.

    ``mode="light"`` samples a bounded number of checkpoints.  ``mode="hard"``
    evaluates the count after every quantum instruction.  ``max_qubits`` is a
    safety bound for statevector/density-matrix simulation.
    """

    mode: str = "light"
    max_qubits: int = 12
    max_snapshots: int = 64
    entropy_threshold: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _normalize_count_mode(self.mode))
        object.__setattr__(self, "max_qubits", int(self.max_qubits))
        object.__setattr__(self, "max_snapshots", int(self.max_snapshots))
        object.__setattr__(self, "entropy_threshold", float(self.entropy_threshold))
        if self.max_qubits < 1:
            raise ValueError("max_qubits must be >= 1")
        if self.max_snapshots < 1:
            raise ValueError("max_snapshots must be >= 1")
        if self.entropy_threshold < 0.0:
            raise ValueError("entropy_threshold must be nonnegative")

    @classmethod
    def light(cls, **kwargs: Any) -> "EntanglementCountConfig":
        """Create a sampled-checkpoint configuration for constrained hardware."""

        kwargs["mode"] = "light"
        return cls(**kwargs)

    @classmethod
    def hard(cls, **kwargs: Any) -> "EntanglementCountConfig":
        """Create an every-step configuration for stronger hardware."""

        kwargs["mode"] = "hard"
        return cls(**kwargs)


def _normalize_count_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_")
    if key not in _COUNT_MODE_ALIASES:
        allowed = ", ".join(sorted({"light", "hard", "full", "heavy", "sampled"}))
        raise ValueError(f"Unsupported entanglement count mode {mode!r}. Choose one of: {allowed}.")
    return _COUNT_MODE_ALIASES[key]


def _trim_terminal_ops(qc: QuantumCircuit) -> list[Any]:
    data = list(qc.data)
    end = len(data)
    while end > 0 and data[end - 1].operation.name in _TERMINAL_ONLY_OPS:
        end -= 1
    return data[:end]


def _iter_quantum_instruction_indices(instructions: list[Any]) -> list[int]:
    return [
        idx
        for idx, inst in enumerate(instructions)
        if inst.operation.num_qubits > 0 and inst.operation.name != "barrier"
    ]


def _sample_instruction_indices(indices: list[int], max_snapshots: int) -> set[int]:
    if not indices:
        return set()
    limit = max(1, int(max_snapshots))
    if len(indices) <= limit:
        return set(indices)
    picks: set[int] = set()
    last = len(indices) - 1
    for slot in range(limit):
        source_idx = round(slot * last / max(1, limit - 1))
        picks.add(indices[source_idx])
    return picks


def _qargs_for_instruction(qc: QuantumCircuit, inst: Any) -> list[int]:
    return [int(qc.find_bit(qubit).index) for qubit in inst.qubits]


def _single_qubit_reduced_states(state: Any, num_qubits: int) -> list[Any]:
    reduced_states: list[Any] = []
    for keep in range(num_qubits):
        traced = [idx for idx in range(num_qubits) if idx != keep]
        reduced_states.append(partial_trace(state, traced))
    return reduced_states


def _single_qubit_entropies(reduced_states: list[Any]) -> list[float]:
    entropies: list[float] = []
    for reduced in reduced_states:
        value = float(np.real_if_close(entropy(reduced, base=2)))
        if abs(value) < 1e-12:
            value = 0.0
        entropies.append(value)
    return entropies


def _single_qubit_purities(reduced_states: list[Any]) -> list[float]:
    purities: list[float] = []
    for reduced in reduced_states:
        data = np.asarray(reduced.data, dtype=complex)
        purity = float(np.real_if_close(np.trace(data @ data)))
        purities.append(max(0.0, min(1.0, purity)))
    return purities


def _step_metrics(
    state: Any,
    num_qubits: int,
    *,
    entropy_threshold: float,
    pure_state_segment: bool,
) -> dict[str, Any]:
    if num_qubits <= 1:
        return {
            "single_qubit_entropies": [0.0] * num_qubits,
            "active_entangled_qubits": 0,
            "active_entanglement_qubits": 0,
            "mean_single_qubit_entropy": 0.0,
            "max_single_qubit_entropy": 0.0,
            "total_single_qubit_entropy": 0.0,
            "meyer_wallach_q": 0.0,
            "metric_interpretation": "pure_state_entanglement" if pure_state_segment else "mixed_state_entropy_proxy",
        }

    reduced_states = _single_qubit_reduced_states(state, num_qubits)
    entropies = _single_qubit_entropies(reduced_states)
    purities = _single_qubit_purities(reduced_states)
    total_entropy = float(sum(entropies))
    active_qubits = sum(1 for value in entropies if value > entropy_threshold)
    mean_purity = sum(purities) / float(num_qubits)
    meyer_wallach_q = float(max(0.0, min(1.0, 2.0 * (1.0 - mean_purity))))

    return {
        "single_qubit_entropies": [float(value) for value in entropies],
        "active_entangled_qubits": int(active_qubits),
        "active_entanglement_qubits": int(active_qubits),
        "mean_single_qubit_entropy": total_entropy / float(num_qubits),
        "max_single_qubit_entropy": max(entropies) if entropies else 0.0,
        "total_single_qubit_entropy": total_entropy,
        "meyer_wallach_q": meyer_wallach_q,
        "metric_interpretation": "pure_state_entanglement" if pure_state_segment else "mixed_state_entropy_proxy",
    }


def _apply_instruction(state: Any, inst: Any, qargs: list[int]) -> tuple[Any, bool]:
    name = inst.operation.name
    if name in _MEASUREMENT_OPS:
        out = state
        for qubit in qargs:
            out = out.evolve(_MEASURE_CHANNEL, qargs=[qubit])
        return out, True
    if name in _RESET_OPS:
        out = state
        for qubit in qargs:
            out = out.evolve(_RESET_CHANNEL, qargs=[qubit])
        return out, True
    return state.evolve(inst.operation, qargs=qargs), False


def profile_entanglement_counts(
    qc: QuantumCircuit,
    config: EntanglementCountConfig | None = None,
) -> dict[str, Any]:
    """Profile active entangled-qubit counts for a circuit.

    The returned dictionary includes an initial snapshot, sampled or every-step
    trace points, peak count fields, entropy aggregates, and a skipped status
    when the circuit exceeds ``config.max_qubits``.
    """

    resolved_config = config or EntanglementCountConfig()
    if qc.num_qubits > resolved_config.max_qubits:
        return {
            "status": "skipped_too_many_qubits",
            "qubits": int(qc.num_qubits),
            "required_entanglement_limit": int(qc.num_qubits),
            "entanglement_count_mode": resolved_config.mode,
            "entanglement_count_strategy": "every_quantum_step"
            if resolved_config.mode == "hard"
            else "sampled_quantum_steps",
        }

    instructions = _trim_terminal_ops(qc)
    quantum_instruction_indices = _iter_quantum_instruction_indices(instructions)
    if not quantum_instruction_indices:
        return {
            "status": "no_quantum_evolution",
            "qubits": int(qc.num_qubits),
            "entanglement_count_mode": resolved_config.mode,
        }

    contains_midcircuit_collapse = any(
        instructions[idx].operation.name in (_MEASUREMENT_OPS | _RESET_OPS)
        for idx in quantum_instruction_indices
    )
    simulation_mode = "density_matrix" if contains_midcircuit_collapse else "statevector"
    if simulation_mode == "statevector":
        state: Any = Statevector.from_int(0, dims=(2,) * qc.num_qubits)
    else:
        state = DensityMatrix.from_int(0, dims=(2,) * qc.num_qubits)

    selected_indices = (
        set(quantum_instruction_indices)
        if resolved_config.mode == "hard"
        else _sample_instruction_indices(quantum_instruction_indices, resolved_config.max_snapshots)
    )

    strategy = "every_quantum_step" if resolved_config.mode == "hard" else "sampled_quantum_steps"
    initial_metrics = _step_metrics(
        state,
        qc.num_qubits,
        entropy_threshold=resolved_config.entropy_threshold,
        pure_state_segment=True,
    )
    trace: list[dict[str, Any]] = [
        {
            "instruction_index": -1,
            "quantum_step": 0,
            "operation": "__initial_state__",
            "qargs": [],
            "pure_state_segment": True,
            **initial_metrics,
        }
    ]

    seen_nonunitary = False
    quantum_step = 0
    for inst_index, inst in enumerate(instructions):
        if inst.operation.num_qubits <= 0 or inst.operation.name == "barrier":
            continue

        qargs = _qargs_for_instruction(qc, inst)
        state, nonunitary = _apply_instruction(state, inst, qargs)
        quantum_step += 1
        if nonunitary:
            seen_nonunitary = True
        pure_state_segment = not seen_nonunitary

        if inst_index not in selected_indices:
            continue
        metrics = _step_metrics(
            state,
            qc.num_qubits,
            entropy_threshold=resolved_config.entropy_threshold,
            pure_state_segment=pure_state_segment,
        )
        trace.append(
            {
                "instruction_index": int(inst_index),
                "quantum_step": int(quantum_step),
                "operation": inst.operation.name,
                "qargs": [int(idx) for idx in qargs],
                "pure_state_segment": bool(pure_state_segment),
                **metrics,
            }
        )

    peak_active = max(point["active_entangled_qubits"] for point in trace)
    peak_mean = max(point["mean_single_qubit_entropy"] for point in trace)
    peak_total = max(point["total_single_qubit_entropy"] for point in trace)
    peak_mw_q = max(point["meyer_wallach_q"] for point in trace)
    final_point = trace[-1]

    return {
        "status": "ok",
        "qubits": int(qc.num_qubits),
        "simulation_mode": simulation_mode,
        "contains_midcircuit_collapse": bool(contains_midcircuit_collapse),
        "entanglement_count_mode": resolved_config.mode,
        "entanglement_count_strategy": strategy,
        "max_snapshots": int(resolved_config.max_snapshots),
        "snapshots_recorded": int(len(trace)),
        "tracked_quantum_steps": int(len(quantum_instruction_indices)),
        "initial_active_entangled_qubits": int(initial_metrics["active_entangled_qubits"]),
        "initial_active_entanglement_qubits": int(initial_metrics["active_entangled_qubits"]),
        "peak_active_entangled_qubits": int(peak_active),
        "peak_active_entanglement_qubits": int(peak_active),
        "final_active_entangled_qubits": int(final_point["active_entangled_qubits"]),
        "final_active_entanglement_qubits": int(final_point["active_entangled_qubits"]),
        "initial_mean_single_qubit_entropy": float(initial_metrics["mean_single_qubit_entropy"]),
        "peak_mean_single_qubit_entropy": float(peak_mean),
        "final_mean_single_qubit_entropy": float(final_point["mean_single_qubit_entropy"]),
        "initial_total_single_qubit_entropy": float(initial_metrics["total_single_qubit_entropy"]),
        "peak_total_single_qubit_entropy": float(peak_total),
        "final_total_single_qubit_entropy": float(final_point["total_single_qubit_entropy"]),
        "initial_meyer_wallach_q": float(initial_metrics["meyer_wallach_q"]),
        "peak_meyer_wallach_q": float(peak_mw_q),
        "final_meyer_wallach_q": float(final_point["meyer_wallach_q"]),
        "trace": trace,
    }

