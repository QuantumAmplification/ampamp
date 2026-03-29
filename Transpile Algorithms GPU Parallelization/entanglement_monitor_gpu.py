from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, entropy, partial_trace

_AER_IMPORT_ERROR: Exception | None = None
try:
    from qiskit_aer import AerSimulator
    import qiskit_aer.library  # noqa: F401
except Exception as exc:  # pragma: no cover
    AerSimulator = None  # type: ignore[assignment]
    _AER_IMPORT_ERROR = exc


_TERMINAL_ONLY_OPS = {"barrier", "measure", "reset"}
_MEASUREMENT_OPS = {"measure"}
_RESET_OPS = {"reset"}
_EPSILON = 1e-9
_COMPLEX_EPSILON = 1e-12
_AER_GPU_DEVICE = "GPU"
_AER_GPU_HINT = (
    "Entanglement/state tracing in this folder expects qiskit-aer-gpu on a CUDA-capable Linux/x86_64 system "
    "(or qiskit-aer-gpu-cu11 for CUDA 11). If you see a libnvidia-ml.so.1 error, "
    "the NVIDIA driver runtime is missing on this machine."
)


@dataclass(frozen=True)
class EntanglementConfig:
    enabled: bool = False
    # <= 0 means no limit.
    max_qubits: int = 0
    max_snapshots: int = 64
    entanglement_every_step: bool = False
    state_enabled: bool = False
    # <= 0 means no limit.
    state_max_qubits: int = 0
    state_top_k: int = 16
    state_include_full: bool = False
    state_every_step: bool = False
    # <= 0 means no limit.
    state_full_density_max_qubits: int = 0


@dataclass(frozen=True)
class _SnapshotSpec:
    label: str
    instruction_index: int
    quantum_step: int
    operation: str
    qargs: list[int]
    pure_state_segment: bool
    capture_phase: str


def _limit_allows(limit: int, num_qubits: int) -> bool:
    return int(limit) <= 0 or num_qubits <= int(limit)


def _trim_terminal_ops(qc: QuantumCircuit) -> list[Any]:
    data = list(qc.data)
    end = len(data)
    while end > 0 and data[end - 1].operation.name in _TERMINAL_ONLY_OPS:
        end -= 1
    return data[:end]


def _iter_quantum_instruction_indices(instructions: list[Any]) -> list[int]:
    out: list[int] = []
    for idx, inst in enumerate(instructions):
        if inst.operation.num_qubits <= 0:
            continue
        if inst.operation.name == "barrier":
            continue
        out.append(idx)
    return out


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


def _complex_payload(value: complex) -> dict[str, float]:
    raw = complex(value)
    real = float(np.real(raw))
    imag = float(np.imag(raw))
    if abs(real) < _COMPLEX_EPSILON:
        real = 0.0
    if abs(imag) < _COMPLEX_EPSILON:
        imag = 0.0
    return {"real": real, "imag": imag}


def _canonicalize_qubit_state(alpha: complex, beta: complex) -> tuple[complex, complex]:
    vec = np.asarray([alpha, beta], dtype=complex)
    for value in vec:
        if abs(value) > _COMPLEX_EPSILON:
            vec *= np.exp(-1j * np.angle(value))
            break
    vec[np.abs(vec) < _COMPLEX_EPSILON] = 0.0
    return complex(vec[0]), complex(vec[1])


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
        if abs(value) < _EPSILON:
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


def _local_qubit_state_payloads(reduced_states: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for qubit, reduced in enumerate(reduced_states):
        data = np.asarray(reduced.data, dtype=complex)
        purity = float(np.real_if_close(np.trace(data @ data)))
        prob_0 = float(np.real_if_close(data[0, 0]))
        prob_1 = float(np.real_if_close(data[1, 1]))
        coherence = complex(data[0, 1])
        item: dict[str, Any] = {
            "qubit": int(qubit),
            "purity": float(max(0.0, min(1.0, purity))),
            "probability_0": prob_0,
            "probability_1": prob_1,
            "coherence": _complex_payload(coherence),
            "alpha_beta_defined": False,
        }
        if purity >= 1.0 - 1e-6:
            eigenvalues, eigenvectors = np.linalg.eigh(data)
            index = int(np.argmax(np.real(eigenvalues)))
            alpha, beta = _canonicalize_qubit_state(eigenvectors[0, index], eigenvectors[1, index])
            item["alpha_beta_defined"] = True
            item["alpha"] = _complex_payload(alpha)
            item["beta"] = _complex_payload(beta)
        payloads.append(item)
    return payloads


def _top_probability_items(probabilities: dict[str, float], limit: int) -> list[tuple[str, float]]:
    ordered = sorted(
        ((basis, float(prob)) for basis, prob in probabilities.items() if float(prob) > _EPSILON),
        key=lambda item: (-item[1], item[0]),
    )
    return ordered[: max(1, int(limit))]


def _state_snapshot_payload(
    state: Any,
    reduced_states: list[Any],
    *,
    num_qubits: int,
    config: EntanglementConfig,
) -> dict[str, Any]:
    local_qubit_states = _local_qubit_state_payloads(reduced_states)
    if isinstance(state, Statevector):
        probabilities = state.probabilities_dict()
        amplitudes = state.to_dict()
        top_entries = []
        for basis, probability in _top_probability_items(probabilities, config.state_top_k):
            amplitude = complex(amplitudes.get(basis, 0.0))
            top_entries.append(
                {
                    "basis": basis,
                    "probability": float(probability),
                    "magnitude": float(abs(amplitude)),
                    "phase_radians": float(np.angle(amplitude)),
                    "amplitude": _complex_payload(amplitude),
                }
            )
        payload: dict[str, Any] = {
            "state_kind": "statevector",
            "top_basis_states": top_entries,
            "local_qubit_states": local_qubit_states,
        }
        if config.state_include_full and _limit_allows(config.state_max_qubits, num_qubits):
            payload["full_amplitudes"] = {
                basis: _complex_payload(value)
                for basis, value in sorted(amplitudes.items())
                if abs(value) > _EPSILON
            }
            payload["full_probabilities"] = {
                basis: float(probability)
                for basis, probability in sorted(probabilities.items())
                if float(probability) > _EPSILON
            }
        return payload

    probabilities = state.probabilities_dict()
    payload = {
        "state_kind": "density_matrix",
        "top_basis_probabilities": [
            {"basis": basis, "probability": float(probability)}
            for basis, probability in _top_probability_items(probabilities, config.state_top_k)
        ],
        "local_qubit_states": local_qubit_states,
    }
    if config.state_include_full and _limit_allows(config.state_max_qubits, num_qubits):
        payload["full_probabilities"] = {
            basis: float(probability)
            for basis, probability in sorted(probabilities.items())
            if float(probability) > _EPSILON
        }
        if _limit_allows(config.state_full_density_max_qubits, num_qubits):
            payload["full_density_matrix"] = {
                basis: _complex_payload(value)
                for basis, value in sorted(state.to_dict().items())
                if abs(value) > _EPSILON
            }
    return payload


def _state_trace_entry(
    state: Any,
    reduced_states: list[Any],
    *,
    instruction_index: int,
    quantum_step: int,
    operation: str,
    qargs: list[int],
    pure_state_segment: bool,
    capture_phase: str,
    num_qubits: int,
    config: EntanglementConfig,
) -> dict[str, Any]:
    return {
        "instruction_index": int(instruction_index),
        "quantum_step": int(quantum_step),
        "operation": operation,
        "qargs": [int(idx) for idx in qargs],
        "pure_state_segment": bool(pure_state_segment),
        "capture_phase": capture_phase,
        "state_snapshot": _state_snapshot_payload(
            state,
            reduced_states,
            num_qubits=num_qubits,
            config=config,
        ),
    }


def _step_metrics(
    reduced_states: list[Any],
    num_qubits: int,
    *,
    pure_state_segment: bool,
) -> dict[str, Any]:
    if num_qubits <= 1:
        return {
            "single_qubit_entropies": [0.0] * num_qubits,
            "mean_single_qubit_entropy": 0.0,
            "max_single_qubit_entropy": 0.0,
            "total_single_qubit_entropy": 0.0,
            "active_entangled_qubits": 0,
            "meyer_wallach_q": 0.0,
            "metric_interpretation": "pure_state_entanglement" if pure_state_segment else "mixed_state_entropy_proxy",
        }

    entropies = _single_qubit_entropies(reduced_states)
    purities = _single_qubit_purities(reduced_states)
    total_entropy = float(sum(entropies))
    mean_entropy = total_entropy / float(num_qubits)
    max_entropy = max(entropies) if entropies else 0.0
    active_qubits = sum(1 for value in entropies if value > 1e-6)
    mean_purity = sum(purities) / float(num_qubits)
    meyer_wallach_q = float(max(0.0, min(1.0, 2.0 * (1.0 - mean_purity))))

    return {
        "single_qubit_entropies": [float(v) for v in entropies],
        "mean_single_qubit_entropy": float(mean_entropy),
        "max_single_qubit_entropy": float(max_entropy),
        "total_single_qubit_entropy": float(total_entropy),
        "active_entangled_qubits": int(active_qubits),
        "meyer_wallach_q": meyer_wallach_q,
        "metric_interpretation": "pure_state_entanglement" if pure_state_segment else "mixed_state_entropy_proxy",
    }


def _snapshot_label(capture_phase: str, instruction_index: int, quantum_step: int) -> str:
    return f"{capture_phase}__inst_{instruction_index}__step_{quantum_step}"


def _append_gpu_snapshot(circuit: QuantumCircuit, simulation_mode: str, label: str) -> None:
    if simulation_mode == "statevector":
        circuit.save_statevector(label=label)
    else:
        circuit.save_density_matrix(label=label)


def _coerce_saved_state(raw: Any, simulation_mode: str) -> Any:
    if isinstance(raw, (Statevector, DensityMatrix)):
        return raw
    if simulation_mode == "statevector":
        return Statevector(raw)
    return DensityMatrix(raw)


def _run_gpu_snapshot_simulation(circuit: QuantumCircuit, simulation_mode: str) -> dict[str, Any]:
    if AerSimulator is None:
        if _AER_IMPORT_ERROR is None:
            raise RuntimeError(_AER_GPU_HINT)
        raise RuntimeError(
            f"{_AER_GPU_HINT} Original error: {type(_AER_IMPORT_ERROR).__name__}: {_AER_IMPORT_ERROR}"
        )
    backend = AerSimulator(method=simulation_mode, device=_AER_GPU_DEVICE)
    return backend.run(circuit, shots=1).result().data(0)


def profile_circuit_entanglement(
    qc: QuantumCircuit,
    *,
    config: EntanglementConfig,
) -> dict[str, Any]:
    if not config.enabled:
        return {"status": "disabled"}

    if not _limit_allows(config.max_qubits, qc.num_qubits):
        return {
            "status": "skipped_too_many_qubits",
            "qubits": int(qc.num_qubits),
            "required_entanglement_limit": int(qc.num_qubits),
            "state_status": "skipped_too_many_qubits" if config.state_enabled else "disabled",
            "required_state_limit": int(qc.num_qubits) if config.state_enabled else None,
        }

    full_instructions = list(qc.data)
    instructions = _trim_terminal_ops(qc)
    quantum_instruction_indices = _iter_quantum_instruction_indices(instructions)
    if not quantum_instruction_indices:
        return {
            "status": "no_quantum_evolution",
            "qubits": int(qc.num_qubits),
            "state_status": "no_quantum_evolution" if config.state_enabled else "disabled",
        }

    full_quantum_indices = _iter_quantum_instruction_indices(full_instructions)
    contains_any_collapse = any(
        full_instructions[idx].operation.name in (_MEASUREMENT_OPS | _RESET_OPS)
        for idx in full_quantum_indices
    )
    contains_midcircuit_collapse = any(
        instructions[idx].operation.name in (_MEASUREMENT_OPS | _RESET_OPS)
        for idx in quantum_instruction_indices
    )
    simulation_mode = "density_matrix" if contains_any_collapse else "statevector"

    state_status = "disabled"
    state_required_limit: int | None = None
    if config.state_enabled:
        if not _limit_allows(config.state_max_qubits, qc.num_qubits):
            state_status = "skipped_too_many_qubits"
            state_required_limit = int(qc.num_qubits)
        else:
            state_status = "ok"

    entanglement_selected_indices = (
        set(quantum_instruction_indices)
        if config.entanglement_every_step
        else _sample_instruction_indices(quantum_instruction_indices, config.max_snapshots)
    )
    state_selected_indices: set[int] = set()
    if state_status == "ok":
        state_selected_indices = (
            set(quantum_instruction_indices)
            if config.state_every_step
            else set(entanglement_selected_indices)
        )

    instrumented = qc.copy_empty_like()
    initial_label = _snapshot_label("initial_state", -1, 0)
    _append_gpu_snapshot(instrumented, simulation_mode, initial_label)

    trace_specs: list[_SnapshotSpec] = []
    state_specs: list[_SnapshotSpec] = []
    seen_nonunitary = False
    quantum_step = 0
    final_pre_measurement_label: str | None = None
    final_pre_measurement_step = 0
    final_post_terminal_label: str | None = None
    final_post_terminal_step: int | None = None
    last_main_quantum_index = quantum_instruction_indices[-1]

    for inst_index, inst in enumerate(full_instructions):
        if inst.operation.num_qubits <= 0 or inst.operation.name == "barrier":
            instrumented.append(inst.operation, inst.qubits, inst.clbits)
            continue

        qargs = _qargs_for_instruction(qc, inst)
        pure_state_segment_before = not seen_nonunitary
        is_main_instruction = inst_index < len(instructions)

        if state_status == "ok" and inst.operation.name in _MEASUREMENT_OPS:
            label = _snapshot_label("before_measurement", inst_index, quantum_step)
            _append_gpu_snapshot(instrumented, simulation_mode, label)
            state_specs.append(
                _SnapshotSpec(
                    label=label,
                    instruction_index=inst_index,
                    quantum_step=quantum_step,
                    operation=inst.operation.name,
                    qargs=qargs,
                    pure_state_segment=pure_state_segment_before,
                    capture_phase="before_measurement",
                )
            )

        instrumented.append(inst.operation, inst.qubits, inst.clbits)
        quantum_step += 1
        if inst.operation.name in (_MEASUREMENT_OPS | _RESET_OPS):
            seen_nonunitary = True

        pure_state_segment = not seen_nonunitary
        if is_main_instruction and inst_index in entanglement_selected_indices:
            label = _snapshot_label("entanglement", inst_index, quantum_step)
            _append_gpu_snapshot(instrumented, simulation_mode, label)
            trace_specs.append(
                _SnapshotSpec(
                    label=label,
                    instruction_index=inst_index,
                    quantum_step=quantum_step,
                    operation=inst.operation.name,
                    qargs=qargs,
                    pure_state_segment=pure_state_segment,
                    capture_phase="after_instruction",
                )
            )

        if state_status == "ok":
            if is_main_instruction and inst_index in state_selected_indices:
                label = _snapshot_label("state", inst_index, quantum_step)
                _append_gpu_snapshot(instrumented, simulation_mode, label)
                state_specs.append(
                    _SnapshotSpec(
                        label=label,
                        instruction_index=inst_index,
                        quantum_step=quantum_step,
                        operation=inst.operation.name,
                        qargs=qargs,
                        pure_state_segment=pure_state_segment,
                        capture_phase="after_instruction",
                    )
                )
            elif not is_main_instruction:
                label = _snapshot_label("terminal", inst_index, quantum_step)
                _append_gpu_snapshot(instrumented, simulation_mode, label)
                state_specs.append(
                    _SnapshotSpec(
                        label=label,
                        instruction_index=inst_index,
                        quantum_step=quantum_step,
                        operation=inst.operation.name,
                        qargs=qargs,
                        pure_state_segment=pure_state_segment,
                        capture_phase="after_terminal_instruction",
                    )
                )
                final_post_terminal_label = label
                final_post_terminal_step = quantum_step

            if is_main_instruction and inst_index == last_main_quantum_index:
                final_pre_measurement_label = _snapshot_label("final_pre_measurement", inst_index, quantum_step)
                _append_gpu_snapshot(instrumented, simulation_mode, final_pre_measurement_label)
                final_pre_measurement_step = quantum_step

    snapshot_payloads = _run_gpu_snapshot_simulation(instrumented, simulation_mode)
    initial_state = _coerce_saved_state(snapshot_payloads[initial_label], simulation_mode)
    initial_reduced_states = _single_qubit_reduced_states(initial_state, qc.num_qubits)
    initial_metrics = _step_metrics(initial_reduced_states, qc.num_qubits, pure_state_segment=True)

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

    for spec in trace_specs:
        state = _coerce_saved_state(snapshot_payloads[spec.label], simulation_mode)
        reduced_states = _single_qubit_reduced_states(state, qc.num_qubits)
        metrics = _step_metrics(reduced_states, qc.num_qubits, pure_state_segment=spec.pure_state_segment)
        trace.append(
            {
                "instruction_index": int(spec.instruction_index),
                "quantum_step": int(spec.quantum_step),
                "operation": spec.operation,
                "qargs": [int(idx) for idx in spec.qargs],
                "pure_state_segment": bool(spec.pure_state_segment),
                **metrics,
            }
        )

    state_trace: list[dict[str, Any]] = []
    if state_status == "ok":
        state_trace.append(
            _state_trace_entry(
                initial_state,
                initial_reduced_states,
                instruction_index=-1,
                quantum_step=0,
                operation="__initial_state__",
                qargs=[],
                pure_state_segment=True,
                capture_phase="initial_state",
                num_qubits=qc.num_qubits,
                config=config,
            )
        )
        for spec in state_specs:
            state = _coerce_saved_state(snapshot_payloads[spec.label], simulation_mode)
            reduced_states = _single_qubit_reduced_states(state, qc.num_qubits)
            state_trace.append(
                _state_trace_entry(
                    state,
                    reduced_states,
                    instruction_index=spec.instruction_index,
                    quantum_step=spec.quantum_step,
                    operation=spec.operation,
                    qargs=spec.qargs,
                    pure_state_segment=spec.pure_state_segment,
                    capture_phase=spec.capture_phase,
                    num_qubits=qc.num_qubits,
                    config=config,
                )
            )

    peak_mean = max(point["mean_single_qubit_entropy"] for point in trace)
    peak_total = max(point["total_single_qubit_entropy"] for point in trace)
    peak_mw_q = max(point["meyer_wallach_q"] for point in trace)
    final_point = trace[-1]
    first_nonunitary_step = next(
        (point["quantum_step"] for point in trace if not point["pure_state_segment"]),
        None,
    )

    result: dict[str, Any] = {
        "status": "ok",
        "qubits": int(qc.num_qubits),
        "simulation_mode": simulation_mode,
        "contains_midcircuit_collapse": bool(contains_midcircuit_collapse),
        "max_snapshots": int(config.max_snapshots),
        "entanglement_every_step": bool(config.entanglement_every_step),
        "snapshots_recorded": int(len(trace)),
        "tracked_quantum_steps": int(len(quantum_instruction_indices)),
        "first_nonunitary_step": first_nonunitary_step,
        "initial_mean_single_qubit_entropy": float(initial_metrics["mean_single_qubit_entropy"]),
        "initial_total_single_qubit_entropy": float(initial_metrics["total_single_qubit_entropy"]),
        "initial_meyer_wallach_q": float(initial_metrics["meyer_wallach_q"]),
        "peak_mean_single_qubit_entropy": float(peak_mean),
        "peak_total_single_qubit_entropy": float(peak_total),
        "peak_meyer_wallach_q": float(peak_mw_q),
        "final_mean_single_qubit_entropy": float(final_point["mean_single_qubit_entropy"]),
        "final_total_single_qubit_entropy": float(final_point["total_single_qubit_entropy"]),
        "final_meyer_wallach_q": float(final_point["meyer_wallach_q"]),
        "trace": trace,
    }

    if config.state_enabled:
        result["state_status"] = state_status
        result["state_top_k"] = int(config.state_top_k)
        result["state_include_full"] = bool(config.state_include_full)
        result["state_every_step"] = bool(config.state_every_step)
        if state_status == "ok":
            result["initial_state_snapshot"] = _state_snapshot_payload(
                initial_state,
                initial_reduced_states,
                num_qubits=qc.num_qubits,
                config=config,
            )
            if final_pre_measurement_label is not None:
                final_pre_state = _coerce_saved_state(snapshot_payloads[final_pre_measurement_label], simulation_mode)
                final_pre_reduced_states = _single_qubit_reduced_states(final_pre_state, qc.num_qubits)
                result["final_pre_measurement_quantum_step"] = int(final_pre_measurement_step)
                result["final_pre_measurement_state_snapshot"] = _state_snapshot_payload(
                    final_pre_state,
                    final_pre_reduced_states,
                    num_qubits=qc.num_qubits,
                    config=config,
                )
            if final_post_terminal_label is not None and final_post_terminal_step is not None:
                final_post_state = _coerce_saved_state(snapshot_payloads[final_post_terminal_label], simulation_mode)
                final_post_reduced_states = _single_qubit_reduced_states(final_post_state, qc.num_qubits)
                result["final_post_terminal_quantum_step"] = int(final_post_terminal_step)
                result["final_post_terminal_state_snapshot"] = _state_snapshot_payload(
                    final_post_state,
                    final_post_reduced_states,
                    num_qubits=qc.num_qubits,
                    config=config,
                )
            result["state_snapshots_recorded"] = int(len(state_trace))
            result["state_trace"] = state_trace
            result["state_measurement_boundaries_recorded"] = int(
                sum(1 for point in state_trace if point.get("capture_phase") == "before_measurement")
            )
        elif state_required_limit is not None:
            result["required_state_limit"] = int(state_required_limit)
    else:
        result["state_status"] = "disabled"

    return result
