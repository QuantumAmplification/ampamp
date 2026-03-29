"""Backend validation utilities for transpiled quantum circuits.

This module adds publishability-style checks for ideal/noisy execution,
count-distribution drift, and structured JSONL logging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from qiskit import QuantumCircuit, transpile

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
except Exception:  # pragma: no cover
    AerSimulator = None  # type: ignore[assignment]
    NoiseModel = None  # type: ignore[assignment]
    ReadoutError = None  # type: ignore[assignment]
    depolarizing_error = None  # type: ignore[assignment]


_NOISE_PRESETS: dict[str, tuple[float, float, float]] = {
    "ideal": (0.0, 0.0, 0.0),
    "light": (0.001, 0.01, 0.02),
    "medium": (0.003, 0.03, 0.05),
    "heavy": (0.01, 0.08, 0.10),
}


@dataclass(frozen=True)
class ValidationNoiseConfig:
    """Noise model configuration for backend validation."""

    noise_level: str = "ideal"
    one_qubit_error: float = 0.0
    two_qubit_error: float = 0.0
    readout_error: float = 0.0


@dataclass(frozen=True)
class ValidationLogConfig:
    """Structured logging configuration."""

    enabled: bool = False
    log_dir: Optional[str] = None
    log_name: str = "backend_validation_log.jsonl"

    @property
    def log_path(self) -> Optional[Path]:
        if not self.enabled or not self.log_dir:
            return None
        return Path(self.log_dir) / self.log_name


@dataclass(frozen=True)
class BackendValidationConfig:
    """Execution and quality thresholds for backend validation."""

    shots: int = 1024
    seed: int = 42
    basis_gates: tuple[str, ...] = ("cx", "id", "rz", "sx", "x")
    optimization_level: int = 3
    max_qubits: int = 20
    tvd_threshold: float = 0.15
    noise: ValidationNoiseConfig = ValidationNoiseConfig()
    logging: ValidationLogConfig = ValidationLogConfig()


class BackendValidationRunner:
    """Runs ideal-vs-noisy validation on transpiled circuits."""

    def __init__(self, config: Optional[BackendValidationConfig] = None):
        if AerSimulator is None:
            raise ImportError("qiskit-aer is required for BackendValidationRunner.")
        self.config = config or BackendValidationConfig()

    @staticmethod
    def _normalize_counts(counts: Mapping[str, int]) -> dict[str, float]:
        total = float(sum(int(v) for v in counts.values()))
        if total <= 0.0:
            return {}
        return {k: float(v) / total for k, v in counts.items()}

    @staticmethod
    def total_variation_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
        support = set(left) | set(right)
        return 0.5 * sum(abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0))) for k in support)

    @staticmethod
    def _dominant_probability(normalized_counts: Mapping[str, float]) -> float:
        if not normalized_counts:
            return 0.0
        return float(max(normalized_counts.values()))

    @staticmethod
    def _ensure_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
        if any(inst.operation.name == "measure" for inst in circuit.data):
            return circuit
        qc = circuit.copy()
        qc.measure_all()
        return qc

    def _resolve_noise_values(self) -> tuple[str, float, float, float]:
        noise_key = str(self.config.noise.noise_level).strip().lower()
        if noise_key in {"none", "off"}:
            noise_key = "ideal"

        if noise_key == "custom":
            return (
                "custom",
                float(self.config.noise.one_qubit_error),
                float(self.config.noise.two_qubit_error),
                float(self.config.noise.readout_error),
            )

        if noise_key not in _NOISE_PRESETS:
            supported = ", ".join(sorted(_NOISE_PRESETS.keys()) + ["custom"])
            raise ValueError(f"Unsupported noise level '{noise_key}'. Use one of: {supported}")

        preset_1q, preset_2q, preset_ro = _NOISE_PRESETS[noise_key]
        # Allow explicit overrides in non-custom modes.
        one_q = float(self.config.noise.one_qubit_error or preset_1q)
        two_q = float(self.config.noise.two_qubit_error or preset_2q)
        readout = float(self.config.noise.readout_error or preset_ro)
        return noise_key, one_q, two_q, readout

    def _build_noise_model(self, *, one_qubit_error_rate: float, two_qubit_error_rate: float, readout_error_rate: float):
        if NoiseModel is None or depolarizing_error is None or ReadoutError is None:
            raise ImportError("qiskit-aer noise components are unavailable.")

        model = NoiseModel()
        if one_qubit_error_rate > 0.0:
            model.add_all_qubit_quantum_error(depolarizing_error(one_qubit_error_rate, 1), ["x", "sx", "rz", "id"])
        if two_qubit_error_rate > 0.0:
            model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error_rate, 2), ["cx", "cz", "ecr", "swap"])
        if readout_error_rate > 0.0:
            readout = ReadoutError([[1 - readout_error_rate, readout_error_rate], [readout_error_rate, 1 - readout_error_rate]])
            model.add_all_qubit_readout_error(readout)
        return model

    def _append_log_record(self, payload: Mapping[str, Any]) -> None:
        log_path = self.config.logging.log_path
        if log_path is None:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    def validate_circuit(self, name: str, circuit: QuantumCircuit) -> dict[str, Any]:
        """Validate one circuit under ideal and noisy Aer execution."""
        if circuit.num_qubits > int(self.config.max_qubits):
            raise ValueError(
                f"Circuit '{name}' uses {circuit.num_qubits} qubits, exceeding max_qubits={self.config.max_qubits}."
            )

        measured = self._ensure_measurements(circuit)
        compiled = transpile(
            measured,
            basis_gates=list(self.config.basis_gates),
            optimization_level=int(self.config.optimization_level),
            seed_transpiler=int(self.config.seed),
        )

        ideal_backend = AerSimulator(seed_simulator=int(self.config.seed))
        ideal_result = ideal_backend.run(compiled, shots=int(self.config.shots)).result()
        ideal_counts = ideal_result.get_counts(compiled)
        ideal_norm = self._normalize_counts(ideal_counts)

        noise_key, one_q, two_q, readout = self._resolve_noise_values()
        if noise_key == "ideal" and one_q == 0.0 and two_q == 0.0 and readout == 0.0:
            noisy_counts = dict(ideal_counts)
        else:
            noise_model = self._build_noise_model(
                one_qubit_error_rate=one_q,
                two_qubit_error_rate=two_q,
                readout_error_rate=readout,
            )
            noisy_backend = AerSimulator(noise_model=noise_model, seed_simulator=int(self.config.seed))
            noisy_result = noisy_backend.run(compiled, shots=int(self.config.shots)).result()
            noisy_counts = noisy_result.get_counts(compiled)

        noisy_norm = self._normalize_counts(noisy_counts)
        tvd = self.total_variation_distance(ideal_norm, noisy_norm)

        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "status": "pass" if tvd <= float(self.config.tvd_threshold) else "fail",
            "num_qubits": int(circuit.num_qubits),
            "shots": int(self.config.shots),
            "noise_level": noise_key,
            "noise_params": {
                "one_qubit_error": float(one_q),
                "two_qubit_error": float(two_q),
                "readout_error": float(readout),
            },
            "metrics": {
                "tvd": float(tvd),
                "tvd_threshold": float(self.config.tvd_threshold),
                "ideal_support_size": int(len(ideal_norm)),
                "noisy_support_size": int(len(noisy_norm)),
                "ideal_dominant_probability": self._dominant_probability(ideal_norm),
                "noisy_dominant_probability": self._dominant_probability(noisy_norm),
            },
            "counts": {
                "ideal": dict(ideal_counts),
                "noisy": dict(noisy_counts),
            },
            "depth": int(compiled.depth()),
            "gates": int(sum(compiled.count_ops().values())),
        }

        self._append_log_record(payload)
        return payload

    def validate_many(self, circuits: Mapping[str, QuantumCircuit]) -> dict[str, dict[str, Any]]:
        """Validate a dictionary of named circuits."""
        return {name: self.validate_circuit(name, qc) for name, qc in circuits.items()}
