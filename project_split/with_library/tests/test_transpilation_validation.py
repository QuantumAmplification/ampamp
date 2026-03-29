import json
import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ampamp.transpilation_validation import (
    BackendValidationConfig,
    BackendValidationRunner,
    ValidationLogConfig,
    ValidationNoiseConfig,
)


class _FakeResult:
    def __init__(self, counts):
        self._counts = counts

    def get_counts(self, _compiled):
        return self._counts


class _FakeJob:
    def __init__(self, counts):
        self._counts = counts

    def result(self):
        return _FakeResult(self._counts)


class _FakeAerSimulator:
    def __init__(self, *args, **kwargs):
        self.noise_model = kwargs.get("noise_model")

    def run(self, _compiled, shots=1024):
        if self.noise_model is None:
            return _FakeJob({"00": shots // 2, "11": shots - (shots // 2)})
        return _FakeJob({"00": int(0.6 * shots), "11": shots - int(0.6 * shots)})


def test_backend_validation_ideal_returns_pass_or_fail_with_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr("ampamp.transpilation_validation.AerSimulator", _FakeAerSimulator)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    cfg = BackendValidationConfig(
        shots=256,
        seed=7,
        max_qubits=4,
        noise=ValidationNoiseConfig(noise_level="ideal"),
        logging=ValidationLogConfig(enabled=True, log_dir=str(tmp_path), log_name="validation.jsonl"),
    )
    runner = BackendValidationRunner(cfg)
    result = runner.validate_circuit("bell", qc)

    assert result["name"] == "bell"
    assert result["status"] in {"pass", "fail"}
    assert "metrics" in result and "tvd" in result["metrics"]
    assert result["metrics"]["tvd"] == pytest.approx(0.0, abs=1e-12)

    log_path = tmp_path / "validation.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["name"] == "bell"


def test_backend_validation_custom_noise_changes_distribution(monkeypatch):
    monkeypatch.setattr("ampamp.transpilation_validation.AerSimulator", _FakeAerSimulator)
    qc = QuantumCircuit(1)
    qc.h(0)

    cfg = BackendValidationConfig(
        shots=512,
        seed=11,
        max_qubits=3,
        noise=ValidationNoiseConfig(
            noise_level="custom",
            one_qubit_error=0.03,
            two_qubit_error=0.05,
            readout_error=0.08,
        ),
    )
    runner = BackendValidationRunner(cfg)
    result = runner.validate_circuit("single_qubit", qc)

    assert result["noise_level"] == "custom"
    assert result["metrics"]["tvd"] >= 0.0
    assert "ideal" in result["counts"] and "noisy" in result["counts"]


def test_backend_validation_rejects_too_many_qubits(monkeypatch):
    monkeypatch.setattr("ampamp.transpilation_validation.AerSimulator", _FakeAerSimulator)
    qc = QuantumCircuit(5)
    cfg = BackendValidationConfig(max_qubits=4)
    runner = BackendValidationRunner(cfg)

    with pytest.raises(ValueError, match="exceeding max_qubits"):
        runner.validate_circuit("too_big", qc)
