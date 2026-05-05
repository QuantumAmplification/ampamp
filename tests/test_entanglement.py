import sys
from pathlib import Path

from qiskit import QuantumCircuit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ampamp import EntanglementCountConfig, profile_entanglement_counts  # noqa: E402


def _entangling_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(0)
    qc.x(1)
    return qc


def test_light_entanglement_count_samples_checkpoints():
    result = profile_entanglement_counts(
        _entangling_circuit(),
        EntanglementCountConfig.light(max_qubits=2, max_snapshots=1),
    )

    assert result["status"] == "ok"
    assert result["entanglement_count_mode"] == "light"
    assert result["entanglement_count_strategy"] == "sampled_quantum_steps"
    assert result["snapshots_recorded"] < result["tracked_quantum_steps"] + 1


def test_hard_entanglement_count_counts_every_step():
    result = profile_entanglement_counts(
        _entangling_circuit(),
        EntanglementCountConfig.hard(max_qubits=2),
    )

    assert result["status"] == "ok"
    assert result["entanglement_count_mode"] == "hard"
    assert result["entanglement_count_strategy"] == "every_quantum_step"
    assert result["snapshots_recorded"] == result["tracked_quantum_steps"] + 1
    assert result["peak_active_entangled_qubits"] == 2
    assert result["peak_active_entanglement_qubits"] == 2


def test_entanglement_count_respects_hardware_limit():
    result = profile_entanglement_counts(
        _entangling_circuit(),
        EntanglementCountConfig.hard(max_qubits=1),
    )

    assert result["status"] == "skipped_too_many_qubits"
    assert result["entanglement_count_mode"] == "hard"
    assert result["required_entanglement_limit"] == 2
