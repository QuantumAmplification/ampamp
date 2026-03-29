import sys
from pathlib import Path

from qiskit import QuantumCircuit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ampamp.transpilation import (
    HardwareCostWeights,
    TranspilationBatchProfiler,
    TranspilationProfileConfig,
    TranspilationProfiler,
)


def test_transpilation_profiler_returns_core_metrics():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    config = TranspilationProfileConfig(
        coupling_map_edges=[[0, 1], [1, 2]],
        cost_weights=HardwareCostWeights(time=1.0, cnot=10.0, distance=5.0),
    )
    profiler = TranspilationProfiler(config)
    metrics = profiler.profile_circuit(qc)

    expected_keys = {
        "logical_depth",
        "logical_gates",
        "num_qubits",
        "basis_gates",
        "initial_distance_penalty",
        "post_routing_depth",
        "routing_swaps",
        "translation_gates",
        "translation_cnots",
        "post_optimization_depth",
        "final_gates",
        "final_cnots",
        "total_time_ns",
        "hardware_penalty_score",
    }

    assert expected_keys.issubset(metrics.keys())
    assert metrics["num_qubits"] == 3
    assert metrics["hardware_penalty_score"] >= 0.0


def test_transpilation_batch_profiler_profiles_named_circuits():
    qc_a = QuantumCircuit(2)
    qc_a.h(0)
    qc_a.cx(0, 1)

    qc_b = QuantumCircuit(2)
    qc_b.x(0)
    qc_b.cz(0, 1)

    profiler = TranspilationBatchProfiler()
    results = profiler.profile_many({"a": qc_a, "b": qc_b})

    assert set(results.keys()) == {"a", "b"}
    assert "final_cnots" in results["a"]
    assert "hardware_penalty_score" in results["b"]
