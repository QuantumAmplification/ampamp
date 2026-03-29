import pytest
from qiskit import QuantumCircuit
from ampamp.transpiler import (
    transpile_for_hardware, 
    CircuitMetrics, 
    AlgorithmProfiler, 
    HardwareReport
)
from ampamp.grover import GroverEngine

def test_transpile_for_hardware_basic():
    # Simple Bell state
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    t_qc, metrics = transpile_for_hardware(qc)
    
    assert isinstance(metrics, CircuitMetrics)
    assert metrics.depth >= qc.depth()
    assert metrics.qubits == 2
    assert metrics.gate_counts.get('cx', 0) == 1

def test_algorithm_profiler_with_grover():
    n_qubits = 4
    marked = [5]
    engine = GroverEngine(n_qubits, marked)
    
    profiler = AlgorithmProfiler("Grover")
    
    # Define a scenario using the engine
    def scenario_a(n=3):
        qc = engine.construct_circuit(1, decompose=True)
        return qc, {"n_qubits": n}
        
    profiler.add_scenario("ScenarioA", scenario_a)
    result = profiler.run_scenario("ScenarioA", n=4)
    
    assert result.label == "SCENARIOA"
    assert isinstance(result.metrics, CircuitMetrics)
    assert result.extra["n_qubits"] == 4

def test_hardware_report(tmp_path):
    report_dir = str(tmp_path / "reports")
    report = HardwareReport(report_dir)
    
    metrics = CircuitMetrics(depth=10, gate_counts={'cx': 5}, qubits=2, clbits=0, blowup_factor=2.0)
    report.log_result("TEST_SCENARIO", metrics, extra={"info": "test"})
    
    summary = report.generate_summary_text()
    assert "Scenario TEST_SCENARIO" in summary
    assert "Depth: 10" in summary
    assert "CX Count: 5" in summary
