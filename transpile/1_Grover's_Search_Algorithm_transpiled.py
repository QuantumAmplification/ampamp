import os
import sys
import numpy as np

# Ensure we can import ampamp from src
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from ampamp.grover import GroverEngine
from ampamp.transpiler import (
    AlgorithmProfiler,
    transpile_for_hardware,
    HardwareProfiles,
    HardwareReport
)

_RESULT_DIR = os.path.join(_HERE, "[RESULT]1_Grover_Modular_Transpilation")
report = HardwareReport(_RESULT_DIR)

def main():
    n_qubits = 6
    good_indices = [10, 25]
    engine = GroverEngine(n_qubits, good_indices)
    profiler = AlgorithmProfiler("Grover")

    print(f"Profiling Grover's Algorithm with {n_qubits} qubits and targets {good_indices}")
    print(f"Optimal Iterations (k*): {engine.k_optimal}\n")

    # --- Scenario A: Unrolling Baseline ---
    def scenario_a():
        # Force k=1 for baseline
        qc = engine.construct_circuit(iterations=1, decompose=True)
        t_qc, metrics = transpile_for_hardware(
            qc,
            basis_gates=HardwareProfiles.ibm_basis_gates(),
            optimization_level=3
        )
        return (t_qc, metrics), {"description": "Unrolling Baseline (k=1)"}

    # --- Scenario B: Topological Routing ---
    def scenario_b(topology_name="Linear"):
        qc = engine.construct_circuit(iterations=1, decompose=True)
        cmap = None
        if topology_name == "Linear":
            cmap = HardwareProfiles.linear_map(n_qubits)
        elif topology_name == "Heavy-Hex":
            cmap = HardwareProfiles.heavy_hex_subgraph_6()
            
        t_qc, metrics = transpile_for_hardware(
            qc,
            coupling_map=cmap,
            basis_gates=HardwareProfiles.ibm_basis_gates(),
            optimization_level=3
        )
        return (t_qc, metrics), {"topology": topology_name}

    # Register and Run Scenarios
    profiler.add_scenario("A", scenario_a)
    profiler.add_scenario("B_Linear", lambda: scenario_b("Linear"))
    profiler.add_scenario("B_HeavyHex", lambda: scenario_b("Heavy-Hex"))

    # Execute
    res_a = profiler.run_scenario("A")
    res_b_lin = profiler.run_scenario("B_Linear")
    
    # Log results
    report.log_result(res_a.label, res_a.metrics, res_a.extra)
    report.log_result(res_b_lin.label, res_b_lin.metrics, res_b_lin.extra)

    # Analysis
    swaps = AlgorithmProfiler.calculate_swap_overhead(res_a.metrics, res_b_lin.metrics)
    penalty = AlgorithmProfiler.calculate_depth_penalty(res_a.metrics, res_b_lin.metrics)

    print(f"Scenario A (All-to-All) Depth: {res_a.metrics.depth}")
    print(f"Scenario B (Linear) Depth: {res_b_lin.metrics.depth}")
    print(f"-> Routing Depth Penalty: {penalty:.2f}x")
    print(f"-> Estimated SWAPs inserted: {swaps}")

    print("\nModular Transpilation Complete. Results logged to:", report.log_path)
    print("\nSummary:")
    print(report.generate_summary_text())

if __name__ == "__main__":
    main()
