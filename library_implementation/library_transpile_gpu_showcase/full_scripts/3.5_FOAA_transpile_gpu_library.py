import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ampamp.foqa import FOQAEngine
from ampamp.transpilation import TranspilationProfileConfig, TranspilationProfiler
from _shared_gpu_library import Logger, run_interactive_scenario_repl, transpile_for_hardware

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)


def _build_proxy(n_qubits=32, mizel_c=1.4, m_content=1):
    return FOQAEngine(0.5).build_proxy_sequence(int(n_qubits), mizel_c=float(mizel_c), m_content=int(m_content))


def run_scenario_a(n_qubits=32, mizel_c=1.4, m_content=1):
    print("\n" + "=" * 70)
    print("SCENARIO A: FOAA PROXY BASELINE")
    print("=" * 70)
    qc = _build_proxy(n_qubits=n_qubits, mizel_c=mizel_c, m_content=m_content)
    _, depth, ops = transpile_for_hardware(qc, None, ["cx", "id", "rz", "sx", "x"], 3)
    print(f"n={n_qubits}, c={mizel_c}, m={m_content}, depth={depth}, cx={int(ops.get('cx', 0))}")


def run_scenario_b(n_qubits=20, mizel_c=1.4, m_content=1):
    print("\n" + "=" * 70)
    print("SCENARIO B: FOAA LINEAR ROUTING")
    print("=" * 70)
    qc = _build_proxy(n_qubits=n_qubits, mizel_c=mizel_c, m_content=m_content)
    linear = [[i, i + 1] for i in range(qc.num_qubits - 1)] + [[i + 1, i] for i in range(qc.num_qubits - 1)]
    _, d_all, o_all = transpile_for_hardware(qc, None, ["cx", "id", "rz", "sx", "x"], 3)
    _, d_lin, o_lin = transpile_for_hardware(qc, linear, ["cx", "id", "rz", "sx", "x"], 3)
    print(f"all-to-all depth={d_all}, cx={int(o_all.get('cx', 0))}")
    print(f"linear depth={d_lin}, cx={int(o_lin.get('cx', 0))}")


def run_scenario_c(n_qubits=24, mizel_c=1.4, m_content=2):
    print("\n" + "=" * 70)
    print("SCENARIO C: FOAA OPTIMIZATION COMPARISON")
    print("=" * 70)
    qc = _build_proxy(n_qubits=n_qubits, mizel_c=mizel_c, m_content=m_content)
    _, d0, o0 = transpile_for_hardware(qc, None, ["cx", "id", "rz", "sx", "x"], 0)
    _, d3, o3 = transpile_for_hardware(qc, None, ["cx", "id", "rz", "sx", "x"], 3)
    print(f"opt0 depth={d0}, cx={int(o0.get('cx', 0))}")
    print(f"opt3 depth={d3}, cx={int(o3.get('cx', 0))}")


def run_scenario_p(n_qubits=20, mizel_c=1.5, m_content=2):
    print("\n" + "=" * 70)
    print("SCENARIO P: FOAA PROFILER SCORE")
    print("=" * 70)
    qc = _build_proxy(n_qubits=n_qubits, mizel_c=mizel_c, m_content=m_content)
    linear = [[i, i + 1] for i in range(qc.num_qubits - 1)] + [[i + 1, i] for i in range(qc.num_qubits - 1)]
    metrics = TranspilationProfiler(
        TranspilationProfileConfig(coupling_map_edges=linear, basis_gates=("cx", "id", "rz", "sx", "x"))
    ).profile_circuit(qc)
    print(f"routing_swaps={metrics['routing_swaps']}, final_cnots={metrics['final_cnots']}")
    print(f"total_time_ns={metrics['total_time_ns']}, score={metrics['hardware_penalty_score']}")


if __name__ == "__main__":
    output_filepath = os.path.join(_RESULT_DIR, "terminal_output.log")
    logger = Logger(output_filepath)
    sys.stdout = logger
    scenarios = [
        ("A", lambda: run_scenario_a(32, 1.4, 1)),
        ("B", lambda: run_scenario_b(20, 1.4, 1)),
        ("C", lambda: run_scenario_c(24, 1.4, 2)),
        ("P", lambda: run_scenario_p(20, 1.5, 2)),
    ]
    interactive = [
        ("A", run_scenario_a),
        ("B", run_scenario_b),
        ("C", run_scenario_c),
        ("P", run_scenario_p),
    ]
    try:
        for _, fn in scenarios:
            fn()
        run_interactive_scenario_repl(interactive, sep="=" * 70)
    finally:
        logger.log.close()
        sys.stdout = logger.terminal
        print("FOAA library GPU suite complete.")
