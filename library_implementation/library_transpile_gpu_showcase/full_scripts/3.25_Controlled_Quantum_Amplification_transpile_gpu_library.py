import os
import sys

from qiskit import QuantumCircuit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ampamp.grover import GroverEngine
from ampamp.transpilation import TranspilationProfileConfig, TranspilationProfiler
from _shared_gpu_library import Logger, run_interactive_scenario_repl, transpile_for_hardware

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)


def _build_controlled_amplifier(n_qubits=5, rounds=4):
    n_qubits = int(n_qubits)
    rounds = int(rounds)
    work = n_qubits
    anc = 1
    total = work + anc

    grover = GroverEngine(work, [0])
    oracle = grover.get_oracle()
    diffusion = grover.get_diffusion()

    qc = QuantumCircuit(total)
    ctrl = total - 1
    qc.h(range(work))
    qc.h(ctrl)

    for _ in range(rounds):
        qc.cx(ctrl, 0)
        qc.append(oracle.control(1), [ctrl] + list(range(work)))
        qc.append(diffusion.control(1), [ctrl] + list(range(work)))
        qc.rz(0.17, ctrl)

    return qc


def run_scenario_a(n_qubits=5, rounds=4):
    print("\n" + "=" * 70)
    print("SCENARIO A: CONTROLLED AMPLIFICATION BASELINE")
    print("=" * 70)
    qc = _build_controlled_amplifier(n_qubits=n_qubits, rounds=rounds)
    _, depth, ops = transpile_for_hardware(qc, None, ["cx", "id", "rz", "sx", "x"], 3)
    print(f"n={n_qubits}, rounds={rounds}, depth={depth}, cx={int(ops.get('cx', 0))}")


def run_scenario_b(n_qubits=5, rounds=4):
    print("\n" + "=" * 70)
    print("SCENARIO B: TOPOLOGY ROUTING PENALTY")
    print("=" * 70)
    qc = _build_controlled_amplifier(n_qubits=n_qubits, rounds=rounds)
    linear = [[i, i + 1] for i in range(qc.num_qubits - 1)] + [[i + 1, i] for i in range(qc.num_qubits - 1)]
    _, d_all, o_all = transpile_for_hardware(qc, None, ["cx", "id", "rz", "sx", "x"], 3)
    _, d_lin, o_lin = transpile_for_hardware(qc, linear, ["cx", "id", "rz", "sx", "x"], 3)
    print(f"all-to-all depth={d_all}, cx={int(o_all.get('cx', 0))}")
    print(f"linear depth={d_lin}, cx={int(o_lin.get('cx', 0))}")
    if d_all > 0:
        print(f"depth multiplier: {d_lin / d_all:.2f}x")


def run_scenario_c(n_qubits=5, rounds=6):
    print("\n" + "=" * 70)
    print("SCENARIO C: OPTIMIZATION LEVEL COMPARISON")
    print("=" * 70)
    qc = _build_controlled_amplifier(n_qubits=n_qubits, rounds=rounds)
    _, d0, o0 = transpile_for_hardware(qc, None, ["cx", "id", "rz", "sx", "x"], 0)
    _, d3, o3 = transpile_for_hardware(qc, None, ["cx", "id", "rz", "sx", "x"], 3)
    print(f"opt0 depth={d0}, cx={int(o0.get('cx', 0))}")
    print(f"opt3 depth={d3}, cx={int(o3.get('cx', 0))}")


def run_scenario_p(n_qubits=5, rounds=6):
    print("\n" + "=" * 70)
    print("SCENARIO P: TRANSPILATION PROFILER SCORE")
    print("=" * 70)
    qc = _build_controlled_amplifier(n_qubits=n_qubits, rounds=rounds)
    linear = [[i, i + 1] for i in range(qc.num_qubits - 1)] + [[i + 1, i] for i in range(qc.num_qubits - 1)]
    profiler = TranspilationProfiler(
        TranspilationProfileConfig(
            coupling_map_edges=linear,
            basis_gates=("cx", "id", "rz", "sx", "x"),
        )
    )
    metrics = profiler.profile_circuit(qc)
    print(f"routing_swaps={metrics['routing_swaps']}, final_cnots={metrics['final_cnots']}")
    print(f"total_time_ns={metrics['total_time_ns']}, score={metrics['hardware_penalty_score']}")


if __name__ == "__main__":
    output_filepath = os.path.join(_RESULT_DIR, "terminal_output.log")
    logger = Logger(output_filepath)
    sys.stdout = logger
    scenarios = [
        ("A", lambda: run_scenario_a(5, 4)),
        ("B", lambda: run_scenario_b(5, 4)),
        ("C", lambda: run_scenario_c(5, 6)),
        ("P", lambda: run_scenario_p(5, 6)),
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
        print("Controlled quantum amplification library GPU suite complete.")
