import ast
import inspect
import os
import sys
import traceback

import numpy as np
from qiskit import QuantumCircuit, transpile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ampamp.grover import GroverEngine
from ampamp.transpilation import TranspilationProfiler, TranspilationProfileConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
STANDARD_QUBITS = 20
_AER_GPU_HINT = (
    "This script requires qiskit-aer-gpu on CUDA-capable Linux/x86_64 "
    "(or qiskit-aer-gpu-cu11 for CUDA 11)."
)


def _parse_cli_value(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return raw


def _parse_kwargs_text(raw):
    kwargs = {}
    text = raw.strip()
    if not text:
        return kwargs
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected key=value pair, got '{item}'")
        key, value = item.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _format_signature_help(fn):
    sig = inspect.signature(fn)
    parts = []
    for name, param in sig.parameters.items():
        if param.default is inspect._empty:
            parts.append(name)
        else:
            parts.append(f"{name}={param.default!r}")
    return ", ".join(parts) if parts else "(no parameters)"


def run_interactive_scenario_repl(scenarios, *, sep):
    if not sys.stdin.isatty():
        return
    scenario_pairs = list(scenarios)
    scenario_map = {label.upper(): fn for label, fn in scenario_pairs}
    print(f"\n{sep}")
    print("INTERACTIVE RE-RUN MODE")
    print(sep)
    print("Select a scenario for rerun with custom parameters.")
    print(f"Available labels: {', '.join(label for label, _ in scenario_pairs)}")
    print("Enter a scenario label such as A or P, or press Enter to exit.")
    while True:
        try:
            choice = input("\nScenario label to rerun: ").strip().upper()
        except EOFError:
            print("\nInteractive mode closed.")
            return
        if not choice:
            print("Interactive rerun mode finished.")
            return
        if choice not in scenario_map:
            print(f"Unknown scenario '{choice}'. Available: {', '.join(scenario_map)}")
            continue
        fn = scenario_map[choice]
        print(f"Selected scenario {choice}: {fn.__name__}")
        print(f"Parameters: {_format_signature_help(fn)}")
        print("Enter overrides as comma-separated key=value pairs.")
        print("Example: n_qubits=32")
        print("Press Enter to use defaults.")
        try:
            raw_kwargs = input("Custom parameters: ")
            kwargs = _parse_kwargs_text(raw_kwargs)
            print(f"\nExecuting scenario {choice} with parameters: {kwargs if kwargs else 'defaults'}")
            fn(**kwargs)
        except Exception as exc:
            print(f"\nScenario {choice} failed during custom execution.")
            print(f"Error: {exc}")
            traceback.print_exc()


class GroverCompiler:
    """Library-backed compiler for physical Grover circuits."""

    def __init__(self, n_qubits, good_indices):
        self.n = int(n_qubits)
        self.good_indices = list(good_indices)
        self.engine = GroverEngine(self.n, self.good_indices)
        self.N = self.engine.N
        self.M = self.engine.M
        self.p = self.engine.solution_density
        self.k_optimal = self.engine.k_optimal

    def get_initialization(self):
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        return qc

    def get_oracle(self):
        return self.engine.get_oracle()

    def get_diffusion(self):
        return self.engine.get_diffusion()

    def generate_ideal_circuit(self):
        qc = self.get_initialization()
        if self.k_optimal > 0:
            oracle = self.get_oracle()
            diff = self.get_diffusion()
            for _ in range(self.k_optimal):
                qc.append(oracle.to_instruction(label="Oracle"), range(self.n))
                qc.append(diff.to_instruction(label="Diffusion"), range(self.n))
        return qc.decompose()


def transpile_for_hardware(qc, coupling_map=None, basis_gates=None, optimization_level=3):
    transpiled_qc = transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )
    return transpiled_qc, transpiled_qc.depth(), transpiled_qc.count_ops()


def run_scenario_a(n_qubits=6, good_indices=[10, 25]):
    print("\n" + "=" * 70)
    print("SCENARIO A: UNROLLING BASELINE")
    print("=" * 70)

    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=good_indices)
    compiler.k_optimal = 1
    raw_qc = compiler.generate_ideal_circuit()
    ideal_depth = raw_qc.depth()
    ideal_ops = dict(raw_qc.count_ops())

    print(f"1. Ideal Logical Circuit (k=1)")
    print(f"   - Target Qubits: {n_qubits}")
    print(f"   - Ideal Depth: {ideal_depth}")
    print(f"   - Ideal Operations: {ideal_ops}")

    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    print(f"\n2. Hardware Constraints")
    print(f"   - Basis Gates: {basis_gates}")
    print(f"   - Coupling Map: None (All-to-All)")

    print("\n3. Transpiling... (Optimization Level 3)")
    t_qc, t_depth, t_ops = transpile_for_hardware(raw_qc, None, basis_gates, 3)

    print(f"\n4. Transpilation Results")
    print(f"   - Physical Circuit Depth: {t_depth}")
    print(f"   - Physical Operations: {dict(t_ops)}")
    depth_ratio = t_depth / max(1, ideal_depth)
    print(f"\n   -> Depth Blowup Factor: {depth_ratio:.2f}x")
    return t_qc


def run_scenario_b(n_qubits=6, good_indices=[10, 25]):
    print("\n" + "=" * 70)
    print("SCENARIO B: RESTRICTED TOPOLOGICAL ROUTING")
    print("=" * 70)

    linear_map = [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]
    heavy_hex_map = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3], [4, 5], [5, 4]]

    architectures = {
        "All-to-All": None,
        "Heavy-Hex Lattice": heavy_hex_map,
        "Linear Topology": linear_map,
    }

    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=good_indices)
    compiler.k_optimal = 1
    raw_qc = compiler.generate_ideal_circuit()
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']

    results = {}
    for name, cmap in architectures.items():
        print(f"\n--- Transpiling for {name} ---")
        _, t_depth, t_ops = transpile_for_hardware(raw_qc, cmap, basis_gates, 3)
        cx_count = int(t_ops.get('cx', 0))
        results[name] = {"depth": int(t_depth), "cx": cx_count, "ops": dict(t_ops)}
        print(f"Depth: {t_depth}")
        print(f"Total CX Gates: {cx_count}")
    return results


def run_scenario_c(n_qubits=5, k=2):
    print("\n" + "=" * 70)
    print("SCENARIO C: COMPILER OPTIMIZATION COMPARISON")
    print("=" * 70)

    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=[10])
    compiler.k_optimal = int(k)
    raw_qc = compiler.generate_ideal_circuit()
    heavy_hex_map_5 = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3]]
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']

    _, depth_0, ops_0 = transpile_for_hardware(raw_qc, heavy_hex_map_5, basis_gates, 0)
    _, depth_3, ops_3 = transpile_for_hardware(raw_qc, heavy_hex_map_5, basis_gates, 3)

    cx_0 = int(ops_0.get('cx', 0))
    cx_3 = int(ops_3.get('cx', 0))
    print(f"Level 0 -> depth={depth_0}, cx={cx_0}")
    print(f"Level 3 -> depth={depth_3}, cx={cx_3}")


def run_scenario_d(max_qubits=8):
    print("\n" + "=" * 70)
    print("SCENARIO D: DIMENSIONAL SCALING ANALYSIS")
    print("=" * 70)

    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    for n in range(3, int(max_qubits) + 1):
        compiler = GroverCompiler(n_qubits=n, good_indices=[0])
        raw_qc = compiler.generate_ideal_circuit()
        linear_map = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
        _, depth, ops = transpile_for_hardware(raw_qc, linear_map, basis_gates, 3)
        print(f"n={n:2d} k*={compiler.k_optimal:2d} depth={depth:5d} cx={int(ops.get('cx',0)):5d}")


def run_scenario_e(n_qubits=4, max_k=10):
    print("\n" + "=" * 70)
    print("SCENARIO E: NOISE-INDUCED PERFORMANCE DEGRADATION")
    print("=" * 70)

    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except Exception as exc:
        print("Missing/incompatible qiskit_aer. Cannot run.")
        print(f"{_AER_GPU_HINT} Original error: {type(exc).__name__}: {exc}")
        return

    error_rate = 0.01
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate, 2), ['cx'])

    try:
        simulator = AerSimulator(noise_model=noise_model, device="GPU")
    except Exception:
        simulator = AerSimulator(noise_model=noise_model)

    heavy_hex_map_4 = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1]]
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']

    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=[0])
    shots = 4096

    for k in range(int(max_k) + 1):
        compiler.k_optimal = k
        raw_qc = compiler.generate_ideal_circuit()
        raw_qc.measure_all()
        t_qc, depth, ops = transpile_for_hardware(raw_qc, heavy_hex_map_4, basis_gates, 3)
        counts = simulator.run(t_qc, shots=shots).result().get_counts()
        success = counts.get("0" * int(n_qubits), 0) / shots
        print(f"k={k:2d} depth={depth:4d} cx={int(ops.get('cx',0)):4d} noisy_success={success:.4f}")


def run_scenario_f(qubit_sizes=[3, 5, 7]):
    print("\n" + "=" * 70)
    print("SCENARIO F: FAULT-TOLERANT COMPILATION OVERHEAD")
    print("=" * 70)

    ft_basis_gates = ['h', 's', 'sdg', 'cx', 't', 'tdg']
    for n in qubit_sizes:
        compiler = GroverCompiler(n_qubits=int(n), good_indices=[0])
        compiler.k_optimal = 1
        raw_qc = compiler.generate_ideal_circuit()
        _, depth, ops = transpile_for_hardware(raw_qc, None, ft_basis_gates, 3)
        total_t = int(ops.get('t', 0) + ops.get('tdg', 0))
        print(f"n={n:2d} ft_depth={depth:5d} t_count={total_t:5d}")


def run_scenario_g(n_qubits=STANDARD_QUBITS):
    print("\n" + "=" * 70)
    print("SCENARIO G: ANCILLA-SPACE TRADE-OFF")
    print("=" * 70)

    compiler = GroverCompiler(n_qubits=int(n_qubits), good_indices=[0])
    raw_qc = compiler.generate_ideal_circuit()
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']

    _, depth_constrained, ops_constrained = transpile_for_hardware(raw_qc, None, basis_gates, 3)
    print(f"Constrained ({n_qubits}q): depth={depth_constrained}, cx={int(ops_constrained.get('cx',0))}")


def run_scenario_h(n_qubits=6, M=48):
    print("\n" + "=" * 70)
    print("SCENARIO H: HIGH-DENSITY REGIME ANALYSIS")
    print("=" * 70)

    N = 2 ** int(n_qubits)
    if not (N / 2 < int(M) < N):
        raise ValueError(f"Scenario H requires N/2 < M < N. Got N={N}, M={M}.")

    good_indices = list(range(int(M)))
    compiler = GroverCompiler(n_qubits=int(n_qubits), good_indices=good_indices)
    compiler.k_optimal = 1

    classical_prob = compiler.p
    theta = 2 * np.arcsin(np.sqrt(classical_prob))
    theoretical_prob = np.sin((3 * theta) / 2) ** 2

    print(f"Classical p={classical_prob:.4f}, theoretical k=1 success={theoretical_prob:.4f}")


def run_profiling_benchmark(n_qubits=5):
    print("\n" + "=" * 70)
    print("UNIFIED HARDWARE PROFILER BENCHMARK")
    print("=" * 70)

    compiler = GroverCompiler(n_qubits=int(n_qubits), good_indices=[0])
    raw_qc = compiler.generate_ideal_circuit()

    coupling_edges = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3]]
    profiler = TranspilationProfiler(
        TranspilationProfileConfig(
            coupling_map_edges=coupling_edges,
            basis_gates=('cx', 'id', 'rz', 'sx', 'x'),
            single_qubit_ns=20,
            two_qubit_ns=100,
            optimize_optimization_level=3,
        )
    )

    metrics = profiler.profile_circuit(raw_qc)
    for k in [
        'initial_distance_penalty',
        'routing_swaps',
        'post_routing_depth',
        'translation_gates',
        'translation_cnots',
        'post_optimization_depth',
        'final_cnots',
        'total_time_ns',
        'hardware_penalty_score',
    ]:
        print(f"{k}: {metrics.get(k)}")


if __name__ == "__main__":
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    output_filepath = os.path.join(_RESULT_DIR, "terminal_output.log")
    os.chdir(_RESULT_DIR)

    logger = Logger(output_filepath)
    sys.stdout = logger

    print("Starting library-backed Grover GPU transpilation benchmark suite...")
    print(f"Saving all results to: {output_filepath}\n")

    scenarios = [
        ("A", lambda: run_scenario_a(n_qubits=6, good_indices=[10, 25])),
        ("B", lambda: run_scenario_b(n_qubits=6, good_indices=[10, 25])),
        ("C", lambda: run_scenario_c(n_qubits=5, k=2)),
        ("D", lambda: run_scenario_d(max_qubits=8)),
        ("E", lambda: run_scenario_e(n_qubits=4, max_k=10)),
        ("F", lambda: run_scenario_f(qubit_sizes=[3, 5, 7])),
        ("G", lambda: run_scenario_g(n_qubits=10)),
        ("H", lambda: run_scenario_h(n_qubits=6, M=48)),
        ("P", lambda: run_profiling_benchmark(n_qubits=5)),
    ]

    interactive = [
        ("A", run_scenario_a),
        ("B", run_scenario_b),
        ("C", run_scenario_c),
        ("D", run_scenario_d),
        ("E", run_scenario_e),
        ("F", run_scenario_f),
        ("G", run_scenario_g),
        ("H", run_scenario_h),
        ("P", run_profiling_benchmark),
    ]

    try:
        for label, fn in scenarios:
            try:
                fn()
            except Exception:
                print(f"\nSCENARIO {label} EXECUTION FAILED")
                traceback.print_exc()

        run_interactive_scenario_repl(interactive, sep="=" * 70)
    finally:
        logger.log.close()
        sys.stdout = logger.terminal
        print("\nBenchmark suite complete. Results saved.")
