import numpy as np
from qiskit import QuantumCircuit, transpile
import importlib.util
import os
import sys
import ast
import inspect
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from aer_publishability_gpu import (
    parse_publishability_cli,
    prepare_backend_validation_artifacts,
    render_backend_validation_summary,
    run_cli_scenario,
    save_figure_with_metadata,
    wrap_scenarios,
)

_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
STANDARD_QUBITS = 6
_AER_GPU_HINT = (
    "This script now requires qiskit-aer-gpu on a CUDA-capable Linux/x86_64 system "
    "(or qiskit-aer-gpu-cu11 for CUDA 11)."
)

def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _parse_cli_value(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.strip().lower()
        if lowered == "true": return True
        if lowered == "false": return False
        return raw

def _parse_kwargs_text(raw):
    kwargs = {}
    text = raw.strip()
    if not text: return kwargs
    for chunk in text.split(","):
        item = chunk.strip()
        if not item: continue
        if "=" not in item: raise ValueError(f"Expected key=value pair, got '{item}'")
        key, value = item.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs

def _format_signature_help(fn):
    sig = inspect.signature(fn)
    parts = []
    for name, param in sig.parameters.items():
        if param.default is inspect._empty: parts.append(name)
        else: parts.append(f"{name}={param.default!r}")
    return ", ".join(parts) if parts else "(no parameters)"

def run_interactive_scenario_repl(scenarios, *, sep):
    if not sys.stdin.isatty(): return
    scenario_pairs = list(scenarios)
    scenario_map = {label.upper(): fn for label, fn in scenario_pairs}
    print(f"\n{sep}\nINTERACTIVE RE-RUN MODE\n{sep}")
    print("Select a scenario for rerun with custom parameters.")
    while True:
        try: choice = input("\nScenario label to rerun (or Enter to exit): ").strip().upper()
        except EOFError: break
        if not choice: break
        if choice not in scenario_map: continue
        fn = scenario_map[choice]
        print(f"Selected scenario {choice}: {fn.__name__}")
        try:
            raw_kwargs = input("Custom parameters (e.g. max_k=8): ")
            kwargs = _parse_kwargs_text(raw_kwargs)
            fn(**kwargs)
        except Exception as exc:
            print(f"\nScenario {choice} failed: {exc}")

class IQAECompiler:
    def __init__(self, n_qubits, good_indices):
        self.n = n_qubits
        self.good_indices = good_indices
        self.N = 2**n_qubits
        self.M = len(good_indices)
        self.p = self.M / self.N

    def get_oracle(self):
        qc = QuantumCircuit(self.n)
        for index in self.good_indices:
            good_bin = format(index, f'0{self.n}b')[::-1]
            for i, bit in enumerate(good_bin):
                if bit == '0': qc.x(i)
            qc.h(self.n - 1)
            qc.mcx(list(range(self.n - 1)), self.n - 1)
            qc.h(self.n - 1)
            for i, bit in enumerate(good_bin):
                if bit == '0': qc.x(i)
        return qc

    def get_diffusion(self):
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        qc.x(range(self.n))
        qc.h(self.n - 1)
        qc.mcx(list(range(self.n - 1)), self.n - 1)
        qc.h(self.n - 1)
        qc.x(range(self.n))
        qc.h(range(self.n))
        return qc

    def generate_ideal_circuit(self, k: int):
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        if k > 0:
            oracle = self.get_oracle()
            diff = self.get_diffusion()
            for _ in range(k):
                qc.append(oracle.to_instruction(label="Oracle"), range(self.n))
                qc.append(diff.to_instruction(label="Diffusion"), range(self.n))
        return qc.decompose()

def transpile_for_hardware(qc, coupling_map=None, basis_gates=None, optimization_level=3):
    t_qc = transpile(qc, coupling_map=coupling_map, basis_gates=basis_gates, optimization_level=optimization_level)
    return t_qc, t_qc.depth(), t_qc.count_ops()

def run_scenario_a(n_qubits=5, good_indices=[10], max_i=4):
    """
    Scenario A: Exponential Depth Blowup in IQAE
    """
    print("\n" + "=" * 70)
    print("SCENARIO A: EXPONENTIAL DEPTH BLOWUP IN IQAE (GPU)")
    print("=" * 70)
    
    compiler = IQAECompiler(n_qubits, good_indices)
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    
    print(f"Target Qubits: {n_qubits}")
    print(f"{'i':<3} | {'k_i (2^i)':<10} | {'Logical Depth':<15} | {'Physical Depth':<15} | {'CX Count'}")
    print("-" * 75)
    
    for i in range(max_i + 1):
        k = 2**i if i > 0 else 0
        raw_qc = compiler.generate_ideal_circuit(k)
        ldepth = raw_qc.depth()
        _, pdepth, ops = transpile_for_hardware(raw_qc, basis_gates=basis_gates, optimization_level=3)
        print(f"{i:<3} | {k:<10} | {ldepth:<15} | {pdepth:<15} | {ops.get('cx', 0)}")

def run_scenario_b(n_qubits=5, k=4):
    """
    Scenario B: Routing Penalty for High-k Circuits
    """
    print("\n" + "=" * 70)
    print("SCENARIO B: ROUTING PENALTY FOR HIGH-k CIRCUITS (GPU)")
    print("=" * 70)
    
    compiler = IQAECompiler(n_qubits, [10])
    raw_qc = compiler.generate_ideal_circuit(k)
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    linear_map = [[i, i+1] for i in range(n_qubits-1)] + [[i+1, i] for i in range(n_qubits-1)]
    heavy_hex_map = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3]]
    
    print(f"Circuit: n={n_qubits}, k={k}")
    
    archs = {"All-to-All": None, "Heavy-Hex": heavy_hex_map, "Linear": linear_map}
    for name, cmap in archs.items():
        _, depth, ops = transpile_for_hardware(raw_qc, coupling_map=cmap, basis_gates=basis_gates)
        print(f"{name:<15} -> Depth: {depth:<6} | CX: {ops.get('cx', 0)}")

def run_scenario_c(n_qubits=4, max_i=3):
    """
    Scenario C: Noise Degradation over IQAE Schedule
    """
    print("\n" + "=" * 70)
    print("SCENARIO C: NOISE DEGRADATION OVER IQAE SCHEDULE (GPU)")
    print("=" * 70)
    
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError as exc:
        print("Missing qiskit_aer module. Cannot run noise simulation.")
        print(f"{_AER_GPU_HINT} Original error: {type(exc).__name__}: {exc}")
        return
        
    error_rate = 0.01
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate, 2), ['cx'])
    sim = AerSimulator(noise_model=noise_model, device="GPU")
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    
    good_indices = [0]
    compiler = IQAECompiler(n_qubits, good_indices)
    
    print(f"Noise Profile: {error_rate*100}% depolarizing error on CX")
    print(f"{'i':<3} | {'k_i':<5} | {'Physical Depth':<15} | {'Ideal Prob':<12} | {'Noisy Prob'}")
    print("-" * 65)
    
    shots = 8192
    for i in range(max_i + 1):
        k = 2**i if i > 0 else 0
        raw_qc = compiler.generate_ideal_circuit(k)
        raw_qc.measure_all()
        t_qc, depth, _ = transpile_for_hardware(raw_qc, basis_gates=basis_gates)
        
        ideal_prob = np.sin((2*k + 1) * np.arcsin(np.sqrt(compiler.p)))**2
        
        job = sim.run(t_qc, shots=shots)
        counts = job.result().get_counts()
        target = format(0, f'0{n_qubits}b')
        noisy_prob = counts.get(target, 0) / shots
        
        print(f"{i:<3} | {k:<5} | {depth:<15} | {ideal_prob:<12.4f} | {noisy_prob:.4f}")

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
    
    cli_argv, publishability = parse_publishability_cli(
        sys.argv[1:],
        default_max_qubits=20,
        default_shots=1024,
        default_log_dir=_RESULT_DIR,
    )
    prepare_backend_validation_artifacts(publishability)
    
    print("Starting IQAE GPU Transpilation Benchmark...")
    print(f"Saving all results to: {output_filepath}\n")
    print(publishability.summary())
    
    scen_funcs = [
        ("A", lambda: run_scenario_a()),
        ("B", lambda: run_scenario_b()),
        ("C", lambda: run_scenario_c()),
    ]
    interactive_funcs = [
        ("A", run_scenario_a),
        ("B", run_scenario_b),
        ("C", run_scenario_c),
    ]
    scenarios = wrap_scenarios(scen_funcs, module_globals=globals(), config=publishability)
    interactive_wrapped = wrap_scenarios(interactive_funcs, module_globals=globals(), config=publishability)

    try:
        cli_executed = run_cli_scenario(cli_argv, interactive_wrapped)
        if not cli_executed:
            for label, fn in scenarios:
                try: fn()
                except Exception as e: print(f"Scenario {label} failed: {e}")
            run_interactive_scenario_repl(interactive_wrapped, sep="=" * 70)
    finally:
        render_backend_validation_summary(publishability)
        logger.log.close()
        sys.stdout = logger.terminal
        print("\nBenchmark complete.")
