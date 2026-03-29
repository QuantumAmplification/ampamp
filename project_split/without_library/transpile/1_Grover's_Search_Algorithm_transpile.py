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
from aer_publishability import (
    parse_publishability_cli,
    prepare_backend_validation_artifacts,
    render_backend_validation_summary,
    run_cli_scenario,
    save_figure_with_metadata,
    wrap_scenarios,
)

_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
STANDARD_QUBITS = 20


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _import_local_module(alias, filename):
    module_path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


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
    """
    Compiler class for generating pure physical hardware circuits for Grover's Algorithm.
    Stripped of all statevector diagnostics, focused strictly on gate and wire generation.
    """
    def __init__(self, n_qubits, good_indices):
        """
        Initializes the compiler with hardware targets in mind.
        
        Args:
            n_qubits (int): Total number of qubits in the register.
            good_indices (list of int): Computational-basis indices spanning H_Good.
        """
        self.n = n_qubits
        self.good_indices = good_indices
        self.N = 2**n_qubits
        self.M = len(good_indices)
        
        # Initial success probability p = M/N
        self.p = self.M / self.N
        
        # Optimal k* based on the theoretical formula
        if self.p == 0:
            self.k_optimal = 0
        else:
            theta0 = np.arcsin(np.sqrt(self.p))
            theta = 2 * theta0
            # k* ≈ floor(pi/(4*theta0) - 1/2) = floor(pi/(2*theta) - 0.5)
            self.k_optimal = int(np.floor(np.pi / (2 * theta) - 0.5))

    def get_initialization(self):
        """
        Step 1: The Initialization Logic (H^⊗n).
        Prepares the uniform superposition |All>.
        """
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        return qc

    def get_oracle(self):
        """
        Step 2: The Oracle Constructor (O).
        Uses X-gates and multi-controlled Z (via mcx) to flip the phase of target states.
        """
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
        """
        Step 3: The Diffusion Constructor (R).
        Reflects about the mean state using H, X, and mcx.
        """
        qc = QuantumCircuit(self.n)
        
        qc.h(range(self.n))
        qc.x(range(self.n))
        
        qc.h(self.n - 1)
        qc.mcx(list(range(self.n - 1)), self.n - 1)
        qc.h(self.n - 1)
        
        qc.x(range(self.n))
        qc.h(range(self.n))
        
        return qc

    def generate_ideal_circuit(self):
        """
        Builds the single perfectly sized, raw QuantumCircuit for exactly k* iterations.
        """
        qc = self.get_initialization()
        
        if self.k_optimal > 0:
            oracle = self.get_oracle()
            diff = self.get_diffusion()
            
            for _ in range(self.k_optimal):
                qc.append(oracle.to_instruction(label="Oracle"), range(self.n))
                qc.append(diff.to_instruction(label="Diffusion"), range(self.n))
                
        # Decompose the custom instructions so the transpiler can optimize internal gates.
        return qc.decompose()


def transpile_for_hardware(qc, coupling_map=None, basis_gates=None, optimization_level=3):
    """
    Passes the raw circuit into a transpilation function that applies a specific hardware 
    coupling map or fault-tolerant basis gate set.
    """
    transpiled_qc = transpile(
        qc, 
        coupling_map=coupling_map, 
        basis_gates=basis_gates, 
        optimization_level=optimization_level
    )
    
    depth = transpiled_qc.depth()
    ops = transpiled_qc.count_ops()
    
    return transpiled_qc, depth, ops

def run_scenario_a(n_qubits=6, good_indices=[10, 25]):
    """
    Scenario A: Unrolling Baseline

    Evaluates the gate-expansion overhead incurred when a logical Grover
    iterate is decomposed into a native hardware gate set.
    """
    print("\n" + "=" * 70)
    print("SCENARIO A: UNROLLING BASELINE")
    print("=" * 70)
    
    # Step 1: Construct the Ideal Logical Circuit for k=1 (one Grover step)
    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=good_indices)
    
    # Override k_optimal to exactly 1 for this scenario
    compiler.k_optimal = 1
    
    raw_qc = compiler.generate_ideal_circuit()
    ideal_depth = raw_qc.depth()
    ideal_ops = dict(raw_qc.count_ops())
    
    print(f"1. Ideal Logical Circuit (k=1)")
    print(f"   - Target Qubits: {n_qubits}")
    print(f"   - Ideal Depth: {ideal_depth}")
    print(f"   - Ideal Operations: {ideal_ops}")
    
    # Step 2: Define the Hardware "Alphabet"
    # A standard IBM-style basis gate set
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    
    print(f"\n2. Hardware Constraints")
    print(f"   - Basis Gates: {basis_gates}")
    print(f"   - Coupling Map: None (All-to-All assumes no routing penalty)")
    
    # Step 3: Execute the Unrolling
    print("\n3. Transpiling... (Highest Optimization Level 3)")
    t_qc, t_depth, t_ops = transpile_for_hardware(
        raw_qc, 
        coupling_map=None, 
        basis_gates=basis_gates, 
        optimization_level=3
    )
    
    # Step 4: Extract and Compare the Metrics
    print(f"\n4. Transpilation Results")
    print(f"   - Physical Circuit Depth: {t_depth}")
    print(f"   - Physical Operations: {dict(t_ops)}")
    
    depth_ratio = t_depth / ideal_depth
    print(f"\n   -> Depth Blowup Factor: {depth_ratio:.2f}x")
    logical_mcx_count = len(good_indices) + 1  # one MCX per marked state in oracle + one in diffusion
    print(f"   -> CNOT count for {logical_mcx_count} logical MCX gates: {t_ops.get('cx', 0)}")
    return t_qc

def run_scenario_b(n_qubits=6, good_indices=[10, 25]):
    """
    Scenario B: Topological Routing Constraints

    Quantifies the additional routing overhead induced by restricted device
    connectivity during transpilation.
    """
    print("\n" + "=" * 70)
    print("SCENARIO B: RESTRICTED TOPOLOGICAL ROUTING")
    print("=" * 70)
    
    # Step 1: Prepare the Architectures (The Coupling Maps)
    # Define coupling maps for a 6-qubit system
    linear_map = [[i, i+1] for i in range(n_qubits-1)] + [[i+1, i] for i in range(n_qubits-1)]
    # Approximation of 6-qubit heavy-hex sub-graph
    heavy_hex_map = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3], [4, 5], [5, 4]]
    
    architectures = {
        "All-to-All": None,
        "Heavy-Hex Lattice": heavy_hex_map,
        "Linear Topology": linear_map
    }
    
    # Step 2: The Routing Execution
    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=good_indices)
    compiler.k_optimal = 1
    raw_qc = compiler.generate_ideal_circuit()
    
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    print(f"Target Circuit: k=1, {n_qubits} Qubits")
    print(f"Basis Gates: {basis_gates}")
    
    results = {}
    
    for name, cmap in architectures.items():
        print(f"\n--- Transpiling for {name} ---")
        t_qc, t_depth, t_ops = transpile_for_hardware(
            raw_qc, 
            coupling_map=cmap, 
            basis_gates=basis_gates, 
            optimization_level=3
        )
        
        cx_count = t_ops.get('cx', 0)
        results[name] = {"depth": t_depth, "cx": cx_count, "ops": dict(t_ops)}
        
        # Step 4: Extract and Compare the Metrics
        print(f"Depth: {t_depth}")
        print(f"Total CX Gates: {cx_count}")
        
    # Step 3: Understanding the SWAP Explosion
    print("\n" + "-" * 70)
    print("TOPOLOGICAL ROUTING PENALTY ANALYSIS")
    print("-" * 70)
    
    base_depth = int(results["All-to-All"]["depth"]) # type: ignore
    base_cx = int(results["All-to-All"]["cx"]) # type: ignore
    
    for name in ["Heavy-Hex Lattice", "Linear Topology"]:
        depth = int(results[name]["depth"]) # type: ignore
        cx = int(results[name]["cx"]) # type: ignore
        
        # Calculate routing overhead
        # Assuming each compiler-inserted SWAP decomposes into 3 CX gates
        extra_cx = cx - base_cx
        approx_swaps = extra_cx // 3
        
        print(f"\n{name} vs All-to-All:")
        print(f"   -> Depth Penalty: {depth / base_depth:.2f}x multiplier ({depth} vs {base_depth})")
        print(f"   -> Extra CNOTs inserted for routing: {extra_cx}")
        print(f"   -> Approximated SWAP count: {max(0, approx_swaps)}")
        
    return results

def run_scenario_c(n_qubits=5, k=2):
    """
    Scenario C: Compiler Optimization Comparison

    Compares low- and high-optimization transpilation settings on a restricted
    topology to quantify the residual physical-depth overhead.
    """
    print("\n" + "=" * 70)
    print("SCENARIO C: COMPILER OPTIMIZATION COMPARISON")
    print("=" * 70)
    
    # Step 1: Construct the Test Subject
    # Use n=5, and a moderately deep k=2 to give the compiler room to optimize.
    # Selecting good_indices arbitrarily, e.g., |10> for 5 qubits
    good_indices = [10]
    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=good_indices)
    compiler.k_optimal = k
    
    raw_qc = compiler.generate_ideal_circuit()
    ideal_depth = raw_qc.depth()
    
    # Define a 5-qubit Heavy-Hex sub-graph approximation
    heavy_hex_map_5 = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3]]
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    
    print(f"1. Problem Instance")
    print(f"   - Qubits: {n_qubits}, Grover Iterations (k): {k}")
    print(f"   - Ideal Logical Depth: {ideal_depth}")
    print(f"   - Architecture: Heavy-Hex Lattice (5 qubits)")
    print(f"   - Basis Gates: {basis_gates}")
    
    # Step 2: The Naive Mapping (Level 0)
    print("\n2. Transpiling at optimization level 0...")
    qc_level_0, depth_0, ops_0 = transpile_for_hardware(
        raw_qc, 
        coupling_map=heavy_hex_map_5, 
        basis_gates=basis_gates, 
        optimization_level=0
    )
    cx_0 = ops_0.get('cx', 0)
    
    print(f"   -> Depth: {depth_0}")
    print(f"   -> Total CX Gates: {cx_0}")
    
    # Step 3: The Aggressive Synthesis (Level 3)
    print("\n3. Transpiling at optimization level 3...")
    qc_level_3, depth_3, ops_3 = transpile_for_hardware(
        raw_qc, 
        coupling_map=heavy_hex_map_5, 
        basis_gates=basis_gates, 
        optimization_level=3
    )
    cx_3 = ops_3.get('cx', 0)
    
    print(f"   -> Depth: {depth_3}")
    print(f"   -> Total CX Gates: {cx_3}")
    
    # Step 4: Extract and Compare the Metrics
    print("\n" + "-" * 70)
    print("MITIGATION EVALUATION RESULTS")
    print("-" * 70)
    
    depth_reduction = (depth_0 - depth_3) / depth_0 * 100 if depth_0 > 0 else 0
    cx_reduction = (cx_0 - cx_3) / cx_0 * 100 if cx_0 > 0 else 0
    
    print(f"Level 3 Optimization Savings:")
    print(f"   -> Depth reduced by {depth_reduction:.2f}% (from {depth_0} down to {depth_3})")
    print(f"   -> CNOTs reduced by {cx_reduction:.2f}% (from {cx_0} down to {cx_3})")
    
    depth_multiplier = depth_3 / ideal_depth
    print(f"\nInterpretation:")
    print(f"   -> Even after aggressive optimization, physical depth remains {depth_multiplier:.2f}x the ideal logical depth.")
    

def run_scenario_d(max_qubits=8):
    """
    Scenario D: Dimensional Scaling Analysis

    Studies how the optimal Grover iteration count and routed circuit depth
    scale with the search-register size.
    """
    print("\n" + "=" * 70)
    print("SCENARIO D: DIMENSIONAL SCALING ANALYSIS")
    print("=" * 70)
    
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    print(f"Basis Gates: {basis_gates}")
    print("Hardware Topology: Linear (worst-case routing proxy)")
    
    print(f"\n{'n':<4} | {'N (2^n)':<8} | {'k*':<4} | {'Logical Depth':<14} | {'Physical Depth':<15} | {'CX Count'}")
    print("-" * 75)
    
    # NISQ Coherence Limit approximation (e.g. 1000 depth)
    coherence_limit = 1000
    breached = False
    
    for n in range(3, max_qubits + 1):
        # M = 1 (single target)
        # Arbitrarily pick the state |0...0> = index 0 as good
        good_indices = [0]
        compiler = GroverCompiler(n_qubits=n, good_indices=good_indices)
        k_star = compiler.k_optimal
        
        raw_qc = compiler.generate_ideal_circuit()
        logical_depth = raw_qc.depth()
        
        # Create a linear coupling map for n qubits
        linear_map = [[i, i+1] for i in range(n-1)] + [[i+1, i] for i in range(n-1)]
        
        # Transpile at max optimization
        t_qc, t_depth, t_ops = transpile_for_hardware(
            raw_qc, 
            coupling_map=linear_map, 
            basis_gates=basis_gates, 
            optimization_level=3
        )
        
        cx_count = t_ops.get('cx', 0)
        
        marker = ""
        if t_depth > coherence_limit and not breached:
            marker = "  <-- Estimated coherence threshold exceeded"
            breached = True
            
        print(f"{n:<4} | {2**n:<8} | {k_star:<4} | {logical_depth:<14} | {t_depth:<15} | {cx_count}{marker}")
        
    print("\n" + "-" * 70)
    print("SCALING CONCLUSION")
    print("-" * 70)
    print("-> Physical depth increases rapidly with n due to MCX decomposition and SWAP routing.")
    print("-> This trend motivates partitioned or distributed amplitude-amplification strategies.")

def run_scenario_e(n_qubits=4, max_k=10):
    """
    Scenario E: Noise-Induced Performance Degradation

    Examines the degradation of Grover amplification under a simple
    depolarizing noise model after routing and gate decomposition.
    """
    print("\n" + "=" * 70)
    print("SCENARIO E: NOISE-INDUCED PERFORMANCE DEGRADATION")
    print("=" * 70)
    
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        print("Missing qiskit_aer module. Cannot run noise simulation.")
        return
        
    # Step 1: Construct the Noise Environment
    # We apply a basic 1% depolarizing error rate to all two-qubit gates
    error_rate = 0.01
    noise_model = NoiseModel()
    
    # cx is our primary two-qubit gate in the target basis
    two_qubit_error = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
    
    simulator = AerSimulator(noise_model=noise_model)
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    
    # 4-qubit heavy-hex snippet for routing
    heavy_hex_map_4 = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1]]
    
    good_indices = [0] # Target |0000>
    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=good_indices)
    
    print(f"Target Qubits: {n_qubits}, Initial Success Probability p: {compiler.p:.4f}")
    print(f"Noise Profile: {error_rate*100}% depolarizing error on CX gates")
    print(f"Architecture: Heavy-Hex (4 qubits)")
    print(f"\n{'k':<3} | {'Logical Depth':<14} | {'Physical Depth':<15} | {'CX Count':<10} | {'Noisy Success Prob'}")
    print("-" * 75)
    
    shots = 8192
    
    # Step 2: The Sweeping Execution
    for k in range(max_k + 1):
        # Override k_optimal to force the iterations
        compiler.k_optimal = k
        raw_qc = compiler.generate_ideal_circuit()
        logical_depth = raw_qc.depth()
        
        # We need to measure all qubits to calculate success
        raw_qc.measure_all()
        
        t_qc, t_depth, t_ops = transpile_for_hardware(
            raw_qc, 
            coupling_map=heavy_hex_map_4, 
            basis_gates=basis_gates, 
            optimization_level=3
        )
        
        cx_count = t_ops.get('cx', 0)
        
        # Step 3: Extracting the Noisy Signal
        job = simulator.run(t_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Look for the target state '0000' (note Qiskit endianness)
        target_bitstring = format(good_indices[0], f'0{n_qubits}b')
        success_count = counts.get(target_bitstring, 0)
        noisy_success_prob = success_count / shots
        
        marker = ""
        if k == int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(compiler.p))) - 0.5)):
            marker = "  <-- Ideal Peak (k*)"
            
        print(f"{k:<3} | {logical_depth:<14} | {t_depth:<15} | {cx_count:<10} | {noisy_success_prob:.4f}{marker}")

    print("\n" + "-" * 70)
    print("NOISE DEGRADATION CONCLUSION")
    print("-" * 70)
    print("-> The routed physical circuits accumulate substantial decomposition and routing overhead.")
    print("-> Under the selected noise model, the ideal peak success probability is significantly reduced.")
    print("-> This behavior motivates fixed-point amplitude amplification with bounded passband behavior.")

def run_scenario_f(qubit_sizes=[3, 5, 7]):
    """
    Scenario F: Fault-Tolerant Compilation Overhead

    Estimates the resource overhead associated with compiling Grover iterates
    into a Clifford+T gate set.
    """
    print("\n" + "=" * 70)
    print("SCENARIO F: FAULT-TOLERANT COMPILATION OVERHEAD")
    print("=" * 70)
    
    # Step 1 & 2: The Fault-Tolerant Setup & Constraints
    ft_basis_gates = ['h', 's', 'sdg', 'cx', 't', 'tdg']
    
    print(f"Target Circuit: k=1 (Single Grover Iterate)")
    print(f"Universal FT Basis Gates: {ft_basis_gates}")
    print("Hardware Topology: All-to-All (Lattice surgery assumed)")
    print(f"\n{'n':<4} | {'Logical Depth':<15} | {'Physical FT Depth':<18} | {'Total T-Count':<15} | {'Approx T-Depth'}")
    print("-" * 80)
    
    # Step 3: The Decomposition Execution
    for n in qubit_sizes:
        # Arbitrary target state |0...0>
        good_indices = [0]
        compiler = GroverCompiler(n_qubits=n, good_indices=good_indices)
        
        # Override for a single iterate
        compiler.k_optimal = 1
        raw_qc = compiler.generate_ideal_circuit()
        logical_depth = raw_qc.depth()
        
        # Transpile into the fault-tolerant basis without topological constraints
        t_qc, t_depth, t_ops = transpile_for_hardware(
            raw_qc, 
            coupling_map=None, 
            basis_gates=ft_basis_gates, 
            optimization_level=3
        )
        
        # Step 4: Extracting the T-Metrics
        # Calculate T-count
        total_t_count = t_ops.get('t', 0) + t_ops.get('tdg', 0)
        
        # Calculate approximate T-depth (Qiskit's built-in depth ignores specific gate timing, 
        # so we extract depth strictly for 't' and 'tdg' gates)
        # We can approximate the T-depth using Qiskit's depth() function by 
        # filtering the circuit to only its T-gates and measuring depth.
        t_only_qc = QuantumCircuit(*t_qc.qregs, *t_qc.cregs)
        for instruction in t_qc.data:
            if instruction.operation.name in ['t', 'tdg']:
                t_only_qc.append(instruction.operation, instruction.qubits, instruction.clbits)
                
        t_depth_approx = t_only_qc.depth()
        
        print(f"{n:<4} | {logical_depth:<15} | {t_depth:<18} | {total_t_count:<15} | {t_depth_approx}")

    print("\n" + "-" * 70)
    print("FAULT-TOLERANT CONCLUSION")
    print("-" * 70)
    print("-> The T-count required to decompose monolithic multi-controlled reflections is substantial.")
    print("-> This motivates block-encoding and QSVT-style constructions that rely more heavily")
    print("   on structured single-qubit phase synthesis.")

def run_scenario_g(n_qubits=STANDARD_QUBITS):
    """
    Scenario G: Ancilla-Space Trade-off

    Compares constrained and ancilla-assisted decompositions of large
    multi-controlled operations to quantify the space-time trade-off.
    """
    print("\n" + "=" * 70)
    print("SCENARIO G: THE ANCILLA-SPACE TRADE-OFF")
    print("=" * 70)
    
    # Target state |0...0> = index 0
    good_indices = [0]
    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=good_indices)
    k_star = compiler.k_optimal
    
    print(f"Target Circuit: k*={k_star} Iterations, {n_qubits} Logical Qubits")
    print(f"Basis Gates: ['cx', 'id', 'rz', 'sx', 'x']")
    print("Hardware Topology: All-to-All (Isolating decomposition space-time tradeoff)")
    print(f"\n{'Architecture':<25} | {'Logical Depth':<15} | {'Physical Depth':<15} | {'CX Count'}")
    print("-" * 75)
    
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    
    # ---------------------------------------------------------
    # Constrained Transpilation (Time-Heavy, Space-Light, 0 Ancilla)
    # ---------------------------------------------------------
    # Re-build the raw circuit to ensure fresh objects
    raw_qc = compiler.generate_ideal_circuit()
    logical_depth = raw_qc.depth()
    
    t_qc_constrained, t_depth_constrained, t_ops_constrained = transpile_for_hardware(
        raw_qc, 
        coupling_map=None, 
        basis_gates=basis_gates, 
        optimization_level=3
    )
    
    cx_constrained = t_ops_constrained.get('cx', 0)
    print(f"{n_qubits} Qubits (0 Ancilla):      | {logical_depth:<15} | {t_depth_constrained:<15} | {cx_constrained}")

    # ---------------------------------------------------------
    # Expanded Transpilation (Space-Heavy, Time-Light, n-3 Ancilla)
    # ---------------------------------------------------------
    num_ctrl_qubits = n_qubits - 1
    num_ancilla = max(0, num_ctrl_qubits - 2)
    total_qubits = n_qubits + num_ancilla
    
    # We must explicitly build the circuit using ancilla-assisted MCX gates (v-chain mode)
    from qiskit.circuit.library import MCXVChain
    
    qc_expanded = QuantumCircuit(total_qubits)
    
    # 1. Initialization
    qc_expanded.h(range(n_qubits))
    
    # Build v-chain MCX gate
    mcx_vchain = MCXVChain(num_ctrl_qubits=num_ctrl_qubits, dirty_ancillas=False)
    
    for _ in range(k_star):
        # Oracle (Targeting 000...0)
        qc_expanded.x(range(n_qubits)) # Flip 0s to 1s
        qc_expanded.h(n_qubits - 1)
        qc_expanded.append(mcx_vchain, list(range(n_qubits - 1)) + [n_qubits - 1] + list(range(n_qubits, total_qubits)))
        qc_expanded.h(n_qubits - 1)
        qc_expanded.x(range(n_qubits)) # Uncompute
        
        # Diffusion
        qc_expanded.h(range(n_qubits))
        qc_expanded.x(range(n_qubits))
        qc_expanded.h(n_qubits - 1)
        qc_expanded.append(mcx_vchain, list(range(n_qubits - 1)) + [n_qubits - 1] + list(range(n_qubits, total_qubits)))
        qc_expanded.h(n_qubits - 1)
        qc_expanded.x(range(n_qubits))
        qc_expanded.h(range(n_qubits))
        
    t_qc_expanded, t_depth_expanded, t_ops_expanded = transpile_for_hardware(
        qc_expanded, 
        coupling_map=None, 
        basis_gates=basis_gates, 
        optimization_level=3
    )
    
    cx_expanded = t_ops_expanded.get('cx', 0)
    print(f"{total_qubits} Qubits ({num_ancilla} Ancilla):    | {qc_expanded.depth():<15} | {t_depth_expanded:<15} | {cx_expanded}")

    # ---------------------------------------------------------
    # Extracting the Trade-off Metrics
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("ANCILLA-SPACE REDUCTION METRICS")
    print("-" * 70)
    depth_saved = t_depth_constrained - t_depth_expanded
    cx_saved = cx_constrained - cx_expanded
    
    print(f"Using {num_ancilla} clean ancilla qubits reduces:")
    print(f"   -> physical depth by {depth_saved}")
    print(f"   -> CNOT count by {cx_saved}")
    print("\n-> These data quantify the ancillary-qubit cost associated with reducing")
    print("   the depth of large multi-controlled constructions.")


def run_scenario_h(n_qubits=6, M=48):
    """
    Scenario H: High-Density Regime Analysis

    Examines the M > N/2 regime, where a single Grover iterate overshoots the
    optimal success region and hardware noise further reduces performance.
    """
    print("\n" + "=" * 70)
    print("SCENARIO H: HIGH-DENSITY REGIME ANALYSIS")
    print("=" * 70)
    
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        print("Missing qiskit_aer module. Cannot run noise simulation.")
        return
        
    N = 2**n_qubits
    if not (N / 2 < M < N):
        raise ValueError(f"Scenario H requires N/2 < M < N. Got N={N}, M={M}.")

    # Arbitrary M > N/2 setup
    good_indices = list(range(M))
    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=good_indices)
    
    # 1. The Classical Baseline
    classical_prob = compiler.p
    
    # Override optimal iterations to force exactly 1 overshooting query
    compiler.k_optimal = 1
    
    # 2. The Theoretical Overshoot (k=1 mathematical calculation)
    # Using P_k = sin^2((2k+1)*theta/2)
    theta = 2 * np.arcsin(np.sqrt(classical_prob))
    theoretical_prob = np.sin((3 * theta) / 2)**2
    
    print(f"Target Qubits: {n_qubits} (N={N})")
    print(f"Target Solutions (M): {M}  (High-Density Regime)")
    print(f"-> 1. Classical Random Guess Probability: {classical_prob:.4f}")
    print(f"-> 2. Theoretical quantum success probability at k=1: {theoretical_prob:.4f}")
    
    # 3. The Noisy Execution
    raw_qc = compiler.generate_ideal_circuit()
    raw_qc.measure_all()
    
    # 6-qubit heavy-hex coupling map
    heavy_hex_map = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3], [4, 5], [5, 4]]
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    
    t_qc, t_depth, t_ops = transpile_for_hardware(
        raw_qc, 
        coupling_map=heavy_hex_map, 
        basis_gates=basis_gates, 
        optimization_level=3
    )
    
    error_rate = 0.01
    noise_model = NoiseModel()
    two_qubit_error = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
    
    simulator = AerSimulator(noise_model=noise_model)
    
    shots = 8192
    job = simulator.run(t_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Sum the counts for all states in good_indices
    success_count = 0
    for idx in good_indices:
        target_bitstring = format(idx, f'0{n_qubits}b')
        success_count += counts.get(target_bitstring, 0)
        
    noisy_prob = success_count / shots
    cx_count = t_ops.get('cx', 0)
    
    print(f"-> 3. Physical noisy hardware success probability at k=1: {noisy_prob:.4f}")
    
    print("\n" + "-" * 70)
    print("NOISE LIMITATION CONCLUSION")
    print("-" * 70)
    print(f"Physical execution cost {t_depth} levels of depth and {cx_count} CNOT gates.")
    print("In this regime, a single Grover iterate rotates the state beyond the optimal")
    print("success region, and hardware noise further suppresses the remaining signal.")
    print("These results motivate fixed-point amplitude amplification, which replaces")
    print("oscillatory over-rotation with bounded passband behavior.")

def run_profiling_benchmark(n_qubits=STANDARD_QUBITS):
    """
    Runs the newly built QuantumProfiler to unify time, depth, distance, and 
    entanglement into a single hardware penalty score.
    """
    print("\n" + "=" * 70)
    print("UNIFIED HARDWARE PROFILER BENCHMARK")
    print("=" * 70)
    
    try:
        profiler_mod = _import_local_module("grover_quantum_profiler_module", "quantum_profiler.py")
        HardwareProfiler = profiler_mod.HardwareProfiler
    except Exception:
        print("Could not import HardwareProfiler. Ensure quantum_profiler.py is available.")
        return
        
    good_indices = [0]
    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=good_indices)
    raw_qc = compiler.generate_ideal_circuit()
    
    # 5-qubit Heavy Hex
    coupling_edges = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3]]
    
    profiler = HardwareProfiler(
        coupling_map_edges=coupling_edges, 
        basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
        single_qubit_ns=20,
        two_qubit_ns=100,
        w_time=1.0, 
        w_cnot=10.0, 
        w_distance=5.0
    )
    
    print(f"Profiling Grover Search (n={n_qubits}, k*={compiler.k_optimal})")
    print("Extracting physical metrics across 5 compiler stages...\n")
    
    metrics = profiler.profile_circuit(raw_qc)
    
    print(f"--- Stage 1: Layout & Routing ---")
    print(f"Topological Strain (Initial Distance Penalty): {metrics['initial_distance_penalty']}")
    print(f"SWAP Gates Injected: {metrics['routing_swaps']}")
    print(f"Post-Routing Depth: {metrics['post_routing_depth']}")
    
    print(f"\n--- Stage 2: Translation (Unrolling) ---")
    print(f"Total Native Gates: {metrics['translation_gates']}")
    print(f"Total CNOTs: {metrics['translation_cnots']}")
    print(f"Post-Translation Depth: {metrics['post_translation_depth']}")
    
    print(f"\n--- Stage 3: Optimization ---")
    print(f"Final Optimized Gates: {metrics['final_gates']}")
    print(f"Final Optimized CNOTs: {metrics['final_cnots']}")
    print(f"Final Optimized Depth: {metrics['post_optimization_depth']}")
    
    print(f"\n--- Stage 4: Scheduling (Physical Time) ---")
    print(f"Total Physical Execution Time (ns): {metrics['total_time_ns']} ns")
    
    print(f"\n======================================================================")
    print(f"Composite hardware penalty score: {metrics['hardware_penalty_score']}")
    print(f"======================================================================")
    print("This composite metric provides a comparative baseline for assessing")
    print("fixed-point amplitude amplification and QSVT-style alternatives.")


def _save_grover_algorithm_figure(
    qubit_sweep=(3, 4, 5, 6, 7),
    *,
    output_name="grover_resource_scaling.png",
):
    plt = _load_pyplot()
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']

    n_vals = []
    k_vals = []
    ideal_depths = []
    physical_depths = []
    cx_counts = []
    depth_multipliers = []

    for n_qubits in qubit_sweep:
        compiler = GroverCompiler(n_qubits=n_qubits, good_indices=[0])
        raw_qc = compiler.generate_ideal_circuit()
        linear_map = [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]
        _, t_depth, t_ops = transpile_for_hardware(
            raw_qc,
            coupling_map=linear_map,
            basis_gates=basis_gates,
            optimization_level=3,
        )

        ideal_depth = float(raw_qc.depth())
        physical_depth = float(t_depth)
        cx_count = float(t_ops.get('cx', 0))

        n_vals.append(int(n_qubits))
        k_vals.append(int(compiler.k_optimal))
        ideal_depths.append(ideal_depth)
        physical_depths.append(physical_depth)
        cx_counts.append(cx_count)
        depth_multipliers.append(physical_depth / max(1.0, ideal_depth))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Grover Transpile Resource Profile", fontsize=14, fontweight="bold")

    ax1, ax2, ax3, ax4 = axes.flat
    ax1.plot(n_vals, k_vals, marker="o", linewidth=2.2, color="#1f77b4")
    ax1.set_title("Optimal Iteration Count")
    ax1.set_ylabel("k*")

    ax2.plot(n_vals, ideal_depths, marker="o", linewidth=2.0, label="ideal logical depth", color="#2ca02c")
    ax2.plot(n_vals, physical_depths, marker="s", linewidth=2.0, label="linear routed depth", color="#d62728")
    ax2.set_title("Logical vs Routed Depth")
    ax2.set_ylabel("Depth")
    ax2.legend(fontsize=8)

    ax3.bar(n_vals, cx_counts, color="#9467bd", alpha=0.85)
    ax3.set_title("Entangling Cost on Linear Topology")
    ax3.set_ylabel("CX count")

    ax4.plot(n_vals, depth_multipliers, marker="D", linewidth=2.2, color="#ff7f0e")
    ax4.set_title("Routing Depth Multiplier")
    ax4.set_ylabel("Physical depth / logical depth")

    for axis in axes.flat:
        axis.set_xlabel("Search qubits")
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "grover_resource_scaling",
            "qubit_sweep": list(map(int, qubit_sweep)),
            "good_indices_policy": "single target at index 0 for each n",
            "basis_gates": ['cx', 'id', 'rz', 'sx', 'x'],
            "routed_topology": "linear",
        },
    )
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


def _save_grover_topology_figure(
    n_qubits: int = 6,
    good_indices=(10, 25),
    *,
    output_name="grover_topology_penalty.png",
):
    plt = _load_pyplot()
    compiler = GroverCompiler(n_qubits=n_qubits, good_indices=list(good_indices))
    compiler.k_optimal = 1
    raw_qc = compiler.generate_ideal_circuit()
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']

    linear_map = [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]
    heavy_hex_map = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3], [4, 5], [5, 4]]
    architectures = {
        "All-to-All": None,
        "Heavy-Hex": heavy_hex_map,
        "Linear": linear_map,
    }

    labels = []
    depths = []
    cx_counts = []
    swap_estimates = []
    depth_multipliers = []
    base_depth = None
    base_cx = None
    for label, cmap in architectures.items():
        _, depth, ops = transpile_for_hardware(
            raw_qc,
            coupling_map=cmap,
            basis_gates=basis_gates,
            optimization_level=3,
        )
        cx = float(ops.get('cx', 0))
        labels.append(label)
        depths.append(float(depth))
        cx_counts.append(cx)
        if base_depth is None:
            base_depth = float(depth)
            base_cx = cx
        extra_cx = max(0.0, cx - float(base_cx or 0.0))
        swap_estimates.append(extra_cx / 3.0)
        depth_multipliers.append(float(depth) / max(1.0, float(base_depth or 1.0)))

    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    fig.suptitle("Grover Topology Penalty Profile", fontsize=14, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.bar(x, depths, color="#1f77b4")
    ax1.set_title("Routed Depth by Topology")
    ax1.set_ylabel("Depth")

    ax2.bar(x, cx_counts, color="#d62728")
    ax2.set_title("Entangling Cost by Topology")
    ax2.set_ylabel("CX count")

    ax3.bar(x, swap_estimates, color="#9467bd")
    ax3.set_title("Approximate SWAP Burden")
    ax3.set_ylabel("Estimated SWAP count")

    ax4.plot(x, depth_multipliers, marker="o", linewidth=2.2, color="#ff7f0e")
    ax4.set_title("Depth Multiplier vs All-to-All")
    ax4.set_ylabel("Multiplier")

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "grover_topology_penalty",
            "n_qubits": int(n_qubits),
            "good_indices": list(map(int, good_indices)),
            "architectures": list(architectures.keys()),
            "basis_gates": basis_gates,
            "iterations_for_figure": 1,
        },
    )
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Class to duplicate standard output to both the console and a file
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            # Handle the flush command by redirecting it to both outputs
            self.terminal.flush()
            self.log.flush()

    # Get the directory of the current script to save the log file alongside it
    output_filepath = os.path.join(_RESULT_DIR, "terminal_output.log")
    os.chdir(_RESULT_DIR)
    
    # Redirect all print output
    logger = Logger(output_filepath)
    sys.stdout = logger
    cli_argv, publishability = parse_publishability_cli(
        sys.argv[1:],
        default_max_qubits=20,
        default_shots=1024,
        default_log_dir=_RESULT_DIR,
    )
    prepare_backend_validation_artifacts(publishability)
    
    print("Starting Grover transpilation benchmark suite...")
    print(f"Saving all results to: {output_filepath}\n")
    print(publishability.summary())
    default_scenarios = [
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
    interactive_scenarios = [
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
    scenarios = wrap_scenarios(default_scenarios, module_globals=globals(), config=publishability)
    interactive_wrapped = wrap_scenarios(interactive_scenarios, module_globals=globals(), config=publishability)

    try:
        cli_executed = run_cli_scenario(cli_argv, interactive_wrapped)
        if not cli_executed:
            for label, fn in scenarios:
                try:
                    fn()
                except Exception:
                    import traceback
                    print(f"\nSCENARIO {label} EXECUTION FAILED")
                    traceback.print_exc()
            run_interactive_scenario_repl(
                interactive_wrapped,
                sep="=" * 70,
            )
    finally:
        render_backend_validation_summary(publishability)
        _save_grover_algorithm_figure()
        _save_grover_topology_figure()
        # Restore standard output and close the log file
        logger.log.close()
        sys.stdout = logger.terminal
        print("\nBenchmark suite complete. Results saved.")

