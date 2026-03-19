import numpy as np
from qiskit import QuantumCircuit, transpile

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
    Scenario A: The Unrolling Baseline (The "Exception" Case)
    
    Proof that scaling standard Grover search on monolithic hardware is 
    infeasible by the sheer weight of multi-controlled gate unrolling.
    """
    print("\n" + "=" * 70)
    print("SCENARIO A: THE UNROLLING BASELINE")
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
    print(f"   -> CNOT count for 2 logical MCX gates: {t_ops.get('cx', 0)}")
    return t_qc

def run_scenario_b(n_qubits=6, good_indices=[10, 25]):
    """
    Scenario B: The Topological Mid-Low (Connectivity Constraints)
    
    Proof that routing multi-controlled gates across restricted physical
    topologies causes a massive explosion in depth due to SWAP operations.
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
    Scenario C: Compiler Optimization Comparative Evaluation (The Mitigation Test)
    
    Demonstrates the limit of classical compiler heuristics (optimization_level 0 vs 3) 
    in reducing the massive logical-to-physical depth blowup on a restricted topology.
    """
    print("\n" + "=" * 70)
    print("SCENARIO C: COMPILER OPTIMIZATION COMPARATIVE EVALUATION")
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
    
    print(f"1. The Test Subject")
    print(f"   - Qubits: {n_qubits}, Grover Iterations (k): {k}")
    print(f"   - Ideal Logical Depth: {ideal_depth}")
    print(f"   - Architecture: Heavy-Hex Lattice (5 qubits)")
    print(f"   - Basis Gates: {basis_gates}")
    
    # Step 2: The Naive Mapping (Level 0)
    print("\n2. Transpiling at Level 0 (Naive Mapping)...")
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
    print("\n3. Transpiling at Level 3 (Aggressive Synthesis)...")
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
    
    blowup_factor = depth_3 / ideal_depth
    print(f"\nThe Hard Mathematical Wall:")
    print(f"   -> Despite max optimization, physical depth is STILL {blowup_factor:.2f}x greater than ideal logical depth.")
    

def run_scenario_d(max_qubits=8):
    """
    Scenario D: The Dimensional Breaking Point (The Extreme Scaling Test)
    
    Demonstrates the double-exponential blowup of algorithm scaling (k*) and 
    hardware scaling (mcx decomposition) as n increases, proving monolithic 
    Grover's impossibility on NISQ.
    """
    print("\n" + "=" * 70)
    print("SCENARIO D: THE DIMENSIONAL BREAKING POINT")
    print("=" * 70)
    
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    print(f"Basis Gates: {basis_gates}")
    print("Hardware Topology: Linear (Worst-case routing)")
    
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
            marker = "  <-- NISQ Coherence Limit Breached!"
            breached = True
            
        print(f"{n:<4} | {2**n:<8} | {k_star:<4} | {logical_depth:<14} | {t_depth:<15} | {cx_count}{marker}")
        
    print("\n" + "-" * 70)
    print("SCALING CONCLUSION")
    print("-" * 70)
    print("-> Physical depth scales severely as n increases due to MCX unrolling + SWAP routing.")
    print("-> Validates the necessity of Distributed Amplitude Amplification (DQAA) to partition the search space.")

def run_scenario_e(n_qubits=4, max_k=10):
    """
    Scenario E: The "Flattened Over-Rotation" (Noise Injection)
    
    Proof that near-term physical noise destroys the fragile geometric 
    rotation of standard Grover amplification before it can reach P=1.0.
    """
    print("\n" + "=" * 70)
    print("SCENARIO E: THE FLATTENED SOUFFLÉ (NOISE INJECTION)")
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
    
    print(f"Target Qubits: {n_qubits}, Base Success Prob p: {compiler.p:.4f}")
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
    print("-> The deep physical circuits accumulate massive SWAP/unrolling errors.")
    print("-> The theoretical P=1.0 peak is crushed by the high CX count.")
    print("-> Validation for Fixed-Point Amplitude Amplification (FPAA) robust passbands!")

def run_scenario_f(qubit_sizes=[3, 5, 7]):
    """
    Scenario F: Fault-Tolerant Compilation (T-Gate Explosion)
    
    Proof that compiling standard multi-controlled continuous reflections into 
    a discrete, fault-tolerant universal basis (Clifford+T) results in an 
    exponential explosion of costly T-gates.
    """
    print("\n" + "=" * 70)
    print("SCENARIO F: FAULT-TOLERANT COMPILATION (T-GATE EXPLOSION)")
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
    print("-> The T-gate factory cost for decomposing monolithic multi-controlled rotations is astronomical.")
    print("-> Explains the transition toward QSVT block-encodings, which parameterize algorithmic primitives")
    print("   to utilize more favorable single-qubit phase rotation synthesis.")

def run_scenario_g(n_qubits=10):
    """
    Scenario G: The Ancilla-Space Trade-off (The Compiler Hack)
    
    Proof that compiling massive multi-controlled unitaries forces a 
    severe spacetime tradeoff. Providing clean ancilla qubits allows 
    the compiler to shortcut quadratic depth scaling using v-chains.
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
    
    print(f"By burning {num_ancilla} clean ancilla qubits, we saved:")
    print(f"   -> {depth_saved} physical gate depth!")
    print(f"   -> {cx_saved} CNOT operations!")
    print("\n-> This quantifies the exact physical real estate cost required to construct")
    print("   efficient block-encodings and oblivious purification primitives natively.")


def run_scenario_h(n_qubits=6, M=48):
    """
    Scenario H: The High-Density Noise Limitation
    
    Demonstrates the paradox of too much success (M > N/2). 
    A single Grover iteration forcibly overshoots the target theoretically, 
    and the required deep physical circuit adds massive thermal routing noise, 
    making it worse than simply random classical guessing.
    """
    print("\n" + "=" * 70)
    print("SCENARIO H: THE HIGH-DENSITY NOISE LIMITATION")
    print("=" * 70)
    
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        print("Missing qiskit_aer module. Cannot run noise simulation.")
        return
        
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
    
    print(f"Target Qubits: {n_qubits} (N=64)")
    print(f"Target Solutions (M): {M}  (High-Density Regime)")
    print(f"-> 1. Classical Random Guess Probability: {classical_prob:.4f}")
    print(f"-> 2. Theoretical Quantum Probability (k=1): {theoretical_prob:.4f}  <-- The Overshoot!")
    
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
    
    print(f"-> 3. Physical Noisy Hardware Probability (k=1): {noisy_prob:.4f}  <-- The Noise Trap")
    
    print("\n" + "-" * 70)
    print("NOISE LIMITATION CONCLUSION")
    print("-" * 70)
    print(f"Physical execution cost {t_depth} levels of depth and {cx_count} CNOT gates.")
    print("Because Grover's iterate is strictly unitary, it forcibly rotates the state")
    print("AWAY from the target. The massive hardware noise then flattens whatever")
    print("remains. This empirically frames the necessity of Fixed-Point Amplitude")
    print("Amplification (FPAA), which utilizes Chebyshev polynomials to create")
    print("bound passbands that do not suffer from oscillatory over-rotation.")

def run_profiling_benchmark(n_qubits=5):
    """
    Runs the newly built QuantumProfiler to unify time, depth, distance, and 
    entanglement into a single hardware penalty score.
    """
    print("\n" + "=" * 70)
    print("UNIFIED HARDWARE PROFILER BENCHMARK")
    print("=" * 70)
    
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from quantum_profiler import HardwareProfiler
    except ImportError:
        print("Could not import HardwareProfiler. Make sure quantum_profiler.py exists.")
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
    print(f"-> UNIFIED HARDWARE PENALTY SCORE: {metrics['hardware_penalty_score']}")
    print(f"======================================================================")
    print("This unified cost function serves as the definitive comparative baseline")
    print("to judge Fixed-Point Amplitude Amplification (FPAA) and QSVT against.")


if __name__ == "__main__":
    import sys
    import os

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filepath = os.path.join(script_dir, "!_Griver's_Search_Algorithm_transpile")
    
    # Redirect all print output
    logger = Logger(output_filepath)
    sys.stdout = logger
    
    print("Starting Transpilation Benchmark Suite...")
    print(f"Saving all results to: {output_filepath}\n")

    try:
        run_scenario_a(n_qubits=6, good_indices=[10, 25])
        run_scenario_b(n_qubits=6, good_indices=[10, 25])
        run_scenario_c(n_qubits=5, k=2)
        run_scenario_d(max_qubits=8)
        run_scenario_e(n_qubits=4, max_k=10)
        run_scenario_f(qubit_sizes=[3, 5, 7])
        run_scenario_g(n_qubits=10)
        run_scenario_h(n_qubits=6, M=48)
        run_profiling_benchmark(n_qubits=5)
    finally:
        # Restore standard output and close the log file
        logger.log.close()
        sys.stdout = logger.terminal
        print("\nBenchmark Suite finished. Results saved.")
