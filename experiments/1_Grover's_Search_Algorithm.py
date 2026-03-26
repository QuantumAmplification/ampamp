import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace, state_fidelity
import sys

try:
    from one_click_utils import start_one_click_session
except Exception:
    def start_one_click_session(script_file, *, figure_prefix=None, log_name="terminal_output.log"):
        import atexit
        import io
        import os
        from pathlib import Path
        script_path = Path(script_file).resolve()
        result_dir = script_path.parent / f"[RESULT]{script_path.stem}"
        result_dir.mkdir(parents=True, exist_ok=True)
        old_stdout, old_stderr, old_cwd = sys.stdout, sys.stderr, Path.cwd()
        log_handle = open(result_dir / log_name, "w", encoding="utf-8")
        class _Tee(io.TextIOBase):
            def __init__(self, *streams): self._streams = streams
            def write(self, data): [s.write(data) or s.flush() for s in self._streams]; return len(data)
            def flush(self): [s.flush() for s in self._streams]
        sys.stdout = _Tee(old_stdout, log_handle)
        sys.stderr = _Tee(old_stderr, log_handle)
        os.chdir(result_dir)
        try:
            import matplotlib.pyplot as plt
            old_show = plt.show
            prefix = figure_prefix or script_path.stem
            counter = {"n": 0}
            def _save_show(*args, **kwargs):
                del args, kwargs
                for fig_id in list(plt.get_fignums()):
                    counter["n"] += 1
                    plt.figure(fig_id).savefig(result_dir / f"{prefix}_figure_{counter['n']:03d}.png", dpi=220, bbox_inches="tight")
                plt.close("all")
            plt.show = _save_show
        except Exception:
            old_show = None
        def _cleanup():
            try:
                if old_show is not None:
                    import matplotlib.pyplot as plt
                    plt.show = old_show
            except Exception:
                pass
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.chdir(old_cwd)
            log_handle.close()
        atexit.register(_cleanup)
        return result_dir

class GroverGeometricLab:
    """
    Module for Section II: Grover's Algorithm.
    A rigorous numerical study of 2D geometry, instability, and cloning barriers.

    Standing notation aligned with final.tex:
    - H_Good / H_Bad: target and non-target subspaces
    - |All> = H^{⊗n}|0>^{⊗n}: prepared input state
    - p = ||Pi_Good |All>||^2 = M/N: initial success probability
    - sin^2(theta0) = p and Grover step angle theta = 2*theta0
    """
    def __init__(self, n_qubits, good_indices):
        """
        Initializes the GroverGeometricLab simulation environment.

        Args:
            n_qubits (int): Total number of qubits in the quantum register.
            good_indices (list of int): Computational-basis indices spanning H_Good.
        """
        # Define the dimensionality of the quantum state space
        self.n = n_qubits
        self.N = 2**n_qubits
        
        # Store properties related to the H_Good subspace
        self.good_indices = good_indices
        self.M = len(good_indices)
        
        # Initial success probability p = M/N = ||Pi_Good |All>||^2
        self.p = self.M / self.N
        self.p_init = self.p
        
        # theta0 satisfies sin^2(theta0)=p; Grover rotation step is theta=2*theta0
        self.theta0 = np.arcsin(np.sqrt(self.p))
        self.theta = 2 * self.theta0
        
        # Handle edge cases p in {0, 1} explicitly to avoid dividing by zero at p=0.
        # For these endpoints, no Grover iterate is needed to maximize success.
        if self.p == 0 or self.p == 1:
            self.k_optimal = 0
        else:
            # Optimal k* ≈ floor(pi/(4*arcsin(sqrt(p))) - 1/2)
            self.k_optimal = int(np.floor(np.pi / (2 * self.theta) - 0.5))
        
        # Initialize the statevector simulator backend from Qiskit Aer
        self.backend = AerSimulator()
        
        # Step A: Initialize the Invariant Vectors
        # These vectors form the orthogonal basis for the 2D invariant subspace where Grover rotation occurs
        self.good_vec = np.zeros(self.N, dtype=complex)
        self.bad_vec = np.zeros(self.N, dtype=complex)
        
        # Populate |Good> uniformly over basis states in H_Good
        for idx in self.good_indices:
            self.good_vec[idx] = 1.0 / np.sqrt(self.M)
            
        # Populate |Bad> uniformly over basis states in H_Bad
        for idx in range(self.N):
            if idx not in self.good_indices:
                self.bad_vec[idx] = 1.0 / np.sqrt(self.N - self.M)

    def grover_success_prob(self, p_value: float, k: int) -> float:
        """Standard Grover success probability in terms of p (with p=M/N in search)."""
        if p_value < 0 or p_value > 1:
            raise ValueError("p_value must be in [0, 1]")
        if p_value == 0:
            return 0.0

        theta = 2.0 * np.arcsin(np.sqrt(p_value))
        angle = (2 * k + 1) * theta / 2.0
        return float(np.sin(angle) ** 2)
        
        # Step A: Initialize the Invariant Vectors
        self.good_vec = np.zeros(self.N, dtype=complex)
        self.bad_vec = np.zeros(self.N, dtype=complex)
        
        for idx in self.good_indices:
            self.good_vec[idx] = 1.0 / np.sqrt(self.M)
            
        for idx in range(self.N):
            if idx not in self.good_indices:
                self.bad_vec[idx] = 1.0 / np.sqrt(self.N - self.M)

    def get_oracle(self):
        """
        Standard Phase Oracle for Section II.
        
        Constructs a phase oracle O = I - 2*Pi_Good that flips phase on H_Good.
        This is accomplished by flipping each marked basis state to all 1s, applying a multi-controlled Z gate 
        (simulated using H, Multi-Controlled X, and H), and uncomputing the bit flips.
        
        Returns:
            QuantumCircuit: The synthesized phase oracle circuit.
        """
        qc = QuantumCircuit(self.n)
        for index in self.good_indices:
            # Convert the good index to its binary string representation (little-endian for Qiskit)
            good_bin = format(index, f'0{self.n}b')[::-1]
            
            # Apply X-gates to transform the zero-bits of the good state into ones
            for i, bit in enumerate(good_bin):
                if bit == '0': qc.x(i)
                
            # Apply a multi-controlled Z gate by wrapping a multi-controlled X (MCX) in Hadamard gates
            qc.h(self.n - 1)
            qc.mcx(list(range(self.n - 1)), self.n - 1)
            qc.h(self.n - 1)
            
            # Uncompute the initial X-gates to restore the state basis
            for i, bit in enumerate(good_bin):
                if bit == '0': qc.x(i)
        return qc

    def get_diffusion(self):
        """
        Standard Diffusion Reflection.
        
        Constructs the Grover diffusion operator (inversion about the mean).
        This operator reflects about |All> to amplify overlap with H_Good.
        
        Returns:
            QuantumCircuit: The synthesized diffusion operator circuit.
        """
        qc = QuantumCircuit(self.n)
        
        # Apply Hadamard gates to transform back to the computational basis
        qc.h(range(self.n))
        
        # Apply X gates to effectively map the zero state |00...0> to |11...1>
        qc.x(range(self.n))
        
        # Apply a multi-controlled Z gate to shift the phase of the |11...1> state
        qc.h(self.n - 1)
        qc.mcx(list(range(self.n - 1)), self.n - 1)
        qc.h(self.n - 1)
        
        # Uncompute the X gates map back from |11...1> to |00...0>
        qc.x(range(self.n))
        
        # Apply Hadamard gates to return to the superposition basis
        qc.h(range(self.n))
        
        return qc

    def analyze_geometry(self, max_k):
        """
        Module 1 & 2: Tracks 2D rotation and Soufflé instability using linear algebra projection.
        
        Observes the precise quantum state vector at each step of Grover's iteration to 
        empirically verify that the operation constitutes a rotation in a 2D invariant subspace 
        spanned by |Good> and |Bad>.
        
        Args:
            max_k (int): Maximum number of iterations to simulate.
            
        Returns:
            tuple: Arrays containing |Good> and |Bad> amplitudes, total subspace probability, 
                   and the empirical success probabilities.
        """
        a_k_vals = []
        b_k_vals = []
        p_total_vals = []
        success_probs = []
        
        # Iterate over increasing numbers of Grover operator applications
        for k in range(max_k + 1):
            qc = QuantumCircuit(self.n)
            
            # Initialize to the uniform superposition state
            qc.h(range(self.n))
            
            if k > 0:
                # Apply the Oracle and Diffusion operators k times
                oracle = self.get_oracle()
                diff = self.get_diffusion()
                for _ in range(k):
                    qc.append(oracle, range(self.n))
                    qc.append(diff, range(self.n))
            
            # Step B: Iterative Statevector Extraction
            # Capture the full statevector to analyze its overlap with the invariant subspace
            qc.save_statevector()
            result = self.backend.run(transpile(qc, self.backend)).result()
            state_k = np.array(result.get_statevector())
            
            # Step C: The Projection Logic
            # Projection coefficients in the {|Good>,|Bad>} plane
            ak = np.dot(self.good_vec.conj(), state_k)
            bk = np.dot(self.bad_vec.conj(), state_k)
            
            # Theoretical invariant subspace conservation check
            prob_in_subspace = np.abs(ak)**2 + np.abs(bk)**2
            
            a_k_vals.append(ak)
            b_k_vals.append(bk)
            p_total_vals.append(prob_in_subspace)
            
            # Success probability is |ak|^2 = ||Pi_Good|psi_k>||^2
            success_probs.append(np.abs(ak)**2)
            
        return np.array(a_k_vals), np.array(b_k_vals), np.array(p_total_vals), success_probs

    def simulate_cloning_barrier(self, max_k):
        """
        Module 3: Proves that naive CNOT cloning fails due to entanglement.
        
        Provides an empirical demonstration of the No-Cloning Theorem by showing how naive 
        attempt at copying via CNOT gates unavoidably destroys the purity of the state 
        due to unwanted system-environment entanglement.
        
        Args:
            max_k (int): Maximum number of iterations to simulate.
            
        Returns:
            tuple: Lists defining the state purity before and after attempting to copy the quantum register.
        """
        purity_original = []
        purity_after_copy = []
        
        for k in range(max_k + 1):
            # 1. Ideal System A Evaluation (Before Copying)
            qc_ideal = QuantumCircuit(self.n)
            qc_ideal.h(range(self.n))
            if k > 0:
                oracle = self.get_oracle()
                diff = self.get_diffusion()
                for _ in range(k):
                    qc_ideal.append(oracle, range(self.n))
                    qc_ideal.append(diff, range(self.n))
            
            qc_ideal.save_statevector()
            state_ideal = self.backend.run(transpile(qc_ideal, self.backend)).result().get_statevector()
            rho_ideal = partial_trace(state_ideal, []) # Full trace returns original
            pur_ideal = np.real(np.trace(np.dot(rho_ideal.data, rho_ideal.data)))
            purity_original.append(pur_ideal)
            
            # 2. Naive Copy System (System A + System B)
            qc = QuantumCircuit(self.n * 2) 
            qc.h(range(self.n))
            
            if k > 0:
                for _ in range(k):
                    qc.append(oracle, range(self.n))
                    qc.append(diff, range(self.n))
                
            # Naive CNOT 'Copy' Phase
            for i in range(self.n):
                qc.cx(i, i + self.n)
                
            qc.save_statevector()
            state_cloned = self.backend.run(transpile(qc, self.backend)).result().get_statevector()
            
            # 3. Analyze Original Register Purity After Copy
            rho_A_after = partial_trace(state_cloned, range(self.n, 2 * self.n))
            pur_collapse = np.real(np.trace(np.dot(rho_A_after.data, rho_A_after.data)))
            purity_after_copy.append(pur_collapse)
            
        return purity_original, purity_after_copy

    def sensitivity_heatmap(self, k_max, p_max, resolution=500):
        """
        Module 5: Generates the 2D Soufflé Instability Heatmap.
        
        Calculates a vectorized surface of Grover success probabilities across varying 
        iteration counts and solution densities to highlight its extreme sensitivity.
        
        Args:
            k_max (int): The upper limit for the number of Grover iterations.
            p_max (float): Upper bound for p = M/N.
            resolution (int): Resolution of the discretized test grid.
            
        Returns:
            tuple: Coordinates and probabilistic success rates defining the instability heatmap.
        """
        # 1. Create the Meshgrid for extensive parametric sampling
        k_range = np.arange(0, k_max)
        p_range = np.linspace(0.001, p_max, resolution)
        K, L = np.meshgrid(k_range, p_range)
        
        # 2. Vectorized Analytic Calculation
        theta = 2 * np.arcsin(np.sqrt(L))
        success_heatmap = np.sin((2*K + 1) * theta / 2)**2
        
        return K, L, success_heatmap

    def recursive_nesting_analysis(self, k1, k2, max_lambda):
        """
        Module 4: Demonstrates self-similar scaling and extreme gate depth in nested Amplitude Amplification (AA).
        
        Examines the theoretical implications of executing nested iterations of standard Grover 
        logic—comparing multi-level success amplifications against corresponding exponential circuit depth.
        
        Args:
            k1 (int): Application count for an inner-layer Grover operator.
            k2 (int): Application count for an outer-layer meta-Grover operator.
            max_lambda (float): Backward-compatible name; interpreted as max_p.
            
        Returns:
            tuple: Solution density arrays, nested level probability curves, and gate depth analysis mappings.
        """
        # 1. Analytical evaluation over p-grid (p = M/N)
        max_p = max_lambda
        p_vals = np.linspace(0.0001, max_p, 500)
        
        # Level 1 probability P1
        theta_1 = 2 * np.arcsin(np.sqrt(p_vals))
        p1_vals = np.sin((2 * k1 + 1) * theta_1 / 2)**2
        
        # Level 2 Probability P2 (Treating P1 as the initial state)
        # Using exact rotation mechanics P_nested = sin^2((2k2 + 1) * arcsin(sqrt(P1)))
        theta_2 = 2 * np.arcsin(np.sqrt(p1_vals))
        p2_vals = np.sin((2 * k2 + 1) * theta_2 / 2)**2
        
        # 2. Gate Depth Analysis (Assuming standard decomposition overhead)
        qc_dummy = QuantumCircuit(self.n)
        oracle = self.get_oracle()
        diff = self.get_diffusion()
        
        oracle_depth = transpile(oracle, basis_gates=['u3', 'cx']).depth()
        diff_depth = transpile(diff, basis_gates=['u3', 'cx']).depth()
        
        # Standard Grover Depth for roughly equivalent success
        # Exact equivalent single-layer iteration count from angle matching:
        # 2*k_eq + 1 = (2*k1 + 1)(2*k2 + 1)
        k_equiv = 2 * k1 * k2 + k1 + k2
        std_depth = k_equiv * (oracle_depth + diff_depth)
        
        # Nested Grover Depth
        l1_depth = k1 * (oracle_depth + diff_depth)
        # Level 2 requires L1, L1_dagger, plus a phase flip
        l2_diff_overhead = l1_depth * 2 + 10 # approximate cost of L1 uncomputation
        nested_depth = k2 * (oracle_depth + l2_diff_overhead)
        
        depth_data = {
            'std_depth': std_depth,
            'nested_depth': nested_depth,
            'k_equiv': k_equiv
        }
        
        return p_vals, p1_vals, p2_vals, depth_data

if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="grover")
    # --------------------------------------------------------------------------------
    # Experimental Execution Block
    # Handles rigorous module evaluation, simulation visualization, and mathematical proof outputs.
    # --------------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("GROVER TEST CASES: M vs N")
    print("=" * 70)

    test_cases = [
        {"label": "Case 1 (M << N)", "N": 1024, "M": 3},
        {"label": "Case 2 (M = N/2)", "N": 1024, "M": 512},
        {"label": "Case 3 (M > N/2)", "N": 1024, "M": 768},
    ]

    for case in test_cases:
        label = case["label"]
        N = case["N"]
        M = case["M"]

        print(f"\n{label}:")
        print(f"  N = {N}, M = {M}")

        if N <= 0 or M < 0:
            continue
        if M > N:
            continue

        p_value = M / N
        # We can leverage the existing formulas inside the Lab
        temp_lab = GroverGeometricLab(n_qubits=10, good_indices=list(range(M)))
        k_star = temp_lab.k_optimal
        success_prob = temp_lab.grover_success_prob(p_value, k_star)

        print(f"  p = M/N = {p_value:.6f}")
        print(f"  k* (near-optimal Grover iterations) = {k_star}")
        print(f"  Grover success probability at k* = {success_prob:.6f}")

    print("\n" + "=" * 70)
    print("Running Grover Geometric Lab Analysis...")
    lab = GroverGeometricLab(n_qubits=6, good_indices=[10, 25])
    max_k = lab.k_optimal * 3

    print("Running Grover Geometric Lab Analysis...")
    a_vals, b_vals, p_total, probs = lab.analyze_geometry(max_k)
    pur_orig, pur_copy = lab.simulate_cloning_barrier(max_k)
    p_vals_nesting, p1_curve, p2_curve, gate_depths = lab.recursive_nesting_analysis(k1=3, k2=3, max_lambda=0.15)
    
    heatmap_K, heatmap_L, success_heatmap = lab.sensitivity_heatmap(k_max=50, p_max=0.5, resolution=500)
    
    # Machine Precision Check
    mean_sq_dev = np.mean((1.0 - p_total)**2)
    print(f"Mean squared deviation from invariant subspace: {mean_sq_dev:.2e}")
    if mean_sq_dev < 1e-15:
        print("-> Confirmed: Invariant Subspace Theorem holds within double-precision limits.")

    print(f"Gate Depth Comparison for Equivalent Boost: Standard={gate_depths['std_depth']} gates vs Nested={gate_depths['nested_depth']} gates")

    # Visualization
    fig = plt.figure(figsize=(24, 12))

    # ---- TOP ROW ----
    # Plot 1: Soufflé Problem
    ax1 = plt.subplot(2, 3, 1)

    # Theoretical Overlay
    k_continuous = np.linspace(0, max_k, 300)
    # The mathematical formula for probability: P_k = sin^2((2k + 1) * theta / 2)
    p_theoretical = np.sin((2 * k_continuous + 1) * lab.theta / 2)**2
    ax1.plot(k_continuous, p_theoretical, color='black', linestyle='--', alpha=0.5, label='Theoretical Curve')

    # Simulated Points
    ax1.plot(range(max_k + 1), probs, color='red', marker='o', linestyle='', label='Simulated Probability')

    # Annotations
    ax1.axvline(lab.k_optimal, color='green', linestyle=':', label='Theoretical k*')
    ax1.text(lab.k_optimal + 0.1, 0.1, 'Theoretical\nOptimum', color='green', ha='left')
    
    crash_k = 2 * lab.k_optimal
    ax1.axvline(crash_k, color='purple', linestyle=':', label='2k* (Crash)')
    ax1.text(crash_k + 0.1, 0.1, 'The Soufflé\nCrash', color='purple', ha='left')
    
    ax1.axhline(0.5, color='orange', linestyle='--', label='Stability Threshold (P=0.5)')

    ax1.set_title("The Soufflé Problem: Instability vs. Iterations")
    ax1.set_xlabel("Iterations (k)")
    ax1.set_ylabel("Success Probability p_k")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()

    # Plot 2: Invariant Subspace Proof
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(
        range(max_k + 1),
        p_total,
        color='blue',
        marker='x',
        label=r'$|\langle H_{\mathrm{Good}}|\psi_k\rangle|^2 + |\langle H_{\mathrm{Bad}}|\psi_k\rangle|^2$',
    )
    ax2.set_title("Invariant Subspace Verification (P = 1.0)")
    ax2.set_xlabel("Iterations (k)")
    ax2.set_ylabel("Total Probability in 2D Plane")
    ax2.set_ylim(0.9, 1.1)
    ax2.legend()

    # Plot 3: Unit Circle (a_k vs b_k)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(np.real(b_vals), np.real(a_vals), marker='o', color='purple', linestyle='-', label=r'$|\psi_k\rangle$ path')
    theta_ideal = np.linspace(0, np.pi/2, 100)
    ax3.plot(np.cos(theta_ideal), np.sin(theta_ideal), color='gray', linestyle='--', alpha=0.5, label='Ideal Unit Circle')
    ax3.set_title("Geometric Rotation in Invariant Subspace")
    ax3.set_xlabel(r"$H_{\mathrm{Bad}}$ Amplitude ($b_k$)")
    ax3.set_ylabel(r"$H_{\mathrm{Good}}$ Amplitude ($a_k$)")
    ax3.set_aspect('equal')
    ax3.grid(True)
    ax3.legend()

    # ---- BOTTOM ROW ----
    # Plot 4: Recursive Nesting
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(p_vals_nesting, p1_curve, color='blue', linestyle='--', label='Level 1 (k=3)')
    ax4.plot(p_vals_nesting, p2_curve, color='red', linestyle='-', label='Level 2 (k1=3, k2=3)')
    ax4.set_title("Self-Similar Scaling (Sharpening Effect)")
    ax4.set_xlabel("Initial Success Probability (p = M/N)")
    ax4.set_ylabel("Success Probability P(p)")
    ax4.grid(True)
    ax4.legend()

    # Plot 5: No-Cloning Barrier
    ax5 = plt.subplot(2, 3, 5)
    x = np.arange(len(pur_orig))
    width = 0.35
    ax5.bar(x - width/2, pur_orig, width, label='Original State Purity', color='lightgreen')
    ax5.bar(x + width/2, pur_copy, width, label='Purity After Naive Copy', color='salmon')
    ax5.axhline(0.833, color='red', linestyle='--', label='UQCM Limit (0.833)')
    ax5.set_title("Cloning Obstruction: Purity Collapse")
    ax5.set_xlabel("Iterations (k)")
    ax5.set_ylabel("Purity Value Tr(ρ^2)")
    ax5.set_ylim(0, 1.1)
    ax5.legend()
    
    # Plot 6: Soufflé Sensitivity Heatmap
    ax6 = plt.subplot(2, 3, 6)
    contour = ax6.pcolormesh(heatmap_L, heatmap_K, success_heatmap, shading='auto', cmap='inferno')
    fig.colorbar(contour, ax=ax6, label='Success Probability p_k')
    ax6.set_title("Grover Sensitivity Heatmap: The Soufflé Islands", fontsize=15)
    ax6.set_xlabel(r"Initial Success Probability ($p = M/N$)", fontsize=12)
    ax6.set_ylabel("Iteration Count ($k$)", fontsize=12)

    plt.tight_layout()
    plt.savefig('grover_geometric_evidence.png')
    print("Saved plot to 'grover_geometric_evidence.png'")
    # plt.show()
    
    # ---------------------------------------------------------
    # Optional 3-qubit Qiskit Demo
    # ---------------------------------------------------------
    def run_optional_qiskit_demo() -> None:
        """
        Optional 3-qubit circuit demo for H_Good state '101'.
        """
        try:
            from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
            from qiskit.visualization import plot_histogram
            from qiskit_aer import AerSimulator
            
            data = QuantumRegister(3, "data")
            anc = QuantumRegister(1, "anc")
            creg = ClassicalRegister(3, "c")
            qc = QuantumCircuit(data, anc, creg)

            def apply_phase_oracle_good_101(qcircuit: QuantumCircuit):
                qcircuit.x(data[1])
                qcircuit.mcp(np.pi, [data[0], data[1], data[2]], anc[0])
                qcircuit.x(data[1])

            def apply_diffusion_operator_3q(qcircuit: QuantumCircuit):
                qcircuit.h(data)
                qcircuit.x(data)
                qcircuit.h(data[2])
                qcircuit.ccx(data[0], data[1], data[2])
                qcircuit.h(data[2])
                qcircuit.x(data)
                qcircuit.h(data)

            qc.h(data)
            qc.x(anc[0])
            apply_phase_oracle_good_101(qc)
            apply_diffusion_operator_3q(qc)
            qc.measure(data, creg)

            backend = AerSimulator()
            compiled = transpile(qc, backend, optimization_level=1)
            shots = 4096
            result = backend.run(compiled, shots=shots).result()
            counts = result.get_counts()

            good_state = "101"
            good_probability = counts.get(good_state, 0) / shots
            print("\n" + "=" * 70)
            print("Qiskit circuit demo (H_Good=101):")
            print("Counts:", counts)
            print(f"p(H_Good={good_state}) = {good_probability:.4f}")

            plot_histogram(counts, title="Grover Search (3-bit, 1 iteration, H_Good=101)")
            plt.tight_layout()
            plt.savefig("grover_circuit_histogram.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Histogram saved: grover_circuit_histogram.png")
            print("=" * 70)
        except Exception as exc:
            print("\nQiskit demo skipped (runtime failed):", exc)

    # Note: Execution is left active or inactive based on user preference.
    run_qiskit_demo = False
    if run_qiskit_demo:
        run_optional_qiskit_demo()
