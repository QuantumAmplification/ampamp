import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace, state_fidelity

class GroverGeometricLab:
    """
    Module for Section II: Grover's Algorithm.
    A rigorous numerical study of 2D geometry, instability, and cloning barriers.
    """
    def __init__(self, n_qubits, marked_indices):
        self.n = n_qubits
        self.N = 2**n_qubits
        self.marked = marked_indices
        self.M = len(marked_indices)
        self.lambda_val = self.M / self.N
        
        self.theta = 2 * np.arcsin(np.sqrt(self.lambda_val))
        # Optimal k* ≈ floor(pi/(4*arcsin(sqrt(lambda))) - 1/2) as provided in test cases
        self.k_optimal = int(np.floor(np.pi / (2 * self.theta) - 0.5))
        self.backend = AerSimulator()
        
        # Step A: Initialize the Invariant Vectors
        self.good_vec = np.zeros(self.N, dtype=complex)
        self.bad_vec = np.zeros(self.N, dtype=complex)
        
        for idx in self.marked:
            self.good_vec[idx] = 1.0 / np.sqrt(self.M)
            
        for idx in range(self.N):
            if idx not in self.marked:
                self.bad_vec[idx] = 1.0 / np.sqrt(self.N - self.M)

    def grover_success_prob(self, lambda_value: float, k: int) -> float:
        """Standard Grover success probability in terms of lambda = M/N."""
        if lambda_value < 0 or lambda_value > 1:
            raise ValueError("lambda_value must be in [0, 1]")
        if lambda_value == 0:
            return 0.0

        theta = 2.0 * np.arcsin(np.sqrt(lambda_value))
        angle = (2 * k + 1) * theta / 2.0
        return float(np.sin(angle) ** 2)
        
        # Step A: Initialize the Invariant Vectors
        self.good_vec = np.zeros(self.N, dtype=complex)
        self.bad_vec = np.zeros(self.N, dtype=complex)
        
        for idx in self.marked:
            self.good_vec[idx] = 1.0 / np.sqrt(self.M)
            
        for idx in range(self.N):
            if idx not in self.marked:
                self.bad_vec[idx] = 1.0 / np.sqrt(self.N - self.M)

    def get_oracle(self):
        """Standard Phase Oracle for Section II."""
        qc = QuantumCircuit(self.n)
        for index in self.marked:
            target_bin = format(index, f'0{self.n}b')[::-1]
            for i, bit in enumerate(target_bin):
                if bit == '0': qc.x(i)
            qc.h(self.n - 1)
            qc.mcx(list(range(self.n - 1)), self.n - 1)
            qc.h(self.n - 1)
            for i, bit in enumerate(target_bin):
                if bit == '0': qc.x(i)
        return qc

    def get_diffusion(self):
        """Standard Diffusion Reflection."""
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        qc.x(range(self.n))
        qc.h(self.n - 1)
        qc.mcx(list(range(self.n - 1)), self.n - 1)
        qc.h(self.n - 1)
        qc.x(range(self.n))
        qc.h(range(self.n))
        return qc

    def analyze_geometry(self, max_k):
        """Module 1 & 2: Tracks 2D rotation and Soufflé instability using linear algebra projection."""
        a_k_vals = []
        b_k_vals = []
        p_total_vals = []
        success_probs = []
        
        for k in range(max_k + 1):
            qc = QuantumCircuit(self.n)
            qc.h(range(self.n))
            
            if k > 0:
                oracle = self.get_oracle()
                diff = self.get_diffusion()
                for _ in range(k):
                    qc.append(oracle, range(self.n))
                    qc.append(diff, range(self.n))
            
            # Step B: Iterative Statevector Extraction
            qc.save_statevector()
            result = self.backend.run(transpile(qc, self.backend)).result()
            state_k = np.array(result.get_statevector())
            
            # Step C: The Projection Logic
            ak = np.dot(self.good_vec.conj(), state_k)
            bk = np.dot(self.bad_vec.conj(), state_k)
            
            prob_in_subspace = np.abs(ak)**2 + np.abs(bk)**2
            
            a_k_vals.append(ak)
            b_k_vals.append(bk)
            p_total_vals.append(prob_in_subspace)
            
            # The actual success probability is |ak|^2
            success_probs.append(np.abs(ak)**2)
            
        return np.array(a_k_vals), np.array(b_k_vals), np.array(p_total_vals), success_probs

    def simulate_cloning_barrier(self, max_k):
        """Module 3: Proves that naive CNOT cloning fails due to entanglement."""
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

    def sensitivity_heatmap(self, k_max, lambda_max, resolution=500):
        """Module 5: Generates the 2D Soufflé Instability Heatmap."""
        # 1. Create the Meshgrid
        k_range = np.arange(0, k_max)
        lambda_range = np.linspace(0.001, lambda_max, resolution)
        K, L = np.meshgrid(k_range, lambda_range)
        
        # 2. Vectorized Analytic Calculation
        theta = 2 * np.arcsin(np.sqrt(L))
        success_heatmap = np.sin((2*K + 1) * theta / 2)**2
        
        return K, L, success_heatmap

    def recursive_nesting_analysis(self, k1, k2, max_lambda):
        """Module 4: Demonstrates self-similar scaling and extreme gate depth in nested AA."""
        # 1. Analytical Evaluation for Smooth Curves
        lambda_vals = np.linspace(0.0001, max_lambda, 500)
        
        # Level 1 probability P1
        theta_1 = 2 * np.arcsin(np.sqrt(lambda_vals))
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
        # Equivalent total iterations for same boost k_eq ~ k1 * k2 (roughly)
        k_equiv = (2*k1 + 1) * k2
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
        
        return lambda_vals, p1_vals, p2_vals, depth_data

if __name__ == "__main__":
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

        lambda_value = M / N
        # We can leverage the existing formulas inside the Lab
        temp_lab = GroverGeometricLab(n_qubits=10, marked_indices=list(range(M)))
        k_star = temp_lab.k_optimal
        success_prob = temp_lab.grover_success_prob(lambda_value, k_star)

        print(f"  λ = M/N = {lambda_value:.6f}")
        print(f"  k* (near-optimal Grover iterations) = {k_star}")
        print(f"  Grover success probability at k* = {success_prob:.6f}")

    print("\n" + "=" * 70)
    print("Running Grover Geometric Lab Analysis...")
    lab = GroverGeometricLab(n_qubits=6, marked_indices=[10, 25])
    max_k = lab.k_optimal * 3

    print("Running Grover Geometric Lab Analysis...")
    a_vals, b_vals, p_total, probs = lab.analyze_geometry(max_k)
    pur_orig, pur_copy = lab.simulate_cloning_barrier(max_k)
    lam_vals, p1_curve, p2_curve, gate_depths = lab.recursive_nesting_analysis(k1=3, k2=3, max_lambda=0.15)
    
    heatmap_K, heatmap_L, success_heatmap = lab.sensitivity_heatmap(k_max=50, lambda_max=0.5, resolution=500)
    
    # Machine Precision Check
    mean_sq_dev = np.mean((1.0 - p_total)**2)
    print(f"Mean squared deviation from invariant subspace: {mean_sq_dev:.2e}")
    if mean_sq_dev < 1e-15:
        print("-> Confirmed: Invariant Subspace Theorem holds within double-precision limits.")

    print(f"Gate Depth Comparison for Equivalent Boost: Standard={gate_depths['std_depth']} gates vs Nested={gate_depths['nested_depth']} gates")

    # Visualization
    # Expand to a 2-row layout to fit the heatmap cleanly
    fig = plt.figure(figsize=(24, 10))

    # ---- TOP ROW (1x4 Grid) ----
    
    # Plot 1: Soufflé Problem
    ax1 = plt.subplot(2, 4, 1)

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
    ax1.set_ylabel("Probability of Success")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()

    # Plot 2: Invariant Subspace Proof
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(range(max_k + 1), p_total, color='blue', marker='x', label='|ak|^2 + |bk|^2')
    ax2.set_title("Invariant Subspace Verification (P = 1.0)")
    ax2.set_xlabel("Iterations (k)")
    ax2.set_ylabel("Total Probability in 2D Plane")
    ax2.set_ylim(0.9, 1.1)
    ax2.legend()

    # Plot 3: Recursive Nesting
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(lam_vals, p1_curve, color='blue', linestyle='--', label='Level 1 (k=3)')
    ax3.plot(lam_vals, p2_curve, color='red', linestyle='-', label='Level 2 (k1=3, k2=3)')
    
    ax3.set_title("Self-Similar Scaling (Sharpening Effect)")
    ax3.set_xlabel("Solution Density (λ = M/N)")
    ax3.set_ylabel("Success Probability P(λ)")
    ax3.grid(True)
    ax3.legend()

    # Plot 4: No-Cloning Barrier
    ax4 = plt.subplot(2, 4, 4)
    x = np.arange(len(pur_orig))
    width = 0.35
    
    ax4.bar(x - width/2, pur_orig, width, label='Original State Purity', color='lightgreen')
    ax4.bar(x + width/2, pur_copy, width, label='Purity After Naive Copy', color='salmon')
    ax4.axhline(0.833, color='red', linestyle='--', label='UQCM Limit (0.833)')
    
    ax4.set_title("Cloning Obstruction: Purity Collapse")
    ax4.set_xlabel("Iterations (k)")
    ax4.set_ylabel("Purity Value Tr(ρ^2)")
    ax4.set_ylim(0, 1.1)
    ax4.legend()

    # ---- BOTTOM ROW (Heatmap spanning multiple columns) ----
    
    # Plot 5: Soufflé Sensitivity Heatmap
    ax5 = plt.subplot(2, 1, 2)  # Spans the entire bottom row
    contour = ax5.pcolormesh(heatmap_L, heatmap_K, success_heatmap, shading='auto', cmap='inferno')
    fig.colorbar(contour, ax=ax5, label='Success Probability')
    
    ax5.set_title("Grover Sensitivity Heatmap: The Soufflé Islands", fontsize=15)
    ax5.set_xlabel(r"Solution Density ($\lambda = M/N$)", fontsize=12)
    ax5.set_ylabel("Iteration Count ($k$)", fontsize=12)

    plt.tight_layout()
    plt.savefig('grover_geometric_evidence.png')
    print("Saved plot to 'grover_geometric_evidence.png'")
    # plt.show()
    
    # ---------------------------------------------------------
    # Optional 3-qubit Qiskit Demo
    # ---------------------------------------------------------
    def run_optional_qiskit_demo() -> None:
        """
        Optional 3-qubit circuit demo for marked state '101'.
        """
        try:
            from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
            from qiskit.visualization import plot_histogram
            from qiskit_aer import AerSimulator
            
            data = QuantumRegister(3, "data")
            anc = QuantumRegister(1, "anc")
            creg = ClassicalRegister(3, "c")
            qc = QuantumCircuit(data, anc, creg)

            def apply_phase_oracle_mark_101(qcircuit: QuantumCircuit):
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
            apply_phase_oracle_mark_101(qc)
            apply_diffusion_operator_3q(qc)
            qc.measure(data, creg)

            backend = AerSimulator()
            compiled = transpile(qc, backend, optimization_level=1)
            shots = 4096
            result = backend.run(compiled, shots=shots).result()
            counts = result.get_counts()

            marked_state = "101"
            marked_probability = counts.get(marked_state, 0) / shots
            print("\n" + "=" * 70)
            print("Qiskit circuit demo (target=101):")
            print("Counts:", counts)
            print(f"P({marked_state}) = {marked_probability:.4f}")

            plot_histogram(counts, title="Grover Search (3-bit, 1 iteration, target=101)")
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
