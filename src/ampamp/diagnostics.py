import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace

class GroverAuditor:
    """
    Diagnostic suite for Grover's Algorithm.
    Handles the 'Soufflé' problem, cloning barriers, and subspace verification.
    """
    def __init__(self, engine):
        """
        Args:
            engine (GroverEngine): An instance of the foundations.GroverEngine
        """
        self.engine = engine
        self.backend = AerSimulator()
        
        # Basis vectors for subspace projection
        self.good_vec = np.zeros(engine.N, dtype=complex)
        self.bad_vec = np.zeros(engine.N, dtype=complex)
        
        for idx in engine.marked:
            self.good_vec[idx] = 1.0 / np.sqrt(engine.M)
        for idx in range(engine.N):
            if idx not in engine.marked:
                self.bad_vec[idx] = 1.0 / np.sqrt(engine.N - engine.M)

    def verify_subspace_rotation(self, max_k: int):
        """
        Audits the Invariant Subspace Theorem.
        Verifies that |a|^2 + |b|^2 = 1.0 throughout the rotation.
        """
        results = {"a_k": [], "b_k": [], "purity": [], "success": []}
        
        for k in range(max_k + 1):
            qc = self.engine.construct_circuit(k)
            qc.save_statevector()
            
            job = self.backend.run(transpile(qc, self.backend))
            state_k = np.array(job.result().get_statevector())
            
            # Project onto good/bad basis
            ak = np.dot(self.good_vec.conj(), state_k)
            bk = np.dot(self.bad_vec.conj(), state_k)
            
            results["a_k"].append(ak)
            results["b_k"].append(bk)
            results["purity"].append(np.abs(ak)**2 + np.abs(bk)**2)
            results["success"].append(np.abs(ak)**2)
            
        return results

    def run_cloning_test(self, max_k: int):
        """
        Empirical proof of the No-Cloning Theorem.
        Measures purity collapse after a naive CNOT copy attempt.
        """
        purity_original = []
        purity_cloned = []
        
        oracle = self.engine.get_oracle()
        diff = self.engine.get_diffusion()

        for k in range(max_k + 1):
            # 1. State before copy
            qc_orig = self.engine.construct_circuit(k)
            qc_orig.save_statevector()
            state_orig = self.backend.run(transpile(qc_orig, self.backend)).result().get_statevector()
            purity_original.append(np.real(np.trace(np.dot(state_orig.data, state_orig.data))))

            # 2. State after naive CNOT copy
            qc_copy = QuantumCircuit(self.engine.n * 2)
            qc_copy.h(range(self.engine.n))
            for _ in range(k):
                qc_copy.append(oracle, range(self.engine.n))
                qc_copy.append(diff, range(self.engine.n))
            
            for i in range(self.engine.n):
                qc_copy.cx(i, i + self.engine.n)
            
            qc_copy.save_statevector()
            state_full = self.backend.run(transpile(qc_copy, self.backend)).result().get_statevector()
            
            # Trace out the copy to see what happened to the original
            rho_after = partial_trace(state_full, range(self.engine.n, 2 * self.engine.n))
            purity_cloned.append(np.real(np.trace(np.dot(rho_after.data, rho_after.data))))
            
        return purity_original, purity_cloned

    @staticmethod
    def generate_souffle_heatmap(k_max: int, lambda_max: float, res: int = 200):
        """Vectorized analytic heatmap of the Soufflé Problem."""
        k_vals = np.arange(0, k_max)
        l_vals = np.linspace(0.001, lambda_max, res)
        K, L = np.meshgrid(k_vals, l_vals)
        
        theta = 2 * np.arcsin(np.sqrt(L))
        Z = np.sin((2*K + 1) * theta / 2)**2
        return K, L, Z
    

# ------------------------------------------------------------------------------------------

class FPAAAuditor:
    """Diagnostic suite for FPAA robustness and resource costs."""
    
    def __init__(self, engine: FixedPointEngine):
        self.engine = engine

    def audit_passband(self, lambda_range=None):
        """Analyzes the 'Plateau' behavior where success prob stays near 1."""
        if lambda_range is None:
            lambda_range = np.linspace(0.001, 1.0, 500)
            
        # Implementation of your SU(2) simulation logic here
        # Verification of the 'Floor' 1 - delta^2
        pass

    def estimate_ftqc_cost(self, synthesis_epsilon=1e-3):
        """Calculates T-gate overhead for Clifford+T fault-tolerant architectures."""
        # Implementation of your Module 6 logic
        pass


class ObliviousAuditor:
    """Diagnostic suite for verifying OAA and LCU block-encodings."""
    
    def __init__(self, engine: ObliviousEngine):
        self.engine = engine

    def run_acid_test(self, num_states=10):
        """
        Verifies the 'Equivalence Theorem': OAA success probability 
        is independent of the input state.
        """
        # Logic from your run_acid_tests function
        pass

    def verify_lcu_distance(self, actual_matrix, target_hamiltonian, alpha):
        """
        Measures ||M_TL - H/alpha||_F to verify Linear Combination of Unitaries accuracy.
        """
        h_norm = target_hamiltonian / alpha
        distance = np.linalg.norm(actual_matrix - h_norm)
        return distance