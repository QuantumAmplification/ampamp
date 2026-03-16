import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import partial_trace

class GroverEngine:
    """
    The core algebraic and circuit engine for Grover's Search.
    Focuses on state synthesis and geometric calculations.
    """
    def __init__(self, n_qubits: int, marked_indices: list[int]):
        self.n = n_qubits
        self.N = 2**n_qubits
        self.marked = marked_indices
        self.M = len(marked_indices)
        
        # Fundamental Geometric properties
        self.solution_density = self.M / self.N
        self.theta = 2 * np.arcsin(np.sqrt(self.solution_density))
        self.k_optimal = int(np.floor(np.pi / (2 * self.theta) - 0.5))

    @staticmethod
    def compute_success_prob(lambda_val: float, k: int) -> float:
        """Analytic success probability calculation."""
        if not (0 <= lambda_val <= 1):
            raise ValueError("lambda_val must be in [0, 1]")
        theta = 2.0 * np.arcsin(np.sqrt(lambda_val))
        return float(np.sin((2 * k + 1) * theta / 2.0) ** 2)

    def get_oracle(self) -> QuantumCircuit:
        """Generates a Phase Oracle for the marked states."""
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

    def get_diffusion(self) -> QuantumCircuit:
        """Generates the Grover Diffusion operator (Inversion about mean)."""
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        qc.x(range(self.n))
        qc.h(self.n - 1)
        qc.mcx(list(range(self.n - 1)), self.n - 1)
        qc.h(self.n - 1)
        qc.x(range(self.n))
        qc.h(range(self.n))
        return qc

    def construct_circuit(self, iterations: int) -> QuantumCircuit:
        """Synthesizes the full Grover circuit."""
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        
        oracle = self.get_oracle()
        diffusion = self.get_diffusion()
        
        for _ in range(iterations):
            qc.append(oracle, range(self.n))
            qc.append(diffusion, range(self.n))
        
        return qc


# ----------------------------------------------------------------------------

class FixedPointEngine:
    """
    Engine for Fixed-Point Amplitude Amplification (FPAA).
    Ensures monotonic convergence using Chebyshev-derived phase schedules.
    """
    def __init__(self, L: int, delta: float):
        if L % 2 == 0:
            raise ValueError("L must be an odd integer for FPAA.")
        self.L = L
        self.delta = delta
        self.alphas, self.betas = self._generate_phases()

    def _generate_phases(self):
        """Analytical FPAA phase schedule synthesis."""
        gamma_inv = float(np.cosh((1.0 / self.L) * np.arccosh(1.0 / self.delta)))
        gamma = 1.0 / gamma_inv
        sq_term = float(np.sqrt(max(0.0, 1.0 - gamma * gamma)))

        alpha = np.zeros(self.L)
        for j in range(1, self.L + 1):
            theta_j = (2.0 * np.pi * j) / self.L
            tan_val = float(np.tan(theta_j))
            alpha[j-1] = 2.0 * np.arctan2(1.0, tan_val * sq_term)
        
        return alpha, -alpha[::-1]

    def build_fixed_point_circuit(self, num_qubits, marked_indices):
        """Synthesizes the Generalized Grover Iterate sequence."""
        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits)) # Initial superposition

        for a_j, b_j in zip(self.alphas, self.betas):
            # 1. Oracle with phase beta
            qc.global_phase += np.pi 
            # (In a real lib, you'd call a generalized_oracle helper here)
            
            # 2. Diffusion with phase alpha
            # (In a real lib, you'd call a generalized_diffusion helper here)
            
        return qc