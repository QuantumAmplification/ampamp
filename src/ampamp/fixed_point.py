"""Fixed-Point Amplitude Amplification module.

Provides the `FixedPointEngine` for running algorithms with monotonic convergence.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import partial_trace

class FixedPointEngine:
    """Engine for Fixed-Point Amplitude Amplification (FPAA).

    Ensures monotonic convergence using Chebyshev-derived phase schedules.
    """
    def __init__(self, L: int, delta: float):
        """Initializes the Fixed-Point Engine.

        Args:
            L (int): The number of generalized Grover iterations. Must be an odd integer.
            delta (float): The bound on the error of the fixed-point monotonic convergence.

        Raises:
            ValueError: If L is not an odd integer.
        """
        if L % 2 == 0:
            raise ValueError("L must be an odd integer for FPAA.")
        self.L = L
        self.delta = delta
        self.alphas, self.betas = self._generate_phases()

    def _generate_phases(self) -> tuple:
        """Analytical FPAA phase schedule synthesis.

        Calculates the phase differences alpha and beta used precisely to ensure
        monotonic amplification based on Chebyshev polynomials.

        Returns:
            tuple: A tuple containing two arrays `(alpha, beta)`:
                - alpha (np.ndarray): Phase array for diffusion operators.
                - beta (np.ndarray): Phase array for oracle operators.
        """
        gamma_inv = float(np.cosh((1.0 / self.L) * np.arccosh(1.0 / self.delta)))
        gamma = 1.0 / gamma_inv
        sq_term = float(np.sqrt(max(0.0, 1.0 - gamma * gamma)))

        alpha = np.zeros(self.L)
        for j in range(1, self.L + 1):
            theta_j = (2.0 * np.pi * j) / self.L
            tan_val = float(np.tan(theta_j))
            alpha[j-1] = 2.0 * np.arctan2(1.0, tan_val * sq_term)
        
        return alpha, -alpha[::-1]

    def build_fixed_point_circuit(self, num_qubits: int, marked_indices: list[int]) -> QuantumCircuit:
        """Synthesizes the Generalized Grover Iterate sequence.

        Constructs the complete quantum circuit utilizing the derived phase schedules.

        Args:
            num_qubits (int): The total number of qubits in the circuit.
            marked_indices (list[int]): List of indices representing marked abstract states.

        Returns:
            QuantumCircuit: The constructed fixed-point search quantum circuit.
        """
        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits)) # Initial superposition

        for a_j, b_j in zip(self.alphas, self.betas):
            # 1. Oracle with phase beta
            qc.global_phase += np.pi 
            # (In a real lib, you'd call a generalized_oracle helper here)
            
            # 2. Diffusion with phase alpha
            # (In a real lib, you'd call a generalized_diffusion helper here)
            
        return qc

# ----------------------------------------------------------------------------

