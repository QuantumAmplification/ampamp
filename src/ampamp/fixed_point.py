from __future__ import annotations

"""Fixed-Point Amplitude Amplification module.

Provides the `FixedPointEngine` for running algorithms with monotonic convergence.
"""

import numpy as np
from qiskit import QuantumCircuit
from .grover import GroverEngine

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
        if L < 1:
            raise ValueError("L must be an odd positive integer (L >= 1).")
        if L % 2 == 0:
            raise ValueError("L must be an odd integer for FPAA.")
        if not (0.0 < delta <= 1.0):
            raise ValueError("delta must be in the range (0, 1].")
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
            theta_j = (2.0 * j - 1.0) * np.pi / (2.0 * self.L)
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
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
            
        try:
            from qiskit.circuit.library import DiagonalGate as Diagonal
        except ImportError:
            from qiskit.circuit.library import Diagonal

        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))

        N = 2 ** num_qubits

        for a_j, b_j in zip(self.alphas, self.betas):
            # Phase oracle: apply e^{i*beta} to marked states
            oracle_diag = np.ones(N, dtype=complex)
            for m in marked_indices:
                oracle_diag[m] = np.exp(1j * float(b_j))
            
            qc.append(Diagonal(oracle_diag), range(num_qubits))

            # Phase diffusion: H^n -> Phase on |0> -> H^n
            qc.h(range(num_qubits))
            diff_diag = np.ones(N, dtype=complex)
            diff_diag[0] = np.exp(1j * float(a_j))
            qc.append(Diagonal(diff_diag), range(num_qubits))
            qc.h(range(num_qubits))

        return qc

# ----------------------------------------------------------------------------
