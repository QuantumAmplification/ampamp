from __future__ import annotations

"""Fixed-Point Amplitude Amplification module.

Provides the `FixedPointEngine` for running algorithms with monotonic convergence.
"""

import numpy as np
from qiskit import QuantumCircuit

class FixedPointEngine:
    """Engine for Fixed-Point Amplitude Amplification (FPAA).

    Ensures monotonic convergence using Chebyshev-derived phase schedules.
    """
    def __init__(self, L: int, delta: float):
        """Initializes the Fixed-Point Engine.

        Args:
            L (int): The odd Chebyshev degree from the Yoder-Low-Chuang
                construction. The synthesized circuit uses (L - 1) / 2
                generalized Grover iterates.
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
        self.num_grover_iterates = (L - 1) // 2
        self.gamma = self._compute_gamma()
        self.lambda_min = float(1.0 - self.gamma * self.gamma)
        self.zetas = self._generate_zetas()
        self.alphas, self.betas = self._generate_phase_pairs()

    def _compute_gamma(self) -> float:
        """Return the Yoder-Low-Chuang gamma parameter.

        The paper defines gamma through gamma^{-1} = T_{1/L}(1 / delta),
        where T_a(x) is the fractional-degree Chebyshev continuation.
        Since delta is in (0, 1], the argument 1 / delta is >= 1 and the
        continuation is expressed with cosh/arccosh.
        """
        chebyshev_fractional = float(np.cosh(np.arccosh(1.0 / self.delta) / self.L))
        return 1.0 / chebyshev_fractional

    @staticmethod
    def _chebyshev_polynomial(degree: int, x: np.ndarray | float) -> np.ndarray | float:
        """Evaluate T_degree(x) on scalars or NumPy arrays."""
        x_arr = np.asarray(x, dtype=float)
        result = np.empty_like(x_arr)

        inside = np.abs(x_arr) <= 1.0
        result[inside] = np.cos(degree * np.arccos(x_arr[inside]))

        outside = ~inside
        if np.any(outside):
            abs_x = np.abs(x_arr[outside])
            continuation = np.cosh(degree * np.arccosh(abs_x))
            signs = np.where((x_arr[outside] < 0.0) & (degree % 2 == 1), -1.0, 1.0)
            result[outside] = signs * continuation

        if np.isscalar(x):
            return float(result.item())
        return result

    def _generate_phase_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the YLC generalized-Grover phase pairs.

        For odd Chebyshev degree L = 2l + 1, the fixed-point sequence uses
        l generalized Grover iterates. Their phases satisfy the matching
        condition alpha_j = -beta_{l-j+1}.

        Returns:
            tuple: A tuple containing two arrays `(alpha, beta)`:
                - alpha (np.ndarray): Phase array for diffusion operators.
                - beta (np.ndarray): Phase array for oracle operators.
        """
        sq_term = float(np.sqrt(max(0.0, 1.0 - self.gamma * self.gamma)))
        alphas = np.zeros(self.num_grover_iterates, dtype=float)

        for j in range(1, self.num_grover_iterates + 1):
            tangent = float(np.tan(2.0 * np.pi * j / self.L))
            alphas[j - 1] = 2.0 * np.arctan2(1.0, tangent * sq_term)

        betas = -alphas[::-1]
        return alphas, betas

    def _generate_zetas(self) -> np.ndarray:
        """Build the palindromic zeta schedule from the YLC recurrence."""
        l = self.num_grover_iterates
        zetas = np.zeros(self.L, dtype=float)
        zetas[l] = ((-1) ** l) * (np.pi / 2.0)

        sq_term = float(np.sqrt(max(0.0, 1.0 - self.gamma * self.gamma)))
        increments = np.zeros(self.L - 1, dtype=float)
        for k in range(1, self.L):
            increments[k - 1] = ((-1) ** k) * np.pi - 2.0 * np.arctan2(
                1.0,
                float(np.tan(k * np.pi / self.L)) * sq_term,
            )

        for k in range(l, 0, -1):
            zetas[k - 1] = zetas[k] - increments[k - 1]
        for k in range(l + 1, self.L):
            zetas[k] = zetas[k - 1] + increments[k - 1]

        # The paper states that zeta_k = zeta_{L-k+1}. We enforce that
        # symmetry numerically to avoid tiny floating-point asymmetries.
        return 0.5 * (zetas + zetas[::-1])

    def success_probability(self, lambda_val: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the exact YLC success polynomial P_L(lambda)."""
        lambda_arr = np.asarray(lambda_val, dtype=float)
        if np.any((lambda_arr < 0.0) | (lambda_arr > 1.0)):
            raise ValueError("lambda_val must be in [0, 1].")

        argument = (1.0 / self.gamma) * np.sqrt(np.clip(1.0 - lambda_arr, 0.0, 1.0))
        chebyshev = self._chebyshev_polynomial(self.L, argument)
        probability = 1.0 - (self.delta ** 2) * np.square(chebyshev)
        probability = np.clip(probability, 0.0, 1.0)

        if np.isscalar(lambda_val):
            return float(probability.item())
        return probability

    def build_fixed_point_circuit(self, num_qubits: int, marked_indices: list[int]) -> QuantumCircuit:
        """Synthesizes the Generalized Grover Iterate sequence.

        Constructs the complete quantum circuit utilizing the derived YLC phase
        pairs. For odd Chebyshev degree L, this sequence contains (L - 1) / 2
        generalized Grover iterates.

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
            # Apply the target-state phase reflection S_t(beta_j).
            oracle_diag = np.ones(N, dtype=complex)
            for m in marked_indices:
                oracle_diag[m] = np.exp(1j * float(b_j))
            qc.append(Diagonal(oracle_diag), range(num_qubits))

            # Apply the source-state phase reflection S_s(alpha_j).
            qc.h(range(num_qubits))
            diff_diag = np.ones(N, dtype=complex)
            diff_diag[0] = np.exp(-1j * float(a_j))
            qc.append(Diagonal(diff_diag), range(num_qubits))
            qc.h(range(num_qubits))

        return qc

# ----------------------------------------------------------------------------
