"""Fixed-Point Oblivious Amplitude Amplification module.

Provides the `FOQAEngine` to analyze and simulate LCU dynamics 
for oblivious amplitude amplification with fixed-point success guarantees.
"""

import numpy as np
from qiskit import QuantumCircuit

class FOQAEngine:
    """Core engine for Fixed-Point Oblivious Amplitude Amplification.

    Handles tripartite LCU dynamics and non-linear success recurrences.
    """
    def __init__(self, theta: float):
        """Initializes the FOQA Engine.

        Args:
            theta (float): The fundamental geometric phase angle bound for the oblivious operation.

        Raises:
            ValueError: If theta is not within [0, pi/2).
        """
        if not (0.0 <= theta < np.pi / 2.0):
            raise ValueError("Theta must be in [0, pi/2).")
        self.theta = theta

    @staticmethod
    def generate_mizel_schedule(c: float, iterations: int) -> np.ndarray:
        """Generates the optimal critical damping schedule: $c/\\sqrt{n+1}$.

        Args:
            c (float): The damping constant modifier.
            iterations (int): The total number of iterations.

        Returns:
            np.ndarray: The computed phase sequence schedule array.
        """
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        return c / np.sqrt(np.arange(1, iterations + 1, dtype=float))

    @staticmethod
    def generate_constant_schedule(alpha: float, iterations: int) -> np.ndarray:
        """Generates a constant Zeno-like or underdamped schedule.

        Args:
            alpha (float): The constant phase angle to apply.
            iterations (int): The total number of iterations.

        Returns:
            np.ndarray: A constant phase sequence schedule array.
        """
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        return np.full(iterations, alpha, dtype=float)

    def build_lcu_split_operator(self, alpha_n: float) -> np.ndarray:
        """Constructs the controlled-$V_n$ operator for the tripartite system.

        Args:
            alpha_n (float): The current phase step angle $\\alpha_n$.

        Returns:
            np.ndarray: The matrix representation of the split operator.
        """
        v_n = np.array([
            [np.cos(alpha_n), -np.sin(alpha_n)],
            [np.sin(alpha_n),  np.cos(alpha_n)],
        ], dtype=complex)

        ket_0 = np.array([1.0, 0.0], dtype=complex)
        ket_1 = np.array([0.0, 1.0], dtype=complex)
        proj_idx_0 = np.outer(ket_0, ket_0.conj())
        proj_idx_1 = np.outer(ket_1, ket_1.conj())
        i_2 = np.eye(2, dtype=complex)

        # Controlled on index == |1> (Marked state): V_n ⊗ |1><1| ⊗ I + I ⊗ |0><0| ⊗ I
        return np.kron(v_n, np.kron(proj_idx_1, i_2)) + \
               np.kron(i_2, np.kron(proj_idx_0, i_2))

    def simulate_recurrence(self, alpha_schedule: np.ndarray) -> np.ndarray:
        """Executes the non-linear FOQA recurrence relations.

        Args:
            alpha_schedule (np.ndarray): The schedule of phase angles $\\alpha_n$ to apply over iterations.

        Returns:
            np.ndarray: The cumulative success probability array containing the likelihood of 
                success at each step of the recurrence.
        """
        t_n = float(np.sin(self.theta))
        s_n = float(np.cos(self.theta))

        prob_already_halted = 0.0
        prob_continue = 1.0
        cumulative_success = np.zeros(len(alpha_schedule), dtype=float)

        for idx, alpha in enumerate(alpha_schedule):
            p_step = (np.sin(alpha) ** 2) * (t_n**2)
            p_step = float(np.clip(p_step, 0.0, 1.0 - 1e-15))

            prob_already_halted += prob_continue * p_step
            prob_continue *= 1.0 - p_step

            norm = np.sqrt(max(1e-15, 1.0 - p_step))
            t_new = (t_n * np.cos(alpha) * np.cos(2.0 * self.theta) + s_n * np.sin(2.0 * self.theta)) / norm
            s_new = (-t_n * np.cos(alpha) * np.sin(2.0 * self.theta) + s_n * np.cos(2.0 * self.theta)) / norm
            
            t_n, s_n = float(t_new), float(s_new)
            cumulative_success[idx] = prob_already_halted + prob_continue * (t_n**2)

        return cumulative_success

    def build_proxy_sequence(
        self,
        n_steps: int,
        mizel_c: float = 1.5,
        m_content: int = 1,
        zeno_alpha: float | None = None,
    ) -> QuantumCircuit:
        """Builds a transpilation-ready FOQA proxy sequence.

        Circuit order: ancilla, index, content[0..m-1].
        Each step applies a ctrl-0 RY(alpha_n) on ancilla conditioned by index qubit.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if m_content < 1:
            raise ValueError("m_content must be >= 1")

        n_total = 2 + int(m_content)
        qc = QuantumCircuit(n_total, name=f"FOQA_proxy_n{n_steps}")

        # Prepare proxy index state sin(theta)|0> + cos(theta)|1>.
        qc.ry(np.pi - 2.0 * float(self.theta), 1)

        for n in range(n_steps):
            if zeno_alpha is None:
                alpha = float(mizel_c) / np.sqrt(float(n + 1))
            else:
                alpha = float(zeno_alpha)

            # ctrl-0 on index qubit (1), target ancilla qubit (0)
            qc.x(1)
            qc.cry(alpha, 1, 0)
            qc.x(1)

        return qc
