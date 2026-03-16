import numpy as np

class FOQAEngine:
    """
    Core engine for Fixed-Point Oblivious Amplitude Amplification.
    Handles tripartite LCU dynamics and non-linear success recurrences.
    """
    def __init__(self, theta: float):
        if not (0.0 <= theta < np.pi / 2.0):
            raise ValueError("Theta must be in [0, pi/2).")
        self.theta = theta

    @staticmethod
    def generate_mizel_schedule(c: float, iterations: int) -> np.ndarray:
        """Generates the optimal critical damping schedule: c/sqrt(n+1)."""
        return c / np.sqrt(np.arange(1, iterations + 1, dtype=float))

    @staticmethod
    def generate_constant_schedule(alpha: float, iterations: int) -> np.ndarray:
        """Generates a constant Zeno-like or underdamped schedule."""
        return np.full(iterations, alpha, dtype=float)

    def build_lcu_split_operator(self, alpha_n: float) -> np.ndarray:
        """Constructs the controlled-V_n operator for the tripartite system."""
        v_n = np.array([
            [np.cos(alpha_n), -np.sin(alpha_n)],
            [np.sin(alpha_n),  np.cos(alpha_n)],
        ], dtype=complex)

        ket_0 = np.array([1.0, 0.0], dtype=complex)
        ket_1 = np.array([0.0, 1.0], dtype=complex)
        proj_idx_0 = np.outer(ket_0, ket_0.conj())
        proj_idx_1 = np.outer(ket_1, ket_1.conj())
        i_2 = np.eye(2, dtype=complex)

        # Controlled on index == |0>: V_n ⊗ |0><0| ⊗ I + I ⊗ |1><1| ⊗ I
        return np.kron(v_n, np.kron(proj_idx_0, i_2)) + \
               np.kron(i_2, np.kron(proj_idx_1, i_2))

    def simulate_recurrence(self, alpha_schedule: np.ndarray) -> np.ndarray:
        """
        Executes the non-linear FOQA recurrence relations.
        Returns the cumulative success probability array.
        """
        t_n = float(np.sin(self.theta))
        s_n = float(np.cos(self.theta))

        prob_already_halted = 0.0
        prob_continue = 1.0
        cumulative_success = np.zeros(len(alpha_schedule), dtype=float)

        for idx, alpha in enumerate(alpha_schedule):
            cumulative_success[idx] = prob_already_halted + prob_continue * (t_n**2)

            p_step = (np.sin(alpha) ** 2) * (t_n**2)
            p_step = float(np.clip(p_step, 0.0, 1.0 - 1e-15))

            prob_already_halted += prob_continue * p_step
            prob_continue *= 1.0 - p_step

            norm = np.sqrt(max(1e-15, 1.0 - p_step))
            t_new = (t_n * np.cos(alpha) * np.cos(2.0 * self.theta) + s_n * np.sin(2.0 * self.theta)) / norm
            s_new = (-t_n * np.cos(alpha) * np.sin(2.0 * self.theta) + s_n * np.cos(2.0 * self.theta)) / norm
            
            t_n, s_n = float(t_new), float(s_new)

        return cumulative_success