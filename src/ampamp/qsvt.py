"""Quantum Singular Value Transformation module.

Provides algebraic and compilation utilities for SU(2) signal processing
and optimal Chebyshev approximating series for QSVT block encodings.
"""

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval, Chebyshev
from scipy.special import jv

class SU2QSPEngine:
    """Forward-model SU(2) compiler for Quantum Signal Processing."""
    
    @staticmethod
    def w_signal(x: float) -> np.ndarray:
        """Return the signal unitary $W(x)$.

        Args:
            x (float): The singular value or signal parameter $x \\in [-1, 1]$.

        Returns:
            np.ndarray: The $2 \\times 2$ unitary block embedding $x$.
        """
        x = float(np.clip(x, -1.0, 1.0))
        y = np.sqrt(max(0.0, 1.0 - x * x))
        return np.array([[x, 1j * y], [1j * y, x]], dtype=complex)

    @staticmethod
    def z_rotation(phi: float) -> np.ndarray:
        """Return the phase rotation $\\exp(i \\phi Z)$.

        Args:
            phi (float): The rotation angle $\\phi$.

        Returns:
            np.ndarray: The $2 \\times 2$ phase rotation unitary.
        """
        return np.array([
            [np.exp(1j * phi), 0.0], 
            [0.0, np.exp(-1j * phi)]
        ], dtype=complex)

    @classmethod
    def evaluate_sequence(cls, phases: np.ndarray, x_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates the QSP sequence and extracts the $P(x)$ and $Q(x)$ polynomials.

        Args:
            phases (np.ndarray): Array of phase shifts representing the sequence.
            x_vals (np.ndarray): Array of target evaluation points $x \\in [-1, 1]$.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing `(P_vals, Q_vals)` 
            corresponding to the output polynomials.
        """
        p_vals = np.zeros(len(x_vals), dtype=complex)
        q_vals = np.zeros(len(x_vals), dtype=complex)

        for idx, x in enumerate(x_vals):
            u = cls.z_rotation(phases[0])
            wx = cls.w_signal(x)
            for phi in phases[1:]:
                u = u @ wx @ cls.z_rotation(phi)

            p_vals[idx] = u[0, 0]
            y = np.sqrt(max(0.0, 1.0 - x * x))
            q_vals[idx] = 0.0j if y <= 1e-15 else u[1, 0] / (1j * y)

        return p_vals, q_vals

class QSVTSynthesizer:
    """Generates optimal Chebyshev approximations for QSVT target functions."""

    @staticmethod
    def synthesize_jacobi_anger(degree: int, time: float) -> tuple[Chebyshev, Chebyshev, float]:
        """Constructs parity-split Chebyshev series for Hamiltonian Simulation.

        Utilizes the Jacobi-Anger expansion to estimate coefficients.

        Args:
            degree (int): The cutoff degree for the expanded series.
            time (float): The evolution time $t$ for simulating $e^{iHt}$.

        Returns:
            tuple[Chebyshev, Chebyshev, float]: A tuple containing the even 
                parity series, odd parity series, and the LCU norm bound $\\alpha$.
        """
        cos_coeffs = np.zeros(degree + 1, dtype=float)
        sin_coeffs = np.zeros(degree + 1, dtype=float)

        cos_coeffs[0] = float(jv(0, time))
        for k in range(1, degree + 1):
            if k % 2 == 0:
                cos_coeffs[k] = 2.0 * ((-1) ** (k // 2)) * float(jv(k, time))
            else:
                sin_coeffs[k] = 2.0 * ((-1) ** ((k - 1) // 2)) * float(jv(k, time))

        lcu_alpha = float(max(1.0, np.sum(np.abs(cos_coeffs)) + np.sum(np.abs(sin_coeffs))))
        return Chebyshev(cos_coeffs), Chebyshev(sin_coeffs), lcu_alpha

    @staticmethod
    def synthesize_matrix_inverse(degree: int, kappa: float, scale_factor: float = 0.8) -> np.ndarray:
        """Constructs an odd Chebyshev polynomial for HHL 2.0 Matrix Inversion.

        Approximates $1/x$ optimally outside the gap region.

        Args:
            degree (int): The odd degree limits.
            kappa (float): The condition number $\\kappa > 1$.
            scale_factor (float): Additional scale reduction buffer factor. Defaults to 0.8.

        Returns:
            np.ndarray: List of coefficients for the optimal Chebyshev sum.

        Raises:
            ValueError: If the input degree is not logically odd.
        """
        if degree % 2 == 0:
            raise ValueError("Degree must be odd for matrix inversion.")
            
        gap = 1.0 / kappa
        x_eval = np.linspace(-1.0, 1.0, 2001)
        y_target = np.zeros_like(x_eval)
        
        outside = np.abs(x_eval) >= gap
        inside = ~outside
        y_target[outside] = scale_factor * (1.0 / (kappa * x_eval[outside]))
        y_target[inside] = scale_factor * (kappa * x_eval[inside])

        fit_weights = np.ones_like(x_eval)
        fit_weights[outside] = 16.0  # Emphasize outside-gap accuracy
        
        coeffs = chebfit(x_eval, y_target, degree, w=fit_weights)
        coeffs[::2] = 0.0  # Enforce strict odd parity
        
        # Unitarity scaling
        max_amp = float(np.max(np.abs(chebval(x_eval, coeffs))))
        if max_amp > 1.0:
            coeffs /= (max_amp + 1e-12)
            
        return coeffs