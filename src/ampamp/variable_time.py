"""Variable-Time Amplitude Amplification module.

Provides classes and utilities for evaluating and constructing 
variable-time quantum search algorithms.
"""

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple
from qiskit import QuantumCircuit, QuantumRegister

@dataclass(frozen=True, init=False)
class VariableTimeBranch:
    """One branch of a variable-time algorithm."""
    stopping_time: float
    weight: float
    p_success: float

    def __init__(
        self,
        stopping_time: float | None = None,
        weight: float | None = None,
        p_success: float | None = None,
        *,
        stop_time: float | None = None,
        success_given_branch: float | None = None,
    ) -> None:
        """Create a VTAA branch with stable public names.

        Preferred field names are `stopping_time` and `p_success`. The legacy
        aliases `stop_time` and `success_given_branch` are accepted for
        backward compatibility.
        """
        if stopping_time is None:
            stopping_time = stop_time
        elif stop_time is not None and float(stopping_time) != float(stop_time):
            raise ValueError("Provide either stopping_time or stop_time, not conflicting values.")

        if p_success is None:
            p_success = success_given_branch
        elif success_given_branch is not None and float(p_success) != float(success_given_branch):
            raise ValueError("Provide either p_success or success_given_branch, not conflicting values.")

        if stopping_time is None or weight is None or p_success is None:
            raise TypeError("VariableTimeBranch requires stopping_time, weight, and p_success.")

        object.__setattr__(self, "stopping_time", float(stopping_time))
        object.__setattr__(self, "weight", float(weight))
        object.__setattr__(self, "p_success", float(p_success))

    @property
    def stop_time(self) -> float:
        """Legacy alias for backward compatibility."""
        return self.stopping_time

    @property
    def success_given_branch(self) -> float:
        """Legacy alias for backward compatibility."""
        return self.p_success

class VTAAEngine:
    """Core engine for Variable-Time Amplitude Amplification.

    Calculates asymptotic scaling, expected costs, and branch mass.
    """
    def __init__(self, branches: Sequence[VariableTimeBranch]):
        """Initializes the VTAA Engine.

        Args:
            branches (Sequence[VariableTimeBranch]): A sequence of branches defining 
                the variable-time algorithm.

        Raises:
            ValueError: If the branches sequence is empty.
        """
        if not branches:
            raise ValueError("At least one branch is required.")
            
        ordered = sorted(branches, key=lambda b: float(b.stopping_time))
        self.stopping_times = np.array([float(b.stopping_time) for b in ordered], dtype=float)
        self.stop_times = self.stopping_times
        
        weights = np.array([float(b.weight) for b in ordered], dtype=float)
        if np.any(weights < 0):
            raise ValueError("Branch weights cannot be negative.")
        total_w = float(np.sum(weights))
        if total_w <= 0.0:
            raise ValueError("Total branch weight must be definitively positive.")
        self.weights = weights / total_w
        
        self.branch_success_probabilities = np.array([float(b.p_success) for b in ordered], dtype=float)
        self.success_given_branch = self.branch_success_probabilities
        if np.any((self.branch_success_probabilities < 0.0) | (self.branch_success_probabilities > 1.0)):
            raise ValueError("p_success probabilities must be in [0, 1].")
        
        self.good_mass = self.weights * self.branch_success_probabilities
        self.bad_mass = self.weights * (1.0 - self.branch_success_probabilities)
        self.p_success = float(np.sum(self.good_mass))
        self.theta = float(np.arcsin(np.sqrt(self.p_success))) if self.p_success > 0 else 0.0

    def stopping_time_moments(self) -> Tuple[float, float, float]:
        """Return the moments of the stopping time distribution.

        Returns:
            Tuple[float, float, float]: A tuple containing $(E[T], \\sqrt{E[T^2]}, T_{max})$ 
                under the branch distribution.
        """
        t_mean = float(np.sum(self.weights * self.stopping_times))
        t_rms = float(np.sqrt(np.sum(self.weights * (self.stopping_times ** 2))))
        t_max = float(np.max(self.stopping_times))
        return t_mean, t_rms, t_max

    def vtaa_asymptotic_bound(self, polylog_factor: float = 1.0, c_tmax: float = 1.0, c_trms: float = 1.0) -> float:
        """Parameterized Ambainis-style VTAA scaling estimate.

        Args:
            polylog_factor (float): The logarithmic overhead factor. Defaults to 1.0.
            c_tmax (float): The constant scaling factor for $T_{max}$. Defaults to 1.0.
            c_trms (float): The constant scaling factor for $T_{rms}$. Defaults to 1.0.

        Returns:
            float: The asymptotic complexity bound. Returns infinity if success probability is non-positive.
        """
        if self.p_success <= 0.0:
            return float("inf")
        _, t_rms, t_max = self.stopping_time_moments()
        return float(polylog_factor * (c_tmax * t_max + c_trms * (t_rms / np.sqrt(self.p_success))))

    @staticmethod
    def build_staged_state_circuit(p_s1: float, p_fail_cond: float) -> QuantumCircuit:
        """Synthesizes a coherent variable-time state with flag registers.

        Encoding convention used:
        - $|00\\rangle$: continue
        - $|01\\rangle$: success
        - $|10\\rangle$: fail

        Args:
            p_s1 (float): The probability of early success at stage 1.
            p_fail_cond (float): The conditional probability of failure given continuation.

        Returns:
            QuantumCircuit: The synthesized variable-time quantum state circuit.
        """
        if not (0.0 <= p_s1 <= 1.0) or not (0.0 <= p_fail_cond <= 1.0):
            raise ValueError("Probabilities must safely reside within bounds [0, 1].")
            
        stage_reg = QuantumRegister(1, "stage_j")
        flag_reg = QuantumRegister(2, "flag")
        data_reg = QuantumRegister(1, "data")
        qc = QuantumCircuit(data_reg, flag_reg, stage_reg)

        # Stage 1
        qc.ry(2.0 * np.arcsin(np.sqrt(p_s1)), flag_reg[0])
        qc.x(flag_reg[0])
        qc.cx(flag_reg[0], stage_reg[0])
        qc.x(flag_reg[0])

        # Stage 2
        qc.x(flag_reg[0])
        qc.cry(2.0 * np.arcsin(np.sqrt(p_fail_cond)), flag_reg[0], flag_reg[1])
        qc.x(flag_reg[0])

        qc.x(flag_reg[1])
        qc.ccx(stage_reg[0], flag_reg[1], flag_reg[0])
        qc.x(flag_reg[1])
        
        return qc
