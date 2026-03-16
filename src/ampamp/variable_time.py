import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple
from qiskit import QuantumCircuit, QuantumRegister

@dataclass(frozen=True)
class VariableTimeBranch:
    """One branch of a variable-time algorithm."""
    stop_time: float
    weight: float
    success_given_branch: float

class VTAAEngine:
    """
    Core engine for Variable-Time Amplitude Amplification.
    Calculates asymptotic scaling, expected costs, and branch mass.
    """
    def __init__(self, branches: Sequence[VariableTimeBranch]):
        if not branches:
            raise ValueError("At least one branch is required.")
            
        ordered = sorted(branches, key=lambda b: float(b.stop_time))
        self.stop_times = np.array([float(b.stop_time) for b in ordered], dtype=float)
        
        weights = np.array([float(b.weight) for b in ordered], dtype=float)
        total_w = float(np.sum(weights))
        self.weights = weights / total_w
        
        self.success_given_branch = np.array([float(b.success_given_branch) for b in ordered], dtype=float)
        
        self.good_mass = self.weights * self.success_given_branch
        self.bad_mass = self.weights * (1.0 - self.success_given_branch)
        self.p_success = float(np.sum(self.good_mass))
        self.theta = float(np.arcsin(np.sqrt(self.p_success))) if self.p_success > 0 else 0.0

    def stopping_time_moments(self) -> Tuple[float, float, float]:
        """Return (E[T], sqrt(E[T^2]), T_max) under the branch distribution."""
        t_mean = float(np.sum(self.weights * self.stop_times))
        t_rms = float(np.sqrt(np.sum(self.weights * (self.stop_times ** 2))))
        t_max = float(np.max(self.stop_times))
        return t_mean, t_rms, t_max

    def vtaa_asymptotic_bound(self, polylog_factor: float = 1.0, c_tmax: float = 1.0, c_trms: float = 1.0) -> float:
        """Parameterized Ambainis-style VTAA scaling estimate."""
        if self.p_success <= 0.0:
            return float("inf")
        _, t_rms, t_max = self.stopping_time_moments()
        return float(polylog_factor * (c_tmax * t_max + c_trms * (t_rms / np.sqrt(self.p_success))))

    @staticmethod
    def build_staged_state_circuit(p_s1: float, p_fail_cond: float) -> QuantumCircuit:
        """
        Synthesizes a coherent variable-time state with flag registers.
        Encoding: |00> continue, |01> success, |10> fail.
        """
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