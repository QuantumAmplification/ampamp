"""Variable-Time Amplitude Amplification (VTAA) laboratory.

This module mirrors the style of the earlier files while keeping the
mathematical objects explicit:

- A variable-time algorithm ``A`` is represented by branches ``i`` with
  stopping time ``t_i``, branch weight ``w_i``, and conditional success
  probability ``s_i``.
- The post-``A`` state is
      |psi> = sum_i sqrt(w_i) (sqrt(s_i)|i,Good> + sqrt(1-s_i)|i,Bad>)
  so total success probability is p = sum_i w_i s_i.
- Standard amplitude amplification acts on this state with the exact
  SU(2) formula
      p_k = sin^2((2k+1) * theta0), where sin^2(theta0)=p.

Standing notation aligned with final.tex:
- H_Good / H_Bad: target and non-target sectors
- Pi_Good / Pi_Bad: orthogonal projectors
- |All> and p are used as the canonical success-parameter view
- complexity discussed in oracle/query-call terms

The file provides:
1) exact branch-level/statistical identities,
2) rigorous Grover/AA probability dynamics,
3) comparison between restart, worst-case AA, and VTAA asymptotic scaling.

Note: The Ambainis VTAA theorem is asymptotic (polylog factors + constants).
This file reports those bounds explicitly as parameterized estimates rather
than claiming exact gate counts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from one_click_utils import start_one_click_session


@dataclass(frozen=True)
class VariableTimeBranch:
    """One branch of a variable-time algorithm.

    Attributes
    ----------
    stop_time:
        Halting time t_i for this branch (must be positive).
    weight:
        Branch probability mass w_i before halting logic; all weights are
        normalized to sum to 1.
    success_given_branch:
        Conditional success probability s_i in [0,1] once this branch halts.
    """

    stop_time: float
    weight: float
    success_given_branch: float


@dataclass
class VTAAReport:
    """Summary of exact metrics and strategy comparisons."""

    p_success: float
    # Legacy field name retained for compatibility: this is theta0 = arcsin(sqrt(p)).
    theta: float
    grover_k_opt: int
    grover_p_at_k_opt: float
    t_mean: float
    t_rms: float
    t_max: float
    expected_time_until_success_restart: float
    expected_time_until_success_worst_case_aa: float
    vtaa_asymptotic_bound: float


@dataclass
class SouffleProblemResults:
    """Over-rotation benchmark when guessed and actual H_Good-subspace counts differ."""

    n: int
    N: int
    guessed_m: int
    actual_m: int
    k_opt_guess: int
    k_values: np.ndarray
    actual_probs: np.ndarray
    guess_probs_theoretical: np.ndarray
    prob_at_halt: float
    peak_prob_before_halt: float
    collapse_from_peak: float


@dataclass
class FTQCScalingResults:
    """Fault-tolerant synthesis scaling for diffusion-core multi-control logic."""

    n_values_vchain: np.ndarray
    t_counts_vchain: np.ndarray
    ancilla_counts_vchain: np.ndarray
    n_values_noancilla: np.ndarray
    t_counts_noancilla: np.ndarray


@dataclass
class ExactAAResults:
    """Exact amplitude amplification via optimized final fractional phases."""

    n: int
    N: int
    m_good: int
    k_base: int
    k_peak_standard: int
    standard_probs: np.ndarray
    exact_oracle_phase: float
    exact_diffusion_phase: float
    exact_prob: float


@dataclass
class PhaseLeakageResults:
    """Rank-growth audit under phase mismatch and analog control skew."""

    n: int
    N: int
    k_max: int
    eps_oracle_deg: float
    eps_diff_deg: float
    crosstalk_oracle_deg: float
    local_z_detune_deg: float
    rank_threshold: float
    rank_ideal: np.ndarray
    rank_mismatch_only: np.ndarray
    rank_leaky: np.ndarray
    final_sv_ideal: np.ndarray
    final_sv_mismatch_only: np.ndarray
    final_sv_leaky: np.ndarray
    mismatch_only_rank2_all: bool
    leaky_exceeds_rank2: bool


@dataclass
class OpenSystemTrajectoryResults:
    """Open-system AA trajectory metrics under dephasing noise."""

    n: int
    N: int
    k_max: int
    phase_damp_1q: float
    phase_damp_2q: float
    good_state: str
    ideal_x: np.ndarray
    ideal_z: np.ndarray
    noisy_x: np.ndarray
    noisy_z: np.ndarray
    noisy_purity: np.ndarray
    trace_distance_to_plane: np.ndarray


@dataclass
class VTAA_StateSynthesisResults:
    """Empirical staged-state synthesis metrics for a variable-time unitary."""

    num_stages: int
    statevector_dimension: int
    success_probability: float
    continue_probability: float
    fail_probability: float


@dataclass
class VTAA_CostSweepResults:
    """VTAA asymptotic cost sweep versus worst-case standard AA."""

    total_ps: float
    early_success_ratios: np.ndarray
    standard_costs: np.ndarray
    vtaa_costs: np.ndarray


@dataclass
class SubspaceAuditResults:
    """SVD audit proving rank-2 trajectory confinement in large Hilbert space."""

    n: int
    N: int
    k_max: int
    good_state: str
    history_shape: tuple[int, int]
    singular_values: np.ndarray
    rank_threshold: float
    empirical_rank: int
    sigma3_to_sigma1: float
    float64_svd_floor: float


@dataclass
class PhaseStaircaseResults:
    """Empirical-vs-theoretical angular staircase in the {|B>,|G>} plane."""

    n: int
    N: int
    k_max: int
    good_state: str
    theta_0: float
    empirical_angles: np.ndarray
    theoretical_angles: np.ndarray
    angle_abs_error: np.ndarray
    max_abs_error: float
    good_fidelity: np.ndarray


class VariableTimeAmplitudeAmplificationLab:
    """Numerical lab for variable-time amplitude amplification."""

    def __init__(self, branches: Sequence[VariableTimeBranch]):
        if not branches:
            raise ValueError("At least one branch is required.")

        # Ensure increasing stopping-time order for deterministic reporting.
        ordered = sorted(branches, key=lambda b: float(b.stop_time))

        times = np.array([float(b.stop_time) for b in ordered], dtype=float)
        weights = np.array([float(b.weight) for b in ordered], dtype=float)
        success = np.array([float(b.success_given_branch) for b in ordered], dtype=float)

        if np.any(times <= 0):
            raise ValueError("All stop times must be > 0.")
        if np.any(weights < 0):
            raise ValueError("All weights must be >= 0.")
        if np.any((success < 0) | (success > 1)):
            raise ValueError("All success_given_branch values must be in [0, 1].")

        total_w = float(np.sum(weights))
        if total_w <= 0:
            raise ValueError("Sum of weights must be positive.")

        # Normalize branch distribution exactly once.
        weights = weights / total_w

        self.stop_times = times
        self.weights = weights
        self.success_given_branch = success

        # Branch-level H_Good/H_Bad mass decomposition.
        self.good_mass = self.weights * self.success_given_branch
        self.bad_mass = self.weights * (1.0 - self.success_given_branch)

        self.p_success = float(np.sum(self.good_mass))
        self.theta = float(np.arcsin(np.sqrt(self.p_success))) if self.p_success > 0 else 0.0
        self.theta0 = self.theta

        self._validate_exact_identities()

    # ------------------------------------------------------------------
    # Exact state model and identities
    # ------------------------------------------------------------------

    def state_after_A(self) -> np.ndarray:
        """Return |psi> in basis {|i,H_Good>, |i,H_Bad>} (dimension 2m)."""
        m = len(self.stop_times)
        psi = np.zeros(2 * m, dtype=complex)
        psi[0::2] = np.sqrt(self.good_mass)
        psi[1::2] = np.sqrt(self.bad_mass)
        return psi

    def success_projector(self) -> np.ndarray:
        """Projector onto H_Good spanned by {|i,H_Good>}."""
        m = len(self.stop_times)
        p = np.zeros((2 * m, 2 * m), dtype=complex)
        for i in range(m):
            p[2 * i, 2 * i] = 1.0
        return p

    def stopping_time_moments(self) -> Tuple[float, float, float]:
        """Return (E[T], sqrt(E[T^2]), T_max) under the branch distribution."""
        t_mean = float(np.sum(self.weights * self.stop_times))
        t_rms = float(np.sqrt(np.sum(self.weights * (self.stop_times ** 2))))
        t_max = float(np.max(self.stop_times))
        return t_mean, t_rms, t_max

    def cdf_by_stopping_stage(self) -> np.ndarray:
        """Cumulative halted mass F_j = sum_{i<=j} w_i in sorted-time order."""
        return np.cumsum(self.weights)

    def success_cdf_by_stage(self) -> np.ndarray:
        """Cumulative H_Good mass G_j = sum_{i<=j} w_i s_i."""
        return np.cumsum(self.good_mass)

    def _validate_exact_identities(self) -> None:
        psi = self.state_after_A()
        norm = float(np.vdot(psi, psi).real)
        if not np.isclose(norm, 1.0, atol=1e-12):
            raise AssertionError(f"State normalization failed: ||psi||^2={norm}")

        p_good_direct = float(np.sum(np.abs(psi[0::2]) ** 2))
        if not np.isclose(p_good_direct, self.p_success, atol=1e-12):
            raise AssertionError(
                f"Success decomposition mismatch: direct={p_good_direct}, mass={self.p_success}"
            )

    # ------------------------------------------------------------------
    # Standard amplitude amplification dynamics (exact)
    # ------------------------------------------------------------------

    def grover_success_probability(self, k: int) -> float:
        """Exact success probability after k Grover/AA iterates."""
        if k < 0:
            raise ValueError("k must be non-negative.")
        if self.p_success <= 0.0:
            return 0.0
        angle = (2 * int(k) + 1) * self.theta
        return float(np.sin(angle) ** 2)

    def optimal_grover_iterations(self) -> int:
        """k* = floor(pi/(4*theta0) - 1/2) with theta0 = arcsin(sqrt(p))."""
        if self.p_success <= 0.0:
            return 0
        theta = self.theta
        return max(0, int(np.floor(np.pi / (4.0 * theta) - 0.5)))

    def verify_su2_rotation(self, max_k: int = 20, atol: float = 1e-12) -> None:
        """Cross-check closed-form p_k against 2D rotation matrix evolution."""
        if self.p_success <= 0.0:
            return

        theta = self.theta
        # Initial coordinates in {|Bad>, |Good>}.
        v = np.array([np.sqrt(1.0 - self.p_success), np.sqrt(self.p_success)], dtype=float)
        # One Grover iterate in invariant plane is rotation by 2*theta0.
        c = np.cos(2.0 * theta)
        s = np.sin(2.0 * theta)
        rot = np.array([[c, -s], [s, c]], dtype=float)

        for k in range(max_k + 1):
            p_matrix = float(v[1] ** 2)
            p_closed = self.grover_success_probability(k)
            if not np.isclose(p_matrix, p_closed, atol=atol):
                raise AssertionError(
                    f"SU(2) check failed at k={k}: matrix={p_matrix}, formula={p_closed}"
                )
            v = rot @ v

    # ------------------------------------------------------------------
    # Runtime/cost models
    # ------------------------------------------------------------------

    def expected_time_until_success_restart(self) -> float:
        """Exact E[cost] for independent restart-until-success.

        Each run costs E[T]; geometric waiting time is 1/p.
        """
        if self.p_success <= 0.0:
            return float("inf")
        t_mean, _, _ = self.stopping_time_moments()
        return float(t_mean / self.p_success)

    def expected_time_until_success_worst_case_aa(self) -> float:
        """Worst-case AA model: each iterate costs T_max, total calls ~ (2k*+1)."""
        if self.p_success <= 0.0:
            return float("inf")
        _, _, t_max = self.stopping_time_moments()
        k_opt = self.optimal_grover_iterations()
        return float((2 * k_opt + 1) * t_max)

    def vtaa_asymptotic_bound(
        self,
        polylog_factor: float = 1.0,
        constant_tmax: float = 1.0,
        constant_trms: float = 1.0,
    ) -> float:
        """Parameterized Ambainis-style VTAA scaling estimate.

        Returns
            polylog_factor * (c1*T_max + c2*T_rms/sqrt(p))

        where constants/polylog are explicit user-chosen knobs (the theorem is
        asymptotic and model-dependent).
        """
        if self.p_success <= 0.0:
            return float("inf")

        _, t_rms, t_max = self.stopping_time_moments()
        return float(
            polylog_factor
            * (constant_tmax * t_max + constant_trms * (t_rms / np.sqrt(self.p_success)))
        )

    # ------------------------------------------------------------------
    # Unified report
    # ------------------------------------------------------------------

    def build_report(
        self,
        polylog_factor: float = 1.0,
        constant_tmax: float = 1.0,
        constant_trms: float = 1.0,
    ) -> VTAAReport:
        t_mean, t_rms, t_max = self.stopping_time_moments()
        k_opt = self.optimal_grover_iterations()

        return VTAAReport(
            p_success=self.p_success,
            theta=self.theta,
            grover_k_opt=k_opt,
            grover_p_at_k_opt=self.grover_success_probability(k_opt),
            t_mean=t_mean,
            t_rms=t_rms,
            t_max=t_max,
            expected_time_until_success_restart=self.expected_time_until_success_restart(),
            expected_time_until_success_worst_case_aa=self.expected_time_until_success_worst_case_aa(),
            vtaa_asymptotic_bound=self.vtaa_asymptotic_bound(
                polylog_factor=polylog_factor,
                constant_tmax=constant_tmax,
                constant_trms=constant_trms,
            ),
        )


# -----------------------------------------------------------------------------
# Convenience API + CLI
# -----------------------------------------------------------------------------


def _parse_csv_floats(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty list provided.")
    return vals


def _build_branches(times: Iterable[float], weights: Iterable[float], success: Iterable[float]) -> List[VariableTimeBranch]:
    t = list(times)
    w = list(weights)
    s = list(success)
    if not (len(t) == len(w) == len(s)):
        raise ValueError("times, weights, and success arrays must have same length.")
    return [
        VariableTimeBranch(stop_time=t_i, weight=w_i, success_given_branch=s_i)
        for t_i, w_i, s_i in zip(t, w, s)
    ]


def example_instance() -> List[VariableTimeBranch]:
    """A small default VT instance with strongly non-uniform stop times."""
    times = [1.0, 2.0, 4.0, 8.0, 16.0]
    weights = [0.36, 0.28, 0.18, 0.12, 0.06]
    success = [0.02, 0.05, 0.10, 0.22, 0.45]
    return _build_branches(times, weights, success)


def format_report(report: VTAAReport) -> str:
    return "\n".join(
        [
            f"p_success                         : {report.p_success:.10f}",
            f"theta0 = arcsin(sqrt(p))          : {report.theta:.10f}",
            f"grover k_opt                      : {report.grover_k_opt}",
            f"grover success at k_opt           : {report.grover_p_at_k_opt:.10f}",
            f"E[T]                              : {report.t_mean:.10f}",
            f"sqrt(E[T^2])                      : {report.t_rms:.10f}",
            f"T_max                             : {report.t_max:.10f}",
            f"restart expected cost (E[T]/p)    : {report.expected_time_until_success_restart:.10f}",
            f"worst-case AA cost ((2k+1)T_max) : {report.expected_time_until_success_worst_case_aa:.10f}",
            f"VTAA asymptotic estimate          : {report.vtaa_asymptotic_bound:.10f}",
        ]
    )


def save_plots(
    lab: VariableTimeAmplitudeAmplificationLab,
    report: VTAAReport,
    output_prefix: str = "vtaa",
    max_k: int = 25,
) -> List[str]:
    """Generate and save three VTAA diagnostic plots."""
    saved: List[str] = []

    # 1) Amplification trajectory p_k vs k.
    k_vals = np.arange(max_k + 1)
    p_vals = np.array([lab.grover_success_probability(int(k)) for k in k_vals], dtype=float)
    fig1, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(k_vals, p_vals, marker="o", linewidth=1.5, markersize=3)
    ax1.axvline(report.grover_k_opt, linestyle="--", linewidth=1.0, label=f"k_opt={report.grover_k_opt}")
    ax1.set_title("Amplitude Amplification Trajectory")
    ax1.set_xlabel("Grover iterations k")
    ax1.set_ylabel("Success probability p_k")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend()
    path1 = f"{output_prefix}_trajectory.png"
    fig1.tight_layout()
    fig1.savefig(path1, dpi=160)
    plt.close(fig1)
    saved.append(path1)

    # 2) CDF curves by stopping stage.
    stage = np.arange(1, len(lab.stop_times) + 1)
    f_vals = lab.cdf_by_stopping_stage()
    g_vals = lab.success_cdf_by_stage()
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    ax2.step(stage, f_vals, where="mid", linewidth=1.8, label="F_j = halted mass")
    ax2.step(stage, g_vals, where="mid", linewidth=1.8, label="G_j = H_Good mass")
    ax2.set_title("Stage-Wise Halt and Success CDF")
    ax2.set_xlabel("Stopping stage index j")
    ax2.set_ylabel("Cumulative mass")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xticks(stage)
    ax2.grid(alpha=0.3)
    ax2.legend()
    path2 = f"{output_prefix}_stage_cdfs.png"
    fig2.tight_layout()
    fig2.savefig(path2, dpi=160)
    plt.close(fig2)
    saved.append(path2)

    # 3) Runtime comparison bars.
    labels = ["Restart E[T]/p", "Worst-case AA", "VTAA bound"]
    values = [
        report.expected_time_until_success_restart,
        report.expected_time_until_success_worst_case_aa,
        report.vtaa_asymptotic_bound,
    ]
    fig3, ax3 = plt.subplots(figsize=(8, 4.8))
    bars = ax3.bar(labels, values, width=0.65)
    ax3.set_title("Expected Cost Comparison")
    ax3.set_ylabel("Cost units")
    ax3.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.2f}", ha="center", va="bottom")
    path3 = f"{output_prefix}_cost_comparison.png"
    fig3.tight_layout()
    fig3.savefig(path3, dpi=160)
    plt.close(fig3)
    saved.append(path3)

    return saved


def _standard_grover_state_history(
    n: int,
    k_max: int,
    good_state: str | None = None,
) -> tuple[List[np.ndarray], str, int, int]:
    """Return full state history |psi_0>..|psi_kmax> for standard AA with one H_Good basis state."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    if k_max < 1:
        raise ValueError("k_max must be >= 1.")

    N = 2**n
    if good_state is None:
        good_state = "1" * n
    if len(good_state) != n or any(ch not in "01" for ch in good_state):
        raise ValueError("good_state must be an n-bit binary string.")

    good_idx = int(good_state, 2)
    oracle_sign = np.ones(N, dtype=float)
    oracle_sign[good_idx] = -1.0

    psi = np.ones(N, dtype=complex) / np.sqrt(N)
    history: List[np.ndarray] = [psi.copy()]
    for _ in range(k_max):
        psi = oracle_sign * psi
        mean_amp = np.mean(psi)
        psi = (2.0 * mean_amp) - psi
        history.append(psi.copy())
    return history, good_state, good_idx, N


def experiment_2d_subspace_extractor(
    n: int = 10,
    k_max: int = 25,
    good_state: str | None = None,
    rank_threshold: float = 1e-12,
) -> SubspaceAuditResults:
    """Construct history matrix H and SVD-audit its empirical rank."""
    if rank_threshold <= 0.0:
        raise ValueError("rank_threshold must be > 0.")

    history, good_state, _, N = _standard_grover_state_history(n=n, k_max=k_max, good_state=good_state)
    history_matrix = np.column_stack(history)
    singular_values = np.linalg.svd(history_matrix, compute_uv=False)
    empirical_rank = int(np.count_nonzero(singular_values > rank_threshold))
    sigma3_to_sigma1 = 0.0
    if len(singular_values) > 2 and singular_values[0] > 0.0:
        sigma3_to_sigma1 = float(singular_values[2] / singular_values[0])
    # Practical floating-point SVD floor estimate: O(eps * max(m,n) * sigma_1).
    float64_svd_floor = float(
        np.finfo(np.float64).eps * max(history_matrix.shape) * float(singular_values[0])
    )

    return SubspaceAuditResults(
        n=n,
        N=N,
        k_max=k_max,
        good_state=good_state,
        history_shape=(int(history_matrix.shape[0]), int(history_matrix.shape[1])),
        singular_values=singular_values,
        rank_threshold=rank_threshold,
        empirical_rank=empirical_rank,
        sigma3_to_sigma1=sigma3_to_sigma1,
        float64_svd_floor=float64_svd_floor,
    )


def save_subspace_audit_plot(result: SubspaceAuditResults, output_prefix: str = "vtaa") -> str:
    """Save singular-value spectrum for the 2D invariant-subspace audit."""
    sv = result.singular_values
    idx = np.arange(1, len(sv) + 1)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.scatter(idx, sv, s=40, color="tab:blue", zorder=4, label="Singular values")
    ax.plot(idx, sv, color="tab:blue", alpha=0.45)
    ax.axhline(
        result.rank_threshold,
        color="tab:red",
        linestyle="--",
        linewidth=1.6,
        label=f"Rank threshold ({result.rank_threshold:.1e})",
    )
    ax.axhline(
        result.float64_svd_floor,
        color="tab:green",
        linestyle=":",
        linewidth=1.5,
        label=f"Float64 SVD floor (~{result.float64_svd_floor:.1e})",
    )
    ax.set_yscale("log")
    ax.set_xlim(0.5, min(16.5, len(sv) + 0.5))
    ax.set_title(
        f"2D Invariant Subspace Audit (n={result.n}, N={result.N}, rank={result.empirical_rank})"
    )
    ax.set_xlabel("Singular value index i")
    ax.set_ylabel("Magnitude sigma_i (log scale)")
    ax.grid(alpha=0.3, which="both", linestyle=":")
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = f"{output_prefix}_subspace_singular_spectrum.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def experiment_geometric_phase_staircase(
    n: int = 8,
    k_max: int = 25,
    good_state: str | None = None,
) -> PhaseStaircaseResults:
    """Extract theta_k and verify linear staircase theta_k=(2k+1)theta0."""
    history, good_state, good_idx, N = _standard_grover_state_history(n=n, k_max=k_max, good_state=good_state)
    theta_0 = float(np.arcsin(np.sqrt(1.0 / N)))

    ket_g = np.zeros(N, dtype=complex)
    ket_g[good_idx] = 1.0
    ket_s = np.ones(N, dtype=complex) / np.sqrt(N)
    ket_b = ket_s - np.vdot(ket_g, ket_s) * ket_g
    ket_b = ket_b / np.linalg.norm(ket_b)

    empirical: List[float] = []
    theoretical: List[float] = []
    fidelity: List[float] = []
    for k, psi in enumerate(history):
        amp_g = complex(np.vdot(ket_g, psi))
        amp_b = complex(np.vdot(ket_b, psi))
        empirical.append(float(np.arctan2(np.real(amp_g), np.real(amp_b))))
        theoretical.append(float((2 * k + 1) * theta_0))
        fidelity.append(float(np.abs(amp_g) ** 2))

    # Keep staircase linear after crossing pi/2 by unwrapping the angle branch.
    empirical_arr = np.unwrap(np.array(empirical, dtype=float))
    theoretical_arr = np.array(theoretical, dtype=float)
    abs_err = np.abs(empirical_arr - theoretical_arr)

    return PhaseStaircaseResults(
        n=n,
        N=N,
        k_max=k_max,
        good_state=good_state,
        theta_0=theta_0,
        empirical_angles=empirical_arr,
        theoretical_angles=theoretical_arr,
        angle_abs_error=abs_err,
        max_abs_error=float(np.max(abs_err)),
        good_fidelity=np.array(fidelity, dtype=float),
    )


def save_phase_staircase_plot(result: PhaseStaircaseResults, output_prefix: str = "vtaa") -> str:
    """Save empirical-vs-theoretical angle staircase plot."""
    k_vals = np.arange(result.k_max + 1, dtype=int)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(
        k_vals,
        result.theoretical_angles,
        color="0.30",
        linewidth=2.4,
        label=r"Theory: $\theta_k=(2k+1)\theta0$",
    )
    ax.scatter(
        k_vals,
        result.empirical_angles,
        color="tab:red",
        s=32,
        label=r"Empirical: $\theta_k=\mathrm{atan2}(\Re\langle H_{Good}|\psi_k\rangle,\Re\langle H_{Bad}|\psi_k\rangle)$",
        zorder=4,
    )
    ax.set_title(
        f"Geometric Phase Staircase (n={result.n}, N={result.N}, max err={result.max_abs_error:.2e})"
    )
    ax.set_xlabel("Grover iteration k")
    ax.set_ylabel("Angle (radians)")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")
    fig.tight_layout()

    path = f"{output_prefix}_phase_staircase.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def experiment_souffle_catastrophe(
    n: int = 8,
    guessed_m: int = 1,
    actual_m: int = 5,
    k_scan_factor: float = 1.5,
) -> SouffleProblemResults:
    """Numerically demonstrate over-rotation from an incorrect H_Good count M estimate."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    if guessed_m < 1 or actual_m < 1:
        raise ValueError("guessed_m and actual_m must be >= 1.")
    if k_scan_factor < 1.0:
        raise ValueError("k_scan_factor must be >= 1.")

    N = 2**n
    if guessed_m > N or actual_m > N:
        raise ValueError("guessed_m and actual_m must be <= 2**n.")

    theta_guess = float(np.arcsin(np.sqrt(guessed_m / N)))
    k_opt_guess = max(0, int(np.floor(np.pi / (4.0 * theta_guess) - 0.5)))
    k_max = max(k_opt_guess + 1, int(np.ceil(k_scan_factor * max(1, k_opt_guess))))
    k_values = np.arange(k_max + 1, dtype=int)

    # Mark the first actual_m basis states as H_Good states.
    good_indices = np.arange(actual_m, dtype=int)
    oracle_sign = np.ones(N, dtype=float)
    oracle_sign[good_indices] = -1.0

    # Uniform start state |s>.
    psi = np.ones(N, dtype=complex) / np.sqrt(N)

    actual_probs: List[float] = []
    for k in k_values:
        p_good = float(np.sum(np.abs(psi[good_indices]) ** 2))
        actual_probs.append(p_good)

        if k < k_max:
            # Oracle phase flip on actual H_Good states.
            psi = oracle_sign * psi
            # Diffusion about |s>: D = 2|s><s| - I.
            mean_amp = np.mean(psi)
            psi = (2.0 * mean_amp) - psi

    actual_probs_arr = np.array(actual_probs, dtype=float)
    guess_probs_theoretical = np.sin((2 * k_values + 1) * theta_guess) ** 2
    prob_at_halt = float(actual_probs_arr[k_opt_guess])
    peak_prob_before_halt = float(np.max(actual_probs_arr[: k_opt_guess + 1]))
    collapse_from_peak = float(peak_prob_before_halt - prob_at_halt)

    return SouffleProblemResults(
        n=n,
        N=N,
        guessed_m=guessed_m,
        actual_m=actual_m,
        k_opt_guess=k_opt_guess,
        k_values=k_values,
        actual_probs=actual_probs_arr,
        guess_probs_theoretical=np.array(guess_probs_theoretical, dtype=float),
        prob_at_halt=prob_at_halt,
        peak_prob_before_halt=peak_prob_before_halt,
        collapse_from_peak=collapse_from_peak,
    )


def save_souffle_plot(result: SouffleProblemResults, output_prefix: str = "vtaa") -> str:
    """Save guessed-vs-actual over-rotation trajectory plot."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    k = result.k_values

    ax.plot(
        k,
        result.guess_probs_theoretical,
        linestyle=":",
        linewidth=2.0,
        color="0.35",
        label=f"Expected (assumed M={result.guessed_m})",
    )
    ax.plot(
        k,
        result.actual_probs,
        marker="o",
        markersize=3,
        linewidth=2.0,
        color="tab:red",
        label=f"Actual (true M={result.actual_m})",
    )
    ax.axvline(
        result.k_opt_guess,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=f"Halt at guessed k*={result.k_opt_guess}",
    )
    ax.set_title("Souffle Problem: Over-Rotation Catastrophe")
    ax.set_xlabel("Grover iterations k")
    ax.set_ylabel("Success probability p")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()

    ax.annotate(
        f"halt={100*result.prob_at_halt:.1f}%\npeak drop={100*result.collapse_from_peak:.1f}%",
        xy=(result.k_opt_guess, result.prob_at_halt),
        xytext=(result.k_opt_guess + 1, min(0.95, result.prob_at_halt + 0.25)),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    path = f"{output_prefix}_souffle_catastrophe.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def experiment_ftqc_diffusion_scaling(
    n_min: int = 5,
    n_max: int = 50,
    noancilla_max: int = 8,
    optimization_level: int = 1,
) -> FTQCScalingResults:
    """Compile MCX-based diffusion core to Clifford+T and record scaling metrics.

    Notes
    -----
    - V-chain uses dirty ancillas and scales roughly linearly in T-count.
    - No-ancilla compilation grows rapidly; by default we cap its sweep.
    """
    if n_min < 3:
        raise ValueError("n_min must be >= 3.")
    if n_max < n_min:
        raise ValueError("n_max must be >= n_min.")
    if noancilla_max < n_min:
        noancilla_max = n_min
    if optimization_level not in (0, 1, 2, 3):
        raise ValueError("optimization_level must be one of {0,1,2,3}.")

    try:
        import warnings

        from qiskit import QuantumCircuit, transpile
    except Exception as exc:
        raise RuntimeError("Qiskit is required for FTQC diffusion scaling.") from exc

    basis = ["cx", "h", "s", "sdg", "t", "tdg", "x", "z"]
    n_values_vchain = np.arange(n_min, n_max + 1, dtype=int)
    t_counts_vchain: List[int] = []
    ancilla_counts_vchain: List[int] = []

    # V-chain sweep for the full n-range.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        for n in n_values_vchain:
            controls = int(n - 1)
            ancillas = max(0, controls - 2)
            ancilla_counts_vchain.append(ancillas)

            num_qubits = n + ancillas
            qc = QuantumCircuit(num_qubits)
            ctrl_idx = list(range(controls))
            good_idx = controls
            anc_idx = list(range(controls + 1, num_qubits))
            qc.mcx(ctrl_idx, good_idx, anc_idx, mode="v-chain")

            t_qc = transpile(
                qc,
                basis_gates=basis,
                optimization_level=optimization_level,
                seed_transpiler=42,
            )
            ops = t_qc.count_ops()
            t_counts_vchain.append(int(ops.get("t", 0) + ops.get("tdg", 0)))

    # No-ancilla sweep (truncated by default for runtime feasibility).
    noancilla_end = min(noancilla_max, n_max)
    n_values_noancilla = np.arange(n_min, noancilla_end + 1, dtype=int)
    t_counts_noancilla: List[int] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        for n in n_values_noancilla:
            controls = int(n - 1)
            qc = QuantumCircuit(n)
            qc.mcx(list(range(controls)), controls, mode="noancilla")

            t_qc = transpile(
                qc,
                basis_gates=basis,
                optimization_level=optimization_level,
                seed_transpiler=42,
            )
            ops = t_qc.count_ops()
            t_counts_noancilla.append(int(ops.get("t", 0) + ops.get("tdg", 0)))

    return FTQCScalingResults(
        n_values_vchain=n_values_vchain,
        t_counts_vchain=np.array(t_counts_vchain, dtype=int),
        ancilla_counts_vchain=np.array(ancilla_counts_vchain, dtype=int),
        n_values_noancilla=n_values_noancilla,
        t_counts_noancilla=np.array(t_counts_noancilla, dtype=int),
    )


def save_ftqc_scaling_plot(result: FTQCScalingResults, output_prefix: str = "vtaa") -> str:
    """Save FTQC T-gate/ancilla scaling plot for diffusion synthesis."""
    fig, ax1 = plt.subplots(figsize=(8.8, 5.2))

    ax1.plot(
        result.n_values_vchain,
        result.t_counts_vchain,
        marker="o",
        linewidth=2.0,
        markersize=3.5,
        color="tab:blue",
        label="T-count (v-chain, with ancillas)",
    )
    if len(result.n_values_noancilla) > 0:
        ax1.plot(
            result.n_values_noancilla,
            result.t_counts_noancilla,
            marker="x",
            linewidth=1.8,
            markersize=4,
            linestyle="--",
            color="tab:red",
            label="T-count (no ancilla)",
        )
    ax1.set_xlabel("Diffusion register size n")
    ax1.set_ylabel("T + Tdg gate count")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(
        result.n_values_vchain,
        result.ancilla_counts_vchain,
        alpha=0.18,
        color="tab:green",
        width=0.75,
        label="dirty ancillas (v-chain)",
    )
    ax2.set_ylabel("Ancilla qubits")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("FTQC Scaling of Diffusion-Operator Synthesis")

    path = f"{output_prefix}_ftqc_diffusion_scaling.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def experiment_exact_amplitude_amplification(
    n: int = 8,
    m_good: int = 3,
    coarse_grid: int = 48,
) -> ExactAAResults:
    """Run standard AA and optimize a fractional final step for exact-AA landing."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    if m_good < 1:
        raise ValueError("m_good must be >= 1.")
    if coarse_grid < 8:
        raise ValueError("coarse_grid must be >= 8.")

    try:
        from scipy.optimize import minimize
    except Exception as exc:
        raise RuntimeError("scipy is required for exact-AA phase optimization.") from exc

    N = 2**n
    if m_good > N:
        raise ValueError("m_good must be <= 2**n.")

    good_indices = np.arange(m_good, dtype=int)
    theta_0 = float(np.arcsin(np.sqrt(m_good / N)))
    k_base = max(0, int(np.floor(np.pi / (4.0 * theta_0) - 0.5)))

    # Build operators in full space for explicit statevector-level evidence.
    proj_good = np.zeros((N, N), dtype=complex)
    proj_good[good_indices, good_indices] = 1.0
    ket_s = np.ones(N, dtype=complex) / np.sqrt(N)
    proj_s = np.outer(ket_s, ket_s.conjugate())
    oracle_pi = np.eye(N, dtype=complex) - 2.0 * proj_good
    diffusion_pi = 2.0 * proj_s - np.eye(N, dtype=complex)

    # Standard AA trajectory through one overshoot step.
    psi = ket_s.copy()
    standard_probs: List[float] = []
    states: List[np.ndarray] = []
    k_max = k_base + 2
    for k in range(k_max + 1):
        states.append(psi.copy())
        standard_probs.append(float(np.sum(np.abs(psi[good_indices]) ** 2)))
        if k < k_max:
            psi = diffusion_pi @ (oracle_pi @ psi)

    standard_probs_arr = np.array(standard_probs, dtype=float)
    k_peak_standard = int(np.argmax(standard_probs_arr))
    psi_base = states[k_base]

    def _prob_after_fractional_step(alpha: float, beta: float) -> float:
        oracle_alpha = np.eye(N, dtype=complex) + (np.exp(1j * alpha) - 1.0) * proj_good
        diffusion_beta = np.eye(N, dtype=complex) + (np.exp(1j * beta) - 1.0) * proj_s
        psi_final = diffusion_beta @ (oracle_alpha @ psi_base)
        return float(np.sum(np.abs(psi_final[good_indices]) ** 2))

    # Coarse search to avoid local minima in local optimizer.
    alpha_grid = np.linspace(0.0, 2.0 * np.pi, coarse_grid, endpoint=False)
    beta_grid = np.linspace(0.0, 2.0 * np.pi, coarse_grid, endpoint=False)
    best_prob = -1.0
    best_alpha = 0.0
    best_beta = 0.0
    for alpha in alpha_grid:
        oracle_alpha = np.eye(N, dtype=complex) + (np.exp(1j * alpha) - 1.0) * proj_good
        oracle_applied = oracle_alpha @ psi_base
        for beta in beta_grid:
            diffusion_beta = np.eye(N, dtype=complex) + (np.exp(1j * beta) - 1.0) * proj_s
            p = float(np.sum(np.abs((diffusion_beta @ oracle_applied)[good_indices]) ** 2))
            if p > best_prob:
                best_prob = p
                best_alpha = float(alpha)
                best_beta = float(beta)

    def _objective(x: np.ndarray) -> float:
        return -_prob_after_fractional_step(float(x[0]), float(x[1]))

    opt = minimize(
        _objective,
        x0=np.array([best_alpha, best_beta], dtype=float),
        method="L-BFGS-B",
        bounds=[(0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)],
        options={"maxiter": 1000},
    )
    exact_alpha = float(opt.x[0])
    exact_beta = float(opt.x[1])
    exact_prob = float(_prob_after_fractional_step(exact_alpha, exact_beta))

    return ExactAAResults(
        n=n,
        N=N,
        m_good=m_good,
        k_base=k_base,
        k_peak_standard=k_peak_standard,
        standard_probs=standard_probs_arr,
        exact_oracle_phase=exact_alpha,
        exact_diffusion_phase=exact_beta,
        exact_prob=exact_prob,
    )


def save_exact_aa_peak_plot(result: ExactAAResults, output_prefix: str = "vtaa") -> str:
    """Save a peak-region comparison: standard AA vs optimized exact-AA final step."""
    k_left = max(0, result.k_base - 1)
    k_right = min(len(result.standard_probs) - 1, result.k_base + 2)
    k_vals = np.arange(k_left, k_right + 1, dtype=int)
    std_zoom = result.standard_probs[k_left : k_right + 1]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(
        k_vals,
        std_zoom,
        color="tab:red",
        marker="o",
        linewidth=2.0,
        markersize=4.5,
        label="Standard AA (pi phases)",
    )

    # Exact-AA endpoint is at k_base + 1 with optimized final phases.
    exact_k = result.k_base + 1
    ax.plot(
        [result.k_base, exact_k],
        [result.standard_probs[result.k_base], result.exact_prob],
        color="tab:green",
        linestyle="--",
        linewidth=1.8,
    )
    ax.scatter(
        [exact_k],
        [result.exact_prob],
        color="tab:green",
        s=80,
        marker="*",
        label="Exact AA (optimized final phases)",
        zorder=5,
    )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="Unity target")
    ax.set_title("Exact Amplitude Amplification: Discretization Fix")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("Success probability p")
    ax.set_ylim(max(0.0, float(np.min(std_zoom)) * 0.97), 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    path = f"{output_prefix}_exact_aa_peak_region.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def experiment_phase_mismatch_leakage(
    n: int = 6,
    k_max: int = 20,
    eps_oracle_deg: float = -5.0,
    eps_diff_deg: float = 2.0,
    crosstalk_oracle_deg: float = 0.6,
    local_z_detune_deg: float = 0.6,
    rank_threshold: float = 1e-10,
) -> PhaseLeakageResults:
    """Track trajectory rank under ideal, mismatch-only, and leaky analog control.

    Important nuance:
    Pure phase mismatch alone remains rank-2 in the idealized generalized-AA model.
    Leakage above rank-2 appears when control errors also introduce non-uniform
    state-dependent phases (modeled here as crosstalk + local detuning).
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    if k_max < 1:
        raise ValueError("k_max must be >= 1.")
    if rank_threshold <= 0.0:
        raise ValueError("rank_threshold must be > 0.")

    N = 2**n
    good_idx = N - 1
    spur_idx = N - 2 if N >= 2 else 0

    ket_t = np.zeros(N, dtype=complex)
    ket_t[good_idx] = 1.0
    proj_t = np.outer(ket_t, ket_t.conjugate())

    ket_spur = np.zeros(N, dtype=complex)
    ket_spur[spur_idx] = 1.0
    proj_spur = np.outer(ket_spur, ket_spur.conjugate())

    ket_s = np.ones(N, dtype=complex) / np.sqrt(N)
    proj_s = np.outer(ket_s, ket_s.conjugate())

    eps_o = float(np.deg2rad(eps_oracle_deg))
    eps_d = float(np.deg2rad(eps_diff_deg))
    gamma_o = float(np.deg2rad(crosstalk_oracle_deg))
    gamma_z = float(np.deg2rad(local_z_detune_deg))

    # Ideal reflections.
    oracle_ideal = np.eye(N, dtype=complex) - 2.0 * proj_t
    diffusion_ideal = 2.0 * proj_s - np.eye(N, dtype=complex)

    # Mismatch-only generalized reflections.
    oracle_mismatch = np.eye(N, dtype=complex) + (np.exp(1j * (np.pi + eps_o)) - 1.0) * proj_t
    diffusion_mismatch = np.eye(N, dtype=complex) + (np.exp(1j * (np.pi + eps_d)) - 1.0) * proj_s

    # Additional analog non-uniformity model (crosstalk + local Z-detuning).
    oracle_crosstalk = np.eye(N, dtype=complex) + (np.exp(1j * gamma_o) - 1.0) * proj_spur
    local_z_sign = np.array([1.0 if (i & 1) == 0 else -1.0 for i in range(N)], dtype=float)
    local_detune = np.diag(np.exp(1j * gamma_z * local_z_sign))

    oracle_leaky = oracle_crosstalk @ oracle_mismatch
    diffusion_leaky = local_detune @ diffusion_mismatch

    def _track_rank(oracle: np.ndarray, diffusion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi = ket_s.copy()
        history = [psi.copy()]
        ranks: List[int] = [1]
        sv_last = np.array([1.0], dtype=float)
        for _ in range(k_max):
            psi = diffusion @ (oracle @ psi)
            history.append(psi.copy())
            sv = np.linalg.svd(np.column_stack(history), compute_uv=False)
            ranks.append(int(np.count_nonzero(sv > rank_threshold)))
            sv_last = sv
        return np.array(ranks, dtype=int), sv_last

    rank_ideal, sv_ideal = _track_rank(oracle_ideal, diffusion_ideal)
    rank_mismatch_only, sv_mismatch = _track_rank(oracle_mismatch, diffusion_mismatch)
    rank_leaky, sv_leaky = _track_rank(oracle_leaky, diffusion_leaky)
    mismatch_only_rank2_all = bool(np.all(rank_mismatch_only <= 2))
    leaky_exceeds_rank2 = bool(np.any(rank_leaky > 2))

    return PhaseLeakageResults(
        n=n,
        N=N,
        k_max=k_max,
        eps_oracle_deg=eps_oracle_deg,
        eps_diff_deg=eps_diff_deg,
        crosstalk_oracle_deg=crosstalk_oracle_deg,
        local_z_detune_deg=local_z_detune_deg,
        rank_threshold=rank_threshold,
        rank_ideal=rank_ideal,
        rank_mismatch_only=rank_mismatch_only,
        rank_leaky=rank_leaky,
        final_sv_ideal=sv_ideal,
        final_sv_mismatch_only=sv_mismatch,
        final_sv_leaky=sv_leaky,
        mismatch_only_rank2_all=mismatch_only_rank2_all,
        leaky_exceeds_rank2=leaky_exceeds_rank2,
    )


def save_phase_leakage_plot(result: PhaseLeakageResults, output_prefix: str = "vtaa") -> str:
    """Save rank-growth evidence plot for phase-mismatch leakage audit."""
    k_vals = np.arange(result.k_max + 1, dtype=int)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    ax.step(
        k_vals,
        result.rank_ideal,
        where="post",
        linewidth=2.2,
        color="tab:blue",
        label="Ideal (180 deg, 180 deg)",
    )
    ax.step(
        k_vals,
        result.rank_mismatch_only,
        where="post",
        linewidth=2.0,
        color="tab:orange",
        label=f"Mismatch only ({180+result.eps_oracle_deg:.1f} deg, {180+result.eps_diff_deg:.1f} deg)",
    )
    ax.step(
        k_vals,
        result.rank_leaky,
        where="post",
        linewidth=2.2,
        color="tab:red",
        label=(
            "Mismatch + analog skew "
            f"(crosstalk {result.crosstalk_oracle_deg:.1f} deg, detune {result.local_z_detune_deg:.1f} deg)"
        ),
    )

    ax.axhline(2, color="black", linestyle=":", linewidth=1.2, label="Invariant-plane rank=2")
    ax.set_title("Phase-Mismatch Leakage: Empirical Trajectory Rank")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("Numerical rank of history matrix")
    ax.set_ylim(0, max(4, int(np.max(result.rank_leaky)) + 1))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    path = f"{output_prefix}_phase_mismatch_leakage.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Trace distance D(rho_a, rho_b) = 1/2 ||rho_a-rho_b||_1 for Hermitian inputs."""
    delta = rho_a - rho_b
    delta = 0.5 * (delta + delta.conjugate().T)
    eigvals = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def experiment_open_system_trajectory(
    n: int = 4,
    k_max: int = 12,
    phase_damp_1q: float = 0.02,
    phase_damp_2q: float = 0.08,
    good_state: str | None = None,
) -> OpenSystemTrajectoryResults:
    """Simulate AA as an open system and track purity/plane-distance decay."""
    if n < 2:
        raise ValueError("n must be >= 2.")
    if k_max < 1:
        raise ValueError("k_max must be >= 1.")
    if not (0.0 <= phase_damp_1q <= 1.0 and 0.0 <= phase_damp_2q <= 1.0):
        raise ValueError("phase damping probabilities must be in [0,1].")

    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, phase_damping_error
    except Exception as exc:
        raise RuntimeError("qiskit + qiskit-aer are required for open-system trajectory module.") from exc

    N = 2**n
    if good_state is None:
        good_state = "1" * n
    if len(good_state) != n or any(ch not in "01" for ch in good_state):
        raise ValueError("good_state must be an n-bit binary string.")
    good_idx = int(good_state, 2)

    # Build the |B>,|G> basis used for 2D projection observables.
    vec_g = np.zeros(N, dtype=complex)
    vec_g[good_idx] = 1.0
    vec_s = np.ones(N, dtype=complex) / np.sqrt(N)
    vec_b = vec_s - np.vdot(vec_g, vec_s) * vec_g
    vec_b = vec_b / np.linalg.norm(vec_b)

    proj_b = np.outer(vec_b, vec_b.conjugate())
    proj_g = np.outer(vec_g, vec_g.conjugate())
    proj_plane = proj_b + proj_g
    z_eff = proj_b - proj_g
    x_eff = np.outer(vec_b, vec_g.conjugate()) + np.outer(vec_g, vec_b.conjugate())

    # Severe dephasing-dominant open-system model.
    noise_model = NoiseModel()
    err_1q = phase_damping_error(phase_damp_1q)
    err_2q = phase_damping_error(phase_damp_2q).tensor(phase_damping_error(phase_damp_2q))
    for gate in ["h", "x", "sx", "u", "u1", "u2", "u3", "p", "rz"]:
        try:
            noise_model.add_all_qubit_quantum_error(err_1q, [gate])
        except Exception:
            pass
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx"])

    sim_ideal = AerSimulator(method="density_matrix")
    sim_noisy = AerSimulator(method="density_matrix", noise_model=noise_model)

    def _append_oracle(qc: "QuantumCircuit", phase: float) -> None:
        rev_bits = good_state[::-1]
        for q, bit in enumerate(rev_bits):
            if bit == "0":
                qc.x(q)
        qc.mcp(phase, list(range(n - 1)), n - 1)
        for q, bit in enumerate(rev_bits):
            if bit == "0":
                qc.x(q)

    def _append_diffusion(qc: "QuantumCircuit", phase: float) -> None:
        qc.h(range(n))
        qc.x(range(n))
        qc.mcp(phase, list(range(n - 1)), n - 1)
        qc.x(range(n))
        qc.h(range(n))

    def _build_circuit(k: int) -> "QuantumCircuit":
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for _ in range(k):
            _append_oracle(qc, np.pi)
            _append_diffusion(qc, np.pi)
        qc.save_density_matrix()
        return qc

    ideal_x: List[float] = []
    ideal_z: List[float] = []
    noisy_x: List[float] = []
    noisy_z: List[float] = []
    purity_noisy: List[float] = []
    trace_dist_plane: List[float] = []

    for k in range(k_max + 1):
        qc = _build_circuit(k)
        tc_ideal = transpile(qc, sim_ideal, optimization_level=0)
        tc_noisy = transpile(qc, sim_noisy, optimization_level=0)

        rho_i = np.asarray(sim_ideal.run(tc_ideal).result().data(0)["density_matrix"], dtype=complex)
        rho_n = np.asarray(sim_noisy.run(tc_noisy).result().data(0)["density_matrix"], dtype=complex)

        ideal_x.append(float(np.real(np.trace(rho_i @ x_eff))))
        ideal_z.append(float(np.real(np.trace(rho_i @ z_eff))))
        noisy_x.append(float(np.real(np.trace(rho_n @ x_eff))))
        noisy_z.append(float(np.real(np.trace(rho_n @ z_eff))))

        purity_noisy.append(float(np.real(np.trace(rho_n @ rho_n))))

        rho_proj = proj_plane @ rho_n @ proj_plane
        in_plane = float(np.real(np.trace(rho_proj)))
        if in_plane > 1e-15:
            rho_plane_state = rho_proj / in_plane
        else:
            rho_plane_state = 0.5 * proj_plane
        trace_dist_plane.append(_trace_distance(rho_n, rho_plane_state))

    return OpenSystemTrajectoryResults(
        n=n,
        N=N,
        k_max=k_max,
        phase_damp_1q=phase_damp_1q,
        phase_damp_2q=phase_damp_2q,
        good_state=good_state,
        ideal_x=np.array(ideal_x, dtype=float),
        ideal_z=np.array(ideal_z, dtype=float),
        noisy_x=np.array(noisy_x, dtype=float),
        noisy_z=np.array(noisy_z, dtype=float),
        noisy_purity=np.array(purity_noisy, dtype=float),
        trace_distance_to_plane=np.array(trace_dist_plane, dtype=float),
    )


def save_open_system_trajectory_plot(result: OpenSystemTrajectoryResults, output_prefix: str = "vtaa") -> str:
    """Save 2-panel open-system evidence: trajectory spiral + purity/distance decay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # Panel A: Effective 2D trajectory (ideal orbit vs noisy spiral).
    circle = plt.Circle((0.0, 0.0), 1.0, fill=False, linestyle=":", linewidth=1.2, color="black", alpha=0.6)
    ax1.add_patch(circle)
    ax1.plot(result.ideal_z, result.ideal_x, "-o", color="tab:blue", linewidth=2.0, markersize=4, label="Ideal closed system")
    ax1.plot(result.noisy_z, result.noisy_x, "-X", color="tab:red", linewidth=2.0, markersize=4, label="Open system (dephasing)")
    ax1.scatter([result.ideal_z[0]], [result.ideal_x[0]], color="tab:blue", s=45, zorder=5)
    ax1.scatter([-1.0], [0.0], color="green", s=60, zorder=5)
    ax1.set_title("Open-System Geometric Trajectory")
    ax1.set_xlabel("Re<Z_eff>")
    ax1.set_ylabel("Re<X_eff>")
    ax1.set_xlim(-1.25, 1.25)
    ax1.set_ylim(-1.25, 1.25)
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel B: Purity and trace-distance diagnostics.
    k_vals = np.arange(result.k_max + 1, dtype=int)
    ax2.plot(k_vals, result.noisy_purity, "-s", color="tab:purple", linewidth=2.0, markersize=4, label="Purity Tr(rho^2)")
    ax2.plot(
        k_vals,
        result.trace_distance_to_plane,
        "-^",
        color="tab:orange",
        linewidth=2.0,
        markersize=4,
        label="Trace distance to ideal 2D plane",
    )
    ax2.axhline(1.0 / result.N, color="black", linestyle="--", linewidth=1.1, label=f"Mixed-state limit 1/N={1/result.N:.3f}")
    ax2.set_title("Purity Decay and Geometric Drift")
    ax2.set_xlabel("Grover iteration k")
    ax2.set_ylabel("Metric value")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    path = f"{output_prefix}_open_system_spiral.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def experiment_vtaa_state_synthesis() -> VTAA_StateSynthesisResults:
    """Construct a coherent staged variable-time state with explicit flags.

    Flag encoding:
    - `|00>` continue
    - `|01>` success
    - `|10>` fail
    """
    try:
        from qiskit import QuantumCircuit, QuantumRegister
        from qiskit.quantum_info import Statevector
    except Exception as exc:
        raise RuntimeError("Qiskit is required for VTAA state synthesis.") from exc

    stage_reg = QuantumRegister(1, "stage_j")
    flag_reg = QuantumRegister(2, "flag")
    data_reg = QuantumRegister(1, "data")
    qc = QuantumCircuit(data_reg, flag_reg, stage_reg, name="A_variable_time")

    # Stage-1 branch model: 20% immediate success, 80% continue.
    p_s1 = 0.20
    qc.ry(2.0 * np.arcsin(np.sqrt(p_s1)), flag_reg[0])

    # Mark stage-2 path on the clock register only for continue branch (flag0=0).
    qc.x(flag_reg[0])
    qc.cx(flag_reg[0], stage_reg[0])
    qc.x(flag_reg[0])

    # Stage-2 conditional split on continue branch:
    # 10% success and 70% fail of total mass (i.e., success/fail conditioned on continue).
    p_continue = 1.0 - p_s1
    p_fail_cond = 0.70 / p_continue
    qc.x(flag_reg[0])
    qc.cry(2.0 * np.arcsin(np.sqrt(p_fail_cond)), flag_reg[0], flag_reg[1])
    qc.x(flag_reg[0])

    # Any stage-2 branch that is not fail becomes success.
    qc.x(flag_reg[1])
    qc.ccx(stage_reg[0], flag_reg[1], flag_reg[0])
    qc.x(flag_reg[1])

    sv = np.asarray(Statevector.from_instruction(qc).data, dtype=complex)
    flag0_idx = qc.find_bit(flag_reg[0]).index
    flag1_idx = qc.find_bit(flag_reg[1]).index

    p_success = 0.0
    p_fail = 0.0
    p_continue_final = 0.0
    for basis_idx, amp in enumerate(sv):
        prob = float(np.abs(amp) ** 2)
        bit0 = (basis_idx >> flag0_idx) & 1
        bit1 = (basis_idx >> flag1_idx) & 1
        flag_val = bit0 + 2 * bit1
        if flag_val == 1:
            p_success += prob
        elif flag_val == 2:
            p_fail += prob
        elif flag_val == 0:
            p_continue_final += prob

    return VTAA_StateSynthesisResults(
        num_stages=2,
        statevector_dimension=int(len(sv)),
        success_probability=float(p_success),
        continue_probability=float(p_continue_final),
        fail_probability=float(p_fail),
    )


def experiment_vtaa_cost_sweep(
    total_ps: float = 0.05,
    t1: float = 100.0,
    t2: float = 1000.0,
    t3: float = 10000.0,
) -> VTAA_CostSweepResults:
    """Sweep early-success ratio and compare VTAA functional to worst-case AA."""
    if not (0.0 < total_ps < 1.0):
        raise ValueError("total_ps must be in (0,1).")
    if min(t1, t2, t3) <= 0.0:
        raise ValueError("all stage times must be positive.")

    ratios = np.linspace(0.01, 0.99, 100)
    standard_costs: List[float] = []
    vtaa_costs: List[float] = []

    for ratio in ratios:
        p_s1 = total_ps * ratio
        p_s2 = total_ps * (1.0 - ratio) * 0.5
        p_s3 = total_ps * (1.0 - ratio) * 0.5

        # Worst-case AA baseline uses T_max in each amplified query block.
        standard_costs.append(float(t3 / np.sqrt(total_ps)))

        # Ambainis-style VTAA functional proxy:
        # T_vtaa ~ sqrt(sum_j p_j * t_j^2), total amplification cost ~ T_vtaa / p_s.
        t_vtaa = float(np.sqrt(p_s1 * (t1**2) + p_s2 * (t2**2) + p_s3 * (t3**2)))
        vtaa_costs.append(float(t_vtaa / total_ps))

    return VTAA_CostSweepResults(
        total_ps=float(total_ps),
        early_success_ratios=ratios,
        standard_costs=np.array(standard_costs, dtype=float),
        vtaa_costs=np.array(vtaa_costs, dtype=float),
    )


def save_vtaa_cost_sweep_plot(result: VTAA_CostSweepResults, output_prefix: str = "vtaa") -> str:
    """Save VTAA vs standard AA asymptotic query-cost sweep."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x_pct = 100.0 * result.early_success_ratios

    ax.plot(
        x_pct,
        result.standard_costs,
        color="tab:red",
        linestyle="--",
        linewidth=2.4,
        label="Standard AA (worst-case T_max baseline)",
    )
    ax.plot(
        x_pct,
        result.vtaa_costs,
        color="tab:blue",
        linewidth=2.6,
        label=r"VTAA cost functional ($T_{vtaa}$)",
    )
    ax.fill_between(
        x_pct,
        result.vtaa_costs,
        result.standard_costs,
        color="tab:green",
        alpha=0.16,
        label="Computational savings region",
    )

    ax.set_title(f"VTAA Cost Sweep (total p_s={100*result.total_ps:.1f}%)")
    ax.set_xlabel("Success amplitude fraction halting at stage 1 (%)")
    ax.set_ylabel("Asymptotic query-cost proxy")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 1.15 * float(np.max(result.standard_costs)))
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")

    path = f"{output_prefix}_vtaa_cost_sweep.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Variable-Time Amplitude Amplification lab")
    parser.add_argument("--times", type=str, default="1,2,4,8,16", help="comma-separated stop times")
    parser.add_argument("--weights", type=str, default="0.36,0.28,0.18,0.12,0.06", help="comma-separated branch weights")
    parser.add_argument("--success", type=str, default="0.02,0.05,0.10,0.22,0.45", help="comma-separated conditional success probabilities")
    parser.add_argument("--polylog", type=float, default=1.0, help="polylog factor in VTAA asymptotic estimate")
    parser.add_argument("--c1", type=float, default=1.0, help="constant multiplying T_max in VTAA estimate")
    parser.add_argument("--c2", type=float, default=1.0, help="constant multiplying T_rms/sqrt(p) in VTAA estimate")
    parser.add_argument("--verify-k", type=int, default=25, help="max k for SU(2) closed-form verification")
    parser.add_argument("--plot-prefix", type=str, default="vtaa", help="prefix for output PNG files")
    parser.add_argument("--no-plots", action="store_true", help="disable plot generation")
    parser.add_argument("--run-subspace-audit", action="store_true", help="run 2D invariant subspace SVD audit")
    parser.add_argument("--subspace-n", type=int, default=10, help="qubit count for subspace audit")
    parser.add_argument("--subspace-k-max", type=int, default=25, help="iterations for subspace audit")
    parser.add_argument("--subspace-good-state", type=str, default=None, help="optional H_Good n-bit state for subspace audit")
    parser.add_argument("--subspace-rank-threshold", type=float, default=1e-12, help="rank threshold for subspace SVD")
    parser.add_argument("--run-phase-staircase", action="store_true", help="run geometric phase staircase audit")
    parser.add_argument("--staircase-n", type=int, default=8, help="qubit count for staircase audit")
    parser.add_argument("--staircase-k-max", type=int, default=25, help="iterations for staircase audit")
    parser.add_argument("--staircase-good-state", type=str, default=None, help="optional H_Good n-bit state for staircase audit")
    parser.add_argument("--run-souffle-catastrophe", action="store_true", help="run over-rotation catastrophe experiment")
    parser.add_argument("--souffle-n", type=int, default=8, help="qubits for souffle experiment")
    parser.add_argument("--souffle-guessed-m", type=int, default=1, help="guessed number of H_Good states")
    parser.add_argument("--souffle-actual-m", type=int, default=5, help="actual number of H_Good states")
    parser.add_argument("--souffle-k-factor", type=float, default=1.5, help="k scan factor past guessed k*")
    parser.add_argument("--run-ftqc-scaling", action="store_true", help="run FTQC ancilla/T-gate diffusion scaling")
    parser.add_argument("--ftqc-n-min", type=int, default=5, help="minimum n for FTQC scaling sweep")
    parser.add_argument("--ftqc-n-max", type=int, default=50, help="maximum n for FTQC scaling sweep")
    parser.add_argument(
        "--ftqc-noancilla-max",
        type=int,
        default=8,
        help="maximum n for no-ancilla sweep (kept low due compile blowup)",
    )
    parser.add_argument("--ftqc-opt-level", type=int, default=1, help="qiskit transpiler optimization level")
    parser.add_argument("--run-exact-aa", action="store_true", help="run exact amplitude amplification benchmark")
    parser.add_argument("--exact-aa-n", type=int, default=8, help="qubit count for exact-AA benchmark")
    parser.add_argument("--exact-aa-m", type=int, default=3, help="H_Good-subspace count for exact-AA benchmark")
    parser.add_argument(
        "--exact-aa-grid",
        type=int,
        default=48,
        help="coarse phase grid per axis before local optimizer (exact-AA)",
    )
    parser.add_argument("--run-phase-leakage", action="store_true", help="run phase-mismatch leakage rank audit")
    parser.add_argument("--leakage-n", type=int, default=6, help="qubit count for phase-leakage audit")
    parser.add_argument("--leakage-k-max", type=int, default=20, help="iterations for phase-leakage rank tracking")
    parser.add_argument("--leakage-eps-oracle-deg", type=float, default=-5.0, help="oracle phase error in degrees")
    parser.add_argument("--leakage-eps-diff-deg", type=float, default=2.0, help="diffusion phase error in degrees")
    parser.add_argument(
        "--leakage-crosstalk-deg",
        type=float,
        default=0.6,
        help="spurious oracle crosstalk phase on an H_Bad basis state (degrees)",
    )
    parser.add_argument(
        "--leakage-local-z-deg",
        type=float,
        default=0.6,
        help="local qubit detuning phase per iteration (degrees)",
    )
    parser.add_argument(
        "--leakage-rank-threshold",
        type=float,
        default=1e-10,
        help="singular-value threshold for numerical rank in leakage audit",
    )
    parser.add_argument("--run-open-system-trajectory", action="store_true", help="run density-matrix open-system AA trajectory")
    parser.add_argument("--open-system-n", type=int, default=4, help="qubit count for open-system trajectory")
    parser.add_argument("--open-system-k-max", type=int, default=12, help="iterations for open-system trajectory")
    parser.add_argument(
        "--open-system-phase-damp-1q",
        type=float,
        default=0.02,
        help="single-qubit phase-damping probability per gate",
    )
    parser.add_argument(
        "--open-system-phase-damp-2q",
        type=float,
        default=0.08,
        help="two-qubit phase-damping probability per entangling gate",
    )
    parser.add_argument(
        "--open-system-good-state",
        type=str,
        default=None,
        help="optional H_Good n-bit state for open-system trajectory",
    )
    parser.add_argument("--run-vtaa-synthesis", action="store_true", help="run VTAA staged state synthesis (Eq. 47-style)")
    parser.add_argument("--run-vtaa-sweep", action="store_true", help="run VTAA cost-functional sweep")
    parser.add_argument("--vtaa-total-ps", type=float, default=0.05, help="total success mass p_s for VTAA sweep")
    parser.add_argument("--vtaa-t1", type=float, default=100.0, help="stage-1 cost for VTAA sweep")
    parser.add_argument("--vtaa-t2", type=float, default=1000.0, help="stage-2 cost for VTAA sweep")
    parser.add_argument("--vtaa-t3", type=float, default=10000.0, help="stage-3 cost for VTAA sweep")
    args = parser.parse_args(argv)

    times = _parse_csv_floats(args.times)
    weights = _parse_csv_floats(args.weights)
    success = _parse_csv_floats(args.success)

    branches = _build_branches(times, weights, success)
    lab = VariableTimeAmplitudeAmplificationLab(branches)

    # Mathematical consistency check for the AA closed form.
    lab.verify_su2_rotation(max_k=args.verify_k)

    report = lab.build_report(polylog_factor=args.polylog, constant_tmax=args.c1, constant_trms=args.c2)
    print(format_report(report))

    subspace_result: SubspaceAuditResults | None = None
    if args.run_subspace_audit:
        subspace_result = experiment_2d_subspace_extractor(
            n=args.subspace_n,
            k_max=args.subspace_k_max,
            good_state=args.subspace_good_state,
            rank_threshold=args.subspace_rank_threshold,
        )
        print("\n2D Subspace SVD Audit:")
        print(f"  N                                 : {subspace_result.N}")
        print(f"  history matrix shape              : {subspace_result.history_shape}")
        print(f"  empirical rank                    : {subspace_result.empirical_rank}")
        print(f"  sigma_1                           : {subspace_result.singular_values[0]:.6e}")
        print(f"  sigma_2                           : {subspace_result.singular_values[1]:.6e}")
        if len(subspace_result.singular_values) > 2:
            print(f"  sigma_3                           : {subspace_result.singular_values[2]:.6e}")
        print(f"  sigma_3 / sigma_1                 : {subspace_result.sigma3_to_sigma1:.6e}")
        print(f"  float64 SVD floor estimate        : {subspace_result.float64_svd_floor:.6e}")

    staircase_result: PhaseStaircaseResults | None = None
    if args.run_phase_staircase:
        staircase_result = experiment_geometric_phase_staircase(
            n=args.staircase_n,
            k_max=args.staircase_k_max,
            good_state=args.staircase_good_state,
        )
        print("\nGeometric Phase Staircase:")
        print(f"  N                                 : {staircase_result.N}")
        print(f"  theta0                            : {staircase_result.theta_0:.10f}")
        print(f"  max |theta_emp-theory|            : {staircase_result.max_abs_error:.3e}")
        print(f"  H_Good fidelity at k_max          : {staircase_result.good_fidelity[-1]:.10f}")

    souffle_result: SouffleProblemResults | None = None
    if args.run_souffle_catastrophe:
        souffle_result = experiment_souffle_catastrophe(
            n=args.souffle_n,
            guessed_m=args.souffle_guessed_m,
            actual_m=args.souffle_actual_m,
            k_scan_factor=args.souffle_k_factor,
        )
        print("\nSouffle Catastrophe:")
        print(f"  guessed M                         : {souffle_result.guessed_m}")
        print(f"  actual M                          : {souffle_result.actual_m}")
        print(f"  guessed optimal k*                : {souffle_result.k_opt_guess}")
        print(f"  peak probability before halt      : {souffle_result.peak_prob_before_halt:.10f}")
        print(f"  probability at halt               : {souffle_result.prob_at_halt:.10f}")
        print(f"  collapse from peak                : {souffle_result.collapse_from_peak:.10f}")

    ftqc_result: FTQCScalingResults | None = None
    if args.run_ftqc_scaling:
        ftqc_result = experiment_ftqc_diffusion_scaling(
            n_min=args.ftqc_n_min,
            n_max=args.ftqc_n_max,
            noancilla_max=args.ftqc_noancilla_max,
            optimization_level=args.ftqc_opt_level,
        )
        print("\nFTQC Diffusion Scaling:")
        print(
            f"  v-chain sweep n                   : {int(ftqc_result.n_values_vchain[0])}"
            f" .. {int(ftqc_result.n_values_vchain[-1])}"
        )
        print(
            f"  v-chain T-count endpoints         : {int(ftqc_result.t_counts_vchain[0])}"
            f" -> {int(ftqc_result.t_counts_vchain[-1])}"
        )
        print(
            f"  v-chain ancilla endpoints         : {int(ftqc_result.ancilla_counts_vchain[0])}"
            f" -> {int(ftqc_result.ancilla_counts_vchain[-1])}"
        )
        if len(ftqc_result.n_values_noancilla) > 0:
            print(
                f"  no-ancilla sweep n                : {int(ftqc_result.n_values_noancilla[0])}"
                f" .. {int(ftqc_result.n_values_noancilla[-1])}"
            )
            print(
                f"  no-ancilla T-count endpoints      : {int(ftqc_result.t_counts_noancilla[0])}"
                f" -> {int(ftqc_result.t_counts_noancilla[-1])}"
            )

    exact_result: ExactAAResults | None = None
    if args.run_exact_aa:
        exact_result = experiment_exact_amplitude_amplification(
            n=args.exact_aa_n,
            m_good=args.exact_aa_m,
            coarse_grid=args.exact_aa_grid,
        )
        print("\nExact Amplitude Amplification:")
        print(f"  N                                 : {exact_result.N}")
        print(f"  M                                 : {exact_result.m_good}")
        print(f"  k_base                            : {exact_result.k_base}")
        print(f"  standard p(k_base)                : {exact_result.standard_probs[exact_result.k_base]:.10f}")
        print(
            f"  standard p(k_base+1)              : "
            f"{exact_result.standard_probs[min(len(exact_result.standard_probs)-1, exact_result.k_base+1)]:.10f}"
        )
        print(
            f"  standard p(k_base+2)              : "
            f"{exact_result.standard_probs[min(len(exact_result.standard_probs)-1, exact_result.k_base+2)]:.10f}"
        )
        print(f"  exact oracle phase alpha          : {exact_result.exact_oracle_phase:.10f} rad")
        print(f"  exact diffusion phase beta        : {exact_result.exact_diffusion_phase:.10f} rad")
        print(f"  exact final probability           : {exact_result.exact_prob:.12f}")

    leakage_result: PhaseLeakageResults | None = None
    if args.run_phase_leakage:
        leakage_result = experiment_phase_mismatch_leakage(
            n=args.leakage_n,
            k_max=args.leakage_k_max,
            eps_oracle_deg=args.leakage_eps_oracle_deg,
            eps_diff_deg=args.leakage_eps_diff_deg,
            crosstalk_oracle_deg=args.leakage_crosstalk_deg,
            local_z_detune_deg=args.leakage_local_z_deg,
            rank_threshold=args.leakage_rank_threshold,
        )
        print("\nPhase-Mismatch Leakage:")
        print(
            f"  mismatch phases                  : oracle={180+leakage_result.eps_oracle_deg:.1f} deg, "
            f"diffusion={180+leakage_result.eps_diff_deg:.1f} deg"
        )
        print(
            f"  analog skew terms                : crosstalk={leakage_result.crosstalk_oracle_deg:.2f} deg, "
            f"local_detune={leakage_result.local_z_detune_deg:.2f} deg"
        )
        print(f"  final rank ideal                 : {int(leakage_result.rank_ideal[-1])}")
        print(f"  final rank mismatch-only         : {int(leakage_result.rank_mismatch_only[-1])}")
        print(f"  final rank leaky                 : {int(leakage_result.rank_leaky[-1])}")
        print(f"  mismatch-only stays rank<=2?     : {leakage_result.mismatch_only_rank2_all}")
        print(f"  non-uniform skew causes leakage? : {leakage_result.leaky_exceeds_rank2}")

    open_result: OpenSystemTrajectoryResults | None = None
    if args.run_open_system_trajectory:
        open_result = experiment_open_system_trajectory(
            n=args.open_system_n,
            k_max=args.open_system_k_max,
            phase_damp_1q=args.open_system_phase_damp_1q,
            phase_damp_2q=args.open_system_phase_damp_2q,
            good_state=args.open_system_good_state,
        )
        print("\nOpen-System Trajectory:")
        print(f"  H_Good state                   : {open_result.good_state}")
        print(f"  phase damping p1q/p2q            : {open_result.phase_damp_1q:.4f} / {open_result.phase_damp_2q:.4f}")
        print(f"  noisy purity (k=0 -> k_max)      : {open_result.noisy_purity[0]:.6f} -> {open_result.noisy_purity[-1]:.6f}")
        print(
            f"  trace distance to 2D plane       : "
            f"{open_result.trace_distance_to_plane[0]:.6f} -> {open_result.trace_distance_to_plane[-1]:.6f}"
        )

    vtaa_state_result: VTAA_StateSynthesisResults | None = None
    if args.run_vtaa_synthesis:
        vtaa_state_result = experiment_vtaa_state_synthesis()
        print("\nVTAA State Synthesis:")
        print(f"  simulated stages                 : {vtaa_state_result.num_stages}")
        print(f"  statevector dimension            : {vtaa_state_result.statevector_dimension}")
        print(f"  success flag |01> probability    : {vtaa_state_result.success_probability:.6f}")
        print(f"  fail flag |10> probability       : {vtaa_state_result.fail_probability:.6f}")
        print(f"  continue flag |00> probability   : {vtaa_state_result.continue_probability:.6f}")

    vtaa_sweep_result: VTAA_CostSweepResults | None = None
    if args.run_vtaa_sweep:
        vtaa_sweep_result = experiment_vtaa_cost_sweep(
            total_ps=args.vtaa_total_ps,
            t1=args.vtaa_t1,
            t2=args.vtaa_t2,
            t3=args.vtaa_t3,
        )
        print("\nVTAA Cost Sweep:")
        print(f"  total p_s                        : {vtaa_sweep_result.total_ps:.6f}")
        print(
            f"  standard/vtaa cost ratio (best)  : "
            f"{float(np.max(vtaa_sweep_result.standard_costs / vtaa_sweep_result.vtaa_costs)):.3f}x"
        )
        print(
            f"  standard/vtaa cost ratio (worst) : "
            f"{float(np.min(vtaa_sweep_result.standard_costs / vtaa_sweep_result.vtaa_costs)):.3f}x"
        )

    if not args.no_plots:
        files = save_plots(lab, report, output_prefix=args.plot_prefix, max_k=args.verify_k)
        if subspace_result is not None:
            files.append(save_subspace_audit_plot(subspace_result, output_prefix=args.plot_prefix))
        if staircase_result is not None:
            files.append(save_phase_staircase_plot(staircase_result, output_prefix=args.plot_prefix))
        if souffle_result is not None:
            files.append(save_souffle_plot(souffle_result, output_prefix=args.plot_prefix))
        if ftqc_result is not None:
            files.append(save_ftqc_scaling_plot(ftqc_result, output_prefix=args.plot_prefix))
        if exact_result is not None:
            files.append(save_exact_aa_peak_plot(exact_result, output_prefix=args.plot_prefix))
        if leakage_result is not None:
            files.append(save_phase_leakage_plot(leakage_result, output_prefix=args.plot_prefix))
        if open_result is not None:
            files.append(save_open_system_trajectory_plot(open_result, output_prefix=args.plot_prefix))
        if vtaa_sweep_result is not None:
            files.append(save_vtaa_cost_sweep_plot(vtaa_sweep_result, output_prefix=args.plot_prefix))
        print("\nSaved plots:")
        for f in files:
            print(f"  - {f}")


if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="vtaa")
    if len(sys.argv) == 1:
        stem = Path(__file__).stem
        main(
            [
                "--plot-prefix",
                stem,
                "--run-subspace-audit",
                "--run-phase-staircase",
                "--run-souffle-catastrophe",
                "--run-ftqc-scaling",
                "--run-exact-aa",
                "--run-phase-leakage",
                "--run-open-system-trajectory",
                "--run-vtaa-synthesis",
                "--run-vtaa-sweep",
            ]
        )
    else:
        main()
