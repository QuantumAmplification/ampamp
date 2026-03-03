"""Variable-Time Amplitude Amplification (VTAA) laboratory.

This module mirrors the style of the earlier files while keeping the
mathematical objects explicit:

- A variable-time algorithm ``A`` is represented by branches ``i`` with
  stopping time ``t_i``, branch weight ``w_i``, and conditional success
  probability ``s_i``.
- The post-``A`` state is
      |psi> = sum_i sqrt(w_i) (sqrt(s_i)|i,good> + sqrt(1-s_i)|i,bad>)
  so total success probability is p = sum_i w_i s_i.
- Standard amplitude amplification acts on this state with the exact
  SU(2) formula
      p_k = sin^2((2k+1) * arcsin(sqrt(p))).

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
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
    theta: float
    grover_k_opt: int
    grover_p_at_k_opt: float
    t_mean: float
    t_rms: float
    t_max: float
    expected_time_until_success_restart: float
    expected_time_until_success_worst_case_aa: float
    vtaa_asymptotic_bound: float


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

        # Branch-level good/bad mass decomposition.
        self.good_mass = self.weights * self.success_given_branch
        self.bad_mass = self.weights * (1.0 - self.success_given_branch)

        self.p_success = float(np.sum(self.good_mass))
        self.theta = float(np.arcsin(np.sqrt(self.p_success))) if self.p_success > 0 else 0.0

        self._validate_exact_identities()

    # ------------------------------------------------------------------
    # Exact state model and identities
    # ------------------------------------------------------------------

    def state_after_A(self) -> np.ndarray:
        """Return |psi> in basis {|i,good>, |i,bad>} (dimension 2m)."""
        m = len(self.stop_times)
        psi = np.zeros(2 * m, dtype=complex)
        psi[0::2] = np.sqrt(self.good_mass)
        psi[1::2] = np.sqrt(self.bad_mass)
        return psi

    def success_projector(self) -> np.ndarray:
        """Projector onto the 'good' subspace spanned by {|i,good>}."""
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
        """Cumulative good mass G_j = sum_{i<=j} w_i s_i."""
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
        """k* = floor(pi/(4*theta) - 1/2) with theta = arcsin(sqrt(p))."""
        if self.p_success <= 0.0:
            return 0
        theta = self.theta
        return max(0, int(np.floor(np.pi / (4.0 * theta) - 0.5)))

    def verify_su2_rotation(self, max_k: int = 20, atol: float = 1e-12) -> None:
        """Cross-check closed-form p_k against 2D rotation matrix evolution."""
        if self.p_success <= 0.0:
            return

        theta = self.theta
        # Initial coordinates in {|bad>, |good>}.
        v = np.array([np.sqrt(1.0 - self.p_success), np.sqrt(self.p_success)], dtype=float)
        # One Grover iterate in invariant plane is rotation by 2*theta.
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
            f"theta = arcsin(sqrt(p))           : {report.theta:.10f}",
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
    ax2.step(stage, g_vals, where="mid", linewidth=1.8, label="G_j = good mass")
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


def main() -> None:
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
    args = parser.parse_args()

    times = _parse_csv_floats(args.times)
    weights = _parse_csv_floats(args.weights)
    success = _parse_csv_floats(args.success)

    branches = _build_branches(times, weights, success)
    lab = VariableTimeAmplitudeAmplificationLab(branches)

    # Mathematical consistency check for the AA closed form.
    lab.verify_su2_rotation(max_k=args.verify_k)

    report = lab.build_report(polylog_factor=args.polylog, constant_tmax=args.c1, constant_trms=args.c2)
    print(format_report(report))
    if not args.no_plots:
        files = save_plots(lab, report, output_prefix=args.plot_prefix, max_k=args.verify_k)
        print("\nSaved plots:")
        for f in files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
