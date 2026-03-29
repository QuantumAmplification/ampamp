"""Fixed-Point Oblivious Amplitude Amplification (FOQA) laboratory.

This module implements a six-experiment FOQA laboratory:
1) LCU oblivious circuit synthesizer
2) Damping-regime sweeper
3) Nonlinear recurrence auditor
4) Asymptotic complexity auditor
5) Adversarial Quantum Zeno failure mode
6) Empty-database boundary stability audit

Standing notation aligned with final.tex:
- H_Good / H_Bad: target and non-target sectors
- Pi_Good / Pi_Bad: orthogonal projectors
- p: success probability parameter
- sin^2(theta0)=p with Grover-step angle 2*theta0
- complexity discussed in oracle/query calls
"""

from __future__ import annotations

import ast
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

try:
    from one_click_utils import start_one_click_session
except Exception:
    def start_one_click_session(script_file, *, figure_prefix=None, log_name="terminal_output.log"):
        import atexit
        import io
        import os
        script_path = Path(script_file).resolve()
        result_dir = script_path.parent / f"[RESULT]{script_path.stem}"
        result_dir.mkdir(parents=True, exist_ok=True)
        old_stdout, old_stderr, old_cwd = sys.stdout, sys.stderr, Path.cwd()
        log_handle = open(result_dir / log_name, "w", encoding="utf-8")
        class _Tee(io.TextIOBase):
            def __init__(self, *streams): self._streams = streams
            def write(self, data): [s.write(data) or s.flush() for s in self._streams]; return len(data)
            def flush(self): [s.flush() for s in self._streams]
        sys.stdout = _Tee(old_stdout, log_handle)
        sys.stderr = _Tee(old_stderr, log_handle)
        os.chdir(result_dir)
        try:
            import matplotlib.pyplot as plt
            old_show = plt.show
            prefix = figure_prefix or script_path.stem
            counter = {"n": 0}
            def _save_show(*args, **kwargs):
                del args, kwargs
                for fig_id in list(plt.get_fignums()):
                    counter["n"] += 1
                    plt.figure(fig_id).savefig(result_dir / f"{prefix}_figure_{counter['n']:03d}.png", dpi=220, bbox_inches="tight")
                plt.close("all")
            plt.show = _save_show
        except Exception:
            old_show = None
        def _cleanup():
            try:
                if old_show is not None:
                    import matplotlib.pyplot as plt
                    plt.show = old_show
            except Exception:
                pass
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.chdir(old_cwd)
            log_handle.close()
        atexit.register(_cleanup)
        return result_dir


def _parse_cli_value(raw: str):
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none":
            return None
        return raw


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote_char = ""
    escaped = False

    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if quote_char:
            current.append(ch)
            if ch == "\\":
                escaped = True
            elif ch == quote_char:
                quote_char = ""
            continue
        if ch in ("'", '"'):
            quote_char = ch
            current.append(ch)
            continue
        if ch in "([{":
            depth += 1
            current.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
            continue
        if ch == "," and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue
        current.append(ch)

    piece = "".join(current).strip()
    if piece:
        parts.append(piece)
    return parts


def _parse_kwargs_text(raw: str) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    for chunk in _split_top_level_commas(raw.strip()):
        if "=" not in chunk:
            raise ValueError(f"Expected key=value pair, got '{chunk}'")
        key, value = chunk.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _parse_kwargs_tokens(tokens: Sequence[str]) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    for token in tokens:
        piece = token.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Expected key=value pair, got '{piece}'")
        key, value = piece.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _parse_command_line(argv: Sequence[str]) -> Optional[Dict[str, object]]:
    if not argv:
        return {}
    if any(token.startswith("-") for token in argv):
        return None
    if len(argv) == 1:
        return _parse_kwargs_text(argv[0])
    return _parse_kwargs_tokens(argv)


@dataclass
class LCUObliviousResults:
    """Container for FOQA Module 1 outputs."""

    theta: float
    alpha_n: float
    state_initial: np.ndarray
    state_post_vn: np.ndarray
    prob_halted: float
    amp_continue: complex
    expected_prob_halted: float
    expected_amp_continue: float
    error_prob: float
    error_amp: float
    math_audit_passed: bool


@dataclass
class DampingRegimeResults:
    """Container for FOQA Module 2 damping-sweep outputs."""

    theta: float
    iterations: int
    alpha_under: float
    alpha_over: float
    mizel_c: float
    probs_underdamped: np.ndarray
    probs_overdamped: np.ndarray
    probs_mizel: np.ndarray
    target_continue_underdamped: np.ndarray
    target_continue_overdamped: np.ndarray
    target_continue_mizel: np.ndarray
    schedule_mizel: np.ndarray
    underdamped_oscillatory: bool
    overdamped_frozen: bool
    mizel_converged: bool
    mizel_monotonic: bool


@dataclass
class FOQARecurrenceResults:
    """Container for FOQA Module 3 recurrence-audit outputs."""

    theta: float
    iterations: int
    recurrence_c: float
    q_n: np.ndarray
    q_n_previous: np.ndarray
    t_n_history: np.ndarray
    monotonic_violations: int
    audit_passed: bool


@dataclass
class FOQAComplexityResults:
    """Container for FOQA Module 4 asymptotic complexity outputs."""

    # Legacy field name retained for compatibility: stores p-values sweep.
    lambdas: np.ndarray
    empirical_steps: np.ndarray
    theoretical_bound: np.ndarray
    grover_baseline: np.ndarray
    empirical_slope: float
    audit_passed: bool


@dataclass
class ZenoCatastropheResults:
    """Container for FOQA Module 5 adversarial Zeno failure mode outputs."""

    theta: float
    iterations: int
    zeno_alpha: float
    mizel_c: float
    probs_zeno: np.ndarray
    probs_mizel: np.ndarray
    probs_classical: np.ndarray
    zeno_to_classical_max_diff: float
    zeno_penalty_ratio: float
    audit_passed: bool


@dataclass
class EmptyDatabaseResults:
    """Container for FOQA Module 6 empty-database boundary audit outputs."""

    iterations: int
    probs_empty: np.ndarray
    probs_control: np.ndarray
    max_noise_amplitude: float
    compiler_crashed: bool
    audit_passed: bool


class FOQALaboratory:
    """Six-module FOQA laboratory.

    Module status:
    1) Implemented
    2) Implemented
    3) Implemented
    4) Implemented
    5) Implemented
    6) Implemented
    """

    def experiment_module1_lcu_oblivious_circuit_synthesizer(
        self,
        theta: float = 0.6,
        alpha_n: float = 1.2,
        show_plot: bool = True,
        save_plot_path: Optional[str] = None,
    ) -> LCUObliviousResults:
        """Module 1: physical proof of LCU damping in FOQA.

        Simulates the tripartite system:
            H_ancilla (2) ⊗ H_index (2) ⊗ H_content (2)

        Initial state:
            |0>_anc ⊗ [sin(theta0)|0>_idx|1>_cont + cos(theta0)|1>_idx|0>_cont]
        """
        if not (0.0 < theta < np.pi / 2.0):
            raise ValueError("theta must be in (0, pi/2).")
        if not (0.0 <= alpha_n <= np.pi):
            raise ValueError("alpha_n should lie in [0, pi] for this module.")

        print("-" * 72)
        print("MODULE 1: THE LCU OBLIVIOUS CIRCUIT SYNTHESIZER (PHYSICAL PROOF)")
        print("-" * 72)
        print(f"Initial overlap theta0: {theta:.6f} rad")
        print(f"Damping angle alpha_n: {alpha_n:.6f} rad")

        ket_0 = np.array([1.0, 0.0], dtype=complex)
        ket_1 = np.array([0.0, 1.0], dtype=complex)

        # |0>_anc|0>_idx|1>_cont (H_Good branch) and |0>_anc|1>_idx|0>_cont (H_Bad branch)
        good_branch = np.kron(ket_0, np.kron(ket_0, ket_1)) * np.sin(theta)
        bad_branch = np.kron(ket_0, np.kron(ket_1, ket_0)) * np.cos(theta)
        state_initial = good_branch + bad_branch

        t_n = np.sin(theta)

        # Primitive FOQA wave-division operator acting on ancilla.
        # The paper defines V_n = exp(-i alpha_n Y / 2), i.e. a half-angle rotation.
        v_n = np.array(
            [
                [np.cos(alpha_n / 2.0), -np.sin(alpha_n / 2.0)],
                [np.sin(alpha_n / 2.0), np.cos(alpha_n / 2.0)],
            ],
            dtype=complex,
        )

        # Controlled on index == |0>: V_n ⊗ |0><0| ⊗ I + I ⊗ |1><1| ⊗ I
        proj_idx_0 = np.outer(ket_0, ket_0.conj())
        proj_idx_1 = np.outer(ket_1, ket_1.conj())
        i_anc = np.eye(2, dtype=complex)
        i_cont = np.eye(2, dtype=complex)

        controlled_vn = np.kron(v_n, np.kron(proj_idx_0, i_cont)) + np.kron(
            i_anc, np.kron(proj_idx_1, i_cont)
        )

        # Unitarity sanity-check: LCU split operator itself must be unitary.
        if not np.allclose(
            controlled_vn.conj().T @ controlled_vn,
            np.eye(8, dtype=complex),
            atol=1e-12,
        ):
            raise AssertionError("Controlled V_n is not unitary; construction is invalid.")

        state_post_vn = controlled_vn @ state_initial

        # LCU branches: halt (anc=|1>) and continue (anc=|0>)
        proj_halt = np.kron(np.outer(ket_1, ket_1.conj()), np.eye(4, dtype=complex))
        proj_cont = np.kron(np.outer(ket_0, ket_0.conj()), np.eye(4, dtype=complex))
        state_halted = proj_halt @ state_post_vn
        state_continue = proj_cont @ state_post_vn

        prob_halted = float(np.sum(np.abs(state_halted) ** 2))

        # Basis ordering from kron is |anc, idx, cont>; H_Good continue = |0,0,1> -> index 1
        amp_continue_target = state_continue[1]

        expected_prob_halted = (np.sin(alpha_n / 2.0) ** 2) * (t_n**2)
        expected_amp_continue = t_n * np.cos(alpha_n / 2.0)

        error_prob = float(np.abs(prob_halted - expected_prob_halted))
        error_amp = float(np.abs(amp_continue_target - expected_amp_continue))
        audit_passed = (error_prob < 1e-12) and (error_amp < 1e-12)

        print(
            f"Prob(halted):         {prob_halted:.12f} "
            f"[expected {expected_prob_halted:.12f}]"
        )
        print(
            f"Amp(H_Good|continue): {amp_continue_target.real:.12f} "
            f"[expected {expected_amp_continue:.12f}]"
        )
        if audit_passed:
            print("Audit: PASS (LCU damping identities verified numerically)")
        else:
            print("Audit: FAIL (check recurrence construction)")
        print("-" * 72)

        self._plot_module1_amplitude_flow(
            state_initial=state_initial,
            state_post_vn=state_post_vn,
            alpha_n=alpha_n,
            show_plot=show_plot,
            save_plot_path=save_plot_path,
        )

        return LCUObliviousResults(
            theta=theta,
            alpha_n=alpha_n,
            state_initial=state_initial,
            state_post_vn=state_post_vn,
            prob_halted=prob_halted,
            amp_continue=amp_continue_target,
            expected_prob_halted=expected_prob_halted,
            expected_amp_continue=expected_amp_continue,
            error_prob=error_prob,
            error_amp=error_amp,
            math_audit_passed=audit_passed,
        )

    def _plot_module1_amplitude_flow(
        self,
        state_initial: np.ndarray,
        state_post_vn: np.ndarray,
        alpha_n: float,
        show_plot: bool,
        save_plot_path: Optional[str],
    ) -> None:
        """Visual evidence for amplitude siphoning in Module 1."""
        fig, ax = plt.subplots(figsize=(11, 6))

        x = np.arange(4)
        labels = [
            r"$|0\rangle_{anc}|0\rangle_{idx}|V\varphi\rangle$ ($H_{\mathrm{Good}}$, Continue)",
            r"$|0\rangle_{anc}|1\rangle_{idx}|\phi\rangle$ ($H_{\mathrm{Bad}}$, Continue)",
            r"$|1\rangle_{anc}|0\rangle_{idx}|V\varphi\rangle$ ($H_{\mathrm{Good}}$, Halted)",
            r"$|1\rangle_{anc}|1\rangle_{idx}|\phi\rangle$ ($H_{\mathrm{Bad}}$, Halted)",
        ]

        # Index map for basis |anc,idx,cont>: 001, 010, 101, 110
        idx_map = [1, 2, 5, 6]
        amps_initial = np.abs(state_initial[idx_map])
        amps_post = np.abs(state_post_vn[idx_map])

        width = 0.36
        ax.bar(
            x - width / 2,
            amps_initial,
            width,
            label="Initial",
            color="#4C78A8",
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            amps_post,
            width,
            label=r"Post $V_n$",
            color="#F58518",
            edgecolor="black",
        )

        ax.annotate(
            r"Siphoned $\propto \sin(\alpha_n/2)$",
            xy=(2.0, amps_post[2]),
            xytext=(2.0, min(0.95, amps_post[2] + 0.18)),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            ha="center",
        )
        ax.annotate(
            r"Retained $\propto \cos(\alpha_n/2)$",
            xy=(0.0, amps_post[0]),
            xytext=(0.0, min(0.95, amps_post[0] + 0.18)),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            ha="center",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=14, ha="right")
        ax.set_ylabel("Absolute Amplitude")
        ax.set_title(f"FOQA LCU Dynamics (alpha_n = {alpha_n:.3f} rad)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
        plt.tight_layout()

        if save_plot_path:
            out = Path(save_plot_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=220)
            print(f"Saved plot: {out}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def experiment_module2_damping_regime_sweeper(
        self,
        theta: float = 0.01,
        iterations: int = 120,
        alpha_under: float = 0.02,
        alpha_over: float = 1.5,
        mizel_c: float = 1.4,
        show_plot: bool = True,
        save_plot_path: Optional[str] = None,
        enforce_audit: bool = True,
    ) -> DampingRegimeResults:
        """Module 2: sweep under/over/critical damping regimes.

        Uses the recurrence described in the manuscript:
            p_{n+1} = sin^2(alpha_n) |t_n|^2
            t_{n+1} = (t_n cos(alpha_n) cos(2theta0) + s_n sin(2theta0)) / sqrt(1-p_{n+1})
            s_{n+1} = (-t_n cos(alpha_n) sin(2theta0) + s_n cos(2theta0)) / sqrt(1-p_{n+1})
        """
        if not (0.0 < theta < np.pi / 2.0):
            raise ValueError("theta must be in (0, pi/2).")
        if iterations < 5:
            raise ValueError("iterations must be >= 5.")

        print("-" * 72)
        print("MODULE 2: THE DAMPING REGIME SWEEPER (PHYSICS OF alpha_n)")
        print("-" * 72)
        print(f"theta0={theta:.6f}, iterations={iterations}")
        print(
            f"Schedules: under={alpha_under:.3f}, over={alpha_over:.3f}, "
            f"mizel=({mizel_c:.3f})/sqrt(n+1)"
        )

        alpha_mizel = mizel_c / np.sqrt(np.arange(1, iterations + 1, dtype=float))
        probs_under, target_under = self._run_module2_observables(
            theta,
            np.full(iterations, alpha_under, dtype=float),
        )
        probs_over, target_over = self._run_module2_observables(
            theta,
            np.full(iterations, alpha_over, dtype=float),
        )
        probs_mizel, target_mizel = self._run_module2_observables(theta, alpha_mizel)

        # Qualitative regime checks (numerical "physics audit").
        # The cumulative halt probability is monotone by construction, so the
        # souffle-style oscillation has to be diagnosed on the continuation
        # branch target weight |t_n|^2 instead.
        under_peak = float(np.max(target_under))
        over_peak = float(np.max(target_over))
        mizel_final_target = float(target_mizel[-1])

        underdamped_oscillatory = bool(
            (under_peak > 0.95)
            and ((under_peak - float(target_under[-1])) > 0.2)
        )
        overdamped_frozen = bool(
            (over_peak < 0.01)
            and (float(np.max(probs_over)) < 0.2)
        )
        mizel_converged = bool(mizel_final_target > 0.99)
        mizel_monotonic = bool(np.all(np.diff(target_mizel) >= -1e-12))

        print(
            "Final success: "
            f"under={probs_under[-1]:.6f}, "
            f"over={probs_over[-1]:.6f}, "
            f"mizel={probs_mizel[-1]:.6f}"
        )
        print(
            "Final |t_n|^2 on continue branch: "
            f"under={target_under[-1]:.6f}, "
            f"over={target_over[-1]:.6f}, "
            f"mizel={target_mizel[-1]:.6f}"
        )
        print(
            "Audit flags: "
            f"underdamped_oscillatory={underdamped_oscillatory}, "
            f"overdamped_frozen={overdamped_frozen}, "
            f"mizel_converged={mizel_converged}, "
            f"mizel_monotonic={mizel_monotonic}"
        )

        if enforce_audit and not (
            underdamped_oscillatory
            and overdamped_frozen
            and mizel_converged
            and mizel_monotonic
        ):
            raise AssertionError("Module 2 damping-regime audit failed.")

        self._plot_module2_damping_regimes(
            theta=theta,
            probs_under=probs_under,
            probs_over=probs_over,
            probs_mizel=probs_mizel,
            target_under=target_under,
            target_over=target_over,
            target_mizel=target_mizel,
            alpha_under=alpha_under,
            alpha_over=alpha_over,
            mizel_c=mizel_c,
            show_plot=show_plot,
            save_plot_path=save_plot_path,
        )
        print("-" * 72)

        return DampingRegimeResults(
            theta=theta,
            iterations=iterations,
            alpha_under=alpha_under,
            alpha_over=alpha_over,
            mizel_c=mizel_c,
            probs_underdamped=probs_under,
            probs_overdamped=probs_over,
            probs_mizel=probs_mizel,
            target_continue_underdamped=target_under,
            target_continue_overdamped=target_over,
            target_continue_mizel=target_mizel,
            schedule_mizel=alpha_mizel,
            underdamped_oscillatory=underdamped_oscillatory,
            overdamped_frozen=overdamped_frozen,
            mizel_converged=mizel_converged,
            mizel_monotonic=mizel_monotonic,
        )

    def _run_module2_observables(
        self,
        theta: float,
        alpha_schedule: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cumulative halt probability and continuation-branch target weight."""
        t_n = np.sin(theta)
        s_n = np.cos(theta)

        prob_already_halted = 0.0
        prob_continue = 1.0
        cumulative_success = np.zeros(len(alpha_schedule), dtype=float)
        target_continue = np.zeros(len(alpha_schedule), dtype=float)

        for idx, alpha in enumerate(alpha_schedule):
            target_continue[idx] = float(np.clip(t_n**2, 0.0, 1.0))

            p_step = (np.sin(alpha) ** 2) * (t_n**2)
            p_step = float(np.clip(p_step, 0.0, 1.0 - 1e-15))

            prob_already_halted += prob_continue * p_step
            prob_continue *= 1.0 - p_step
            cumulative_success[idx] = prob_already_halted

            norm = np.sqrt(max(1e-15, 1.0 - p_step))
            t_new = (
                t_n * np.cos(alpha) * np.cos(2.0 * theta)
                + s_n * np.sin(2.0 * theta)
            ) / norm
            s_new = (
                -t_n * np.cos(alpha) * np.sin(2.0 * theta)
                + s_n * np.cos(2.0 * theta)
            ) / norm
            t_n, s_n = float(t_new), float(s_new)

        return cumulative_success, target_continue

    def _run_module2_recurrence(
        self,
        theta: float,
        alpha_schedule: np.ndarray,
    ) -> np.ndarray:
        """Recurrence engine for Module 2."""
        cumulative_success, _ = self._run_module2_observables(theta, alpha_schedule)
        return cumulative_success

    def _plot_module2_damping_regimes(
        self,
        theta: float,
        probs_under: np.ndarray,
        probs_over: np.ndarray,
        probs_mizel: np.ndarray,
        target_under: np.ndarray,
        target_over: np.ndarray,
        target_mizel: np.ndarray,
        alpha_under: float,
        alpha_over: float,
        mizel_c: float,
        show_plot: bool,
        save_plot_path: Optional[str],
    ) -> None:
        """Comparative 3-regime evidence plot for Module 2."""
        fig, (ax_halt, ax_target) = plt.subplots(1, 2, figsize=(14, 6))
        steps = np.arange(1, len(probs_under) + 1)

        ax_halt.plot(
            steps,
            probs_under,
            color="#D62728",
            linestyle=":",
            linewidth=2.6,
            label=rf"Underdamped ($\alpha_n={alpha_under:.2f}$): Souffle oscillation",
        )
        ax_halt.plot(
            steps,
            probs_over,
            color="#FF7F0E",
            linestyle="--",
            linewidth=2.6,
            label=rf"Overdamped ($\alpha_n={alpha_over:.2f}$): Zeno-like freezing",
        )
        ax_halt.plot(
            steps,
            probs_mizel,
            color="#2CA02C",
            linewidth=3.2,
            label=rf"Critical (Mizel) ($\alpha_n={mizel_c:.1f}/\sqrt{{n+1}}$)",
        )

        ax_halt.annotate(
            "Zeno-like plateau",
            xy=(len(steps) * 0.70, probs_over[int(len(steps) * 0.70)]),
            xytext=(len(steps) * 0.52, min(1.0, probs_over.max() + 0.12)),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            color="#B85C00",
        )
        ax_halt.annotate(
            "Monotone halt gain",
            xy=(len(steps) * 0.80, probs_mizel[int(len(steps) * 0.80)]),
            xytext=(len(steps) * 0.63, 0.80),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            color="#1E7A1E",
        )

        ax_halt.axhline(1.0, color="black", linewidth=1.2)
        ax_halt.set_ylim(0.0, 1.05)
        ax_halt.set_xlabel("Iteration n")
        ax_halt.set_ylabel("Cumulative halt probability")
        ax_halt.set_title(f"FOQA Damping Regimes: Halt Probability (theta0={theta:.4f})")
        ax_halt.grid(alpha=0.30)
        ax_halt.legend(loc="lower right")

        ax_target.plot(
            steps,
            target_under,
            color="#D62728",
            linestyle=":",
            linewidth=2.6,
            label="Underdamped continuation target weight",
        )
        ax_target.plot(
            steps,
            target_over,
            color="#FF7F0E",
            linestyle="--",
            linewidth=2.6,
            label="Overdamped continuation target weight",
        )
        ax_target.plot(
            steps,
            target_mizel,
            color="#2CA02C",
            linewidth=3.2,
            label="Mizel continuation target weight",
        )

        under_peak_idx = int(np.argmax(target_under))
        mizel_anchor_idx = min(len(steps) - 1, int(len(steps) * 0.80))
        ax_target.annotate(
            "Souffle overshoot",
            xy=(steps[under_peak_idx], target_under[under_peak_idx]),
            xytext=(steps[under_peak_idx] * 0.55, 0.85),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            color="#9E1B1B",
        )
        ax_target.annotate(
            "Fixed-point lock",
            xy=(steps[mizel_anchor_idx], target_mizel[mizel_anchor_idx]),
            xytext=(steps[mizel_anchor_idx] * 0.58, 0.62),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            color="#1E7A1E",
        )

        ax_target.set_ylim(0.0, 1.05)
        ax_target.set_xlabel("Iteration n")
        ax_target.set_ylabel(r"Continuation-branch target weight $|t_n|^2$")
        ax_target.set_title("FOQA Damping Regimes: Continuation Dynamics")
        ax_target.grid(alpha=0.30)
        ax_target.legend(loc="lower right")
        plt.tight_layout()

        if save_plot_path:
            out = Path(save_plot_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=220)
            print(f"Saved plot: {out}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def experiment_module3_nonlinear_recurrence_auditor(
        self,
        theta: float = 0.15,
        iterations: int = 6,
        recurrence_c: float = 0.5,
        show_plot: bool = True,
        save_plot_path: Optional[str] = None,
        enforce_audit: bool = True,
    ) -> FOQARecurrenceResults:
        """Module 3: strict numerical audit of the FOQA continuation probability."""
        if not (0.0 < theta < np.pi / 2.0):
            raise ValueError("theta must be in (0, pi/2).")
        if iterations < 2:
            raise ValueError("iterations must be >= 2.")

        print("-" * 72)
        print("MODULE 3: THE NON-LINEAR RECURRENCE AUDITOR (MATH PROOF)")
        print("-" * 72)
        print(f"theta0={theta:.6f}, iterations={iterations}, c={recurrence_c:.6f}")

        t_n = np.sin(theta)
        s_n = np.cos(theta)
        prob_already_halted = 0.0
        prob_continue = 1.0

        q_n_history = np.ones(iterations, dtype=float)
        t_n_history = np.zeros(iterations, dtype=float)
        t_n_history[0] = float(t_n)
        alpha_schedule = recurrence_c / np.sqrt(np.arange(1, iterations, dtype=float))

        for n in range(1, iterations):
            alpha_n = alpha_schedule[n - 1]
            p_step = (np.sin(alpha_n) ** 2) * (t_n**2)
            p_step = float(np.clip(p_step, 0.0, 1.0 - 1e-15))

            prob_already_halted += prob_continue * p_step
            prob_continue *= 1.0 - p_step

            norm_factor = np.sqrt(max(1e-15, 1.0 - p_step))
            t_new = (t_n * np.cos(alpha_n) * np.cos(2.0 * theta) + s_n * np.sin(2.0 * theta)) / norm_factor
            s_new = (-t_n * np.cos(alpha_n) * np.sin(2.0 * theta) + s_n * np.cos(2.0 * theta)) / norm_factor
            t_n, s_n = float(t_new), float(s_new)
            q_n_history[n] = float(np.clip(prob_continue, 0.0, 1.0))
            t_n_history[n] = float(t_n)

        q_n_previous = np.ones_like(q_n_history)
        q_n_previous[1:] = q_n_history[:-1]

        monotonic_violations = int(np.sum(np.diff(q_n_history) > 1e-12))
        audit_passed = monotonic_violations == 0

        print(f"Initial failure q0: {q_n_history[0]:.6f}")
        print(f"Final recorded failure q_n: {q_n_history[-1]:.6e}")
        print(f"Monotonicity violations: {monotonic_violations}")
        if audit_passed:
            print("Audit: PASS (the continuation probability q_n is nonincreasing)")
        else:
            print("Audit: FAIL (the continuation probability increased on some step)")

        if enforce_audit and not audit_passed:
            raise AssertionError("Module 3 recurrence audit failed.")

        self._plot_module3_recurrence_audit(
            q_n_history=q_n_history,
            q_n_previous=q_n_previous,
            show_plot=show_plot,
            save_plot_path=save_plot_path,
        )
        print("-" * 72)

        return FOQARecurrenceResults(
            theta=theta,
            iterations=iterations,
            recurrence_c=recurrence_c,
            q_n=q_n_history,
            q_n_previous=q_n_previous,
            t_n_history=t_n_history,
            monotonic_violations=monotonic_violations,
            audit_passed=audit_passed,
        )

    def _plot_module3_recurrence_audit(
        self,
        q_n_history: np.ndarray,
        q_n_previous: np.ndarray,
        show_plot: bool,
        save_plot_path: Optional[str],
    ) -> None:
        """Log-scale failure-probability audit plot for Module 3."""
        fig, ax = plt.subplots(figsize=(11, 6))
        steps = np.arange(len(q_n_history))

        q_plot = np.clip(q_n_history, 1e-14, 1.0)
        q_prev_plot = np.clip(q_n_previous, 1e-14, 1.0)

        ax.plot(
            steps,
            q_plot,
            color="#D62728",
            marker="o",
            linewidth=2.8,
            label=r"Actual failure probability $q_n$",
        )
        ax.plot(
            steps,
            q_prev_plot,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label=r"Previous-step reference $q_{n-1}$",
        )

        ax.set_yscale("log")
        ax.set_ylim(1e-12, 1.2)
        ax.set_xlabel("Iteration n")
        ax.set_ylabel("Failure Probability (log scale)")
        ax.set_title("FOQA Recurrence Audit: Continuation-Probability Convergence")
        ax.grid(True, which="both", alpha=0.30)
        ax.legend(loc="upper right")
        plt.tight_layout()

        if save_plot_path:
            out = Path(save_plot_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=220)
            print(f"Saved plot: {out}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def experiment_module4_asymptotic_complexity_auditor(
        self,
        num_p_values: int = 20,
        p_max_exponent: int = -1,
        p_min_exponent: int = -6,
        target_success: float = 0.99,
        complexity_c: float = 1.5,
        max_iterations: int = 50_000,
        slope_tolerance: float = 0.02,
        show_plot: bool = True,
        save_plot_path: Optional[str] = None,
        enforce_audit: bool = True,
    ) -> FOQAComplexityResults:
        """Module 4: asymptotic query-complexity audit under unknown p."""
        if num_p_values < 5:
            raise ValueError("num_p_values must be >= 5.")
        if not (0.0 < target_success < 1.0):
            raise ValueError("target_success must lie in (0, 1).")
        if complexity_c <= 0.0:
            raise ValueError("complexity_c must be > 0.")
        if max_iterations < 100:
            raise ValueError("max_iterations must be >= 100.")

        print("-" * 72)
        print("MODULE 4: THE ASYMPTOTIC COMPLEXITY AUDITOR (OPTIMALITY PROOF)")
        print("-" * 72)

        p_values = np.logspace(
            p_max_exponent,
            p_min_exponent,
            num_p_values,
            dtype=float,
        )
        empirical_steps: list[int] = []

        print(
            "Sweep range: "
            f"10^{p_max_exponent} -> 10^{p_min_exponent} "
            f"over {num_p_values} p values"
        )
        print(
            f"Target success >= {target_success:.4f}; max_iterations={max_iterations}"
        )

        for p in p_values:
            theta = float(np.arcsin(np.sqrt(p)))
            t_n = float(np.sin(theta))
            s_n = float(np.cos(theta))

            prob_already_halted = 0.0
            prob_continue = 1.0
            n = 0

            while True:
                current_success = prob_already_halted
                if current_success >= target_success:
                    break

                alpha_n = complexity_c / np.sqrt(n + 1.0)
                p_step = (np.sin(alpha_n) ** 2) * (t_n**2)
                p_step = float(np.clip(p_step, 0.0, 1.0 - 1e-15))

                prob_already_halted += prob_continue * p_step
                prob_continue *= 1.0 - p_step

                norm = np.sqrt(max(1e-15, 1.0 - p_step))
                t_new = (
                    t_n * np.cos(alpha_n) * np.cos(2.0 * theta) + s_n * np.sin(2.0 * theta)
                ) / norm
                s_new = (
                    -t_n * np.cos(alpha_n) * np.sin(2.0 * theta) + s_n * np.cos(2.0 * theta)
                ) / norm
                t_n, s_n = float(t_new), float(s_new)
                n += 1

                if n > max_iterations:
                    raise RuntimeError(
                        f"Module 4 convergence failure for p={p:.6e} "
                        f"within {max_iterations} iterations."
                    )

            empirical_steps.append(n)

        empirical_steps_np = np.asarray(empirical_steps, dtype=float)
        grover_baseline = (np.pi / 4.0) * np.sqrt(1.0 / p_values)
        theoretical_bound = complexity_c * np.sqrt(1.0 / p_values)

        log_p_values = np.log10(p_values)
        log_steps = np.log10(empirical_steps_np)
        empirical_slope, _ = np.polyfit(log_p_values, log_steps, 1)
        empirical_slope = float(empirical_slope)

        audit_passed = bool(np.abs(empirical_slope - (-0.5)) < slope_tolerance)

        print(f"Empirical scaling exponent: {empirical_slope:.6f} (target -0.5)")
        print(
            "Audit: "
            + (
                "PASS (quadratic speedup preserved)"
                if audit_passed
                else "FAIL (slope deviates from quadratic scaling)"
            )
        )

        if enforce_audit and not audit_passed:
            raise AssertionError("Module 4 asymptotic complexity audit failed.")

        self._plot_module4_asymptotic_complexity(
            lambdas=p_values,
            empirical_steps=empirical_steps_np,
            theoretical_bound=theoretical_bound,
            grover_baseline=grover_baseline,
            empirical_slope=empirical_slope,
            complexity_c=complexity_c,
            target_success=target_success,
            show_plot=show_plot,
            save_plot_path=save_plot_path,
        )
        print("-" * 72)

        return FOQAComplexityResults(
            lambdas=p_values,
            empirical_steps=empirical_steps_np,
            theoretical_bound=theoretical_bound,
            grover_baseline=grover_baseline,
            empirical_slope=empirical_slope,
            audit_passed=audit_passed,
        )

    def _plot_module4_asymptotic_complexity(
        self,
        lambdas: np.ndarray,
        empirical_steps: np.ndarray,
        theoretical_bound: np.ndarray,
        grover_baseline: np.ndarray,
        empirical_slope: float,
        complexity_c: float,
        target_success: float,
        show_plot: bool,
        save_plot_path: Optional[str],
    ) -> None:
        """Log-log scaling evidence for Module 4."""
        fig, ax = plt.subplots(figsize=(11, 6))

        ax.scatter(
            lambdas,
            empirical_steps,
            color="#2CA02C",
            marker="o",
            s=80,
            zorder=3,
            label=rf"Empirical halting step ($P_{{succ}}>{target_success:.2f}$)",
        )
        ax.plot(
            lambdas,
            theoretical_bound,
            color="black",
            linestyle="-",
            linewidth=2.6,
            zorder=2,
            label=rf"FOQA bound: ${complexity_c:.2f}\sqrt{{1/p}}$",
        )
        ax.plot(
            lambdas,
            grover_baseline,
            color="#D62728",
            linestyle="--",
            linewidth=2.1,
            zorder=1,
            label=r"Grover baseline: $\frac{\pi}{4}\sqrt{1/p}$",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.30)
        ax.set_title("FOQA Asymptotic Complexity Audit")
        ax.set_xlabel(r"Initial success probability $p$ (decreasing $\rightarrow$)")
        ax.set_ylabel("Queries / iterations to high success")
        ax.legend(loc="lower left")

        anchor_idx = int(len(lambdas) * 0.5)
        anchor_idx = max(0, min(anchor_idx, len(lambdas) - 1))
        ax.annotate(
            rf"Empirical scaling: $p^{{{empirical_slope:.3f}}}$",
            xy=(lambdas[anchor_idx], theoretical_bound[anchor_idx]),
            xytext=(lambdas[max(0, anchor_idx - 3)], theoretical_bound[anchor_idx] * 2.5),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            fontsize=11,
        )

        plt.tight_layout()
        if save_plot_path:
            out = Path(save_plot_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=220)
            print(f"Saved plot: {out}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def experiment_module5_adversarial_zeno_catastrophe(
        self,
        theta: float = 0.01,
        iterations: int = 100,
        zeno_alpha: float = 1.5,
        mizel_c: float = 1.5,
        classical_tolerance: float = 0.05,
        penalty_threshold: float = 5.0,
        show_plot: bool = True,
        save_plot_path: Optional[str] = None,
        enforce_audit: bool = True,
    ) -> ZenoCatastropheResults:
        """Module 5: adversarial FOQA audit under extreme constant damping."""
        if not (0.0 < theta < np.pi / 2.0):
            raise ValueError("theta must be in (0, pi/2).")
        if iterations < 10:
            raise ValueError("iterations must be >= 10.")
        if not (0.0 <= zeno_alpha <= np.pi):
            raise ValueError("zeno_alpha must lie in [0, pi].")
        if mizel_c <= 0.0:
            raise ValueError("mizel_c must be > 0.")
        if classical_tolerance <= 0.0:
            raise ValueError("classical_tolerance must be > 0.")
        if penalty_threshold <= 1.0:
            raise ValueError("penalty_threshold must be > 1.")

        print("-" * 72)
        print("MODULE 5: ADVERSARIAL FOQA - THE QUANTUM ZENO CATASTROPHE")
        print("-" * 72)

        initial_prob = float(np.sin(theta) ** 2)
        print(
            f"theta0={theta:.6f}, iterations={iterations}, "
            f"initial_prob={initial_prob:.8f}"
        )
        print(
            f"Schedules: zeno(alpha={zeno_alpha:.4f}), "
            f"mizel(c/sqrt(n+1), c={mizel_c:.4f})"
        )

        alpha_zeno = np.full(iterations, zeno_alpha, dtype=float)
        alpha_mizel = mizel_c / np.sqrt(np.arange(1, iterations + 1, dtype=float))

        probs_zeno = self._run_module2_recurrence(theta=theta, alpha_schedule=alpha_zeno)
        probs_mizel = self._run_module2_recurrence(theta=theta, alpha_schedule=alpha_mizel)

        # Shift by +1 so n=0 aligns with the recurrence's first-point convention.
        probs_classical = 1.0 - (1.0 - initial_prob) ** (np.arange(iterations) + 1.0)

        zeno_to_classical_max_diff = float(np.max(np.abs(probs_zeno - probs_classical)))
        zeno_penalty_ratio = float(probs_mizel[-1] / max(probs_zeno[-1], 1e-15))

        print(f"Final success p (Mizel):     {probs_mizel[-1]:.6f}")
        print(f"Final success p (Zeno):      {probs_zeno[-1]:.6f}")
        print(f"Final success p (Classical): {probs_classical[-1]:.6f}")
        print(f"Max |Zeno-Classical|:      {zeno_to_classical_max_diff:.6f}")
        print(f"Mizel/Zeno penalty ratio:  {zeno_penalty_ratio:.6f}")

        audit_passed = bool(
            (zeno_to_classical_max_diff < classical_tolerance)
            and (zeno_penalty_ratio > penalty_threshold)
        )

        if audit_passed:
            print("Audit: PASS (heavy damping induces near-classical Zeno behavior)")
        else:
            print("Audit: FAIL (adversarial Zeno collapse not strong under current settings)")

        if enforce_audit and not audit_passed:
            raise AssertionError("Module 5 Zeno failure mode audit failed.")

        self._plot_module5_zeno_catastrophe(
            probs_zeno=probs_zeno,
            probs_mizel=probs_mizel,
            probs_classical=probs_classical,
            zeno_alpha=zeno_alpha,
            mizel_c=mizel_c,
            theta=theta,
            show_plot=show_plot,
            save_plot_path=save_plot_path,
        )
        print("-" * 72)

        return ZenoCatastropheResults(
            theta=theta,
            iterations=iterations,
            zeno_alpha=zeno_alpha,
            mizel_c=mizel_c,
            probs_zeno=probs_zeno,
            probs_mizel=probs_mizel,
            probs_classical=probs_classical,
            zeno_to_classical_max_diff=zeno_to_classical_max_diff,
            zeno_penalty_ratio=zeno_penalty_ratio,
            audit_passed=audit_passed,
        )

    def _plot_module5_zeno_catastrophe(
        self,
        probs_zeno: np.ndarray,
        probs_mizel: np.ndarray,
        probs_classical: np.ndarray,
        zeno_alpha: float,
        mizel_c: float,
        theta: float,
        show_plot: bool,
        save_plot_path: Optional[str],
    ) -> None:
        """Comparative evidence plot for adversarial Zeno damping."""
        fig, ax = plt.subplots(figsize=(11, 6))
        steps = np.arange(1, len(probs_zeno) + 1)

        ax.plot(
            steps,
            probs_mizel,
            color="#2CA02C",
            linewidth=3.0,
            label=rf"Mizel schedule ($\alpha_n={mizel_c:.2f}/\sqrt{{n+1}}$)",
        )
        ax.plot(
            steps,
            probs_zeno,
            color="#FF7F0E",
            linestyle="--",
            linewidth=3.0,
            label=rf"Zeno schedule ($\alpha_n={zeno_alpha:.2f}$ constant)",
        )
        ax.plot(
            steps,
            probs_classical,
            color="black",
            linestyle=":",
            linewidth=2.2,
            label=r"Classical guessing baseline $1-(1-\sin^2\theta)^n$",
        )

        marker_step = int(len(steps) * 0.65)
        marker_step = max(0, min(marker_step, len(steps) - 1))
        ax.annotate(
            "Coherent buildup suppressed",
            xy=(steps[marker_step], probs_zeno[marker_step]),
            xytext=(steps[max(0, marker_step - 25)], min(0.95, probs_zeno[marker_step] + 0.35)),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            color="#C66A00",
            fontsize=11,
        )

        ax.set_title(f"FOQA Adversarial Audit: Quantum Zeno Catastrophe (theta0={theta:.3f})")
        ax.set_xlabel("Iteration n")
        ax.set_ylabel("Cumulative halt probability")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.30)
        ax.legend(loc="upper left")
        plt.tight_layout()

        if save_plot_path:
            out = Path(save_plot_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=220)
            print(f"Saved plot: {out}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def experiment_module6_empty_database_paradox(
        self,
        iterations: int = 50,
        control_theta: float = 0.1,
        mizel_c: float = 1.5,
        noise_floor: float = 1e-15,
        show_plot: bool = True,
        save_plot_path: Optional[str] = None,
        enforce_audit: bool = True,
    ) -> EmptyDatabaseResults:
        """Module 6: empty-database boundary audit for theta0 = 0.

        The audit verifies three safety requirements under a null oracle:
        - no NaN/Inf propagation in state or probability traces,
        - no runtime crash/exception in the recurrence loop,
        - strict suppression of false positives under `noise_floor`.
        """
        if iterations < 2:
            raise ValueError("iterations must be >= 2.")
        if not (0.0 < control_theta < np.pi / 2.0):
            raise ValueError("control_theta must be in (0, pi/2).")
        if mizel_c <= 0.0:
            raise ValueError("mizel_c must be > 0.")
        if noise_floor <= 0.0:
            raise ValueError("noise_floor must be > 0.")

        print("-" * 72)
        print("MODULE 6: ADVERSARIAL FOQA - THE EMPTY DATABASE PARADOX")
        print("-" * 72)
        print(
            f"iterations={iterations}, control_theta0={control_theta:.6f}, "
            f"schedule={mizel_c:.3f}/sqrt(n+1)"
        )

        alpha_schedule = mizel_c / np.sqrt(np.arange(1, iterations + 1, dtype=float))

        def run_foqa_safe(theta: float) -> tuple[np.ndarray, bool]:
            """Run one FOQA recurrence trajectory with defensive numerics."""
            t_n = float(np.sin(theta))
            s_n = float(np.cos(theta))
            prob_already_halted = 0.0
            prob_continue = 1.0
            cumulative_success = np.full(iterations, np.nan, dtype=float)

            for idx, alpha in enumerate(alpha_schedule):
                try:
                    p_step = (np.sin(alpha) ** 2) * (t_n**2)
                    p_step = float(np.clip(p_step, 0.0, 1.0 - 1e-15))

                    prob_already_halted += prob_continue * p_step
                    prob_continue *= 1.0 - p_step
                    cumulative_success[idx] = float(prob_already_halted)

                    # Mathematically norm = sqrt(1 - p_step); clamp for numerical safety.
                    norm = np.sqrt(max(np.finfo(float).tiny, 1.0 - p_step))
                    t_new = (
                        t_n * np.cos(alpha) * np.cos(2.0 * theta)
                        + s_n * np.sin(2.0 * theta)
                    ) / norm
                    s_new = (
                        -t_n * np.cos(alpha) * np.sin(2.0 * theta)
                        + s_n * np.cos(2.0 * theta)
                    ) / norm

                    if not np.isfinite(t_new) or not np.isfinite(s_new):
                        return cumulative_success, True
                    if not np.isfinite(prob_already_halted) or not np.isfinite(prob_continue):
                        return cumulative_success, True

                    t_n, s_n = float(t_new), float(s_new)
                except Exception:
                    return cumulative_success, True

            return cumulative_success, False

        print("Executing null-oracle boundary (theta0 = 0.0) ...")
        probs_empty, crashed_empty = run_foqa_safe(theta=0.0)
        print(f"Executing control baseline (theta0 = {control_theta:.3f}) ...")
        probs_control, crashed_control = run_foqa_safe(theta=control_theta)

        finite_empty = bool(np.all(np.isfinite(probs_empty)))
        finite_control = bool(np.all(np.isfinite(probs_control)))
        compiler_crashed = bool(
            crashed_empty or crashed_control or (not finite_empty) or (not finite_control)
        )
        max_noise = float(np.max(np.abs(probs_empty))) if finite_empty else float("inf")

        print(f"Compiler crashed (NaN/Inf/exception): {compiler_crashed}")
        print(f"Max empty-database signal:            {max_noise:.4e}")

        audit_passed = bool((not compiler_crashed) and (max_noise < noise_floor))
        if audit_passed:
            print("Audit: PASS (strict zero maintained without numerical instability)")
        else:
            print("Audit: FAIL (boundary handling produced noise or numerical instability)")

        if enforce_audit and not audit_passed:
            raise AssertionError("Module 6 empty-database audit failed.")

        self._plot_module6_empty_database_paradox(
            probs_empty=probs_empty,
            probs_control=probs_control,
            control_theta=control_theta,
            show_plot=show_plot,
            save_plot_path=save_plot_path,
        )
        print("-" * 72)

        return EmptyDatabaseResults(
            iterations=iterations,
            probs_empty=probs_empty,
            probs_control=probs_control,
            max_noise_amplitude=max_noise,
            compiler_crashed=compiler_crashed,
            audit_passed=audit_passed,
        )

    def _plot_module6_empty_database_paradox(
        self,
        probs_empty: np.ndarray,
        probs_control: np.ndarray,
        control_theta: float,
        show_plot: bool,
        save_plot_path: Optional[str],
    ) -> None:
        """Boundary evidence plot for safe empty-database handling."""
        fig, ax = plt.subplots(figsize=(11, 5.5))
        steps = np.arange(1, len(probs_empty) + 1)

        ax.plot(
            steps,
            probs_control,
            color="#7F7F7F",
            linestyle=":",
            linewidth=2.1,
            label=rf"Control ($\theta={control_theta:.2f}$): normal convergence",
        )
        ax.plot(
            steps,
            probs_empty,
            color="#D62728",
            linewidth=3.0,
            label=r"Empty database ($\theta0=0$): stable flatline",
        )

        anchor = max(1, len(steps) // 2)
        ax.annotate(
            "Strict zero preserved\n(no NaN/Inf propagation)",
            xy=(anchor, 0.0),
            xytext=(anchor, 0.28),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            color="#B22222",
            ha="center",
            fontsize=11,
        )

        ax.set_title("FOQA Boundary Audit: Empty Database Paradox")
        ax.set_xlabel("Iteration n")
        ax.set_ylabel("Cumulative halt probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.30)
        ax.legend(loc="center right")
        plt.tight_layout()

        if save_plot_path:
            out = Path(save_plot_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=220)
            print(f"Saved plot: {out}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FOQA (3.5) module runner")
    parser.add_argument("--module", type=int, default=1, help="Module index to run.")
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Initial overlap theta0. Defaults: module1=0.6, module2=0.01",
    )
    parser.add_argument("--alpha", type=float, default=1.2, help="Damping angle alpha_n.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Iteration count. Defaults: module2=120, module3=6.",
    )
    parser.add_argument(
        "--alpha-under",
        type=float,
        default=0.02,
        help="Constant underdamped alpha_n for Module 2.",
    )
    parser.add_argument(
        "--alpha-over",
        type=float,
        default=1.5,
        help="Constant overdamped alpha_n for Module 2.",
    )
    parser.add_argument(
        "--mizel-c",
        type=float,
        default=1.4,
        help="Constant c in alpha_n = c/sqrt(n+1) for Module 2.",
    )
    parser.add_argument(
        "--recurrence-c",
        type=float,
        default=0.5,
        help="Constant c in alpha_n = c/sqrt(n+1) for Module 3.",
    )
    parser.add_argument(
        "--complexity-c",
        type=float,
        default=1.5,
        help="Constant c in alpha_n = c/sqrt(n+1) for Module 4.",
    )
    parser.add_argument(
        "--num-p-values",
        type=int,
        default=20,
        dest="num_lambdas",
        help="Number of p points for Module 4 log sweep.",
    )
    parser.add_argument(
        "--num-lambdas",
        type=int,
        dest="num_lambdas",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--p-max-exp",
        type=int,
        default=-1,
        help="Largest exponent for p sweep in Module 4 (10^exp).",
    )
    parser.add_argument(
        "--p-min-exp",
        type=int,
        default=-6,
        help="Smallest exponent for p sweep in Module 4 (10^exp).",
    )
    parser.add_argument(
        "--target-success-p",
        type=float,
        default=0.99,
        dest="good_success",
        help="Cumulative success probability threshold p_target for Module 4 halting.",
    )
    parser.add_argument(
        "--good-success",
        type=float,
        dest="good_success",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50_000,
        help="Iteration failsafe per p in Module 4.",
    )
    parser.add_argument(
        "--slope-tolerance",
        type=float,
        default=0.02,
        help="Tolerance around -0.5 slope for Module 4 audit.",
    )
    parser.add_argument(
        "--zeno-theta",
        type=float,
        default=0.01,
        help="Initial overlap theta0 for Module 5 adversarial Zeno audit.",
    )
    parser.add_argument(
        "--zeno-iterations",
        type=int,
        default=100,
        help="Iteration count for Module 5 adversarial Zeno audit.",
    )
    parser.add_argument(
        "--zeno-alpha",
        type=float,
        default=1.5,
        help="Constant heavy damping alpha for Module 5 Zeno schedule.",
    )
    parser.add_argument(
        "--zeno-mizel-c",
        type=float,
        default=1.5,
        help="Mizel schedule constant c in alpha_n = c/sqrt(n+1) for Module 5.",
    )
    parser.add_argument(
        "--zeno-classical-tolerance",
        type=float,
        default=0.05,
        help="Max |Zeno-Classical| tolerance for Module 5 failure mode audit.",
    )
    parser.add_argument(
        "--zeno-penalty-threshold",
        type=float,
        default=5.0,
        help="Required final Mizel/Zeno success ratio for Module 5 audit.",
    )
    parser.add_argument(
        "--empty-iterations",
        type=int,
        default=50,
        help="Iteration count for Module 6 empty-database audit.",
    )
    parser.add_argument(
        "--empty-control-theta",
        type=float,
        default=0.1,
        help="Control-theta0 baseline for Module 6 empty-database audit.",
    )
    parser.add_argument(
        "--empty-mizel-c",
        type=float,
        default=1.5,
        help="Mizel schedule constant c in alpha_n = c/sqrt(n+1) for Module 6.",
    )
    parser.add_argument(
        "--empty-noise-floor",
        type=float,
        default=1e-15,
        help="Maximum allowed empty-database noise amplitude for Module 6 audit.",
    )
    parser.add_argument(
        "--no-enforce-audit",
        action="store_true",
        help="Disable strict audit assertions for Modules 2-6.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive plotting (useful in headless runs).",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Optional path to save the module plot.",
    )
    return parser


def experiment_adversarial_empty_database(
    iterations: int = 50,
    control_theta: float = 0.1,
    mizel_c: float = 1.5,
    noise_floor: float = 1e-15,
    show_plot: bool = True,
    save_plot_path: Optional[str] = None,
    enforce_audit: bool = True,
) -> EmptyDatabaseResults:
    """Standalone wrapper for Module 6 empty-database audit."""
    return FOQALaboratory().experiment_module6_empty_database_paradox(
        iterations=iterations,
        control_theta=control_theta,
        mizel_c=mizel_c,
        noise_floor=noise_floor,
        show_plot=show_plot,
        save_plot_path=save_plot_path,
        enforce_audit=enforce_audit,
    )


def run_full_analysis(
    module1_theta: float = 0.6,
    module1_alpha: float = 1.2,
    module2_theta: float = 0.01,
    module2_iterations: int = 120,
    module2_alpha_under: float = 0.02,
    module2_alpha_over: float = 1.5,
    module2_mizel_c: float = 1.4,
    module3_theta: float = 0.15,
    module3_iterations: int = 6,
    module3_recurrence_c: float = 0.5,
    module4_num_p_values: int = 20,
    module4_p_max_exp: int = -1,
    module4_p_min_exp: int = -6,
    module4_target_success: float = 0.99,
    module4_complexity_c: float = 1.5,
    module4_max_iterations: int = 50_000,
    module4_slope_tolerance: float = 0.02,
    module5_theta: float = 0.01,
    module5_iterations: int = 100,
    module5_zeno_alpha: float = 1.5,
    module5_mizel_c: float = 1.5,
    module5_classical_tolerance: float = 0.05,
    module5_penalty_threshold: float = 5.0,
    module6_iterations: int = 50,
    module6_control_theta: float = 0.1,
    module6_mizel_c: float = 1.5,
    module6_noise_floor: float = 1e-15,
    out_prefix: Optional[str] = None,
    show_plot: bool = False,
    enforce_audit: bool = True,
) -> None:
    """Run all six FOQA modules and save artifacts with configurable parameters."""
    lab = FOQALaboratory()
    stem = Path(__file__).stem if out_prefix is None else str(out_prefix)

    lab.experiment_module1_lcu_oblivious_circuit_synthesizer(
        theta=module1_theta,
        alpha_n=module1_alpha,
        show_plot=show_plot,
        save_plot_path=f"{stem}_module1_lcu_oblivious.png",
    )
    lab.experiment_module2_damping_regime_sweeper(
        theta=module2_theta,
        iterations=module2_iterations,
        alpha_under=module2_alpha_under,
        alpha_over=module2_alpha_over,
        mizel_c=module2_mizel_c,
        show_plot=show_plot,
        save_plot_path=f"{stem}_module2_damping_regimes.png",
        enforce_audit=enforce_audit,
    )
    lab.experiment_module3_nonlinear_recurrence_auditor(
        theta=module3_theta,
        iterations=module3_iterations,
        recurrence_c=module3_recurrence_c,
        show_plot=show_plot,
        save_plot_path=f"{stem}_module3_recurrence_audit.png",
        enforce_audit=enforce_audit,
    )
    lab.experiment_module4_asymptotic_complexity_auditor(
        num_p_values=module4_num_p_values,
        p_max_exponent=module4_p_max_exp,
        p_min_exponent=module4_p_min_exp,
        target_success=module4_target_success,
        complexity_c=module4_complexity_c,
        max_iterations=module4_max_iterations,
        slope_tolerance=module4_slope_tolerance,
        show_plot=show_plot,
        save_plot_path=f"{stem}_module4_asymptotic_complexity.png",
        enforce_audit=enforce_audit,
    )
    lab.experiment_module5_adversarial_zeno_catastrophe(
        theta=module5_theta,
        iterations=module5_iterations,
        zeno_alpha=module5_zeno_alpha,
        mizel_c=module5_mizel_c,
        classical_tolerance=module5_classical_tolerance,
        penalty_threshold=module5_penalty_threshold,
        show_plot=show_plot,
        save_plot_path=f"{stem}_module5_zeno_catastrophe.png",
        enforce_audit=enforce_audit,
    )
    lab.experiment_module6_empty_database_paradox(
        iterations=module6_iterations,
        control_theta=module6_control_theta,
        mizel_c=module6_mizel_c,
        noise_floor=module6_noise_floor,
        show_plot=show_plot,
        save_plot_path=f"{stem}_module6_empty_database.png",
        enforce_audit=enforce_audit,
    )
    print("Fixed-point oblivious amplitude amplification analysis complete.")


def run_all_modules_one_click(show_plot: bool = False) -> None:
    """Backward-compatible wrapper for the default six-module FOQA analysis."""
    run_full_analysis(show_plot=show_plot, out_prefix=Path(__file__).stem, enforce_audit=True)


def _interactive_rerun_prompt(defaults: Dict[str, object]) -> None:
    if not sys.stdin.isatty():
        return

    print("\n" + "=" * 72)
    print("INTERACTIVE RE-RUN MODE")
    print("=" * 72)
    print("Press Enter to finish, or enter custom key=value pairs to rerun.")
    print("Example: module2_theta=0.02, module2_iterations=90")
    print("Example: module4_num_p_values=30, module4_target_success=0.995")
    print("Example: module5_theta=0.02, module6_iterations=80")

    try:
        raw = input("Custom parameters: ").strip()
    except EOFError:
        print("\nInteractive mode closed.")
        return

    if not raw:
        print("Interactive mode finished.")
        return
    if "=" not in raw:
        print("No key=value parameters detected. Interactive mode finished.")
        return

    try:
        kwargs = _parse_kwargs_text(raw)
    except Exception as exc:
        print(f"Could not parse custom parameters: {exc}")
        print("Interactive mode finished without rerun.")
        return

    allowed = set(defaults)
    unknown = set(kwargs) - allowed
    if unknown:
        print(f"Unknown argument(s): {', '.join(sorted(unknown))}")
        print("Interactive mode finished without rerun.")
        return

    merged = dict(defaults)
    merged.update(kwargs)
    print(f"\nRe-running with parameters: {merged}")
    run_full_analysis(**merged)


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_cli().parse_args(argv)
    lab = FOQALaboratory()

    if args.module == 1:
        theta = 0.6 if args.theta is None else float(args.theta)
        result = lab.experiment_module1_lcu_oblivious_circuit_synthesizer(
            theta=theta,
            alpha_n=args.alpha,
            show_plot=not args.no_show,
            save_plot_path=args.save_plot,
        )
        if not result.math_audit_passed:
            raise SystemExit("Module 1 math audit failed.")
        return
    if args.module == 2:
        theta = 0.01 if args.theta is None else float(args.theta)
        iterations = 120 if args.iterations is None else int(args.iterations)
        result = lab.experiment_module2_damping_regime_sweeper(
            theta=theta,
            iterations=iterations,
            alpha_under=args.alpha_under,
            alpha_over=args.alpha_over,
            mizel_c=args.mizel_c,
            show_plot=not args.no_show,
            save_plot_path=args.save_plot,
            enforce_audit=not args.no_enforce_audit,
        )
        if (not args.no_enforce_audit) and not (
            result.underdamped_oscillatory
            and result.overdamped_frozen
            and result.mizel_converged
            and result.mizel_monotonic
        ):
            raise SystemExit("Module 2 damping-regime audit failed.")
        return
    if args.module == 3:
        theta = 0.15 if args.theta is None else float(args.theta)
        iterations = 6 if args.iterations is None else int(args.iterations)
        result = lab.experiment_module3_nonlinear_recurrence_auditor(
            theta=theta,
            iterations=iterations,
            recurrence_c=args.recurrence_c,
            show_plot=not args.no_show,
            save_plot_path=args.save_plot,
            enforce_audit=not args.no_enforce_audit,
        )
        if (not args.no_enforce_audit) and (not result.audit_passed):
            raise SystemExit("Module 3 recurrence audit failed.")
        return
    if args.module == 4:
        result = lab.experiment_module4_asymptotic_complexity_auditor(
            num_p_values=args.num_p_values,
            p_max_exponent=args.p_max_exp,
            p_min_exponent=args.p_min_exp,
            target_success=args.good_success,
            complexity_c=args.complexity_c,
            max_iterations=args.max_iterations,
            slope_tolerance=args.slope_tolerance,
            show_plot=not args.no_show,
            save_plot_path=args.save_plot,
            enforce_audit=not args.no_enforce_audit,
        )
        if (not args.no_enforce_audit) and (not result.audit_passed):
            raise SystemExit("Module 4 asymptotic complexity audit failed.")
        return
    if args.module == 5:
        result = lab.experiment_module5_adversarial_zeno_catastrophe(
            theta=args.zeno_theta,
            iterations=args.zeno_iterations,
            zeno_alpha=args.zeno_alpha,
            mizel_c=args.zeno_mizel_c,
            classical_tolerance=args.zeno_classical_tolerance,
            penalty_threshold=args.zeno_penalty_threshold,
            show_plot=not args.no_show,
            save_plot_path=args.save_plot,
            enforce_audit=not args.no_enforce_audit,
        )
        if (not args.no_enforce_audit) and (not result.audit_passed):
            raise SystemExit("Module 5 Zeno failure mode audit failed.")
        return
    if args.module == 6:
        result = lab.experiment_module6_empty_database_paradox(
            iterations=args.empty_iterations,
            control_theta=args.empty_control_theta,
            mizel_c=args.empty_mizel_c,
            noise_floor=args.empty_noise_floor,
            show_plot=not args.no_show,
            save_plot_path=args.save_plot,
            enforce_audit=not args.no_enforce_audit,
        )
        if (not args.no_enforce_audit) and (not result.audit_passed):
            raise SystemExit("Module 6 empty-database audit failed.")
        return
    raise SystemExit(f"Module {args.module} is not implemented yet.")


if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="foaa")
    cli_kwargs = _parse_command_line(sys.argv[1:])
    if cli_kwargs is None:
        main()
    elif cli_kwargs:
        default_kwargs: Dict[str, object] = {
            "module1_theta": 0.6,
            "module1_alpha": 1.2,
            "module2_theta": 0.01,
            "module2_iterations": 120,
            "module2_alpha_under": 0.02,
            "module2_alpha_over": 1.5,
            "module2_mizel_c": 1.4,
            "module3_theta": 0.15,
            "module3_iterations": 6,
            "module3_recurrence_c": 0.5,
            "module4_num_p_values": 20,
            "module4_p_max_exp": -1,
            "module4_p_min_exp": -6,
            "module4_target_success": 0.99,
            "module4_complexity_c": 1.5,
            "module4_max_iterations": 50_000,
            "module4_slope_tolerance": 0.02,
            "module5_theta": 0.01,
            "module5_iterations": 100,
            "module5_zeno_alpha": 1.5,
            "module5_mizel_c": 1.5,
            "module5_classical_tolerance": 0.05,
            "module5_penalty_threshold": 5.0,
            "module6_iterations": 50,
            "module6_control_theta": 0.1,
            "module6_mizel_c": 1.5,
            "module6_noise_floor": 1e-15,
            "out_prefix": Path(__file__).stem,
            "show_plot": False,
            "enforce_audit": True,
        }
        unknown = set(cli_kwargs) - set(default_kwargs)
        if unknown:
            raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
        merged = dict(default_kwargs)
        merged.update(cli_kwargs)
        run_full_analysis(**merged)
    else:
        default_kwargs = {
            "module1_theta": 0.6,
            "module1_alpha": 1.2,
            "module2_theta": 0.01,
            "module2_iterations": 120,
            "module2_alpha_under": 0.02,
            "module2_alpha_over": 1.5,
            "module2_mizel_c": 1.4,
            "module3_theta": 0.15,
            "module3_iterations": 6,
            "module3_recurrence_c": 0.5,
            "module4_num_p_values": 20,
            "module4_p_max_exp": -1,
            "module4_p_min_exp": -6,
            "module4_target_success": 0.99,
            "module4_complexity_c": 1.5,
            "module4_max_iterations": 50_000,
            "module4_slope_tolerance": 0.02,
            "module5_theta": 0.01,
            "module5_iterations": 100,
            "module5_zeno_alpha": 1.5,
            "module5_mizel_c": 1.5,
            "module5_classical_tolerance": 0.05,
            "module5_penalty_threshold": 5.0,
            "module6_iterations": 50,
            "module6_control_theta": 0.1,
            "module6_mizel_c": 1.5,
            "module6_noise_floor": 1e-15,
            "out_prefix": Path(__file__).stem,
            "show_plot": False,
            "enforce_audit": True,
        }
        run_full_analysis(**default_kwargs)
        _interactive_rerun_prompt(default_kwargs)

