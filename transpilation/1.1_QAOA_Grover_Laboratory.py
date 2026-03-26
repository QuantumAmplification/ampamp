"""QAOA-inspired Grover analysis in the ideal unitary model.

This module is the theory-side companion to the transpile-side QAOA file.
Its role is to study the search dynamics in the ideal statevector setting,
before hardware constraints, routing overhead, or proxy chip topologies are
introduced. The analysis focuses on four questions:

1. How quickly does a Grover-style global mixer recover the target state?
2. How much does a local transverse-field mixer leak out of the Grover plane?
3. Is the fast factorized transverse update numerically faithful to the
   dense-matrix exponential?
4. How robust are the optimized results across targets and optimizer seeds?

Standing notation aligned with final.tex:

- |w> is the marked basis state.
- |s> = N^{-1/2} sum_x |x> is the uniform superposition.
- H_C = I - |w><w| is the search cost Hamiltonian.
- H_B^Grover = I - |s><s| is the global mixer.
- H_B^TF = -sum_j X_j is the local transverse-field mixer.

The leakage diagnostic is

    1 - |<w|psi>|^2 - |<s_perp|psi>|^2,

where |s_perp> is the normalized component of |s> orthogonal to |w>. This
measures the probability mass that has left the two-dimensional Grover plane.
"""

from __future__ import annotations

import ast
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

try:
    from one_click_utils import start_one_click_session
except Exception:
    def start_one_click_session(script_file, *, figure_prefix=None, log_name="terminal_output.log"):
        import atexit
        import io

        script_path = Path(script_file).resolve()
        result_dir = script_path.parent / f"[RESULT]{script_path.stem}"
        result_dir.mkdir(parents=True, exist_ok=True)
        old_stdout, old_stderr, old_cwd = sys.stdout, sys.stderr, Path.cwd()
        log_handle = open(result_dir / log_name, "w", encoding="utf-8")

        class _Tee(io.TextIOBase):
            def __init__(self, *streams):
                self._streams = streams

            def write(self, data):
                for stream in self._streams:
                    stream.write(data)
                    stream.flush()
                return len(data)

            def flush(self):
                for stream in self._streams:
                    stream.flush()

        sys.stdout = _Tee(old_stdout, log_handle)
        sys.stderr = _Tee(old_stderr, log_handle)
        os.chdir(result_dir)
        old_show = plt.show
        prefix = figure_prefix or script_path.stem
        counter = {"n": 0}

        def _save_show(*args, **kwargs):
            del args, kwargs
            for fig_id in list(plt.get_fignums()):
                counter["n"] += 1
                plt.figure(fig_id).savefig(
                    result_dir / f"{prefix}_figure_{counter['n']:03d}.png",
                    dpi=220,
                    bbox_inches="tight",
                )
            plt.close("all")

        plt.show = _save_show

        def _cleanup():
            try:
                plt.show = old_show
            except Exception:
                pass
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.chdir(old_cwd)
            log_handle.close()

        atexit.register(_cleanup)
        return result_dir


@dataclass
class QAOADepthSweep:
    num_qubits: int
    target_idx: int
    transverse_backend: str
    depths: np.ndarray
    grover_success: np.ndarray
    transverse_success: np.ndarray
    transverse_leakage: np.ndarray


@dataclass
class QAOABackendAgreement:
    depth_p: int
    max_random_state_difference: float
    max_random_success_gap: float
    optimized_state_difference: float
    optimized_success_gap: float


@dataclass
class QAOATargetSymmetry:
    target_indices: np.ndarray
    optimized_success: np.ndarray
    optimized_leakage: np.ndarray
    success_span: float
    leakage_span: float


@dataclass
class QAOARobustnessAnalysis:
    depth_p: int
    seeds: np.ndarray
    transverse_success: np.ndarray
    transverse_leakage: np.ndarray
    best_success: float
    median_success: float
    worst_success: float


QAOARobustnessAudit = QAOARobustnessAnalysis


@dataclass
class QAOAShellSpread:
    depth_p: int
    shell_indices: np.ndarray
    grover_distribution: np.ndarray
    transverse_distribution: np.ndarray


class QAOAGroverTheoryLab:
    """Exact numerical analysis of QAOA-inspired unstructured search."""

    def __init__(self, num_qubits: int = 5, target_idx: int = 0, n_restarts: int = 4, seed: int = 42):
        self.n = int(num_qubits)
        self.N = 2 ** self.n
        self.target_idx = int(target_idx)
        self.n_restarts = int(n_restarts)
        self.seed = int(seed)

        if not 0 <= self.target_idx < self.N:
            raise ValueError(f"target_idx must be in [0, {self.N - 1}]")

        self.s_state = np.ones(self.N, dtype=complex) / math.sqrt(self.N)
        self.target_state = np.zeros(self.N, dtype=complex)
        self.target_state[self.target_idx] = 1.0

        self.h_c_diagonal = np.ones(self.N, dtype=float)
        self.h_c_diagonal[self.target_idx] = 0.0

        self.s_ortho = self.s_state.copy()
        self.s_ortho[self.target_idx] = 0.0
        self.s_ortho /= np.linalg.norm(self.s_ortho)

        self._dense_transverse_hamiltonian: Optional[np.ndarray] = None

    def _apply_problem_phase(self, state: np.ndarray, gamma: float) -> np.ndarray:
        return state * np.exp(-1j * gamma * self.h_c_diagonal)

    def _apply_grover_mixer(self, state: np.ndarray, beta: float) -> np.ndarray:
        phase = np.exp(-1j * beta)
        overlap = np.vdot(self.s_state, state)
        return phase * state + (1.0 - phase) * overlap * self.s_state

    def _transverse_unitary(self, beta: float) -> np.ndarray:
        c = math.cos(beta)
        s = 1j * math.sin(beta)
        single_qubit = np.array([[c, s], [s, c]], dtype=complex)
        unitary = single_qubit
        for _ in range(self.n - 1):
            unitary = np.kron(unitary, single_qubit)
        return unitary

    def _get_dense_transverse_hamiltonian(self) -> np.ndarray:
        if self._dense_transverse_hamiltonian is not None:
            return self._dense_transverse_hamiltonian

        h_b = np.zeros((self.N, self.N), dtype=complex)
        for qubit in range(self.n):
            for basis_index in range(self.N):
                flipped = basis_index ^ (1 << qubit)
                h_b[basis_index, flipped] -= 1.0
        self._dense_transverse_hamiltonian = h_b
        return h_b

    def _apply_transverse_mixer(self, state: np.ndarray, beta: float, *, backend: str = "factorized") -> np.ndarray:
        if backend == "factorized":
            return self._transverse_unitary(beta) @ state
        if backend == "dense":
            return la.expm(-1j * beta * self._get_dense_transverse_hamiltonian()) @ state
        raise ValueError(f"Unknown backend: {backend}")

    def apply_qaoa(self, params: Sequence[float], mixer_type: str, *, transverse_backend: str = "factorized") -> np.ndarray:
        params_array = np.asarray(params, dtype=float)
        if params_array.size % 2 != 0:
            raise ValueError("QAOA parameter vector must have even length")

        depth_p = params_array.size // 2
        gammas = params_array[:depth_p]
        betas = params_array[depth_p:]

        state = self.s_state.copy()
        for gamma, beta in zip(gammas, betas):
            state = self._apply_problem_phase(state, float(gamma))
            if mixer_type == "grover":
                state = self._apply_grover_mixer(state, float(beta))
            elif mixer_type == "transverse":
                state = self._apply_transverse_mixer(state, float(beta), backend=transverse_backend)
            else:
                raise ValueError(f"Unknown mixer_type: {mixer_type}")
        return state

    def success_probability(self, state: np.ndarray) -> float:
        return float(abs(state[self.target_idx]) ** 2)

    def plane_leakage(self, state: np.ndarray) -> float:
        prob_target = abs(np.vdot(self.target_state, state)) ** 2
        prob_s_ortho = abs(np.vdot(self.s_ortho, state)) ** 2
        leakage = 1.0 - (prob_target + prob_s_ortho)
        return float(np.clip(leakage.real, 0.0, 1.0))

    def shell_distribution(self, state: np.ndarray) -> np.ndarray:
        distribution = np.zeros(self.n + 1, dtype=float)
        for basis_index, amplitude in enumerate(state):
            distance = (basis_index ^ self.target_idx).bit_count()
            distribution[distance] += float(abs(amplitude) ** 2)
        return distribution

    def optimize_depth(
        self,
        depth_p: int,
        mixer_type: str,
        *,
        transverse_backend: str = "factorized",
        seed_override: Optional[int] = None,
        n_restarts_override: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
        if depth_p <= 0:
            raise ValueError("depth_p must be positive")

        def objective(params: np.ndarray) -> float:
            state = self.apply_qaoa(params, mixer_type, transverse_backend=transverse_backend)
            return -self.success_probability(state)

        best_prob = -1.0
        best_params: Optional[np.ndarray] = None
        n_restarts = self.n_restarts if n_restarts_override is None else int(n_restarts_override)
        base_seed = self.seed if seed_override is None else int(seed_override)
        bounds = [(0.0, 2.0 * math.pi) for _ in range(2 * depth_p)]

        for restart in range(n_restarts):
            rng = np.random.default_rng(base_seed + 101 * depth_p + 1009 * restart + (0 if mixer_type == "grover" else 1))
            initial_guess = rng.uniform(0.0, math.pi, size=2 * depth_p)
            result = minimize(
                objective,
                initial_guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300},
            )
            probability = float(-result.fun)
            if probability > best_prob:
                best_prob = probability
                best_params = np.asarray(result.x, dtype=float)

        if best_params is None:
            raise RuntimeError("QAOA optimization failed to produce parameters")
        return best_prob, best_params

    def sweep_depths(self, max_p: int = 6, *, transverse_backend: str = "factorized") -> QAOADepthSweep:
        depths = np.arange(1, int(max_p) + 1, dtype=int)
        grover_success = []
        transverse_success = []
        transverse_leakage = []

        for depth in depths:
            grover_prob, _ = self.optimize_depth(depth, "grover")
            transverse_prob, transverse_params = self.optimize_depth(
                depth,
                "transverse",
                transverse_backend=transverse_backend,
            )
            transverse_state = self.apply_qaoa(
                transverse_params,
                "transverse",
                transverse_backend=transverse_backend,
            )
            grover_success.append(grover_prob)
            transverse_success.append(transverse_prob)
            transverse_leakage.append(self.plane_leakage(transverse_state))

        return QAOADepthSweep(
            num_qubits=self.n,
            target_idx=self.target_idx,
            transverse_backend=transverse_backend,
            depths=depths,
            grover_success=np.asarray(grover_success, dtype=float),
            transverse_success=np.asarray(transverse_success, dtype=float),
            transverse_leakage=np.asarray(transverse_leakage, dtype=float),
        )

    def backend_agreement(self, depth_p: int = 3, *, n_samples: int = 6) -> QAOABackendAgreement:
        rng = np.random.default_rng(self.seed + 17 * depth_p)
        max_state_difference = 0.0
        max_success_gap = 0.0

        for _ in range(int(n_samples)):
            params = rng.uniform(0.0, math.pi, size=2 * depth_p)
            factorized_state = self.apply_qaoa(params, "transverse", transverse_backend="factorized")
            dense_state = self.apply_qaoa(params, "transverse", transverse_backend="dense")
            max_state_difference = max(max_state_difference, float(np.linalg.norm(factorized_state - dense_state)))
            max_success_gap = max(
                max_success_gap,
                abs(self.success_probability(factorized_state) - self.success_probability(dense_state)),
            )

        factorized_opt, factorized_params = self.optimize_depth(depth_p, "transverse", transverse_backend="factorized")
        dense_opt, dense_params = self.optimize_depth(depth_p, "transverse", transverse_backend="dense")
        optimized_state_difference = float(
            np.linalg.norm(
                self.apply_qaoa(factorized_params, "transverse", transverse_backend="factorized")
                - self.apply_qaoa(dense_params, "transverse", transverse_backend="dense")
            )
        )

        return QAOABackendAgreement(
            depth_p=depth_p,
            max_random_state_difference=max_state_difference,
            max_random_success_gap=max_success_gap,
            optimized_state_difference=optimized_state_difference,
            optimized_success_gap=float(abs(factorized_opt - dense_opt)),
        )

    def target_symmetry_analysis(
        self,
        depth_p: int = 4,
        *,
        target_indices: Optional[Iterable[int]] = None,
        transverse_backend: str = "factorized",
    ) -> QAOATargetSymmetry:
        if target_indices is None:
            candidate_indices = [0, 1, 3, 7, self.N // 2]
            target_indices = tuple(idx for idx in candidate_indices if 0 <= idx < self.N)
        indices = np.asarray(list(target_indices), dtype=int)
        success = []
        leakage = []

        for idx in indices:
            lab = QAOAGroverTheoryLab(
                num_qubits=self.n,
                target_idx=int(idx),
                n_restarts=self.n_restarts,
                seed=self.seed + 11 * int(idx),
            )
            prob, params = lab.optimize_depth(depth_p, "transverse", transverse_backend=transverse_backend)
            state = lab.apply_qaoa(params, "transverse", transverse_backend=transverse_backend)
            success.append(prob)
            leakage.append(lab.plane_leakage(state))

        success_arr = np.asarray(success, dtype=float)
        leakage_arr = np.asarray(leakage, dtype=float)
        return QAOATargetSymmetry(
            target_indices=indices,
            optimized_success=success_arr,
            optimized_leakage=leakage_arr,
            success_span=float(np.max(success_arr) - np.min(success_arr)),
            leakage_span=float(np.max(leakage_arr) - np.min(leakage_arr)),
        )

    def target_symmetry_audit(self, *args, **kwargs) -> QAOATargetSymmetry:
        return self.target_symmetry_analysis(*args, **kwargs)

    def optimizer_robustness_analysis(
        self,
        depth_p: int = 4,
        *,
        seeds: Sequence[int] = (11, 23, 37, 41, 53),
        transverse_backend: str = "factorized",
        n_restarts_per_seed: int = 2,
    ) -> QAOARobustnessAnalysis:
        success = []
        leakage = []

        for seed in seeds:
            trial_lab = QAOAGroverTheoryLab(
                num_qubits=self.n,
                target_idx=self.target_idx,
                n_restarts=n_restarts_per_seed,
                seed=int(seed),
            )
            prob, params = trial_lab.optimize_depth(depth_p, "transverse", transverse_backend=transverse_backend)
            state = trial_lab.apply_qaoa(params, "transverse", transverse_backend=transverse_backend)
            success.append(prob)
            leakage.append(trial_lab.plane_leakage(state))

        success_arr = np.asarray(success, dtype=float)
        leakage_arr = np.asarray(leakage, dtype=float)
        return QAOARobustnessAnalysis(
            depth_p=depth_p,
            seeds=np.asarray(seeds, dtype=int),
            transverse_success=success_arr,
            transverse_leakage=leakage_arr,
            best_success=float(np.max(success_arr)),
            median_success=float(np.median(success_arr)),
            worst_success=float(np.min(success_arr)),
        )

    def optimizer_robustness(self, *args, **kwargs) -> QAOARobustnessAnalysis:
        return self.optimizer_robustness_analysis(*args, **kwargs)

    def shell_spread_analysis(self, depth_p: int = 4, *, transverse_backend: str = "factorized") -> QAOAShellSpread:
        _, grover_params = self.optimize_depth(depth_p, "grover")
        _, transverse_params = self.optimize_depth(depth_p, "transverse", transverse_backend=transverse_backend)

        grover_state = self.apply_qaoa(grover_params, "grover")
        transverse_state = self.apply_qaoa(transverse_params, "transverse", transverse_backend=transverse_backend)

        return QAOAShellSpread(
            depth_p=depth_p,
            shell_indices=np.arange(0, self.n + 1, dtype=int),
            grover_distribution=self.shell_distribution(grover_state),
            transverse_distribution=self.shell_distribution(transverse_state),
        )

    def shell_spread_audit(self, *args, **kwargs) -> QAOAShellSpread:
        return self.shell_spread_analysis(*args, **kwargs)


def plot_depth_sweep(results: QAOADepthSweep) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(results.depths, results.grover_success, marker="o", linewidth=2.5, color="tab:blue", label="Global Grover mixer")
    ax1.plot(
        results.depths,
        results.transverse_success,
        marker="s",
        linewidth=2.5,
        linestyle="--",
        color="tab:red",
        label=f"Transverse mixer ({results.transverse_backend})",
    )
    ax1.set_title("Optimized Success Probability vs Depth")
    ax1.set_xlabel("QAOA depth p")
    ax1.set_ylabel("Target success probability")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    ax2.bar(results.depths, results.transverse_leakage, color="tab:purple", edgecolor="black", alpha=0.8)
    ax2.set_title("Leakage out of the Grover Plane")
    ax2.set_xlabel("QAOA depth p")
    ax2.set_ylabel(r"Mass outside $\mathrm{span}\{|w\rangle, |s\rangle\}$")
    ax2.set_ylim(0.0, max(1.05 * float(np.max(results.transverse_leakage)), 0.2))
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"QAOA-Inspired Search Analysis in the Ideal Model (n={results.num_qubits}, target={results.target_idx})",
        fontsize=15,
    )
    plt.tight_layout()
    plt.show()


def plot_shell_spread(results: QAOAShellSpread) -> None:
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(results.shell_indices - width / 2, results.grover_distribution, width=width, color="tab:blue", alpha=0.8, label="Grover mixer")
    ax.bar(
        results.shell_indices + width / 2,
        results.transverse_distribution,
        width=width,
        color="tab:red",
        alpha=0.8,
        label="Transverse mixer",
    )
    ax.set_title(f"Hamming-Shell Probability Spread at depth p={results.depth_p}")
    ax.set_xlabel("Hamming distance from the marked state")
    ax.set_ylabel("Probability mass")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _parse_cli_value(raw: str):
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return raw


def _parse_kwargs_text(raw: str) -> dict:
    kwargs = {}
    text = raw.strip()
    if not text:
        return kwargs
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected key=value pair, got '{item}'")
        key, value = item.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _parse_kwargs_tokens(tokens: Sequence[str]) -> dict:
    kwargs = {}
    for item in tokens:
        piece = item.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Expected key=value pair, got '{piece}'")
        key, value = piece.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _parse_command_line(argv: Sequence[str]) -> dict:
    if not argv:
        return {}
    if len(argv) == 1 and "," in argv[0]:
        return _parse_kwargs_text(argv[0])
    return _parse_kwargs_tokens(argv)


def _interactive_rerun_prompt() -> None:
    if not sys.stdin.isatty():
        return

    print("\n" + "=" * 72)
    print("INTERACTIVE RE-RUN MODE")
    print("=" * 72)
    print("Press Enter to finish, or enter custom key=value pairs to rerun.")
    print("Example: num_qubits=4, max_p=3, target_idx=2, n_restarts=2, seed=9")

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

    allowed = {"num_qubits", "max_p", "target_idx", "n_restarts", "seed"}
    unknown = set(kwargs) - allowed
    if unknown:
        print(f"Unknown argument(s): {', '.join(sorted(unknown))}")
        print("Interactive mode finished without rerun.")
        return

    print(f"\nRe-running with parameters: {kwargs}")
    run_full_theory_analysis(**kwargs)


def run_full_theory_analysis(
    num_qubits: int = 5,
    max_p: int = 6,
    target_idx: int = 0,
    n_restarts: int = 4,
    seed: int = 42,
) -> None:
    lab = QAOAGroverTheoryLab(
        num_qubits=num_qubits,
        target_idx=target_idx,
        n_restarts=n_restarts,
        seed=seed,
    )

    print("=" * 72)
    print("THEORY MODULE 1.1: QAOA-INSPIRED GROVER ANALYSIS")
    print("=" * 72)
    print(f"Qubits: {num_qubits}")
    print(f"Search space size: {lab.N}")
    print(f"Marked basis index: {target_idx}")
    print("Model: ideal statevector / unitary evolution")

    depth_sweep = lab.sweep_depths(max_p=max_p, transverse_backend="factorized")
    print("\nDepth sweep:")
    for depth, g_prob, t_prob, leak in zip(
        depth_sweep.depths,
        depth_sweep.grover_success,
        depth_sweep.transverse_success,
        depth_sweep.transverse_leakage,
    ):
        print(
            f"  p={int(depth)} | "
            f"Grover={g_prob:.6f} | "
            f"Transverse={t_prob:.6f} | "
            f"Leakage={leak:.6f}"
        )

    backend = lab.backend_agreement(depth_p=min(3, max_p))
    print("\nBackend agreement analysis:")
    print(f"  max ||psi_factorized - psi_dense|| on random samples = {backend.max_random_state_difference:.3e}")
    print(f"  max success gap on random samples = {backend.max_random_success_gap:.3e}")
    print(f"  optimized state difference = {backend.optimized_state_difference:.3e}")
    print(f"  optimized success gap = {backend.optimized_success_gap:.3e}")

    target_analysis = lab.target_symmetry_analysis(depth_p=min(4, max_p))
    print("\nTarget symmetry analysis:")
    for idx, success, leak in zip(target_analysis.target_indices, target_analysis.optimized_success, target_analysis.optimized_leakage):
        print(f"  target={int(idx):>2} | success={success:.6f} | leakage={leak:.6f}")
    print(f"  success span across targets = {target_analysis.success_span:.3e}")
    print(f"  leakage span across targets = {target_analysis.leakage_span:.3e}")

    robustness = lab.optimizer_robustness_analysis(depth_p=min(4, max_p))
    print("\nOptimizer robustness analysis:")
    print(f"  best / median / worst transverse success = {robustness.best_success:.6f} / {robustness.median_success:.6f} / {robustness.worst_success:.6f}")

    shell_spread = lab.shell_spread_analysis(depth_p=min(4, max_p))
    print("\nShell-spread analysis:")
    for shell, g_mass, t_mass in zip(shell_spread.shell_indices, shell_spread.grover_distribution, shell_spread.transverse_distribution):
        print(f"  distance={int(shell)} | Grover={g_mass:.6f} | Transverse={t_mass:.6f}")

    plot_depth_sweep(depth_sweep)
    plot_shell_spread(shell_spread)


if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="1.1_qaoa_theory")
    cli_kwargs = _parse_command_line(sys.argv[1:])
    allowed = {"num_qubits", "max_p", "target_idx", "n_restarts", "seed"}
    unknown = set(cli_kwargs) - allowed
    if unknown:
        raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
    if cli_kwargs:
        run_full_theory_analysis(**cli_kwargs)
    else:
        run_full_theory_analysis()
        _interactive_rerun_prompt()
