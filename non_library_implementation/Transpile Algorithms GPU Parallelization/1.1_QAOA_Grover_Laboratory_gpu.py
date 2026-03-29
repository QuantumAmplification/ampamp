"""QAOA-inspired search compilation study.

This file is the transpilation-side counterpart to the theory module. Its role
is not to model the ideal unitary dynamics from first principles, but to run
compact numerical and compilation-oriented studies in a hardware-proxy setting.
In the broader repository, the transpile folder is where circuits and numerical
surrogates are prepared for simulated chip constraints, while the theory folder
keeps the ideal-state analysis.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import os
import sys
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
import scipy.linalg as la
from qiskit import QuantumCircuit, transpile

_BOOTSTRAP_HERE = os.path.dirname(os.path.abspath(__file__))
if _BOOTSTRAP_HERE not in sys.path:
    sys.path.insert(0, _BOOTSTRAP_HERE)

from aer_publishability_gpu import (
    parse_publishability_cli,
    prepare_backend_validation_artifacts,
    render_backend_validation_summary,
    run_cli_scenario,
    save_figure_with_metadata,
    wrap_scenarios,
)
from transpile_path_utils_gpu import ensure_directory_on_syspath, resolve_project_file

_HERE = os.fspath(ensure_directory_on_syspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))
_THEORY_PATH = os.fspath(
    resolve_project_file(__file__, "1.1_QAOA_Grover_Laboratory.py", preferred_dirs=("Theory Algorithms",))
)

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
_AER_GPU_HINT = (
    "This script now requires qiskit-aer-gpu on a CUDA-capable Linux/x86_64 system "
    "(or qiskit-aer-gpu-cu11 for CUDA 11)."
)


def _gpu_statevector_data(qc: QuantumCircuit, *, seed: int | None = None) -> np.ndarray:
    try:
        from qiskit_aer import AerSimulator
        import qiskit_aer.library  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"{_AER_GPU_HINT} Original error: {type(exc).__name__}: {exc}") from exc

    probe = qc.copy()
    probe.save_statevector(label="gpu_statevector")
    backend_kwargs: dict[str, Any] = {"method": "statevector", "device": "GPU"}
    if seed is not None:
        backend_kwargs["seed_simulator"] = seed
    saved = AerSimulator(**backend_kwargs).run(probe, shots=1).result().data(0)["gpu_statevector"]
    return np.asarray(getattr(saved, "data", saved), dtype=complex)


class Logger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def _import_theory_module():
    spec = importlib.util.spec_from_file_location("qaoa_theory_module", _THEORY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load theory module from {_THEORY_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qaoa_theory_module"] = mod
    spec.loader.exec_module(mod)
    return mod


_QAOA_THEORY = _import_theory_module()
QAOAGroverTheoryLab = _QAOA_THEORY.QAOAGroverTheoryLab


@dataclass
class QAOAGroverResults:
    depths: List[int]
    grover_success: List[float]
    transverse_success: List[float]
    leakage: List[float]
    grover_params: List[np.ndarray]
    transverse_params: List[np.ndarray]
    transverse_backend: str
    num_qubits: int
    max_p: int


class QAOAGroverLab:
    """
    Numerical comparison of two QAOA-style mixers for unstructured search.

    - Global Grover mixer:     H_B = I - |s><s|
    - Local transverse mixer:  H_B = -sum_j X_j

    The study tracks:
    1. optimized target success probability, and
    2. how much probability leaks out of the Grover plane span{|w>, |s>}.
    """

    def __init__(self, num_qubits: int = 5, target_idx: int = 0, n_restarts: int = 4, seed: int = 42):
        self.n = int(num_qubits)
        self.N = 2 ** self.n
        self.target_idx = int(target_idx)
        self.n_restarts = int(n_restarts)
        self.seed = int(seed)

        self.s_state = np.ones(self.N, dtype=complex) / math.sqrt(self.N)

        # We use H_C = I - |w><w|, equivalent to -|w><w| up to a global phase.
        self.h_c_diagonal = np.ones(self.N, dtype=float)
        self.h_c_diagonal[self.target_idx] = 0.0

        self.target_state = np.zeros(self.N, dtype=complex)
        self.target_state[self.target_idx] = 1.0

        self.s_ortho = self.s_state.copy()
        self.s_ortho[self.target_idx] = 0.0
        self.s_ortho /= np.linalg.norm(self.s_ortho)

        self._dense_transverse_hamiltonian: Optional[np.ndarray] = None

    def _apply_problem_phase(self, state: np.ndarray, gamma: float) -> np.ndarray:
        return state * np.exp(-1j * gamma * self.h_c_diagonal)

    def _apply_grover_mixer(self, state: np.ndarray, beta: float) -> np.ndarray:
        # exp(-i beta (I - |s><s|)) = e^{-i beta} I + (1 - e^{-i beta}) |s><s|
        phase = np.exp(-1j * beta)
        overlap = np.vdot(self.s_state, state)
        return phase * state + (1.0 - phase) * overlap * self.s_state

    def _transverse_unitary(self, beta: float) -> np.ndarray:
        # H_B = -sum_j X_j, so exp(-i beta H_B) = exp(i beta sum_j X_j) = kron_j exp(i beta X).
        c = math.cos(beta)
        s = 1j * math.sin(beta)
        single = np.array([[c, s], [s, c]], dtype=complex)
        unitary = single
        for _ in range(self.n - 1):
            unitary = np.kron(unitary, single)
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

    def _apply_transverse_mixer(self, state: np.ndarray, beta: float, backend: str = "factorized") -> np.ndarray:
        if backend == "factorized":
            return self._transverse_unitary(beta) @ state
        if backend == "dense":
            h_b = self._get_dense_transverse_hamiltonian()
            return la.expm(-1j * beta * h_b) @ state
        raise ValueError(f"Unknown transverse backend: {backend}")

    def _apply_qaoa(self, params: np.ndarray, mixer_type: str, transverse_backend: str = "factorized") -> np.ndarray:
        p = len(params) // 2
        gammas = params[:p]
        betas = params[p:]

        state = self.s_state.copy()
        for step in range(p):
            state = self._apply_problem_phase(state, float(gammas[step]))
            if mixer_type == "grover":
                state = self._apply_grover_mixer(state, float(betas[step]))
            elif mixer_type == "transverse":
                state = self._apply_transverse_mixer(state, float(betas[step]), backend=transverse_backend)
            else:
                raise ValueError(f"Unknown mixer_type: {mixer_type}")
        return state

    def _success_probability(self, state: np.ndarray) -> float:
        return float(abs(state[self.target_idx]) ** 2)

    @staticmethod
    def _phase_invariant_state_gap(reference: np.ndarray, candidate: np.ndarray) -> float:
        overlap = np.vdot(reference, candidate)
        if abs(overlap) > 0.0:
            candidate = candidate * np.exp(-1j * np.angle(overlap))
        return float(np.linalg.norm(reference - candidate))

    def _apply_basis_state_phase(self, qc: QuantumCircuit, basis_index: int, phase: float) -> None:
        if self.n == 1:
            if basis_index == 0:
                qc.x(0)
                qc.p(phase, 0)
                qc.x(0)
            else:
                qc.p(phase, 0)
            return

        flipped: list[int] = []
        for qubit in range(self.n):
            if ((basis_index >> qubit) & 1) == 0:
                qc.x(qubit)
                flipped.append(qubit)
        qc.mcp(phase, list(range(self.n - 1)), self.n - 1)
        for qubit in reversed(flipped):
            qc.x(qubit)

    def _append_problem_phase_circuit(self, qc: QuantumCircuit, gamma: float) -> None:
        # Up to a global phase, exp(-i gamma (I - |w><w|)) equals
        # a phase e^{i gamma} on the marked basis state.
        self._apply_basis_state_phase(qc, self.target_idx, gamma)

    def _append_grover_mixer_circuit(self, qc: QuantumCircuit, beta: float) -> None:
        qc.h(range(self.n))
        self._apply_basis_state_phase(qc, 0, beta)
        qc.h(range(self.n))

    def _append_transverse_mixer_circuit(self, qc: QuantumCircuit, beta: float) -> None:
        for qubit in range(self.n):
            qc.rx(-2.0 * beta, qubit)

    def build_qaoa_circuit(
        self,
        params: Sequence[float],
        mixer_type: str,
        *,
        measure: bool = False,
    ) -> QuantumCircuit:
        params_array = np.asarray(params, dtype=float)
        if params_array.size % 2 != 0:
            raise ValueError("QAOA parameter vector must have even length")

        depth_p = params_array.size // 2
        gammas = params_array[:depth_p]
        betas = params_array[depth_p:]

        qc = QuantumCircuit(self.n, self.n if measure else 0, name=f"qaoa_{mixer_type}_p{depth_p}")
        qc.h(range(self.n))
        for gamma, beta in zip(gammas, betas):
            self._append_problem_phase_circuit(qc, float(gamma))
            if mixer_type == "grover":
                self._append_grover_mixer_circuit(qc, float(beta))
            elif mixer_type == "transverse":
                self._append_transverse_mixer_circuit(qc, float(beta))
            else:
                raise ValueError(f"Unknown mixer_type: {mixer_type}")
        if measure:
            qc.measure(range(self.n), range(self.n))
        return qc

    def exact_circuit_state(self, params: Sequence[float], mixer_type: str) -> np.ndarray:
        return _gpu_statevector_data(self.build_qaoa_circuit(params, mixer_type), seed=self.seed)

    def sample_success_probability(
        self,
        params: Sequence[float],
        mixer_type: str,
        *,
        shots: int = 2048,
    ) -> float:
        try:
            from qiskit_aer import AerSimulator
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "qiskit-aer is required for QAOA backend sampling. "
                + f"{_AER_GPU_HINT} Original error: {type(exc).__name__}: {exc}"
            ) from exc

        qc = self.build_qaoa_circuit(params, mixer_type, measure=True)
        backend = AerSimulator(seed_simulator=self.seed, device="GPU")
        tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=self.seed)
        counts = backend.run(tqc, shots=shots).result().get_counts()
        target_bitstring = format(self.target_idx, f"0{self.n}b")
        return float(counts.get(target_bitstring, 0) / shots)

    def circuit_backend_evidence(
        self,
        params: Sequence[float],
        mixer_type: str,
        *,
        transverse_backend: str = "factorized",
        basis_gates: Sequence[str] = ("cx", "id", "rz", "sx", "x"),
        optimization_level: int = 3,
        shots: int = 2048,
    ) -> Dict[str, float]:
        qc = self.build_qaoa_circuit(params, mixer_type, measure=False)
        ideal_state = self._apply_qaoa(np.asarray(params, dtype=float), mixer_type, transverse_backend=transverse_backend)
        circuit_state = self.exact_circuit_state(params, mixer_type)
        compiled = transpile(
            qc,
            basis_gates=list(basis_gates),
            optimization_level=optimization_level,
            seed_transpiler=self.seed,
        )
        sampled_success = self.sample_success_probability(params, mixer_type, shots=shots)
        return {
            "depth": float(compiled.depth()),
            "size": float(compiled.size()),
            "cx_count": float(compiled.count_ops().get("cx", 0)),
            "state_gap": self._phase_invariant_state_gap(ideal_state, circuit_state),
            "sampled_success": sampled_success,
        }

    def optimize_qaoa(
        self,
        depth_p: int,
        mixer_type: str,
        *,
        transverse_backend: str = "factorized",
    ) -> Tuple[float, np.ndarray]:
        def cost_function(params: np.ndarray) -> float:
            final_state = self._apply_qaoa(params, mixer_type, transverse_backend=transverse_backend)
            return -self._success_probability(final_state)

        best_prob = -1.0
        best_params: Optional[np.ndarray] = None
        bounds = [(0.0, 2.0 * math.pi) for _ in range(2 * depth_p)]

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(self.seed + 101 * depth_p + 1009 * restart + (0 if mixer_type == "grover" else 1))
            initial_guess = rng.uniform(0.0, math.pi, 2 * depth_p)
            result = minimize(
                cost_function,
                initial_guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300},
            )
            probability = float(-result.fun)
            if probability > best_prob:
                best_prob = probability
                best_params = result.x

        assert best_params is not None
        return best_prob, best_params

    def analyze_leakage(self, depth_p: int, optimal_params: np.ndarray, *, transverse_backend: str = "factorized") -> float:
        state = self._apply_qaoa(optimal_params, "transverse", transverse_backend=transverse_backend)
        prob_w = abs(np.vdot(self.target_state, state)) ** 2
        prob_s_ortho = abs(np.vdot(self.s_ortho, state)) ** 2
        leakage = 1.0 - (prob_w + prob_s_ortho)
        return float(np.clip(leakage.real, 0.0, 1.0))

    def run_full_experiment(self, max_p: int = 6, *, transverse_backend: str = "factorized") -> QAOAGroverResults:
        depths = list(range(1, max_p + 1))
        grover_successes: List[float] = []
        transverse_successes: List[float] = []
        leakages: List[float] = []
        grover_params: List[np.ndarray] = []
        transverse_params: List[np.ndarray] = []

        print(f"Transverse mixer backend: {transverse_backend}")

        for depth in depths:
            print(f"Optimizing depth p={depth}...")
            grover_prob, grover_opt_params = self.optimize_qaoa(depth, "grover")
            transverse_prob, transverse_opt_params = self.optimize_qaoa(
                depth,
                "transverse",
                transverse_backend=transverse_backend,
            )
            leakage = self.analyze_leakage(depth, transverse_opt_params, transverse_backend=transverse_backend)

            grover_successes.append(grover_prob)
            transverse_successes.append(transverse_prob)
            leakages.append(leakage)
            grover_params.append(np.asarray(grover_opt_params, dtype=float))
            transverse_params.append(np.asarray(transverse_opt_params, dtype=float))

            print(
                f"  p={depth} | "
                f"Grover mixer success={grover_prob:.4f} | "
                f"Transverse mixer success={transverse_prob:.4f} | "
                f"Leakage={leakage:.4f}"
            )

        return QAOAGroverResults(
            depths=depths,
            grover_success=grover_successes,
            transverse_success=transverse_successes,
            leakage=leakages,
            grover_params=grover_params,
            transverse_params=transverse_params,
            transverse_backend=transverse_backend,
            num_qubits=self.n,
            max_p=max_p,
        )

    def save_summary(self, results: QAOAGroverResults, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"qaoa_grover_summary_{results.transverse_backend}.txt"
        output_path = os.path.join(_RESULT_DIR, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("MODULE 1.1: QAOA SEARCH TRANSPILATION ANALYSIS\n")
            handle.write("=" * 64 + "\n")
            handle.write(f"Qubits: {self.n}\n")
            handle.write(f"Search space size N: {self.N}\n")
            handle.write(f"Target index: {self.target_idx}\n\n")
            handle.write(f"Transverse backend: {results.transverse_backend}\n")
            handle.write(f"Maximum depth: {results.max_p}\n\n")
            for i, depth in enumerate(results.depths):
                handle.write(
                    f"Depth p={depth}: "
                    f"Grover success={results.grover_success[i]:.6f}, "
                    f"Transverse success={results.transverse_success[i]:.6f}, "
                    f"Leakage={results.leakage[i]:.6f}\n"
                )
        return output_path

    def plot_results(
        self,
        results: QAOAGroverResults,
        filename: Optional[str] = None,
        *,
        backend_summary: Optional[Dict[str, List[float]]] = None,
    ) -> str:
        if filename is None:
            filename = f"1.1_QAOA_Grover_Leakage_Evidence_{results.transverse_backend}.png"
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax1, ax2, ax3, ax4 = axes.flat

        ax1.plot(
            results.depths,
            results.grover_success,
            marker="o",
            color="tab:blue",
            linewidth=2.5,
            markersize=8,
            label=r"Global Grover Mixer ($H_B = I - |s\rangle\langle s|$)",
        )
        ax1.plot(
            results.depths,
            results.transverse_success,
            marker="s",
            color="tab:red",
            linewidth=2.5,
            markersize=8,
            linestyle="--",
            label=r"Local Transverse Mixer ($H_B = -\sum_j X_j$)",
        )
        ax1.axhline(1.0, color="black", linestyle="--", alpha=0.5)
        ax1.set_title("QAOA Unstructured Search: Mixer Efficiency", fontsize=14)
        ax1.set_xlabel("QAOA Circuit Depth (p)", fontsize=12)
        ax1.set_ylabel("Maximized Target Probability", fontsize=12)
        ax1.set_ylim(0.0, 1.05)
        ax1.set_xticks(results.depths)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right", fontsize=10)

        ax2.bar(results.depths, results.leakage, color="tab:purple", edgecolor="black", alpha=0.8, width=0.65)
        ax2.set_title("Invariant Subspace Shattering", fontsize=14)
        ax2.set_xlabel("QAOA Circuit Depth (p)", fontsize=12)
        ax2.set_ylabel(r"Probability Mass Outside $\mathrm{span}\{|w\rangle, |s\rangle\}$", fontsize=12)
        ax2.set_ylim(0.0, 1.05)
        ax2.set_xticks(results.depths)
        ax2.grid(axis="y", alpha=0.3)

        for depth, leak in zip(results.depths, results.leakage):
            ax2.text(depth, leak + 0.02, f"{100 * leak:.1f}%", ha="center", fontsize=10, color="indigo")

        if len(results.depths) >= 3:
            x_idx = min(2, len(results.depths) - 1)
            ax2.annotate(
                "State vector escapes\nthe Grover plane",
                xy=(results.depths[x_idx], results.leakage[x_idx]),
                xytext=(results.depths[x_idx], min(0.95, results.leakage[x_idx] + 0.18)),
                arrowprops={"arrowstyle": "->", "lw": 1.5},
                ha="center",
                fontsize=10,
            )

        if backend_summary is None:
            ax3.text(0.5, 0.5, "Backend resource summary unavailable", ha="center", va="center", transform=ax3.transAxes)
            ax4.text(0.5, 0.5, "Aer agreement summary unavailable", ha="center", va="center", transform=ax4.transAxes)
        else:
            grover_depths = backend_summary["grover_depth"]
            transverse_depths = backend_summary["transverse_depth"]
            grover_cx = backend_summary["grover_cx"]
            transverse_cx = backend_summary["transverse_cx"]
            grover_aer = backend_summary["grover_aer_success"]
            transverse_aer = backend_summary["transverse_aer_success"]

            ax3.plot(results.depths, grover_depths, marker="o", color="tab:blue", linewidth=2.2, label="Grover depth")
            ax3.plot(
                results.depths,
                transverse_depths,
                marker="s",
                color="tab:red",
                linewidth=2.2,
                linestyle="--",
                label="Transverse depth",
            )
            ax3.set_title("Transpiled Resource Growth", fontsize=14)
            ax3.set_xlabel("QAOA Circuit Depth (p)", fontsize=12)
            ax3.set_ylabel("Transpiled circuit depth", fontsize=12)
            ax3.set_xticks(results.depths)
            ax3.grid(True, alpha=0.3)

            ax3b = ax3.twinx()
            ax3b.scatter(results.depths, grover_cx, color="navy", s=45, marker="D", label="Grover CX")
            ax3b.scatter(results.depths, transverse_cx, color="darkred", s=45, marker="^", label="Transverse CX")
            ax3b.set_ylabel("CNOT count", fontsize=12)

            handles_l, labels_l = ax3.get_legend_handles_labels()
            handles_r, labels_r = ax3b.get_legend_handles_labels()
            ax3.legend(handles_l + handles_r, labels_l + labels_r, loc="upper left", fontsize=9)

            ax4.plot(results.depths, results.grover_success, marker="o", color="tab:blue", linewidth=2.2, label="Grover ideal")
            ax4.plot(
                results.depths,
                grover_aer,
                marker="o",
                color="tab:blue",
                linewidth=2.0,
                linestyle=":",
                label="Grover Aer",
            )
            ax4.plot(
                results.depths,
                results.transverse_success,
                marker="s",
                color="tab:red",
                linewidth=2.2,
                linestyle="--",
                label="Transverse ideal",
            )
            ax4.plot(
                results.depths,
                transverse_aer,
                marker="s",
                color="tab:red",
                linewidth=2.0,
                linestyle="-.",
                label="Transverse Aer",
            )
            ax4.set_title("Ideal vs Aer Success", fontsize=14)
            ax4.set_xlabel("QAOA Circuit Depth (p)", fontsize=12)
            ax4.set_ylabel("Target probability", fontsize=12)
            ax4.set_ylim(0.0, 1.05)
            ax4.set_xticks(results.depths)
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc="lower right", fontsize=9)

            grover_gap = float(np.mean(np.abs(np.asarray(results.grover_success) - np.asarray(grover_aer))))
            transverse_gap = float(np.mean(np.abs(np.asarray(results.transverse_success) - np.asarray(transverse_aer))))
            ax4.text(
                0.03,
                0.04,
                f"Mean |ideal-Aer| gap\nGrover={grover_gap:.3e}, Transverse={transverse_gap:.3e}",
                transform=ax4.transAxes,
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.9},
            )

        fig.suptitle(
            f"QAOA-Inspired Search Comparison (n={self.n} qubits, backend={results.transverse_backend})",
            fontsize=16,
        )
        fig.tight_layout()

        output_path = os.path.join(_RESULT_DIR, filename)
        metadata = {
            "figure_kind": "qaoa_grover_leakage_evidence",
            "num_qubits": int(self.n),
            "target_idx": int(self.target_idx),
            "n_restarts": int(self.n_restarts),
            "seed": int(self.seed),
            "transverse_backend": str(results.transverse_backend),
            "depths": [int(x) for x in results.depths],
            "grover_success": [float(x) for x in results.grover_success],
            "transverse_success": [float(x) for x in results.transverse_success],
            "leakage": [float(x) for x in results.leakage],
            "backend_summary_present": bool(backend_summary is not None),
        }
        if backend_summary is not None:
            metadata["backend_summary"] = {
                key: [float(v) for v in values]
                for key, values in backend_summary.items()
            }
        save_figure_with_metadata(fig, output_path, metadata)
        plt.close(fig)
        return output_path


def _save_target_symmetry_plot(analysis, *, num_qubits: int, depth_p: int) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    targets = [str(int(idx)) for idx in analysis.target_indices]

    ax1.bar(targets, analysis.optimized_success, color="tab:blue", edgecolor="black", alpha=0.82)
    ax1.set_title("Optimized Success by Marked Target", fontsize=14)
    ax1.set_xlabel("Marked basis state")
    ax1.set_ylabel("Target success probability")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(targets, analysis.optimized_leakage, color="tab:purple", edgecolor="black", alpha=0.82)
    ax2.set_title("Leakage by Marked Target", fontsize=14)
    ax2.set_xlabel("Marked basis state")
    ax2.set_ylabel("Leakage outside Grover plane")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(axis="y", alpha=0.3)
    ax2.text(
        0.03,
        0.92,
        f"success span={analysis.success_span:.3e}\nleakage span={analysis.leakage_span:.3e}",
        transform=ax2.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )

    fig.suptitle(f"Case C: Target Symmetry Audit (n={num_qubits}, p={depth_p})", fontsize=15)
    fig.tight_layout()
    output_path = os.path.join(_RESULT_DIR, "qaoa_target_symmetry_analysis.png")
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "qaoa_target_symmetry_analysis",
            "num_qubits": int(num_qubits),
            "depth_p": int(depth_p),
            "target_indices": [int(x) for x in analysis.target_indices],
            "optimized_success": [float(x) for x in analysis.optimized_success],
            "optimized_leakage": [float(x) for x in analysis.optimized_leakage],
            "success_span": float(analysis.success_span),
            "leakage_span": float(analysis.leakage_span),
        },
    )
    plt.close(fig)
    return output_path


def _save_optimizer_sensitivity_plot(analysis, *, num_qubits: int, depth_p: int, target_idx: int) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    seeds = [str(int(seed)) for seed in analysis.seeds]

    ax1.bar(seeds, analysis.transverse_success, color="tab:green", edgecolor="black", alpha=0.82)
    ax1.axhline(analysis.best_success, color="black", linestyle=":", alpha=0.6)
    ax1.set_title("Success by Optimizer Seed", fontsize=14)
    ax1.set_xlabel("Optimizer seed")
    ax1.set_ylabel("Optimized target success")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(seeds, analysis.transverse_leakage, color="tab:orange", edgecolor="black", alpha=0.82)
    ax2.set_title("Leakage by Optimizer Seed", fontsize=14)
    ax2.set_xlabel("Optimizer seed")
    ax2.set_ylabel("Leakage outside Grover plane")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(axis="y", alpha=0.3)
    ax2.text(
        0.03,
        0.92,
        f"best={analysis.best_success:.4f}\nmedian={analysis.median_success:.4f}\nworst={analysis.worst_success:.4f}",
        transform=ax2.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )

    fig.suptitle(
        f"Case D: Optimization Sensitivity (n={num_qubits}, target={target_idx}, p={depth_p})",
        fontsize=15,
    )
    fig.tight_layout()
    output_path = os.path.join(_RESULT_DIR, "qaoa_optimization_sensitivity_analysis.png")
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "qaoa_optimization_sensitivity_analysis",
            "num_qubits": int(num_qubits),
            "depth_p": int(depth_p),
            "target_idx": int(target_idx),
            "seeds": [int(x) for x in analysis.seeds],
            "transverse_success": [float(x) for x in analysis.transverse_success],
            "transverse_leakage": [float(x) for x in analysis.transverse_leakage],
            "best_success": float(analysis.best_success),
            "median_success": float(analysis.median_success),
            "worst_success": float(analysis.worst_success),
        },
    )
    plt.close(fig)
    return output_path


def _save_backend_agreement_plot(
    agreement,
    factorized_evidence: Dict[str, float],
    dense_evidence: Dict[str, float],
    *,
    num_qubits: int,
    target_idx: int,
) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    gap_labels = [
        "random state",
        "random success",
        "opt state",
        "opt success",
    ]
    gap_values = [
        agreement.max_random_state_difference,
        agreement.max_random_success_gap,
        agreement.optimized_state_difference,
        agreement.optimized_success_gap,
    ]
    ax1.bar(gap_labels, gap_values, color=["tab:blue", "tab:cyan", "tab:red", "tab:pink"], edgecolor="black", alpha=0.82)
    ax1.set_title("Dense vs Factorized Backend Gaps", fontsize=14)
    ax1.set_ylabel("Difference magnitude")
    ax1.tick_params(axis="x", rotation=15)
    ax1.grid(axis="y", alpha=0.3)
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    compare_labels = ["depth", "cx", "Aer success"]
    factorized_vals = [
        factorized_evidence["depth"],
        factorized_evidence["cx_count"],
        factorized_evidence["sampled_success"],
    ]
    dense_vals = [
        dense_evidence["depth"],
        dense_evidence["cx_count"],
        dense_evidence["sampled_success"],
    ]
    x = np.arange(len(compare_labels), dtype=float)
    width = 0.34
    ax2.bar(x - width / 2, factorized_vals, width=width, color="tab:green", edgecolor="black", alpha=0.82, label="factorized")
    ax2.bar(x + width / 2, dense_vals, width=width, color="tab:purple", edgecolor="black", alpha=0.82, label="dense")
    ax2.set_title("Circuit Evidence for Optimized Solutions", fontsize=14)
    ax2.set_xticks(x, compare_labels)
    ax2.set_ylabel("Value")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    fig.suptitle(
        f"Case E: Backend Agreement (n={num_qubits}, target={target_idx}, p={agreement.depth_p})",
        fontsize=15,
    )
    fig.tight_layout()
    output_path = os.path.join(_RESULT_DIR, "qaoa_backend_agreement_analysis.png")
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "qaoa_backend_agreement_analysis",
            "num_qubits": int(num_qubits),
            "target_idx": int(target_idx),
            "depth_p": int(agreement.depth_p),
            "factorized_evidence": {key: float(value) for key, value in factorized_evidence.items()},
            "dense_evidence": {key: float(value) for key, value in dense_evidence.items()},
            "max_random_state_difference": float(agreement.max_random_state_difference),
            "max_random_success_gap": float(agreement.max_random_success_gap),
            "optimized_state_difference": float(agreement.optimized_state_difference),
            "optimized_success_gap": float(agreement.optimized_success_gap),
        },
    )
    plt.close(fig)
    return output_path


def _parse_cli_value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return raw


def _parse_kwargs_text(raw: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
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


def _parse_kwargs_tokens(tokens: List[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for item in tokens:
        piece = item.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Expected key=value pair, got '{piece}'")
        key, value = piece.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _format_signature_help(fn) -> str:
    sig = inspect.signature(fn)
    parts = []
    for name, param in sig.parameters.items():
        if param.default is inspect._empty:
            parts.append(name)
        else:
            parts.append(f"{name}={param.default!r}")
    return ", ".join(parts) if parts else "(no parameters)"


def _write_report(filename: str, title: str, lines: List[str]) -> str:
    output_path = os.path.join(_RESULT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(title + "\n")
        handle.write("=" * len(title) + "\n\n")
        for line in lines:
            handle.write(line + "\n")
    return output_path


def _format_backend_evidence_line(prefix: str, evidence: Dict[str, float]) -> str:
    return (
        f"{prefix} | depth={int(evidence['depth']):>3} | size={int(evidence['size']):>3} | "
        f"cx={int(evidence['cx_count']):>3} | state_gap={evidence['state_gap']:.3e} | "
        f"aer_success={evidence['sampled_success']:.6f}"
    )


def _parse_command_line(argv: List[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    if not argv:
        return None, {}

    case_label: Optional[str] = None
    tokens = list(argv)
    if tokens and "=" not in tokens[0]:
        case_label = tokens.pop(0).strip().upper()

    if len(tokens) == 1 and "," in tokens[0]:
        config = _parse_kwargs_text(tokens[0])
    else:
        config = _parse_kwargs_tokens(tokens)
    if "case" in config:
        case_label = str(config.pop("case")).strip().upper()
    return case_label, config


def _run_lab(
    *,
    label: str = "baseline",
    num_qubits: int = 5,
    max_p: int = 6,
    target_idx: int = 0,
    n_restarts: int = 4,
    seed: int = 42,
    transverse_backend: str = "factorized",
) -> QAOAGroverResults:
    lab = QAOAGroverLab(
        num_qubits=num_qubits,
        target_idx=target_idx,
        n_restarts=n_restarts,
        seed=seed,
    )
    results = lab.run_full_experiment(max_p=max_p, transverse_backend=transverse_backend)
    backend_summary: Dict[str, List[float]] = {
        "grover_depth": [],
        "grover_cx": [],
        "grover_aer_success": [],
        "transverse_depth": [],
        "transverse_cx": [],
        "transverse_aer_success": [],
    }

    print("\n--- SUMMARY RESULTS ---")
    for i, depth in enumerate(results.depths):
        print(
            f"Depth p={depth}: "
            f"Grover Succ={results.grover_success[i]:.4f} | "
            f"Transverse Succ={results.transverse_success[i]:.4f} | "
            f"Leakage={results.leakage[i]:.4f}"
        )

    backend_lines = ["Circuit backend evidence (exact circuit build -> transpile -> Aer sampling):", ""]
    print("\n--- CIRCUIT BACKEND EVIDENCE ---")
    for depth, grover_params, transverse_params in zip(
        results.depths,
        results.grover_params,
        results.transverse_params,
    ):
        grover_evidence = lab.circuit_backend_evidence(
            grover_params,
            "grover",
            transverse_backend=transverse_backend,
            shots=1024,
        )
        transverse_evidence = lab.circuit_backend_evidence(
            transverse_params,
            "transverse",
            transverse_backend=transverse_backend,
            shots=1024,
        )
        grover_line = _format_backend_evidence_line(f"p={depth} Grover    ", grover_evidence)
        transverse_line = _format_backend_evidence_line(f"p={depth} Transverse", transverse_evidence)
        print("  " + grover_line)
        print("  " + transverse_line)
        backend_lines.extend([grover_line, transverse_line, ""])
        backend_summary["grover_depth"].append(float(grover_evidence["depth"]))
        backend_summary["grover_cx"].append(float(grover_evidence["cx_count"]))
        backend_summary["grover_aer_success"].append(float(grover_evidence["sampled_success"]))
        backend_summary["transverse_depth"].append(float(transverse_evidence["depth"]))
        backend_summary["transverse_cx"].append(float(transverse_evidence["cx_count"]))
        backend_summary["transverse_aer_success"].append(float(transverse_evidence["sampled_success"]))

    safe_label = label.lower().replace(" ", "_")
    summary_path = lab.save_summary(results, filename=f"qaoa_grover_summary_{safe_label}_{transverse_backend}.txt")
    plot_path = lab.plot_results(
        results,
        filename=f"1.1_QAOA_Grover_Leakage_Evidence_{safe_label}_{transverse_backend}.png",
        backend_summary=backend_summary,
    )
    backend_path = _write_report(
        f"qaoa_circuit_backend_{safe_label}_{transverse_backend}.txt",
        "QAOA Circuit Backend Evidence",
        backend_lines,
    )

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved plot to:    {plot_path}")
    print(f"Saved backend report to: {backend_path}")
    return results


def scenario_a_baseline(
    num_qubits: int = 5,
    max_p: int = 6,
    target_idx: int = 0,
    n_restarts: int = 4,
    seed: int = 42,
):
    print("\n" + "=" * 72)
    print("CASE A: BASELINE DEPTH DEPENDENCE")
    print("=" * 72)
    return _run_lab(
        label="A_baseline",
        num_qubits=num_qubits,
        max_p=max_p,
        target_idx=target_idx,
        n_restarts=n_restarts,
        seed=seed,
        transverse_backend="factorized",
    )


def scenario_b_medium_scale(
    num_qubits: int = 6,
    max_p: int = 5,
    target_idx: int = 0,
    n_restarts: int = 3,
    seed: int = 84,
):
    print("\n" + "=" * 72)
    print("CASE B: MEDIUM-SCALE DEPTH DEPENDENCE")
    print("=" * 72)
    return _run_lab(
        label="B_medium_scale",
        num_qubits=num_qubits,
        max_p=max_p,
        target_idx=target_idx,
        n_restarts=n_restarts,
        seed=seed,
        transverse_backend="factorized",
    )


def scenario_c_target_symmetry(
    num_qubits: int = 5,
    depth_p: int = 4,
    target_indices: Tuple[int, ...] = (0, 1, 3, 7, 16),
    n_restarts: int = 4,
    seed: int = 42,
):
    print("\n" + "=" * 72)
    print("CASE C: TARGET-SYMMETRY ANALYSIS")
    print("=" * 72)

    lab = QAOAGroverTheoryLab(num_qubits=num_qubits, n_restarts=n_restarts, seed=seed)
    analysis = lab.target_symmetry_analysis(depth_p=depth_p, target_indices=target_indices)

    print(f"Depth p={depth_p}, targets={tuple(int(x) for x in analysis.target_indices)}")
    lines = [f"num_qubits={num_qubits}", f"depth_p={depth_p}", ""]
    backend_lines = [
        f"num_qubits={num_qubits}",
        f"depth_p={depth_p}",
        "",
        "Circuit backend evidence for the same optimized transverse-QAOA object:",
        "",
    ]
    for idx, success, leak in zip(analysis.target_indices, analysis.optimized_success, analysis.optimized_leakage):
        line = f"target={int(idx):>2} | success={success:.6f} | leakage={leak:.6f}"
        print("  " + line)
        lines.append(line)
        local_lab = QAOAGroverLab(
            num_qubits=num_qubits,
            target_idx=int(idx),
            n_restarts=n_restarts,
            seed=seed + 11 * int(idx),
        )
        _, params = local_lab.optimize_qaoa(depth_p, "transverse", transverse_backend="factorized")
        evidence = local_lab.circuit_backend_evidence(
            params,
            "transverse",
            transverse_backend="factorized",
            shots=1024,
        )
        backend_line = _format_backend_evidence_line(f"target={int(idx):>2}", evidence)
        print("    " + backend_line)
        backend_lines.append(backend_line)
    print(f"Success span: {analysis.success_span:.3e}")
    print(f"Leakage span: {analysis.leakage_span:.3e}")
    lines.extend(["", f"success_span={analysis.success_span:.6e}", f"leakage_span={analysis.leakage_span:.6e}"])

    output_path = _write_report("qaoa_target_symmetry_analysis.txt", "Case C: Target Symmetry Analysis", lines)
    backend_path = _write_report(
        "qaoa_target_symmetry_backend.txt",
        "Case C: Target Symmetry Backend Evidence",
        backend_lines,
    )
    plot_path = _save_target_symmetry_plot(analysis, num_qubits=num_qubits, depth_p=depth_p)
    print(f"Saved report to: {output_path}")
    print(f"Saved backend report to: {backend_path}")
    print(f"Saved plot to: {plot_path}")
    return analysis


def scenario_d_optimizer_luck(
    num_qubits: int = 5,
    depth_p: int = 4,
    target_idx: int = 0,
    seeds: Tuple[int, ...] = (11, 23, 37, 41, 53),
    n_restarts_per_seed: int = 2,
):
    print("\n" + "=" * 72)
    print("CASE D: OPTIMIZATION-SENSITIVITY ANALYSIS")
    print("=" * 72)

    lab = QAOAGroverTheoryLab(num_qubits=num_qubits, target_idx=target_idx)
    analysis = lab.optimizer_robustness_analysis(
        depth_p=depth_p,
        seeds=seeds,
        n_restarts_per_seed=n_restarts_per_seed,
    )

    lines = [
        f"num_qubits={num_qubits}",
        f"depth_p={depth_p}",
        f"target_idx={target_idx}",
        f"n_restarts_per_seed={n_restarts_per_seed}",
        "",
    ]
    backend_lines = [
        f"num_qubits={num_qubits}",
        f"depth_p={depth_p}",
        f"target_idx={target_idx}",
        "",
        "Circuit backend evidence for each optimizer seed:",
        "",
    ]
    for seed_value, success, leak in zip(analysis.seeds, analysis.transverse_success, analysis.transverse_leakage):
        line = f"seed={int(seed_value):>3} | success={success:.6f} | leakage={leak:.6f}"
        print("  " + line)
        lines.append(line)
        local_lab = QAOAGroverLab(
            num_qubits=num_qubits,
            target_idx=target_idx,
            n_restarts=n_restarts_per_seed,
            seed=int(seed_value),
        )
        _, params = local_lab.optimize_qaoa(depth_p, "transverse", transverse_backend="factorized")
        evidence = local_lab.circuit_backend_evidence(
            params,
            "transverse",
            transverse_backend="factorized",
            shots=1024,
        )
        backend_line = _format_backend_evidence_line(f"seed={int(seed_value):>3}", evidence)
        print("    " + backend_line)
        backend_lines.append(backend_line)
    print(
        "Best / Median / Worst success = "
        f"{analysis.best_success:.6f} / {analysis.median_success:.6f} / {analysis.worst_success:.6f}"
    )
    lines.extend(
        [
            "",
            f"best_success={analysis.best_success:.6f}",
            f"median_success={analysis.median_success:.6f}",
            f"worst_success={analysis.worst_success:.6f}",
        ]
    )

    output_path = _write_report("qaoa_optimization_sensitivity_analysis.txt", "Case D: Optimization-Sensitivity Analysis", lines)
    backend_path = _write_report(
        "qaoa_optimization_sensitivity_backend.txt",
        "Case D: Optimization-Sensitivity Backend Evidence",
        backend_lines,
    )
    plot_path = _save_optimizer_sensitivity_plot(
        analysis,
        num_qubits=num_qubits,
        depth_p=depth_p,
        target_idx=target_idx,
    )
    print(f"Saved report to: {output_path}")
    print(f"Saved backend report to: {backend_path}")
    print(f"Saved plot to: {plot_path}")
    return analysis


def scenario_e_backend_agreement(
    num_qubits: int = 5,
    depth_p: int = 3,
    target_idx: int = 0,
    n_restarts: int = 4,
    seed: int = 42,
):
    print("\n" + "=" * 72)
    print("CASE E: FACTORIZED VS DENSE BACKEND COMPARISON")
    print("=" * 72)

    lab = QAOAGroverTheoryLab(num_qubits=num_qubits, target_idx=target_idx, n_restarts=n_restarts, seed=seed)
    agreement = lab.backend_agreement(depth_p=depth_p)

    print(f"depth_p={agreement.depth_p}")
    print(f"max random state difference   = {agreement.max_random_state_difference:.3e}")
    print(f"max random success gap       = {agreement.max_random_success_gap:.3e}")
    print(f"optimized state difference   = {agreement.optimized_state_difference:.3e}")
    print(f"optimized success gap        = {agreement.optimized_success_gap:.3e}")

    local_lab = QAOAGroverLab(num_qubits=num_qubits, target_idx=target_idx, n_restarts=n_restarts, seed=seed)
    _, factorized_params = local_lab.optimize_qaoa(depth_p, "transverse", transverse_backend="factorized")
    _, dense_params = local_lab.optimize_qaoa(depth_p, "transverse", transverse_backend="dense")
    factorized_evidence = local_lab.circuit_backend_evidence(
        factorized_params,
        "transverse",
        transverse_backend="factorized",
        shots=1024,
    )
    dense_evidence = local_lab.circuit_backend_evidence(
        dense_params,
        "transverse",
        transverse_backend="dense",
        shots=1024,
    )
    factorized_line = _format_backend_evidence_line("factorized-opt", factorized_evidence)
    dense_line = _format_backend_evidence_line("dense-opt     ", dense_evidence)
    print("\nCircuit backend evidence:")
    print("  " + factorized_line)
    print("  " + dense_line)

    output_path = _write_report(
        "qaoa_backend_agreement_analysis.txt",
        "Case E: Backend Agreement Analysis",
        [
            f"num_qubits={num_qubits}",
            f"target_idx={target_idx}",
            f"depth_p={agreement.depth_p}",
            "",
            f"max_random_state_difference={agreement.max_random_state_difference:.6e}",
            f"max_random_success_gap={agreement.max_random_success_gap:.6e}",
            f"optimized_state_difference={agreement.optimized_state_difference:.6e}",
            f"optimized_success_gap={agreement.optimized_success_gap:.6e}",
        ],
    )
    backend_path = _write_report(
        "qaoa_backend_agreement_circuit_evidence.txt",
        "Case E: Backend Agreement Circuit Evidence",
        [
            f"num_qubits={num_qubits}",
            f"target_idx={target_idx}",
            f"depth_p={agreement.depth_p}",
            "",
            factorized_line,
            dense_line,
        ],
    )
    plot_path = _save_backend_agreement_plot(
        agreement,
        factorized_evidence,
        dense_evidence,
        num_qubits=num_qubits,
        target_idx=target_idx,
    )
    print(f"Saved report to: {output_path}")
    print(f"Saved backend report to: {backend_path}")
    print(f"Saved plot to: {plot_path}")
    return agreement


def scenario_f_dense_reference(
    num_qubits: int = 5,
    max_p: int = 4,
    target_idx: int = 0,
    n_restarts: int = 3,
    seed: int = 42,
):
    print("\n" + "=" * 72)
    print("CASE F: DENSE REFERENCE COMPUTATION")
    print("=" * 72)
    return _run_lab(
        label="F_dense_reference",
        num_qubits=num_qubits,
        max_p=max_p,
        target_idx=target_idx,
        n_restarts=n_restarts,
        seed=seed,
        transverse_backend="dense",
    )


def run_interactive_scenario_repl(scenarios) -> None:
    if not sys.stdin.isatty():
        return

    scenario_pairs = list(scenarios)
    scenario_map = {label.upper(): fn for label, fn in scenario_pairs}
    print("\n" + "=" * 72)
    print("INTERACTIVE RE-RUN MODE")
    print("=" * 72)
    print("Select a case for rerun with custom parameters.")
    print(f"Available labels: {', '.join(label for label, _ in scenario_pairs)}")
    print("Press Enter to exit.")

    while True:
        try:
            choice = input("\nCase label to rerun: ").strip().upper()
        except EOFError:
            print("\nInteractive mode closed.")
            return

        if not choice:
            print("Interactive rerun mode finished.")
            return
        if choice not in scenario_map:
            print(f"Unknown case '{choice}'. Available: {', '.join(scenario_map)}")
            continue

        fn = scenario_map[choice]
        print(f"Selected case {choice}: {fn.__name__}")
        print(f"Parameters: {_format_signature_help(fn)}")
        print("Enter overrides as comma-separated key=value pairs.")
        print("Example: num_qubits=6, max_p=4, n_restarts=2, seed=7")
        print("For tuple-like inputs, use Python syntax.")
        print("Example: target_indices=(0,1,3,7)")

        try:
            raw_kwargs = input("Custom parameters: ")
            if raw_kwargs.strip() and "=" not in raw_kwargs:
                print("No key=value parameters detected. Interactive mode finished.")
                return
            kwargs = _parse_kwargs_text(raw_kwargs)
            print(f"\nExecuting case {choice} with parameters: {kwargs if kwargs else 'defaults'}")
            fn(**kwargs)
        except Exception as exc:
            print(f"\nCase {choice} failed during custom execution.")
            print(f"Error: {exc}")
            print("Interactive rerun mode finished without execution.")
            return


def main() -> None:
    log_path = os.path.join(_RESULT_DIR, "terminal_output.log")
    logger = Logger(log_path)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    publishability = None
    sys.stdout = logger
    sys.stderr = logger
    try:
        print("-" * 65)
        print("MODULE 1.1: QAOA SEARCH TRANSPILATION ANALYSIS")
        print("-" * 65)
        print("Transpile-side role: compact comparative studies in a hardware-proxy setting.")
        print("Theory-side role: ideal unitary/statevector analysis without hardware constraints.")
        print("The default run covers baseline and medium-scale depth dependence,")
        print("target symmetry, optimization sensitivity, backend agreement,")
        print("and leaves the dense reference computation as an opt-in case.")
        print(f"Run log: {log_path}")
        cli_argv, publishability = parse_publishability_cli(
            sys.argv[1:],
            default_max_qubits=20,
            default_shots=1024,
            default_log_dir=_RESULT_DIR,
        )
        prepare_backend_validation_artifacts(publishability)
        print(publishability.summary())

        raw_scenarios = [
            ("A", scenario_a_baseline),
            ("B", scenario_b_medium_scale),
            ("C", scenario_c_target_symmetry),
            ("D", scenario_d_optimizer_luck),
            ("E", scenario_e_backend_agreement),
            ("F", scenario_f_dense_reference),
        ]
        scenarios = wrap_scenarios(raw_scenarios, module_globals=globals(), config=publishability)

        if run_cli_scenario(cli_argv, scenarios, label_name="case"):
            return

        for label, fn in scenarios:
            if label == "F":
                continue
            fn()

        if sys.stdin.isatty():
            print("\nDefault analysis suite complete.")
            print("Case F is more computationally demanding and is therefore executed on demand.")
            print("Any case may now be rerun with custom parameters.")
        run_interactive_scenario_repl(scenarios)
    finally:
        if publishability is not None:
            render_backend_validation_summary(publishability)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.close()


if __name__ == "__main__":
    main()
