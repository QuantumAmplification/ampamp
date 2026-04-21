"""Library-native QAOA/Grover transpilation study (GPU-oriented workflow).

This module mirrors the scenario intent of the non-library QAOA transpile suite,
but circuit construction and transpilation diagnostics are implemented through the
ampamp library.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ampamp.grover import GroverEngine
from ampamp.transpilation import TranspilationProfileConfig, TranspilationProfiler
from _shared_gpu_library import Logger, run_interactive_scenario_repl

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)


def _transpile_metrics(
    qc: QuantumCircuit,
    *,
    coupling_map: List[List[int]] | None = None,
    basis_gates: Sequence[str] = ("cx", "id", "rz", "sx", "x"),
    optimization_level: int = 3,
) -> Dict[str, float]:
    t_qc = transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=list(basis_gates),
        optimization_level=int(optimization_level),
    )
    ops = t_qc.count_ops()
    profiler = TranspilationProfiler(
        TranspilationProfileConfig(
            coupling_map_edges=coupling_map,
            basis_gates=tuple(str(g) for g in basis_gates),
        )
    )
    profile = profiler.profile_circuit(qc)
    return {
        "depth": float(t_qc.depth()),
        "size": float(t_qc.size()),
        "cx": float(ops.get("cx", 0)),
        "profile_score": float(profile["hardware_penalty_score"]),
        "profile_time_ns": float(profile["total_time_ns"]),
        "profile_final_cnots": float(profile["final_cnots"]),
        "profile_swaps": float(profile["routing_swaps"]),
    }


@dataclass
class QAOAResults:
    depths: List[int]
    grover_success: List[float]
    transverse_success: List[float]
    leakage: List[float]
    grover_params: List[np.ndarray]
    transverse_params: List[np.ndarray]


class QAOALibraryLab:
    def __init__(self, num_qubits: int = 5, target_idx: int = 0, n_restarts: int = 4, seed: int = 42):
        self.n = int(num_qubits)
        self.N = 2 ** self.n
        self.target_idx = int(target_idx)
        self.n_restarts = int(n_restarts)
        self.seed = int(seed)

        self.s_state = np.ones(self.N, dtype=complex) / math.sqrt(self.N)
        self.target_state = np.zeros(self.N, dtype=complex)
        self.target_state[self.target_idx] = 1.0

        self.s_ortho = self.s_state.copy()
        self.s_ortho[self.target_idx] = 0.0
        self.s_ortho /= np.linalg.norm(self.s_ortho)

    def _apply_cost_phase(self, state: np.ndarray, gamma: float) -> np.ndarray:
        out = state.copy()
        out[self.target_idx] *= np.exp(1j * float(gamma))
        return out

    def _apply_transverse(self, state: np.ndarray, beta: float) -> np.ndarray:
        c = math.cos(beta)
        s = 1j * math.sin(beta)
        single = np.array([[c, s], [s, c]], dtype=complex)
        unitary = single
        for _ in range(self.n - 1):
            unitary = np.kron(unitary, single)
        return unitary @ state

    def _apply_qaoa(self, params: np.ndarray, mixer_type: str) -> np.ndarray:
        p = len(params) // 2
        gammas = params[:p]
        betas = params[p:]
        state = self.s_state.copy()
        for step in range(p):
            state = self._apply_cost_phase(state, float(gammas[step]))
            if mixer_type == "grover":
                state = self._apply_grover_mixer_state(state, float(betas[step]))
            elif mixer_type == "transverse":
                state = self._apply_transverse(state, float(betas[step]))
            else:
                raise ValueError(f"Unknown mixer_type: {mixer_type}")
        return state

    def _apply_grover_mixer_state(self, state: np.ndarray, beta: float) -> np.ndarray:
        phase = np.exp(-1j * beta)
        overlap = np.vdot(self.s_state, state)
        return phase * state + (1.0 - phase) * overlap * self.s_state

    def _success(self, state: np.ndarray) -> float:
        return float(abs(state[self.target_idx]) ** 2)

    def _cost_fn(self, params: np.ndarray, mixer_type: str) -> float:
        return -self._success(self._apply_qaoa(params, mixer_type))

    def optimize(self, depth_p: int, mixer_type: str) -> Tuple[float, np.ndarray]:
        bounds = [(0.0, 2.0 * math.pi) for _ in range(2 * int(depth_p))]
        best_prob = -1.0
        best_params = None
        for restart in range(self.n_restarts):
            rng = np.random.default_rng(self.seed + 131 * int(depth_p) + 1009 * restart + (0 if mixer_type == "grover" else 1))
            x0 = rng.uniform(0.0, math.pi, 2 * int(depth_p))
            result = minimize(lambda x: self._cost_fn(x, mixer_type), x0, method="L-BFGS-B", bounds=bounds)
            prob = float(-result.fun)
            if prob > best_prob:
                best_prob = prob
                best_params = np.asarray(result.x, dtype=float)
        assert best_params is not None
        return best_prob, best_params

    def leakage_from_params(self, params: np.ndarray) -> float:
        state = self._apply_qaoa(params, "transverse")
        pw = abs(np.vdot(self.target_state, state)) ** 2
        ps = abs(np.vdot(self.s_ortho, state)) ** 2
        return float(np.clip(1.0 - (pw + ps), 0.0, 1.0))

    def build_qaoa_circuit(self, params: Sequence[float], mixer_type: str, measure: bool = False) -> QuantumCircuit:
        params_arr = np.asarray(params, dtype=float)
        if params_arr.size % 2 != 0:
            raise ValueError("parameter vector must have even length")
        p = params_arr.size // 2
        gammas = params_arr[:p]
        betas = params_arr[p:]

        qc = QuantumCircuit(self.n, self.n if measure else 0, name=f"qaoa_{mixer_type}_p{p}")
        qc.h(range(self.n))

        grover = GroverEngine(self.n, [self.target_idx])
        oracle = grover.get_oracle()
        diffusion = grover.get_diffusion()

        for gamma, beta in zip(gammas, betas):
            qc.append(oracle.to_instruction(label="Cost"), range(self.n))
            qc.rz(float(gamma), self.n - 1)
            if mixer_type == "grover":
                # Library-native global mixer through Grover diffusion
                qc.rz(float(beta), self.n - 1)
                qc.append(diffusion.to_instruction(label="GroverMixer"), range(self.n))
            elif mixer_type == "transverse":
                for q in range(self.n):
                    qc.rx(-2.0 * float(beta), q)
            else:
                raise ValueError(f"Unknown mixer_type: {mixer_type}")

        if measure:
            qc.measure(range(self.n), range(self.n))
        return qc

    def circuit_backend_evidence(self, params: Sequence[float], mixer_type: str, *, coupling_map: List[List[int]] | None = None) -> Dict[str, float]:
        qc = self.build_qaoa_circuit(params, mixer_type, measure=False)
        return _transpile_metrics(qc, coupling_map=coupling_map)

    def run_full(self, max_p: int = 6) -> QAOAResults:
        depths = list(range(1, int(max_p) + 1))
        grover_success: List[float] = []
        transverse_success: List[float] = []
        leakage: List[float] = []
        grover_params: List[np.ndarray] = []
        transverse_params: List[np.ndarray] = []

        for p in depths:
            gp, gparams = self.optimize(p, "grover")
            tp, tparams = self.optimize(p, "transverse")
            lk = self.leakage_from_params(tparams)
            grover_success.append(float(gp))
            transverse_success.append(float(tp))
            leakage.append(float(lk))
            grover_params.append(gparams)
            transverse_params.append(tparams)
            print(f"p={p} | grover={gp:.4f} | transverse={tp:.4f} | leakage={lk:.4f}")

        return QAOAResults(depths, grover_success, transverse_success, leakage, grover_params, transverse_params)


def _write_report(filename: str, title: str, lines: List[str]) -> str:
    out = os.path.join(_RESULT_DIR, filename)
    with open(out, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        for line in lines:
            f.write(line + "\n")
    return out


def run_scenario_a_baseline(
    num_qubits: int = 5,
    max_p: int = 6,
    target_idx: int = 0,
    n_restarts: int = 4,
    seed: int = 42,
    report_name: str = "qaoa_case_a_baseline_library.txt",
):
    print("\n" + "=" * 72)
    print("CASE A: BASELINE DEPTH DEPENDENCE")
    print("=" * 72)
    lab = QAOALibraryLab(num_qubits=num_qubits, target_idx=target_idx, n_restarts=n_restarts, seed=seed)
    results = lab.run_full(max_p=max_p)
    lines = [f"num_qubits={num_qubits}", f"target_idx={target_idx}", f"max_p={max_p}", ""]
    for i, p in enumerate(results.depths):
        ev_g = lab.circuit_backend_evidence(results.grover_params[i], "grover")
        ev_t = lab.circuit_backend_evidence(results.transverse_params[i], "transverse")
        lines.append(
            f"p={p} | grover_success={results.grover_success[i]:.6f} | transverse_success={results.transverse_success[i]:.6f} | leakage={results.leakage[i]:.6f}"
        )
        lines.append(
            f"  grover depth={int(ev_g['depth'])} cx={int(ev_g['cx'])} score={ev_g['profile_score']:.2f}"
        )
        lines.append(
            f"  transverse depth={int(ev_t['depth'])} cx={int(ev_t['cx'])} score={ev_t['profile_score']:.2f}"
        )
    path = _write_report(report_name, "Case A: Baseline Depth Dependence (Library)", lines)
    print(f"Saved report to: {path}")


def run_scenario_b_medium_scale(num_qubits: int = 6, max_p: int = 5, target_idx: int = 0, n_restarts: int = 3, seed: int = 84):
    print("\n" + "=" * 72)
    print("CASE B: MEDIUM-SCALE DEPTH DEPENDENCE")
    print("=" * 72)
    return run_scenario_a_baseline(
        num_qubits=num_qubits,
        max_p=max_p,
        target_idx=target_idx,
        n_restarts=n_restarts,
        seed=seed,
        report_name="qaoa_case_b_medium_scale_library.txt",
    )


def run_scenario_c_target_symmetry(num_qubits: int = 5, depth_p: int = 4, target_indices: Tuple[int, ...] = (0, 1, 3, 7, 16), n_restarts: int = 4, seed: int = 42):
    print("\n" + "=" * 72)
    print("CASE C: TARGET-SYMMETRY ANALYSIS")
    print("=" * 72)
    lines = [f"num_qubits={num_qubits}", f"depth_p={depth_p}", f"targets={target_indices}", ""]
    successes = []
    leaks = []
    for idx in target_indices:
        lab = QAOALibraryLab(num_qubits=num_qubits, target_idx=int(idx), n_restarts=n_restarts, seed=seed + 11 * int(idx))
        prob, params = lab.optimize(int(depth_p), "transverse")
        leak = lab.leakage_from_params(params)
        ev = lab.circuit_backend_evidence(params, "transverse")
        successes.append(prob)
        leaks.append(leak)
        print(f"target={idx:>2} | success={prob:.6f} | leakage={leak:.6f} | depth={int(ev['depth'])} | cx={int(ev['cx'])}")
        lines.append(f"target={idx:>2} | success={prob:.6f} | leakage={leak:.6f} | depth={int(ev['depth'])} | cx={int(ev['cx'])} | score={ev['profile_score']:.2f}")
    lines.append("")
    lines.append(f"success_span={(max(successes) - min(successes)):.6e}")
    lines.append(f"leakage_span={(max(leaks) - min(leaks)):.6e}")
    path = _write_report("qaoa_case_c_target_symmetry_library.txt", "Case C: Target Symmetry (Library)", lines)
    print(f"Saved report to: {path}")


def run_scenario_d_optimizer_sensitivity(num_qubits: int = 5, depth_p: int = 4, target_idx: int = 0, seeds: Tuple[int, ...] = (11, 23, 37, 41, 53), n_restarts_per_seed: int = 2):
    print("\n" + "=" * 72)
    print("CASE D: OPTIMIZATION-SENSITIVITY ANALYSIS")
    print("=" * 72)
    lines = [f"num_qubits={num_qubits}", f"depth_p={depth_p}", f"target_idx={target_idx}", ""]
    vals = []
    for s in seeds:
        lab = QAOALibraryLab(num_qubits=num_qubits, target_idx=target_idx, n_restarts=n_restarts_per_seed, seed=int(s))
        prob, params = lab.optimize(depth_p, "transverse")
        leak = lab.leakage_from_params(params)
        vals.append(prob)
        ev = lab.circuit_backend_evidence(params, "transverse")
        print(f"seed={s:>3} | success={prob:.6f} | leakage={leak:.6f} | depth={int(ev['depth'])} | cx={int(ev['cx'])}")
        lines.append(f"seed={s:>3} | success={prob:.6f} | leakage={leak:.6f} | depth={int(ev['depth'])} | cx={int(ev['cx'])} | score={ev['profile_score']:.2f}")
    lines.append("")
    lines.append(f"best_success={max(vals):.6f}")
    lines.append(f"median_success={float(np.median(vals)):.6f}")
    lines.append(f"worst_success={min(vals):.6f}")
    path = _write_report("qaoa_case_d_optimizer_sensitivity_library.txt", "Case D: Optimizer Sensitivity (Library)", lines)
    print(f"Saved report to: {path}")


def run_scenario_e_backend_agreement(num_qubits: int = 5, depth_p: int = 3, target_idx: int = 0, n_restarts: int = 4, seed: int = 42):
    print("\n" + "=" * 72)
    print("CASE E: BACKEND AGREEMENT (LIBRARY TRANSPILE PROFILES)")
    print("=" * 72)
    lab = QAOALibraryLab(num_qubits=num_qubits, target_idx=target_idx, n_restarts=n_restarts, seed=seed)
    _, gparams = lab.optimize(depth_p, "grover")
    _, tparams = lab.optimize(depth_p, "transverse")

    all_to_all = None
    linear = [[i, i + 1] for i in range(num_qubits - 1)] + [[i + 1, i] for i in range(num_qubits - 1)]

    ev_g_all = lab.circuit_backend_evidence(gparams, "grover", coupling_map=all_to_all)
    ev_g_lin = lab.circuit_backend_evidence(gparams, "grover", coupling_map=linear)
    ev_t_all = lab.circuit_backend_evidence(tparams, "transverse", coupling_map=all_to_all)
    ev_t_lin = lab.circuit_backend_evidence(tparams, "transverse", coupling_map=linear)

    print(f"Grover all-to-all depth={int(ev_g_all['depth'])}, linear depth={int(ev_g_lin['depth'])}")
    print(f"Transverse all-to-all depth={int(ev_t_all['depth'])}, linear depth={int(ev_t_lin['depth'])}")

    lines = [
        f"num_qubits={num_qubits}",
        f"depth_p={depth_p}",
        f"target_idx={target_idx}",
        "",
        f"grover all_to_all depth={int(ev_g_all['depth'])} cx={int(ev_g_all['cx'])} score={ev_g_all['profile_score']:.2f}",
        f"grover linear     depth={int(ev_g_lin['depth'])} cx={int(ev_g_lin['cx'])} score={ev_g_lin['profile_score']:.2f}",
        f"transverse all_to_all depth={int(ev_t_all['depth'])} cx={int(ev_t_all['cx'])} score={ev_t_all['profile_score']:.2f}",
        f"transverse linear     depth={int(ev_t_lin['depth'])} cx={int(ev_t_lin['cx'])} score={ev_t_lin['profile_score']:.2f}",
    ]
    path = _write_report("qaoa_case_e_backend_agreement_library.txt", "Case E: Backend Agreement (Library)", lines)
    print(f"Saved report to: {path}")


def run_scenario_f_dense_reference(num_qubits: int = 5, max_p: int = 4, target_idx: int = 0, n_restarts: int = 3, seed: int = 42):
    print("\n" + "=" * 72)
    print("CASE F: DENSE REFERENCE COMPUTATION")
    print("=" * 72)
    return run_scenario_a_baseline(
        num_qubits=num_qubits,
        max_p=max_p,
        target_idx=target_idx,
        n_restarts=n_restarts,
        seed=seed,
        report_name="qaoa_case_f_dense_reference_library.txt",
    )


if __name__ == "__main__":
    output_filepath = os.path.join(_RESULT_DIR, "terminal_output.log")
    logger = Logger(output_filepath)
    sys.stdout = logger

    scenarios = [
        ("A", lambda: run_scenario_a_baseline(5, 6, 0, 4, 42)),
        ("B", lambda: run_scenario_b_medium_scale(6, 5, 0, 3, 84)),
        ("C", lambda: run_scenario_c_target_symmetry(5, 4, (0, 1, 3, 7, 16), 4, 42)),
        ("D", lambda: run_scenario_d_optimizer_sensitivity(5, 4, 0, (11, 23, 37, 41, 53), 2)),
        ("E", lambda: run_scenario_e_backend_agreement(5, 3, 0, 4, 42)),
        ("F", lambda: run_scenario_f_dense_reference(5, 4, 0, 3, 42)),
    ]
    interactive = [
        ("A", run_scenario_a_baseline),
        ("B", run_scenario_b_medium_scale),
        ("C", run_scenario_c_target_symmetry),
        ("D", run_scenario_d_optimizer_sensitivity),
        ("E", run_scenario_e_backend_agreement),
        ("F", run_scenario_f_dense_reference),
    ]

    try:
        for _, fn in scenarios:
            fn()
        run_interactive_scenario_repl(interactive, sep="=" * 72)
    finally:
        logger.log.close()
        sys.stdout = logger.terminal
        print("QAOA library GPU suite complete.")
