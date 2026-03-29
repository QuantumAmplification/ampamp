"""Controlled Quantum Amplification: ideal-model analysis companion.

This module studies the 2017 Controlled Quantum Amplification (CQAA) framework
of Dohotaru and Høyer in an ideal unitary setting. The focus is on the paper's
core claims:

1. the controlled circuit U admits a principal (+1)-eigenvector with balanced
   support on the starting configuration and the marked configuration,
2. the quantum hitting time of U is of the same order as the hitting time of
   the abstract search operator A = W G,
3. the choice of control angle \tilde{\theta} matters structurally,
4. repeated applications of U can turn detection-style spectral structure into
   finding-style amplitude on the enlarged Hilbert space,
5. the multiple-target continuation can be studied through the marked
   subspace projector.

The implementation keeps the analysis deliberately small-dimensional so that it
remains reproducible and fast as a paper-companion script.
"""

from __future__ import annotations

import ast
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

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
            plt.show = old_show
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


def _parse_kwargs_text(raw: str) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for chunk in _split_top_level_commas(raw.strip()):
        if "=" not in chunk:
            raise ValueError(f"Expected key=value pair, got '{chunk}'")
        key, value = chunk.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _parse_kwargs_tokens(tokens: Sequence[str]) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for token in tokens:
        piece = token.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Expected key=value pair, got '{piece}'")
        key, value = piece.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _parse_command_line(argv: Sequence[str]) -> Optional[dict[str, object]]:
    # Preserve the existing argparse interface whenever standard option flags
    # are used; reserve key=value parsing for the lightweight custom rerun mode.
    if not argv:
        return {}
    if any(token.startswith("-") for token in argv):
        return None
    if len(argv) == 1:
        return _parse_kwargs_text(argv[0])
    return _parse_kwargs_tokens(argv)


def _parse_float_tuple(value: object) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(x.strip()) for x in value.split(",") if x.strip())
    return tuple(float(x) for x in value)


@dataclass
class CQAAInstance:
    """Concrete small-dimensional CQAA instance."""

    search_dim: int
    epsilon: float
    theta: float
    theta_tilde: float
    raw_initial: np.ndarray
    initial_perp: np.ndarray
    target_state: np.ndarray
    target_projector: np.ndarray
    g_pi_state: np.ndarray
    walk_operator: np.ndarray
    detection_operator: np.ndarray
    controlled_operator: np.ndarray
    target_subspace_dimension: int


@dataclass
class PrincipalEigenvectorAuditResults:
    """Audit of the principal (+1)-eigenvector balancing property."""

    epsilons: np.ndarray
    fidelities: np.ndarray
    overlap_initial: np.ndarray
    overlap_target: np.ndarray
    balance_gap: np.ndarray
    qht_u: np.ndarray


@dataclass
class QHTComparisonResults:
    """Theorem-style sweep comparing QHT(U) and QHT(A)."""

    epsilons: np.ndarray
    qht_u: np.ndarray
    qht_a: np.ndarray
    qht_w_on_target: np.ndarray
    ratio_u_over_a: np.ndarray
    ratio_u_over_w_scaled: np.ndarray


@dataclass
class AngleSweepResults:
    """Sensitivity of CQAA to the control angle."""

    angle_factors: np.ndarray
    thetas_tilde: np.ndarray
    principal_fidelity: np.ndarray
    balance_gap: np.ndarray
    qht_u: np.ndarray


@dataclass
class AmplificationDynamicsResults:
    """Repeated-application dynamics for U and A."""

    iterations: np.ndarray
    cqaa_success: np.ndarray
    detection_direct_success: np.ndarray
    cqaa_optimal_iteration: int
    detection_optimal_iteration: int
    cqaa_peak_success: float
    detection_peak_success: float


@dataclass
class MultipleTargetResults:
    """Marked-subspace continuation for multiple targets."""

    iterations: np.ndarray
    marked_subspace_dimension: int
    epsilon_pi: float
    cqaa_marked_success: np.ndarray
    peak_iteration: int
    peak_success: float


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError("Cannot normalize the zero vector.")
    return np.asarray(vec, dtype=complex) / norm


def _rotation_block(angle: float) -> np.ndarray:
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=complex,
    )


def _build_real_walk(search_dim: int, rotation_angles: tuple[float, ...]) -> np.ndarray:
    if search_dim < 3:
        raise ValueError("search_dim must be at least 3.")
    blocks: list[np.ndarray] = [np.array([[1.0]], dtype=complex)]
    remaining = search_dim - 1
    angle_idx = 0
    while remaining >= 2:
        angle = rotation_angles[min(angle_idx, len(rotation_angles) - 1)]
        blocks.append(_rotation_block(float(angle)))
        remaining -= 2
        angle_idx += 1
    if remaining == 1:
        blocks.append(np.array([[-1.0]], dtype=complex))
    matrix = np.block(
        [
            [
                blocks[i] if i == j else np.zeros((blocks[i].shape[0], blocks[j].shape[0]), dtype=complex)
                for j in range(len(blocks))
            ]
            for i in range(len(blocks))
        ]
    )
    return matrix


def _make_target_direction(search_dim: int, weights: tuple[float, ...]) -> np.ndarray:
    direction = np.zeros(search_dim, dtype=complex)
    usable = min(search_dim - 1, len(weights))
    for idx in range(usable):
        direction[idx + 1] = float(weights[idx])
    if np.linalg.norm(direction) <= 0.0:
        direction[1] = 1.0
    return _normalize(direction)


def _tilde_basis(theta_tilde: float) -> tuple[np.ndarray, np.ndarray]:
    ket0_tilde = np.array([math.cos(theta_tilde), math.sin(theta_tilde)], dtype=complex)
    ket1_tilde = np.array([-math.sin(theta_tilde), math.cos(theta_tilde)], dtype=complex)
    return ket0_tilde, ket1_tilde


def _projector(vec: np.ndarray) -> np.ndarray:
    vec_n = _normalize(vec)
    return np.outer(vec_n, np.conjugate(vec_n))


def _qht_alpha(unitary: np.ndarray, state: np.ndarray, tol: float = 1e-8) -> float:
    vals, vecs = np.linalg.eig(unitary)
    total = 0.0
    for idx, eigenvalue in enumerate(vals):
        angle = float(np.angle(eigenvalue))
        if angle <= tol or abs(abs(angle) - math.pi) <= tol:
            continue
        coeff = np.vdot(vecs[:, idx], state)
        total += 2.0 * (abs(coeff) ** 2) / (angle ** 2)
    return float(np.sqrt(max(total, 0.0)))


def _principal_plus_one_eigenvector(unitary: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    vals, vecs = np.linalg.eig(unitary)
    close = [idx for idx, value in enumerate(vals) if abs(value - 1.0) <= 1e-7]
    if not close:
        idx = int(np.argmin(np.abs(vals - 1.0)))
        return _normalize(vecs[:, idx])
    if reference is None:
        idx = close[0]
        return _normalize(vecs[:, idx])
    idx = max(close, key=lambda j: abs(np.vdot(reference, vecs[:, j])))
    return _normalize(vecs[:, idx])


def build_cqaa_instance(
    search_dim: int = 5,
    epsilon: float = 0.10,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
) -> CQAAInstance:
    """Construct a small CQAA instance with one marked state."""
    if not (0.0 < epsilon < 0.5):
        raise ValueError("epsilon must satisfy 0 < epsilon < 1/2 for the CQAA angle choice.")

    raw_initial = np.zeros(search_dim, dtype=complex)
    raw_initial[0] = 1.0
    theta = math.asin(math.sqrt(epsilon))
    theta_tilde = math.asin(math.sin(theta) / math.cos(theta))
    walk_operator = _build_real_walk(search_dim, rotation_angles)
    target_direction = _make_target_direction(search_dim, target_weights)
    target_state = math.sin(theta) * raw_initial + math.cos(theta) * target_direction
    target_state = _normalize(target_state)
    initial_perp = _normalize(raw_initial - math.sin(theta) * target_state)

    target_projector = _projector(target_state)
    reflection_g = np.eye(search_dim, dtype=complex) - 2.0 * target_projector
    detection_operator = walk_operator @ reflection_g

    ket0_tilde, ket1_tilde = _tilde_basis(theta_tilde)
    ctrl_zero = np.array([1.0, 0.0], dtype=complex)
    ctrl_one = np.array([0.0, 1.0], dtype=complex)
    controlled_reflection = (
        np.kron(_projector(ket0_tilde), reflection_g)
        + np.kron(_projector(ket1_tilde), np.eye(search_dim, dtype=complex))
    )
    controlled_walk = (
        np.kron(_projector(ctrl_zero), walk_operator)
        + np.kron(_projector(ctrl_one), np.eye(search_dim, dtype=complex))
    )
    controlled_operator = controlled_walk @ controlled_reflection

    return CQAAInstance(
        search_dim=search_dim,
        epsilon=float(epsilon),
        theta=float(theta),
        theta_tilde=float(theta_tilde),
        raw_initial=raw_initial,
        initial_perp=initial_perp,
        target_state=target_state,
        target_projector=target_projector,
        g_pi_state=target_state,
        walk_operator=walk_operator,
        detection_operator=detection_operator,
        controlled_operator=controlled_operator,
        target_subspace_dimension=1,
    )


def build_cqaa_multiple_target_instance(
    search_dim: int = 5,
    epsilon_pi: float = 0.12,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.90, 0.45, 0.20, 0.15),
) -> CQAAInstance:
    """Construct a marked-subspace CQAA continuation with two target states."""
    if search_dim < 4:
        raise ValueError("search_dim must be at least 4 for the multiple-target continuation.")
    if not (0.0 < epsilon_pi < 0.5):
        raise ValueError("epsilon_pi must satisfy 0 < epsilon_pi < 1/2.")

    raw_initial = np.zeros(search_dim, dtype=complex)
    raw_initial[0] = 1.0
    theta = math.asin(math.sqrt(epsilon_pi))
    theta_tilde = math.asin(math.sin(theta) / math.cos(theta))
    walk_operator = _build_real_walk(search_dim, rotation_angles)

    g_pi_direction = _make_target_direction(search_dim, target_weights)
    g_pi_state = _normalize(math.sin(theta) * raw_initial + math.cos(theta) * g_pi_direction)
    initial_perp = _normalize(raw_initial - math.sin(theta) * g_pi_state)

    aux = np.zeros(search_dim, dtype=complex)
    aux[min(2, search_dim - 1)] = 1.0
    aux = aux - np.vdot(g_pi_state, aux) * g_pi_state
    if np.linalg.norm(aux) <= 1e-12:
        aux = np.zeros(search_dim, dtype=complex)
        aux[min(3, search_dim - 1)] = 1.0
        aux = aux - np.vdot(g_pi_state, aux) * g_pi_state
    aux = _normalize(aux)

    target_projector = _projector(g_pi_state) + _projector(aux)
    ket0_tilde, ket1_tilde = _tilde_basis(theta_tilde)
    ctrl_zero = np.array([1.0, 0.0], dtype=complex)
    ctrl_one = np.array([0.0, 1.0], dtype=complex)

    reflection_g = np.eye(search_dim, dtype=complex) - 2.0 * target_projector
    detection_operator = walk_operator @ reflection_g
    controlled_reflection = (
        np.kron(_projector(ket0_tilde), reflection_g)
        + np.kron(_projector(ket1_tilde), np.eye(search_dim, dtype=complex))
    )
    controlled_walk = (
        np.kron(_projector(ctrl_zero), walk_operator)
        + np.kron(_projector(ctrl_one), np.eye(search_dim, dtype=complex))
    )
    controlled_operator = controlled_walk @ controlled_reflection

    return CQAAInstance(
        search_dim=search_dim,
        epsilon=float(epsilon_pi),
        theta=float(theta),
        theta_tilde=float(theta_tilde),
        raw_initial=raw_initial,
        initial_perp=initial_perp,
        target_state=g_pi_state,
        target_projector=target_projector,
        g_pi_state=g_pi_state,
        walk_operator=walk_operator,
        detection_operator=detection_operator,
        controlled_operator=controlled_operator,
        target_subspace_dimension=2,
    )


def _controlled_initial_state(instance: CQAAInstance) -> np.ndarray:
    ctrl_zero = np.array([1.0, 0.0], dtype=complex)
    return np.kron(ctrl_zero, instance.initial_perp)


def _controlled_target_state(instance: CQAAInstance) -> np.ndarray:
    _, ket1_tilde = _tilde_basis(instance.theta_tilde)
    return np.kron(ket1_tilde, instance.g_pi_state)


def experiment_principal_eigenvector_audit(
    epsilons: tuple[float, ...] = (0.02, 0.05, 0.10, 0.18),
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
) -> PrincipalEigenvectorAuditResults:
    fidelities = []
    overlap_initial = []
    overlap_target = []
    balance_gap = []
    qht_u = []

    for epsilon in epsilons:
        instance = build_cqaa_instance(
            search_dim=search_dim,
            epsilon=float(epsilon),
            rotation_angles=rotation_angles,
            target_weights=target_weights,
        )
        ideal = _normalize(_controlled_initial_state(instance) - _controlled_target_state(instance))
        principal = _principal_plus_one_eigenvector(instance.controlled_operator, reference=ideal)
        if abs(np.vdot(principal, ideal)) < abs(np.vdot(-principal, ideal)):
            principal = -principal
        fidelity = float(abs(np.vdot(principal, ideal)) ** 2)
        init_overlap = float(abs(np.vdot(principal, _controlled_initial_state(instance))) ** 2)
        target_overlap = float(abs(np.vdot(principal, _controlled_target_state(instance))) ** 2)
        fidelities.append(fidelity)
        overlap_initial.append(init_overlap)
        overlap_target.append(target_overlap)
        balance_gap.append(abs(init_overlap - target_overlap))
        qht_u.append(_qht_alpha(instance.controlled_operator, _controlled_initial_state(instance)))

    return PrincipalEigenvectorAuditResults(
        epsilons=np.asarray(epsilons, dtype=float),
        fidelities=np.asarray(fidelities, dtype=float),
        overlap_initial=np.asarray(overlap_initial, dtype=float),
        overlap_target=np.asarray(overlap_target, dtype=float),
        balance_gap=np.asarray(balance_gap, dtype=float),
        qht_u=np.asarray(qht_u, dtype=float),
    )


def experiment_qht_comparison(
    epsilons: tuple[float, ...] = (0.02, 0.05, 0.10, 0.18),
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
) -> QHTComparisonResults:
    qht_u = []
    qht_a = []
    qht_w = []
    for epsilon in epsilons:
        instance = build_cqaa_instance(
            search_dim=search_dim,
            epsilon=float(epsilon),
            rotation_angles=rotation_angles,
            target_weights=target_weights,
        )
        qht_u_val = _qht_alpha(instance.controlled_operator, _controlled_initial_state(instance))
        qht_a_val = _qht_alpha(instance.detection_operator, instance.initial_perp)
        qht_w_val = _qht_alpha(instance.walk_operator, instance.g_pi_state)
        qht_u.append(qht_u_val)
        qht_a.append(qht_a_val)
        qht_w.append(qht_w_val)

    qht_u_arr = np.asarray(qht_u, dtype=float)
    qht_a_arr = np.asarray(qht_a, dtype=float)
    qht_w_arr = np.asarray(qht_w, dtype=float)
    eps_arr = np.asarray(epsilons, dtype=float)
    return QHTComparisonResults(
        epsilons=eps_arr,
        qht_u=qht_u_arr,
        qht_a=qht_a_arr,
        qht_w_on_target=qht_w_arr,
        ratio_u_over_a=qht_u_arr / np.maximum(qht_a_arr, 1e-12),
        ratio_u_over_w_scaled=qht_u_arr / np.maximum(qht_w_arr / np.sqrt(eps_arr), 1e-12),
    )


def experiment_angle_sweep(
    epsilon: float = 0.10,
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
    angle_factors: tuple[float, ...] = (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40),
) -> AngleSweepResults:
    reference = build_cqaa_instance(
        search_dim=search_dim,
        epsilon=epsilon,
        rotation_angles=rotation_angles,
        target_weights=target_weights,
    )
    fidelities = []
    balance_gaps = []
    qht_values = []
    theta_values = []

    for factor in angle_factors:
        theta_tilde = float(reference.theta_tilde * factor)
        ket0_tilde, ket1_tilde = _tilde_basis(theta_tilde)
        ctrl_zero = np.array([1.0, 0.0], dtype=complex)
        ctrl_one = np.array([0.0, 1.0], dtype=complex)
        reflection_g = np.eye(reference.search_dim, dtype=complex) - 2.0 * reference.target_projector
        controlled_reflection = (
            np.kron(_projector(ket0_tilde), reflection_g)
            + np.kron(_projector(ket1_tilde), np.eye(reference.search_dim, dtype=complex))
        )
        controlled_walk = (
            np.kron(_projector(ctrl_zero), reference.walk_operator)
            + np.kron(_projector(ctrl_one), np.eye(reference.search_dim, dtype=complex))
        )
        unitary = controlled_walk @ controlled_reflection
        ideal = _normalize(np.kron(ctrl_zero, reference.initial_perp) - np.kron(ket1_tilde, reference.g_pi_state))
        principal = _principal_plus_one_eigenvector(unitary, reference=ideal)
        if abs(np.vdot(principal, ideal)) < abs(np.vdot(-principal, ideal)):
            principal = -principal
        init_overlap = abs(np.vdot(principal, np.kron(ctrl_zero, reference.initial_perp))) ** 2
        target_overlap = abs(np.vdot(principal, np.kron(ket1_tilde, reference.g_pi_state))) ** 2
        theta_values.append(theta_tilde)
        fidelities.append(abs(np.vdot(principal, ideal)) ** 2)
        balance_gaps.append(abs(init_overlap - target_overlap))
        qht_values.append(_qht_alpha(unitary, np.kron(ctrl_zero, reference.initial_perp)))

    return AngleSweepResults(
        angle_factors=np.asarray(angle_factors, dtype=float),
        thetas_tilde=np.asarray(theta_values, dtype=float),
        principal_fidelity=np.asarray(fidelities, dtype=float),
        balance_gap=np.asarray(balance_gaps, dtype=float),
        qht_u=np.asarray(qht_values, dtype=float),
    )


def experiment_amplification_dynamics(
    epsilon: float = 0.10,
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
    max_iterations: int = 24,
) -> AmplificationDynamicsResults:
    instance = build_cqaa_instance(
        search_dim=search_dim,
        epsilon=epsilon,
        rotation_angles=rotation_angles,
        target_weights=target_weights,
    )
    cqaa_success = []
    direct_success = []
    initial_controlled = _controlled_initial_state(instance)
    target_controlled = _controlled_target_state(instance)
    for t in range(max_iterations + 1):
        state_u = np.linalg.matrix_power(instance.controlled_operator, t) @ initial_controlled
        state_a = np.linalg.matrix_power(instance.detection_operator, t) @ instance.initial_perp
        cqaa_success.append(abs(np.vdot(target_controlled, state_u)) ** 2)
        direct_success.append(abs(np.vdot(instance.target_state, state_a)) ** 2)

    cqaa_arr = np.asarray(cqaa_success, dtype=float)
    direct_arr = np.asarray(direct_success, dtype=float)
    return AmplificationDynamicsResults(
        iterations=np.arange(max_iterations + 1, dtype=int),
        cqaa_success=cqaa_arr,
        detection_direct_success=direct_arr,
        cqaa_optimal_iteration=int(np.argmax(cqaa_arr)),
        detection_optimal_iteration=int(np.argmax(direct_arr)),
        cqaa_peak_success=float(np.max(cqaa_arr)),
        detection_peak_success=float(np.max(direct_arr)),
    )


def experiment_multiple_target_continuation(
    epsilon_pi: float = 0.12,
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.90, 0.45, 0.20, 0.15),
    max_iterations: int = 24,
) -> MultipleTargetResults:
    instance = build_cqaa_multiple_target_instance(
        search_dim=search_dim,
        epsilon_pi=epsilon_pi,
        rotation_angles=rotation_angles,
        target_weights=target_weights,
    )
    initial_controlled = _controlled_initial_state(instance)
    ket1_tilde = _tilde_basis(instance.theta_tilde)[1]
    marked_success = []
    marked_projector_controlled = np.kron(_projector(ket1_tilde), instance.target_projector)
    for t in range(max_iterations + 1):
        state = np.linalg.matrix_power(instance.controlled_operator, t) @ initial_controlled
        marked_success.append(float(np.real(np.vdot(state, marked_projector_controlled @ state))))
    marked_arr = np.asarray(marked_success, dtype=float)
    return MultipleTargetResults(
        iterations=np.arange(max_iterations + 1, dtype=int),
        marked_subspace_dimension=int(instance.target_subspace_dimension),
        epsilon_pi=float(epsilon_pi),
        cqaa_marked_success=marked_arr,
        peak_iteration=int(np.argmax(marked_arr)),
        peak_success=float(np.max(marked_arr)),
    )


def plot_qht_comparison(results: QHTComparisonResults, save_path: Optional[str] = None, show_plot: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(results.epsilons, results.qht_u, marker="o", linewidth=2.2, label="QHT(U, |0, init_perp>)")
    ax.plot(results.epsilons, results.qht_a, marker="s", linewidth=2.0, label="QHT(A, |init_perp>)")
    ax.plot(
        results.epsilons,
        results.qht_w_on_target / np.sqrt(results.epsilons),
        marker="^",
        linewidth=2.0,
        label="QHT(W, |g>)/sqrt(epsilon)",
    )
    ax.set_title("CQAA Hitting-Time Comparison")
    ax.set_xlabel(r"Initial success probability $\epsilon$")
    ax.set_ylabel("Quantum hitting time")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_angle_sweep(results: AngleSweepResults, save_path: Optional[str] = None, show_plot: bool = False) -> None:
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    ax1.plot(results.angle_factors, results.principal_fidelity, marker="o", color="tab:blue", linewidth=2.2, label="principal-eigenvector fidelity")
    ax1.set_xlabel(r"Angle factor $\tilde{\theta}/\tilde{\theta}_{\mathrm{opt}}$")
    ax1.set_ylabel("Fidelity / QHT balance metric")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(results.angle_factors, results.balance_gap, marker="s", color="tab:red", linewidth=2.0, label="balance gap")
    ax2.plot(results.angle_factors, results.qht_u, marker="^", color="tab:green", linewidth=2.0, label="QHT(U)")
    ax1.set_title("CQAA Control-Angle Sensitivity")
    handles = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in handles]
    ax1.legend(handles, labels, loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_amplification_dynamics(
    single_target: AmplificationDynamicsResults,
    multi_target: MultipleTargetResults,
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax1.plot(single_target.iterations, single_target.cqaa_success, marker="o", linewidth=2.2, label="CQAA success on |1_tilde, g>")
    ax1.plot(single_target.iterations, single_target.detection_direct_success, marker="s", linewidth=2.0, linestyle="--", label="direct A^t overlap with |g>")
    ax1.set_title("Detection-to-Finding Dynamics")
    ax1.set_xlabel("Number of applications")
    ax1.set_ylabel("Success probability")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.plot(multi_target.iterations, multi_target.cqaa_marked_success, marker="o", color="tab:purple", linewidth=2.2)
    ax2.set_title("Marked-Subspace Continuation")
    ax2.set_xlabel("Number of applications")
    ax2.set_ylabel("Probability in marked subspace")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def summarize_principal_eigenvector_audit(results: PrincipalEigenvectorAuditResults) -> dict[str, Any]:
    return {
        "epsilons": results.epsilons.astype(float).tolist(),
        "fidelities": results.fidelities.astype(float).tolist(),
        "overlap_initial": results.overlap_initial.astype(float).tolist(),
        "overlap_target": results.overlap_target.astype(float).tolist(),
        "balance_gap": results.balance_gap.astype(float).tolist(),
        "qht_u": results.qht_u.astype(float).tolist(),
    }


def summarize_qht_comparison(results: QHTComparisonResults) -> dict[str, Any]:
    return {
        "epsilons": results.epsilons.astype(float).tolist(),
        "qht_u": results.qht_u.astype(float).tolist(),
        "qht_a": results.qht_a.astype(float).tolist(),
        "qht_w_on_target": results.qht_w_on_target.astype(float).tolist(),
        "ratio_u_over_a": results.ratio_u_over_a.astype(float).tolist(),
        "ratio_u_over_w_scaled": results.ratio_u_over_w_scaled.astype(float).tolist(),
    }


def summarize_angle_sweep(results: AngleSweepResults) -> dict[str, Any]:
    return {
        "angle_factors": results.angle_factors.astype(float).tolist(),
        "thetas_tilde": results.thetas_tilde.astype(float).tolist(),
        "principal_fidelity": results.principal_fidelity.astype(float).tolist(),
        "balance_gap": results.balance_gap.astype(float).tolist(),
        "qht_u": results.qht_u.astype(float).tolist(),
    }


def summarize_amplification_dynamics(results: AmplificationDynamicsResults) -> dict[str, Any]:
    return {
        "iterations": results.iterations.astype(int).tolist(),
        "cqaa_success": results.cqaa_success.astype(float).tolist(),
        "detection_direct_success": results.detection_direct_success.astype(float).tolist(),
        "cqaa_optimal_iteration": results.cqaa_optimal_iteration,
        "detection_optimal_iteration": results.detection_optimal_iteration,
        "cqaa_peak_success": results.cqaa_peak_success,
        "detection_peak_success": results.detection_peak_success,
    }


def summarize_multiple_target_continuation(results: MultipleTargetResults) -> dict[str, Any]:
    return {
        "iterations": results.iterations.astype(int).tolist(),
        "marked_subspace_dimension": results.marked_subspace_dimension,
        "epsilon_pi": results.epsilon_pi,
        "cqaa_marked_success": results.cqaa_marked_success.astype(float).tolist(),
        "peak_iteration": results.peak_iteration,
        "peak_success": results.peak_success,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled Quantum Amplification theory companion.")
    parser.add_argument("--search-dim", type=int, default=5, help="Search-space dimension for the ideal CQAA instance.")
    parser.add_argument("--epsilon", type=float, default=0.10, help="Initial success probability epsilon for the single-target instance.")
    parser.add_argument("--epsilon-pi", type=float, default=0.12, help="Total marked-subspace overlap for the multiple-target continuation.")
    parser.add_argument("--rotation-angles", type=str, default="0.42,0.91", help="Comma-separated real rotation angles for W.")
    parser.add_argument("--target-weights", type=str, default="0.85,0.45,0.25,0.10", help="Comma-separated weights defining the target direction.")
    parser.add_argument("--eps-sweep", type=str, default="0.02,0.05,0.10,0.18", help="Comma-separated epsilon values for CQAA sweeps.")
    parser.add_argument("--angle-factors", type=str, default="0.60,0.75,0.90,1.00,1.10,1.25,1.40", help="Comma-separated factors multiplying the optimal control angle.")
    parser.add_argument("--max-iterations", type=int, default=24, help="Maximum number of repeated applications in the dynamics study.")
    parser.add_argument("--run-all", action="store_true", help="Run all CQAA analyses.")
    parser.add_argument("--run-principal-audit", action="store_true", help="Run the principal-eigenvector balance audit.")
    parser.add_argument("--run-qht-comparison", action="store_true", help="Run the hitting-time comparison sweep.")
    parser.add_argument("--run-angle-sweep", action="store_true", help="Run the control-angle sensitivity sweep.")
    parser.add_argument("--run-dynamics", action="store_true", help="Run repeated-application dynamics for single and multiple targets.")
    parser.add_argument("--out-prefix", type=str, default="cqaa", help="Prefix for saved artifacts.")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path for the JSON summary.")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively.")
    return parser


def run_full_analysis(
    search_dim: int = 5,
    epsilon: float = 0.10,
    epsilon_pi: float = 0.12,
    rotation_angles: object = "0.42,0.91",
    target_weights: object = "0.85,0.45,0.25,0.10",
    eps_sweep: object = "0.02,0.05,0.10,0.18",
    angle_factors: object = "0.60,0.75,0.90,1.00,1.10,1.25,1.40",
    max_iterations: int = 24,
    run_all: bool = False,
    run_principal_audit: bool = False,
    run_qht_comparison: bool = False,
    run_angle_sweep: bool = False,
    run_dynamics: bool = False,
    out_prefix: str = "cqaa",
    save_json: Optional[str] = None,
    show_plots: bool = False,
) -> int:
    rotation_angles_tuple = _parse_float_tuple(rotation_angles)
    target_weights_tuple = _parse_float_tuple(target_weights)
    eps_sweep_tuple = _parse_float_tuple(eps_sweep)
    angle_factors_tuple = _parse_float_tuple(angle_factors)

    if (
        not run_all
        and not run_principal_audit
        and not run_qht_comparison
        and not run_angle_sweep
        and not run_dynamics
    ):
        audit = experiment_principal_eigenvector_audit(
            epsilons=eps_sweep_tuple,
            search_dim=search_dim,
            rotation_angles=rotation_angles_tuple,
            target_weights=target_weights_tuple,
        )
        print(json.dumps(summarize_principal_eigenvector_audit(audit), indent=2))
        return 0

    summary: dict[str, Any] = {}
    run_principal = run_all or run_principal_audit
    run_qht = run_all or run_qht_comparison
    run_angle = run_all or run_angle_sweep
    run_dyn = run_all or run_dynamics

    if run_principal:
        audit = experiment_principal_eigenvector_audit(
            epsilons=eps_sweep_tuple,
            search_dim=search_dim,
            rotation_angles=rotation_angles_tuple,
            target_weights=target_weights_tuple,
        )
        summary["principal_eigenvector_audit"] = summarize_principal_eigenvector_audit(audit)

    if run_qht:
        qht = experiment_qht_comparison(
            epsilons=eps_sweep_tuple,
            search_dim=search_dim,
            rotation_angles=rotation_angles_tuple,
            target_weights=target_weights_tuple,
        )
        qht_plot = f"{out_prefix}_qht_comparison.png"
        plot_qht_comparison(qht, save_path=qht_plot, show_plot=bool(show_plots))
        summary["qht_comparison"] = summarize_qht_comparison(qht)
        summary["qht_comparison_artifact"] = {"plot": qht_plot}

    if run_angle:
        angle = experiment_angle_sweep(
            epsilon=float(epsilon),
            search_dim=search_dim,
            rotation_angles=rotation_angles_tuple,
            target_weights=target_weights_tuple,
            angle_factors=angle_factors_tuple,
        )
        angle_plot = f"{out_prefix}_angle_sweep.png"
        plot_angle_sweep(angle, save_path=angle_plot, show_plot=bool(show_plots))
        summary["angle_sweep"] = summarize_angle_sweep(angle)
        summary["angle_sweep_artifact"] = {"plot": angle_plot}

    if run_dyn:
        dynamics = experiment_amplification_dynamics(
            epsilon=float(epsilon),
            search_dim=search_dim,
            rotation_angles=rotation_angles_tuple,
            target_weights=target_weights_tuple,
            max_iterations=int(max_iterations),
        )
        multi = experiment_multiple_target_continuation(
            epsilon_pi=float(epsilon_pi),
            search_dim=search_dim,
            rotation_angles=rotation_angles_tuple,
            target_weights=target_weights_tuple,
            max_iterations=int(max_iterations),
        )
        dyn_plot = f"{out_prefix}_dynamics.png"
        plot_amplification_dynamics(dynamics, multi, save_path=dyn_plot, show_plot=bool(show_plots))
        summary["single_target_dynamics"] = summarize_amplification_dynamics(dynamics)
        summary["multiple_target_continuation"] = summarize_multiple_target_continuation(multi)
        summary["dynamics_artifact"] = {"plot": dyn_plot}

    if save_json:
        out = Path(save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


def _interactive_rerun_prompt(defaults: dict[str, object]) -> None:
    if not sys.stdin.isatty():
        return

    print("\n" + "=" * 72)
    print("INTERACTIVE RE-RUN MODE")
    print("=" * 72)
    print("Press Enter to finish, or enter custom key=value pairs to rerun.")
    print("Example: run_all=True, search_dim=6, epsilon=0.08")
    print("Example: run_qht_comparison=True, eps_sweep=[0.02, 0.08, 0.16]")
    print("Example: run_dynamics=True, max_iterations=30, out_prefix='cqaa_custom'")

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

    unknown = set(kwargs) - set(defaults)
    if unknown:
        print(f"Unknown argument(s): {', '.join(sorted(unknown))}")
        print("Interactive mode finished without rerun.")
        return

    merged = dict(defaults)
    merged.update(kwargs)
    print(f"\nRe-running with parameters: {merged}")
    run_full_analysis(**merged)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_full_analysis(
        search_dim=args.search_dim,
        epsilon=args.epsilon,
        epsilon_pi=args.epsilon_pi,
        rotation_angles=args.rotation_angles,
        target_weights=args.target_weights,
        eps_sweep=args.eps_sweep,
        angle_factors=args.angle_factors,
        max_iterations=args.max_iterations,
        run_all=args.run_all,
        run_principal_audit=args.run_principal_audit,
        run_qht_comparison=args.run_qht_comparison,
        run_angle_sweep=args.run_angle_sweep,
        run_dynamics=args.run_dynamics,
        out_prefix=args.out_prefix,
        save_json=args.save_json,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="cqaa")
    cli_kwargs = _parse_command_line(sys.argv[1:])
    default_kwargs: dict[str, object] = {
        "search_dim": 5,
        "epsilon": 0.10,
        "epsilon_pi": 0.12,
        "rotation_angles": "0.42,0.91",
        "target_weights": "0.85,0.45,0.25,0.10",
        "eps_sweep": "0.02,0.05,0.10,0.18",
        "angle_factors": "0.60,0.75,0.90,1.00,1.10,1.25,1.40",
        "max_iterations": 24,
        "run_all": True,
        "run_principal_audit": False,
        "run_qht_comparison": False,
        "run_angle_sweep": False,
        "run_dynamics": False,
        "out_prefix": Path(__file__).stem,
        "save_json": f"{Path(__file__).stem}_summary.json",
        "show_plots": False,
    }
    if cli_kwargs is None:
        raise SystemExit(main())
    if cli_kwargs:
        unknown = set(cli_kwargs) - set(default_kwargs)
        if unknown:
            raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
        merged = dict(default_kwargs)
        merged.update(cli_kwargs)
        raise SystemExit(run_full_analysis(**merged))

    # No explicit CLI arguments means: run the paper's default analysis once,
    # then allow an optional interactive rerun with custom overrides.
    exit_code = run_full_analysis(**default_kwargs)
    _interactive_rerun_prompt(default_kwargs)
    raise SystemExit(exit_code)

"""Controlled Quantum Amplification: ideal-model analysis companion.

This module studies the 2017 Controlled Quantum Amplification (CQAA) framework
of Dohotaru and Høyer in an ideal unitary setting. The focus is on the paper's
core claims:

1. the controlled circuit U admits a principal (+1)-eigenvector with balanced
   support on the starting configuration and the marked configuration,
2. the quantum hitting time of U is of the same order as the hitting time of
   the abstract search operator A = W G,
3. the choice of control angle \tilde{\theta} matters structurally,
4. repeated applications of U can turn detection-style spectral structure into
   finding-style amplitude on the enlarged Hilbert space,
5. the multiple-target continuation can be studied through the marked
   subspace projector.

The implementation keeps the analysis deliberately small-dimensional so that it
remains reproducible and fast as a paper-companion script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
            plt.show = old_show
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.chdir(old_cwd)
            log_handle.close()

        atexit.register(_cleanup)
        return result_dir


@dataclass
class CQAAInstance:
    """Concrete small-dimensional CQAA instance."""

    search_dim: int
    epsilon: float
    theta: float
    theta_tilde: float
    raw_initial: np.ndarray
    initial_perp: np.ndarray
    target_state: np.ndarray
    target_projector: np.ndarray
    g_pi_state: np.ndarray
    walk_operator: np.ndarray
    detection_operator: np.ndarray
    controlled_operator: np.ndarray
    target_subspace_dimension: int


@dataclass
class PrincipalEigenvectorAuditResults:
    """Audit of the principal (+1)-eigenvector balancing property."""

    epsilons: np.ndarray
    fidelities: np.ndarray
    overlap_initial: np.ndarray
    overlap_target: np.ndarray
    balance_gap: np.ndarray
    qht_u: np.ndarray


@dataclass
class QHTComparisonResults:
    """Theorem-style sweep comparing QHT(U) and QHT(A)."""

    epsilons: np.ndarray
    qht_u: np.ndarray
    qht_a: np.ndarray
    qht_w_on_target: np.ndarray
    ratio_u_over_a: np.ndarray
    ratio_u_over_w_scaled: np.ndarray


@dataclass
class AngleSweepResults:
    """Sensitivity of CQAA to the control angle."""

    angle_factors: np.ndarray
    thetas_tilde: np.ndarray
    principal_fidelity: np.ndarray
    balance_gap: np.ndarray
    qht_u: np.ndarray


@dataclass
class AmplificationDynamicsResults:
    """Repeated-application dynamics for U and A."""

    iterations: np.ndarray
    cqaa_success: np.ndarray
    detection_direct_success: np.ndarray
    cqaa_optimal_iteration: int
    detection_optimal_iteration: int
    cqaa_peak_success: float
    detection_peak_success: float


@dataclass
class MultipleTargetResults:
    """Marked-subspace continuation for multiple targets."""

    iterations: np.ndarray
    marked_subspace_dimension: int
    epsilon_pi: float
    cqaa_marked_success: np.ndarray
    peak_iteration: int
    peak_success: float


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError("Cannot normalize the zero vector.")
    return np.asarray(vec, dtype=complex) / norm


def _rotation_block(angle: float) -> np.ndarray:
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=complex,
    )


def _build_real_walk(search_dim: int, rotation_angles: tuple[float, ...]) -> np.ndarray:
    if search_dim < 3:
        raise ValueError("search_dim must be at least 3.")
    blocks: list[np.ndarray] = [np.array([[1.0]], dtype=complex)]
    remaining = search_dim - 1
    angle_idx = 0
    while remaining >= 2:
        angle = rotation_angles[min(angle_idx, len(rotation_angles) - 1)]
        blocks.append(_rotation_block(float(angle)))
        remaining -= 2
        angle_idx += 1
    if remaining == 1:
        blocks.append(np.array([[-1.0]], dtype=complex))
    matrix = np.block(
        [
            [
                blocks[i] if i == j else np.zeros((blocks[i].shape[0], blocks[j].shape[0]), dtype=complex)
                for j in range(len(blocks))
            ]
            for i in range(len(blocks))
        ]
    )
    return matrix


def _make_target_direction(search_dim: int, weights: tuple[float, ...]) -> np.ndarray:
    direction = np.zeros(search_dim, dtype=complex)
    usable = min(search_dim - 1, len(weights))
    for idx in range(usable):
        direction[idx + 1] = float(weights[idx])
    if np.linalg.norm(direction) <= 0.0:
        direction[1] = 1.0
    return _normalize(direction)


def _tilde_basis(theta_tilde: float) -> tuple[np.ndarray, np.ndarray]:
    ket0_tilde = np.array([math.cos(theta_tilde), math.sin(theta_tilde)], dtype=complex)
    ket1_tilde = np.array([-math.sin(theta_tilde), math.cos(theta_tilde)], dtype=complex)
    return ket0_tilde, ket1_tilde


def _projector(vec: np.ndarray) -> np.ndarray:
    vec_n = _normalize(vec)
    return np.outer(vec_n, np.conjugate(vec_n))


def _qht_alpha(unitary: np.ndarray, state: np.ndarray, tol: float = 1e-8) -> float:
    vals, vecs = np.linalg.eig(unitary)
    total = 0.0
    for idx, eigenvalue in enumerate(vals):
        angle = float(np.angle(eigenvalue))
        if angle <= tol or abs(abs(angle) - math.pi) <= tol:
            continue
        coeff = np.vdot(vecs[:, idx], state)
        total += 2.0 * (abs(coeff) ** 2) / (angle ** 2)
    return float(np.sqrt(max(total, 0.0)))


def _principal_plus_one_eigenvector(unitary: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    vals, vecs = np.linalg.eig(unitary)
    close = [idx for idx, value in enumerate(vals) if abs(value - 1.0) <= 1e-7]
    if not close:
        idx = int(np.argmin(np.abs(vals - 1.0)))
        return _normalize(vecs[:, idx])
    if reference is None:
        idx = close[0]
        return _normalize(vecs[:, idx])
    idx = max(close, key=lambda j: abs(np.vdot(reference, vecs[:, j])))
    return _normalize(vecs[:, idx])


def build_cqaa_instance(
    search_dim: int = 5,
    epsilon: float = 0.10,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
) -> CQAAInstance:
    """Construct a small CQAA instance with one marked state."""
    if not (0.0 < epsilon < 0.5):
        raise ValueError("epsilon must satisfy 0 < epsilon < 1/2 for the CQAA angle choice.")

    raw_initial = np.zeros(search_dim, dtype=complex)
    raw_initial[0] = 1.0
    theta = math.asin(math.sqrt(epsilon))
    theta_tilde = math.asin(math.sin(theta) / math.cos(theta))
    walk_operator = _build_real_walk(search_dim, rotation_angles)
    target_direction = _make_target_direction(search_dim, target_weights)
    target_state = math.sin(theta) * raw_initial + math.cos(theta) * target_direction
    target_state = _normalize(target_state)
    initial_perp = _normalize(raw_initial - math.sin(theta) * target_state)

    target_projector = _projector(target_state)
    reflection_g = np.eye(search_dim, dtype=complex) - 2.0 * target_projector
    detection_operator = walk_operator @ reflection_g

    ket0_tilde, ket1_tilde = _tilde_basis(theta_tilde)
    ctrl_zero = np.array([1.0, 0.0], dtype=complex)
    ctrl_one = np.array([0.0, 1.0], dtype=complex)
    controlled_reflection = (
        np.kron(_projector(ket0_tilde), reflection_g)
        + np.kron(_projector(ket1_tilde), np.eye(search_dim, dtype=complex))
    )
    controlled_walk = (
        np.kron(_projector(ctrl_zero), walk_operator)
        + np.kron(_projector(ctrl_one), np.eye(search_dim, dtype=complex))
    )
    controlled_operator = controlled_walk @ controlled_reflection

    return CQAAInstance(
        search_dim=search_dim,
        epsilon=float(epsilon),
        theta=float(theta),
        theta_tilde=float(theta_tilde),
        raw_initial=raw_initial,
        initial_perp=initial_perp,
        target_state=target_state,
        target_projector=target_projector,
        g_pi_state=target_state,
        walk_operator=walk_operator,
        detection_operator=detection_operator,
        controlled_operator=controlled_operator,
        target_subspace_dimension=1,
    )


def build_cqaa_multiple_target_instance(
    search_dim: int = 5,
    epsilon_pi: float = 0.12,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.90, 0.45, 0.20, 0.15),
) -> CQAAInstance:
    """Construct a marked-subspace CQAA continuation with two target states."""
    if search_dim < 4:
        raise ValueError("search_dim must be at least 4 for the multiple-target continuation.")
    if not (0.0 < epsilon_pi < 0.5):
        raise ValueError("epsilon_pi must satisfy 0 < epsilon_pi < 1/2.")

    raw_initial = np.zeros(search_dim, dtype=complex)
    raw_initial[0] = 1.0
    theta = math.asin(math.sqrt(epsilon_pi))
    theta_tilde = math.asin(math.sin(theta) / math.cos(theta))
    walk_operator = _build_real_walk(search_dim, rotation_angles)

    g_pi_direction = _make_target_direction(search_dim, target_weights)
    g_pi_state = _normalize(math.sin(theta) * raw_initial + math.cos(theta) * g_pi_direction)
    initial_perp = _normalize(raw_initial - math.sin(theta) * g_pi_state)

    aux = np.zeros(search_dim, dtype=complex)
    aux[min(2, search_dim - 1)] = 1.0
    aux = aux - np.vdot(g_pi_state, aux) * g_pi_state
    if np.linalg.norm(aux) <= 1e-12:
        aux = np.zeros(search_dim, dtype=complex)
        aux[min(3, search_dim - 1)] = 1.0
        aux = aux - np.vdot(g_pi_state, aux) * g_pi_state
    aux = _normalize(aux)

    target_projector = _projector(g_pi_state) + _projector(aux)
    ket0_tilde, ket1_tilde = _tilde_basis(theta_tilde)
    ctrl_zero = np.array([1.0, 0.0], dtype=complex)
    ctrl_one = np.array([0.0, 1.0], dtype=complex)

    reflection_g = np.eye(search_dim, dtype=complex) - 2.0 * target_projector
    detection_operator = walk_operator @ reflection_g
    controlled_reflection = (
        np.kron(_projector(ket0_tilde), reflection_g)
        + np.kron(_projector(ket1_tilde), np.eye(search_dim, dtype=complex))
    )
    controlled_walk = (
        np.kron(_projector(ctrl_zero), walk_operator)
        + np.kron(_projector(ctrl_one), np.eye(search_dim, dtype=complex))
    )
    controlled_operator = controlled_walk @ controlled_reflection

    return CQAAInstance(
        search_dim=search_dim,
        epsilon=float(epsilon_pi),
        theta=float(theta),
        theta_tilde=float(theta_tilde),
        raw_initial=raw_initial,
        initial_perp=initial_perp,
        target_state=g_pi_state,
        target_projector=target_projector,
        g_pi_state=g_pi_state,
        walk_operator=walk_operator,
        detection_operator=detection_operator,
        controlled_operator=controlled_operator,
        target_subspace_dimension=2,
    )


def _controlled_initial_state(instance: CQAAInstance) -> np.ndarray:
    ctrl_zero = np.array([1.0, 0.0], dtype=complex)
    return np.kron(ctrl_zero, instance.initial_perp)


def _controlled_target_state(instance: CQAAInstance) -> np.ndarray:
    _, ket1_tilde = _tilde_basis(instance.theta_tilde)
    return np.kron(ket1_tilde, instance.g_pi_state)


def experiment_principal_eigenvector_audit(
    epsilons: tuple[float, ...] = (0.02, 0.05, 0.10, 0.18),
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
) -> PrincipalEigenvectorAuditResults:
    fidelities = []
    overlap_initial = []
    overlap_target = []
    balance_gap = []
    qht_u = []

    for epsilon in epsilons:
        instance = build_cqaa_instance(
            search_dim=search_dim,
            epsilon=float(epsilon),
            rotation_angles=rotation_angles,
            target_weights=target_weights,
        )
        ideal = _normalize(_controlled_initial_state(instance) - _controlled_target_state(instance))
        principal = _principal_plus_one_eigenvector(instance.controlled_operator, reference=ideal)
        if abs(np.vdot(principal, ideal)) < abs(np.vdot(-principal, ideal)):
            principal = -principal
        fidelity = float(abs(np.vdot(principal, ideal)) ** 2)
        init_overlap = float(abs(np.vdot(principal, _controlled_initial_state(instance))) ** 2)
        target_overlap = float(abs(np.vdot(principal, _controlled_target_state(instance))) ** 2)
        fidelities.append(fidelity)
        overlap_initial.append(init_overlap)
        overlap_target.append(target_overlap)
        balance_gap.append(abs(init_overlap - target_overlap))
        qht_u.append(_qht_alpha(instance.controlled_operator, _controlled_initial_state(instance)))

    return PrincipalEigenvectorAuditResults(
        epsilons=np.asarray(epsilons, dtype=float),
        fidelities=np.asarray(fidelities, dtype=float),
        overlap_initial=np.asarray(overlap_initial, dtype=float),
        overlap_target=np.asarray(overlap_target, dtype=float),
        balance_gap=np.asarray(balance_gap, dtype=float),
        qht_u=np.asarray(qht_u, dtype=float),
    )


def experiment_qht_comparison(
    epsilons: tuple[float, ...] = (0.02, 0.05, 0.10, 0.18),
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
) -> QHTComparisonResults:
    qht_u = []
    qht_a = []
    qht_w = []
    for epsilon in epsilons:
        instance = build_cqaa_instance(
            search_dim=search_dim,
            epsilon=float(epsilon),
            rotation_angles=rotation_angles,
            target_weights=target_weights,
        )
        qht_u_val = _qht_alpha(instance.controlled_operator, _controlled_initial_state(instance))
        qht_a_val = _qht_alpha(instance.detection_operator, instance.initial_perp)
        qht_w_val = _qht_alpha(instance.walk_operator, instance.g_pi_state)
        qht_u.append(qht_u_val)
        qht_a.append(qht_a_val)
        qht_w.append(qht_w_val)

    qht_u_arr = np.asarray(qht_u, dtype=float)
    qht_a_arr = np.asarray(qht_a, dtype=float)
    qht_w_arr = np.asarray(qht_w, dtype=float)
    eps_arr = np.asarray(epsilons, dtype=float)
    return QHTComparisonResults(
        epsilons=eps_arr,
        qht_u=qht_u_arr,
        qht_a=qht_a_arr,
        qht_w_on_target=qht_w_arr,
        ratio_u_over_a=qht_u_arr / np.maximum(qht_a_arr, 1e-12),
        ratio_u_over_w_scaled=qht_u_arr / np.maximum(qht_w_arr / np.sqrt(eps_arr), 1e-12),
    )


def experiment_angle_sweep(
    epsilon: float = 0.10,
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
    angle_factors: tuple[float, ...] = (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40),
) -> AngleSweepResults:
    reference = build_cqaa_instance(
        search_dim=search_dim,
        epsilon=epsilon,
        rotation_angles=rotation_angles,
        target_weights=target_weights,
    )
    fidelities = []
    balance_gaps = []
    qht_values = []
    theta_values = []

    for factor in angle_factors:
        theta_tilde = float(reference.theta_tilde * factor)
        ket0_tilde, ket1_tilde = _tilde_basis(theta_tilde)
        ctrl_zero = np.array([1.0, 0.0], dtype=complex)
        ctrl_one = np.array([0.0, 1.0], dtype=complex)
        reflection_g = np.eye(reference.search_dim, dtype=complex) - 2.0 * reference.target_projector
        controlled_reflection = (
            np.kron(_projector(ket0_tilde), reflection_g)
            + np.kron(_projector(ket1_tilde), np.eye(reference.search_dim, dtype=complex))
        )
        controlled_walk = (
            np.kron(_projector(ctrl_zero), reference.walk_operator)
            + np.kron(_projector(ctrl_one), np.eye(reference.search_dim, dtype=complex))
        )
        unitary = controlled_walk @ controlled_reflection
        ideal = _normalize(np.kron(ctrl_zero, reference.initial_perp) - np.kron(ket1_tilde, reference.g_pi_state))
        principal = _principal_plus_one_eigenvector(unitary, reference=ideal)
        if abs(np.vdot(principal, ideal)) < abs(np.vdot(-principal, ideal)):
            principal = -principal
        init_overlap = abs(np.vdot(principal, np.kron(ctrl_zero, reference.initial_perp))) ** 2
        target_overlap = abs(np.vdot(principal, np.kron(ket1_tilde, reference.g_pi_state))) ** 2
        theta_values.append(theta_tilde)
        fidelities.append(abs(np.vdot(principal, ideal)) ** 2)
        balance_gaps.append(abs(init_overlap - target_overlap))
        qht_values.append(_qht_alpha(unitary, np.kron(ctrl_zero, reference.initial_perp)))

    return AngleSweepResults(
        angle_factors=np.asarray(angle_factors, dtype=float),
        thetas_tilde=np.asarray(theta_values, dtype=float),
        principal_fidelity=np.asarray(fidelities, dtype=float),
        balance_gap=np.asarray(balance_gaps, dtype=float),
        qht_u=np.asarray(qht_values, dtype=float),
    )


def experiment_amplification_dynamics(
    epsilon: float = 0.10,
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.85, 0.45, 0.25, 0.10),
    max_iterations: int = 24,
) -> AmplificationDynamicsResults:
    instance = build_cqaa_instance(
        search_dim=search_dim,
        epsilon=epsilon,
        rotation_angles=rotation_angles,
        target_weights=target_weights,
    )
    cqaa_success = []
    direct_success = []
    initial_controlled = _controlled_initial_state(instance)
    target_controlled = _controlled_target_state(instance)
    for t in range(max_iterations + 1):
        state_u = np.linalg.matrix_power(instance.controlled_operator, t) @ initial_controlled
        state_a = np.linalg.matrix_power(instance.detection_operator, t) @ instance.initial_perp
        cqaa_success.append(abs(np.vdot(target_controlled, state_u)) ** 2)
        direct_success.append(abs(np.vdot(instance.target_state, state_a)) ** 2)

    cqaa_arr = np.asarray(cqaa_success, dtype=float)
    direct_arr = np.asarray(direct_success, dtype=float)
    return AmplificationDynamicsResults(
        iterations=np.arange(max_iterations + 1, dtype=int),
        cqaa_success=cqaa_arr,
        detection_direct_success=direct_arr,
        cqaa_optimal_iteration=int(np.argmax(cqaa_arr)),
        detection_optimal_iteration=int(np.argmax(direct_arr)),
        cqaa_peak_success=float(np.max(cqaa_arr)),
        detection_peak_success=float(np.max(direct_arr)),
    )


def experiment_multiple_target_continuation(
    epsilon_pi: float = 0.12,
    search_dim: int = 5,
    rotation_angles: tuple[float, ...] = (0.42, 0.91),
    target_weights: tuple[float, ...] = (0.90, 0.45, 0.20, 0.15),
    max_iterations: int = 24,
) -> MultipleTargetResults:
    instance = build_cqaa_multiple_target_instance(
        search_dim=search_dim,
        epsilon_pi=epsilon_pi,
        rotation_angles=rotation_angles,
        target_weights=target_weights,
    )
    initial_controlled = _controlled_initial_state(instance)
    ket1_tilde = _tilde_basis(instance.theta_tilde)[1]
    marked_success = []
    marked_projector_controlled = np.kron(_projector(ket1_tilde), instance.target_projector)
    for t in range(max_iterations + 1):
        state = np.linalg.matrix_power(instance.controlled_operator, t) @ initial_controlled
        marked_success.append(float(np.real(np.vdot(state, marked_projector_controlled @ state))))
    marked_arr = np.asarray(marked_success, dtype=float)
    return MultipleTargetResults(
        iterations=np.arange(max_iterations + 1, dtype=int),
        marked_subspace_dimension=int(instance.target_subspace_dimension),
        epsilon_pi=float(epsilon_pi),
        cqaa_marked_success=marked_arr,
        peak_iteration=int(np.argmax(marked_arr)),
        peak_success=float(np.max(marked_arr)),
    )


def plot_qht_comparison(results: QHTComparisonResults, save_path: Optional[str] = None, show_plot: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(results.epsilons, results.qht_u, marker="o", linewidth=2.2, label="QHT(U, |0, init_perp>)")
    ax.plot(results.epsilons, results.qht_a, marker="s", linewidth=2.0, label="QHT(A, |init_perp>)")
    ax.plot(
        results.epsilons,
        results.qht_w_on_target / np.sqrt(results.epsilons),
        marker="^",
        linewidth=2.0,
        label="QHT(W, |g>)/sqrt(epsilon)",
    )
    ax.set_title("CQAA Hitting-Time Comparison")
    ax.set_xlabel(r"Initial success probability $\epsilon$")
    ax.set_ylabel("Quantum hitting time")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_angle_sweep(results: AngleSweepResults, save_path: Optional[str] = None, show_plot: bool = False) -> None:
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    ax1.plot(results.angle_factors, results.principal_fidelity, marker="o", color="tab:blue", linewidth=2.2, label="principal-eigenvector fidelity")
    ax1.set_xlabel(r"Angle factor $\tilde{\theta}/\tilde{\theta}_{\mathrm{opt}}$")
    ax1.set_ylabel("Fidelity / QHT balance metric")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(results.angle_factors, results.balance_gap, marker="s", color="tab:red", linewidth=2.0, label="balance gap")
    ax2.plot(results.angle_factors, results.qht_u, marker="^", color="tab:green", linewidth=2.0, label="QHT(U)")
    ax1.set_title("CQAA Control-Angle Sensitivity")
    handles = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in handles]
    ax1.legend(handles, labels, loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_amplification_dynamics(
    single_target: AmplificationDynamicsResults,
    multi_target: MultipleTargetResults,
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax1.plot(single_target.iterations, single_target.cqaa_success, marker="o", linewidth=2.2, label="CQAA success on |1_tilde, g>")
    ax1.plot(single_target.iterations, single_target.detection_direct_success, marker="s", linewidth=2.0, linestyle="--", label="direct A^t overlap with |g>")
    ax1.set_title("Detection-to-Finding Dynamics")
    ax1.set_xlabel("Number of applications")
    ax1.set_ylabel("Success probability")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.plot(multi_target.iterations, multi_target.cqaa_marked_success, marker="o", color="tab:purple", linewidth=2.2)
    ax2.set_title("Marked-Subspace Continuation")
    ax2.set_xlabel("Number of applications")
    ax2.set_ylabel("Probability in marked subspace")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def summarize_principal_eigenvector_audit(results: PrincipalEigenvectorAuditResults) -> dict[str, Any]:
    return {
        "epsilons": results.epsilons.astype(float).tolist(),
        "fidelities": results.fidelities.astype(float).tolist(),
        "overlap_initial": results.overlap_initial.astype(float).tolist(),
        "overlap_target": results.overlap_target.astype(float).tolist(),
        "balance_gap": results.balance_gap.astype(float).tolist(),
        "qht_u": results.qht_u.astype(float).tolist(),
    }


def summarize_qht_comparison(results: QHTComparisonResults) -> dict[str, Any]:
    return {
        "epsilons": results.epsilons.astype(float).tolist(),
        "qht_u": results.qht_u.astype(float).tolist(),
        "qht_a": results.qht_a.astype(float).tolist(),
        "qht_w_on_target": results.qht_w_on_target.astype(float).tolist(),
        "ratio_u_over_a": results.ratio_u_over_a.astype(float).tolist(),
        "ratio_u_over_w_scaled": results.ratio_u_over_w_scaled.astype(float).tolist(),
    }


def summarize_angle_sweep(results: AngleSweepResults) -> dict[str, Any]:
    return {
        "angle_factors": results.angle_factors.astype(float).tolist(),
        "thetas_tilde": results.thetas_tilde.astype(float).tolist(),
        "principal_fidelity": results.principal_fidelity.astype(float).tolist(),
        "balance_gap": results.balance_gap.astype(float).tolist(),
        "qht_u": results.qht_u.astype(float).tolist(),
    }


def summarize_amplification_dynamics(results: AmplificationDynamicsResults) -> dict[str, Any]:
    return {
        "iterations": results.iterations.astype(int).tolist(),
        "cqaa_success": results.cqaa_success.astype(float).tolist(),
        "detection_direct_success": results.detection_direct_success.astype(float).tolist(),
        "cqaa_optimal_iteration": results.cqaa_optimal_iteration,
        "detection_optimal_iteration": results.detection_optimal_iteration,
        "cqaa_peak_success": results.cqaa_peak_success,
        "detection_peak_success": results.detection_peak_success,
    }


def summarize_multiple_target_continuation(results: MultipleTargetResults) -> dict[str, Any]:
    return {
        "iterations": results.iterations.astype(int).tolist(),
        "marked_subspace_dimension": results.marked_subspace_dimension,
        "epsilon_pi": results.epsilon_pi,
        "cqaa_marked_success": results.cqaa_marked_success.astype(float).tolist(),
        "peak_iteration": results.peak_iteration,
        "peak_success": results.peak_success,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled Quantum Amplification theory companion.")
    parser.add_argument("--search-dim", type=int, default=5, help="Search-space dimension for the ideal CQAA instance.")
    parser.add_argument("--epsilon", type=float, default=0.10, help="Initial success probability epsilon for the single-target instance.")
    parser.add_argument("--epsilon-pi", type=float, default=0.12, help="Total marked-subspace overlap for the multiple-target continuation.")
    parser.add_argument("--rotation-angles", type=str, default="0.42,0.91", help="Comma-separated real rotation angles for W.")
    parser.add_argument("--target-weights", type=str, default="0.85,0.45,0.25,0.10", help="Comma-separated weights defining the target direction.")
    parser.add_argument("--eps-sweep", type=str, default="0.02,0.05,0.10,0.18", help="Comma-separated epsilon values for CQAA sweeps.")
    parser.add_argument("--angle-factors", type=str, default="0.60,0.75,0.90,1.00,1.10,1.25,1.40", help="Comma-separated factors multiplying the optimal control angle.")
    parser.add_argument("--max-iterations", type=int, default=24, help="Maximum number of repeated applications in the dynamics study.")
    parser.add_argument("--run-all", action="store_true", help="Run all CQAA analyses.")
    parser.add_argument("--run-principal-audit", action="store_true", help="Run the principal-eigenvector balance audit.")
    parser.add_argument("--run-qht-comparison", action="store_true", help="Run the hitting-time comparison sweep.")
    parser.add_argument("--run-angle-sweep", action="store_true", help="Run the control-angle sensitivity sweep.")
    parser.add_argument("--run-dynamics", action="store_true", help="Run repeated-application dynamics for single and multiple targets.")
    parser.add_argument("--out-prefix", type=str, default="cqaa", help="Prefix for saved artifacts.")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path for the JSON summary.")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    rotation_angles = tuple(float(x.strip()) for x in args.rotation_angles.split(",") if x.strip())
    target_weights = tuple(float(x.strip()) for x in args.target_weights.split(",") if x.strip())
    eps_sweep = tuple(float(x.strip()) for x in args.eps_sweep.split(",") if x.strip())
    angle_factors = tuple(float(x.strip()) for x in args.angle_factors.split(",") if x.strip())

    if (
        not args.run_all
        and not args.run_principal_audit
        and not args.run_qht_comparison
        and not args.run_angle_sweep
        and not args.run_dynamics
    ):
        audit = experiment_principal_eigenvector_audit(
            epsilons=eps_sweep,
            search_dim=args.search_dim,
            rotation_angles=rotation_angles,
            target_weights=target_weights,
        )
        print(json.dumps(summarize_principal_eigenvector_audit(audit), indent=2))
        return 0

    summary: dict[str, Any] = {}
    run_principal = args.run_all or args.run_principal_audit
    run_qht = args.run_all or args.run_qht_comparison
    run_angle = args.run_all or args.run_angle_sweep
    run_dyn = args.run_all or args.run_dynamics

    if run_principal:
        audit = experiment_principal_eigenvector_audit(
            epsilons=eps_sweep,
            search_dim=args.search_dim,
            rotation_angles=rotation_angles,
            target_weights=target_weights,
        )
        summary["principal_eigenvector_audit"] = summarize_principal_eigenvector_audit(audit)

    if run_qht:
        qht = experiment_qht_comparison(
            epsilons=eps_sweep,
            search_dim=args.search_dim,
            rotation_angles=rotation_angles,
            target_weights=target_weights,
        )
        qht_plot = f"{args.out_prefix}_qht_comparison.png"
        plot_qht_comparison(qht, save_path=qht_plot, show_plot=bool(args.show_plots))
        summary["qht_comparison"] = summarize_qht_comparison(qht)
        summary["qht_comparison_artifact"] = {"plot": qht_plot}

    if run_angle:
        angle = experiment_angle_sweep(
            epsilon=float(args.epsilon),
            search_dim=args.search_dim,
            rotation_angles=rotation_angles,
            target_weights=target_weights,
            angle_factors=angle_factors,
        )
        angle_plot = f"{args.out_prefix}_angle_sweep.png"
        plot_angle_sweep(angle, save_path=angle_plot, show_plot=bool(args.show_plots))
        summary["angle_sweep"] = summarize_angle_sweep(angle)
        summary["angle_sweep_artifact"] = {"plot": angle_plot}

    if run_dyn:
        dynamics = experiment_amplification_dynamics(
            epsilon=float(args.epsilon),
            search_dim=args.search_dim,
            rotation_angles=rotation_angles,
            target_weights=target_weights,
            max_iterations=int(args.max_iterations),
        )
        multi = experiment_multiple_target_continuation(
            epsilon_pi=float(args.epsilon_pi),
            search_dim=args.search_dim,
            rotation_angles=rotation_angles,
            target_weights=target_weights,
            max_iterations=int(args.max_iterations),
        )
        dyn_plot = f"{args.out_prefix}_dynamics.png"
        plot_amplification_dynamics(dynamics, multi, save_path=dyn_plot, show_plot=bool(args.show_plots))
        summary["single_target_dynamics"] = summarize_amplification_dynamics(dynamics)
        summary["multiple_target_continuation"] = summarize_multiple_target_continuation(multi)
        summary["dynamics_artifact"] = {"plot": dyn_plot}

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="cqaa")
    if len(sys.argv) == 1:
        stem = Path(__file__).stem
        raise SystemExit(
            main(
                [
                    "--run-all",
                    "--out-prefix",
                    stem,
                    "--save-json",
                    f"{stem}_summary.json",
                ]
            )
        )
    raise SystemExit(main())
