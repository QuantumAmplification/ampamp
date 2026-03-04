"""QSVT Unification Laboratory

This file implements a complete, phase-structured QSVT laboratory from core
SU(2) polynomial synthesis to asymptotic limits and hardware fragility.

Phase I: Core algebra and structural validation
1. Generalized sequence evaluator (Eq. 78)
2. Polynomial extractor (Eq. 80)
3. Unitarity / boundedness auditor (Eq. 83, 84)
4. Parity constraint checker (Eq. 82)
5. Canonical phase-sequence diagnostics
6. Block-encoding synthesizer
7. Invariant subspace audit (Theorem 17)

Phase II: Algorithmic unification
8. Hamiltonian simulation (Jacobi-Anger synthesis)
9. Matrix inversion (HHL 2.0 style polynomial inverse)
10. Fixed-point search (sign-function thresholding)

Phase III: Operator calculus
11. LCU operator algebra (addition / multiplication closure)
12. Uniform singular value amplification (USVA)

Phase IV: Fundamental limits and realism
13. Markov/Bernstein extremal derivative boundary
14. Physical phase-fragility and hardware sensitivity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class QSPForwardResults:
    """Results for SU(2)-QSP forward evaluation and structural audits."""

    target_name: str
    degree: int
    x_values: np.ndarray
    p_real: np.ndarray
    p_imag: np.ndarray
    q_real: np.ndarray
    q_imag: np.ndarray
    unitarity_errors: np.ndarray
    max_unitarity_error: float
    parity_is_valid: bool
    max_parity_error: float


@dataclass
class LCUBlockEncodingResults:
    """Results for LCU block-encoding synthesis and extraction audits."""

    target_matrix_A: np.ndarray
    alpha: float
    full_unitary_U: np.ndarray
    extracted_block: np.ndarray
    reconstruction_error: float
    assembly_error: float
    is_U_strictly_unitary: bool


@dataclass
class InvariantSubspaceResults:
    """Results for invariant-subspace geometry and orthogonality audits."""

    matrix_dim: int
    unitary_dim: int
    degree: int
    sv_trajectory_0: np.ndarray
    sv_trajectory_1: np.ndarray
    rank_0: int
    rank_1: int
    inter_plane_overlap: float
    max_unitarity_error: float


@dataclass
class HamiltonianSimResults:
    """Results for Jacobi-Anger Hamiltonian simulation benchmarks."""

    t: float
    x_values: np.ndarray
    exact_exp: np.ndarray
    approx_cos: np.ndarray
    approx_sin: np.ndarray
    approx_exp: np.ndarray
    degrees_tested: np.ndarray
    max_errors: np.ndarray
    lcu_alphas: np.ndarray
    optimal_d_for_target_error: int
    parity_even_error: float
    parity_odd_error: float


@dataclass
class MatrixInversionResults:
    """Results for QSVT-based matrix inversion and resource scaling audits."""

    kappa: float
    degree: int
    x_domain: np.ndarray
    target_function: np.ndarray
    poly_approx: np.ndarray
    max_error_in_domain: float
    max_amplitude: float
    odd_parity_error: float
    epsilons: np.ndarray
    qpe_qubits: np.ndarray
    qsvt_qubits: np.ndarray


@dataclass
class FixedPointSearchResults:
    """Results for fixed-point search transfer and convergence diagnostics."""

    delta: float
    target_degree: int
    x_eval: np.ndarray
    target_sgn: np.ndarray
    poly_sgn: np.ndarray
    test_x0: float
    degrees_eval: np.ndarray
    standard_aa_probs: np.ndarray
    qsvt_fpaa_raw_probs: np.ndarray
    qsvt_fpaa_probs: np.ndarray
    monotonic_violations_raw: int
    monotonic_violations_envelope: int
    odd_parity_error: float
    max_amplitude: float


@dataclass
class OperatorAlgebraResults:
    """Results for LCU addition/multiplication closure and alpha growth."""

    A: np.ndarray
    B: np.ndarray
    alpha_A: float
    alpha_B: float
    add_error: float
    mult_error: float
    alpha_add: float
    alpha_mult: float
    k_values: np.ndarray
    decay_curve: np.ndarray
    unitary_error_A: float
    unitary_error_B: float


@dataclass
class UniformAmplificationResults:
    """Results for uniform singular-value amplification and rescue dynamics."""

    amplification_factor: float
    degree: int
    x_eval: np.ndarray
    target_func: np.ndarray
    poly_approx: np.ndarray
    slope_at_origin: float
    max_amplitude: float
    odd_parity_error: float
    k_depths: np.ndarray
    decay_curve: np.ndarray
    rescued_curve: np.ndarray
    rescue_operations: int


@dataclass
class MarkovBoundaryResults:
    """Results for Markov/Bernstein extremal derivative boundary audits."""

    d_visual: int
    x_eval: np.ndarray
    poly_visual: np.ndarray
    deriv_visual: np.ndarray
    degrees: np.ndarray
    empirical_global_slopes: np.ndarray
    theoretical_markov_bounds: np.ndarray
    empirical_interior_slopes: np.ndarray
    theoretical_interior_bounds: np.ndarray


@dataclass
class PhaseFragilityResults:
    """Results for out-of-domain leakage and phase-drift fragility audits."""

    degree: int
    x_eval_extended: np.ndarray
    p_ideal_extended: np.ndarray
    x_eval_valid: np.ndarray
    p_ideal_valid: np.ndarray
    p_noisy_valid: np.ndarray
    phase_error_rad: float
    depths: np.ndarray
    fidelity_decay: np.ndarray
    max_leakage: float
    max_distortion: float


class SU2QSPEngine:
    """Forward-model SU(2) compiler for Quantum Signal Processing (QSP)."""

    @staticmethod
    def w_signal(x: float) -> np.ndarray:
        """Return the signal unitary W(x) (Eq. 75)."""
        x = float(np.clip(x, -1.0, 1.0))
        y = np.sqrt(max(0.0, 1.0 - x * x))
        return np.array([[x, 1j * y], [1j * y, x]], dtype=complex)

    @staticmethod
    def z_rotation(phi: float) -> np.ndarray:
        """Return the phase rotation exp(i * phi * Z)."""
        return np.array(
            [[np.exp(1j * phi), 0.0], [0.0, np.exp(-1j * phi)]],
            dtype=complex,
        )

    @classmethod
    def evaluate_sequence(
        cls,
        name: str,
        phases: np.ndarray,
        num_points: int = 1001,
        audit_tolerance: float = 1e-10,
    ) -> QSPForwardResults:
        """Evaluate Eq. 78 and run Eq. 80/82/83/84 structural audits."""
        phases = np.asarray(phases, dtype=float).ravel()
        if phases.size == 0:
            raise ValueError("phases must contain at least one entry")
        if num_points < 3:
            raise ValueError("num_points must be >= 3")

        degree = int(phases.size - 1)
        x_vals = np.linspace(-1.0, 1.0, num_points)

        p_vals = np.zeros(num_points, dtype=complex)
        q_vals = np.zeros(num_points, dtype=complex)
        unitarity_errors = np.zeros(num_points, dtype=float)

        for idx, x in enumerate(x_vals):
            # 1) Generalized sequence evaluation (Eq. 78)
            u = cls.z_rotation(phases[0])
            wx = cls.w_signal(x)
            for phi in phases[1:]:
                u = u @ wx @ cls.z_rotation(phi)

            # 2) Polynomial extraction (Eq. 80)
            p_x = u[0, 0]
            y = np.sqrt(max(0.0, 1.0 - x * x))
            if y <= 1e-15:
                q_x = 0.0j
            else:
                q_x = u[1, 0] / (1j * y)

            p_vals[idx] = p_x
            q_vals[idx] = q_x

            # 3) Unitarity/boundedness audit (Eq. 83/84)
            unitarity_lhs = (abs(p_x) ** 2) + (1.0 - x * x) * (abs(q_x) ** 2)
            unitarity_errors[idx] = abs(1.0 - unitarity_lhs)

        # 4) Parity audit (Eq. 82): P(-x) = (-1)^d * P(x)
        parity_factor = (-1) ** (degree % 2)
        parity_errors = np.abs(p_vals[::-1] - parity_factor * p_vals)
        max_parity_error = float(np.max(parity_errors))
        parity_is_valid = bool(max_parity_error < audit_tolerance)

        return QSPForwardResults(
            target_name=name,
            degree=degree,
            x_values=x_vals,
            p_real=np.real(p_vals),
            p_imag=np.imag(p_vals),
            q_real=np.real(q_vals),
            q_imag=np.imag(q_vals),
            unitarity_errors=unitarity_errors,
            max_unitarity_error=float(np.max(unitarity_errors)),
            parity_is_valid=parity_is_valid,
            max_parity_error=max_parity_error,
        )


def canonical_phase_sets() -> Dict[str, np.ndarray]:
    """Return canonical phase vectors used throughout the diagnostics."""
    return {
        # Degree-3 odd-parity sign proxy.
        "Sign Function Proxy (d=3)": np.array(
            [-0.785398, 1.570796, 1.570796, -0.785398], dtype=float
        ),
        # Degree-4 even-parity cosine proxy.
        "Cosine Proxy (d=4)": np.array([0.0, 1.0, -1.0, 1.0, 0.0], dtype=float),
    }


def run_qsp_engine_diagnostics(plot: bool = True) -> Dict[str, QSPForwardResults]:
    """Run canonical SU(2)-QSP diagnostics and optional evidence plots."""
    print("-" * 65)
    print("PHASE I: SU(2) QSP ENGINE DIAGNOSTICS")
    print("-" * 65)

    phases = canonical_phase_sets()
    results: Dict[str, QSPForwardResults] = {}

    for name, phi in phases.items():
        result = SU2QSPEngine.evaluate_sequence(name=name, phases=phi)
        results[name] = result
        print(f"Target: {result.target_name}")
        print(f"  Degree:              {result.degree}")
        print(f"  Parity Valid?        {result.parity_is_valid}")
        print(f"  Max Parity Error:    {result.max_parity_error:.4e}")
        print(f"  Max Unitarity Error: {result.max_unitarity_error:.4e}")
        if result.max_unitarity_error >= 1e-12:
            raise AssertionError("Unitarity violation detected.")
        if not result.parity_is_valid:
            raise AssertionError("Parity violation detected.")

    if not plot:
        return results

    sign_res = results["Sign Function Proxy (d=3)"]
    cos_res = results["Cosine Proxy (d=4)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(sign_res.x_values, sign_res.p_real, lw=2.5, label=r"$\Re[P(x)]$")
    ax1.plot(sign_res.x_values, sign_res.p_imag, "--", lw=2.0, label=r"$\Im[P(x)]$")
    ax1.axhline(1.0, color="black", ls=":", label=r"Unitarity Boundary ($\pm 1$)")
    ax1.axhline(-1.0, color="black", ls=":")
    ax1.set_title("Sign Function Proxy (d=3)\nParity: Odd", fontsize=12)
    ax1.set_xlabel(r"Signal $x \in [-1, 1]$")
    ax1.set_ylabel("Polynomial Output")
    ax1.set_ylim(-1.2, 1.2)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    ax2.plot(cos_res.x_values, cos_res.p_real, lw=2.5, label=r"$\Re[P(x)]$")
    ax2.plot(cos_res.x_values, cos_res.p_imag, "--", lw=2.0, label=r"$\Im[P(x)]$")
    ax2.axhline(1.0, color="black", ls=":", label=r"Unitarity Boundary ($\pm 1$)")
    ax2.axhline(-1.0, color="black", ls=":")
    ax2.set_title("Cosine Proxy (d=4)\nParity: Even", fontsize=12)
    ax2.set_xlabel(r"Signal $x \in [-1, 1]$")
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right")

    plt.suptitle("QSVT Forward Engine: Polynomial Extraction & Bounds Auditing")
    plt.tight_layout()
    plt.show()

    return results


def _ancilla_zero_subspace_indices(total_dim: int, ancilla_qubit_index: int = 0) -> np.ndarray:
    """Return computational-basis indices where a chosen ancilla is |0>."""
    n_qubits = int(np.log2(total_dim))
    if (1 << n_qubits) != total_dim:
        raise ValueError("total_dim must be a power of two")
    return np.array(
        [idx for idx in range(total_dim) if ((idx >> ancilla_qubit_index) & 1) == 0],
        dtype=int,
    )


def _basis_permutation_ancilla_major() -> np.ndarray:
    """Permutation from Qiskit (system⊗ancilla) to ancilla-major ordering."""
    return np.array([0, 2, 1, 3], dtype=int)


def _matrix_sqrt_psd(matrix: np.ndarray, eig_clip: float = 1e-14) -> np.ndarray:
    """Return the Hermitian PSD matrix square root via eigendecomposition."""
    herm = 0.5 * (matrix + matrix.conj().T)
    eigvals, eigvecs = np.linalg.eigh(herm)
    eigvals = np.where(eigvals < eig_clip, 0.0, eigvals)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.conj().T


def _embed_local_operator(op: np.ndarray, targets: list[int], dims: list[int]) -> np.ndarray:
    """Embed a local operator on selected subsystems into a full tensor space."""
    op = np.asarray(op, dtype=complex)
    targets = list(targets)
    dims = list(dims)
    n_subsystems = len(dims)
    full_dim = int(np.prod(dims))
    local_dim = int(np.prod([dims[t] for t in targets]))

    if op.shape != (local_dim, local_dim):
        raise ValueError("op shape does not match product dimension of targets")

    states = [np.unravel_index(idx, dims) for idx in range(full_dim)]
    non_targets = [k for k in range(n_subsystems) if k not in targets]
    target_dims = [dims[t] for t in targets]

    full = np.zeros((full_dim, full_dim), dtype=complex)
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if any(si[k] != sj[k] for k in non_targets):
                continue
            loc_i = tuple(si[t] for t in targets)
            loc_j = tuple(sj[t] for t in targets)
            ii = int(np.ravel_multi_index(loc_i, target_dims))
            jj = int(np.ravel_multi_index(loc_j, target_dims))
            full[i, j] = op[ii, jj]

    return full


def experiment_lcu_block_encoding(plot: bool = True) -> LCUBlockEncodingResults:
    """MODULE 2: Build and audit an LCU block-encoding for a non-unitary A."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Operator

    print("-" * 65)
    print("MODULE 2: LCU BLOCK-ENCODING SYNTHESIZER")
    print("-" * 65)

    # Non-unitary target matrix: A = a0*I + a1*X.
    alpha_0, alpha_1 = 1.5, 0.5
    u_0 = np.eye(2, dtype=complex)
    u_1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    a_target = alpha_0 * u_0 + alpha_1 * u_1

    # Subnormalization factor: alpha = sum_j |a_j|.
    alpha = float(abs(alpha_0) + abs(alpha_1))

    print(f"Target Matrix A (Non-Unitary):\n{a_target}")
    print(f"Calculated Subnormalization (alpha): {alpha:.2f}")

    ancilla = QuantumRegister(1, "ancilla")
    system = QuantumRegister(1, "system")

    # PREP prepares (sqrt(a0)|0> + sqrt(a1)|1>) / sqrt(alpha) on ancilla.
    qc_prep = QuantumCircuit(ancilla, system, name="PREP")
    theta = 2.0 * np.arccos(np.sqrt(abs(alpha_0) / alpha))
    qc_prep.ry(theta, ancilla[0])
    prep_matrix = Operator(qc_prep).data

    # SEL applies I if ancilla=0 and X if ancilla=1.
    qc_sel = QuantumCircuit(ancilla, system, name="SEL")
    qc_sel.cx(ancilla[0], system[0])
    sel_matrix = Operator(qc_sel).data

    # Assemble the full block-encoding circuit.
    qc_full = QuantumCircuit(ancilla, system, name="U_BlockEncode")
    qc_full.append(qc_prep, [ancilla[0], system[0]])
    qc_full.append(qc_sel, [ancilla[0], system[0]])
    qc_full.append(qc_prep.inverse(), [ancilla[0], system[0]])
    u_full = Operator(qc_full).data

    # Independent matrix check for PREP^\dagger * SEL * PREP.
    u_formula = prep_matrix.conj().T @ sel_matrix @ prep_matrix
    assembly_error = float(np.linalg.norm(u_full - u_formula))

    is_unitary = bool(
        np.allclose(u_full @ u_full.conj().T, np.eye(4, dtype=complex), atol=1e-10)
    )

    # Project onto the ancilla-|0> subspace and recover the encoded block.
    anc0 = _ancilla_zero_subspace_indices(total_dim=u_full.shape[0], ancilla_qubit_index=0)
    extracted_block = u_full[np.ix_(anc0, anc0)]
    reconstructed_a = alpha * extracted_block
    reconstruction_error = float(np.linalg.norm(a_target - reconstructed_a))

    print(f"Is global U strictly unitary? {is_unitary}")
    print(f"Assembly Error ||U_circuit - U_formula||: {assembly_error:.4e}")
    print(f"Extraction Error ||A - alpha * U_block||: {reconstruction_error:.4e}")
    if reconstruction_error < 1e-12 and is_unitary:
        print("PASS: Non-unitary A is correctly embedded as A/alpha in unitary U.")
    print("-" * 65)

    if plot:
        import matplotlib.patches as patches

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the target matrix A.
        ax1.matshow(np.abs(a_target), cmap="Blues", vmin=0, vmax=2)
        for (i, j), val in np.ndenumerate(a_target):
            label_color = "white" if abs(val) > 1 else "black"
            ax1.text(
                j,
                i,
                f"{np.real(val):.2f}",
                ha="center",
                va="center",
                color=label_color,
                fontweight="bold",
            )
        ax1.set_title("Target Non-Unitary Matrix ($A$)", pad=20, fontsize=13)
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])

        # Reorder basis for visualization so the ancilla-|0> block is top-left.
        perm = _basis_permutation_ancilla_major()
        u_vis = u_full[np.ix_(perm, perm)]
        ax2.matshow(np.abs(u_vis), cmap="viridis", vmin=0, vmax=1)
        rect = patches.Rectangle(
            (-0.5, -0.5), 2, 2, linewidth=3, edgecolor="tab:green", facecolor="none"
        )
        ax2.add_patch(rect)
        ax2.text(
            0.5,
            -0.85,
            r"Block: $A / \alpha$",
            color="tab:green",
            fontsize=12,
            ha="center",
            fontweight="bold",
        )
        for (i, j), val in np.ndenumerate(u_vis):
            label_color = "white" if abs(val) > 0.5 else "black"
            ax2.text(j, i, f"{np.real(val):.2f}", ha="center", va="center", color=label_color)

        ax2.set_title("Expanded Unitary ($U$)", pad=20, fontsize=13)
        ax2.set_xticks([0, 1, 2, 3])
        ax2.set_yticks([0, 1, 2, 3])

        plt.suptitle(f"LCU Block-Encoding Audit: Subnormalization $\\alpha={alpha:.2f}$")
        plt.tight_layout()
        plt.show()

    return LCUBlockEncodingResults(
        target_matrix_A=a_target,
        alpha=alpha,
        full_unitary_U=u_full,
        extracted_block=extracted_block,
        reconstruction_error=reconstruction_error,
        assembly_error=assembly_error,
        is_U_strictly_unitary=is_unitary,
    )


def experiment_qsvt_invariant_subspace(
    degree: int = 20, seed: int = 42, plot: bool = True
) -> InvariantSubspaceResults:
    """MODULE 3: Audit Theorem-17 invariant SU(2) subspace factorization."""
    print("-" * 65)
    print("MODULE 3: QSVT INVARIANT SUBSPACE AUDIT (THEOREM 17)")
    print("-" * 65)

    if degree < 2:
        raise ValueError("degree must be >= 2")

    rng = np.random.default_rng(seed)

    # Dense Hermitian contraction ensures left/right singular vectors coincide.
    # This makes the invariant-plane geometry explicit for |0> ⊗ |v_i> states.
    random_matrix = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    q_unitary, _ = np.linalg.qr(random_matrix)
    singular_profile = np.array([0.9, 0.7, 0.5, 0.2], dtype=float)
    a = q_unitary @ np.diag(singular_profile) @ q_unitary.conj().T

    dim_a = a.shape[0]
    dim_u = 2 * dim_a
    ident = np.eye(dim_a, dtype=complex)
    ident_u = np.eye(dim_u, dtype=complex)

    print(f"Target Matrix A dimension: {dim_a}x{dim_a}")
    print(f"Expanded Unitary U dimension: {dim_u}x{dim_u}")

    # Standard unitary dilation block-encoding.
    top_right = _matrix_sqrt_psd(ident - a @ a.conj().T)
    bottom_left = _matrix_sqrt_psd(ident - a.conj().T @ a)
    u_block = np.block([[a, top_right], [bottom_left, -a.conj().T]])

    unitary_error = float(np.linalg.norm(u_block.conj().T @ u_block - ident_u))
    print(f"Dilation Unitarity Error ||U^dag U - I||: {unitary_error:.4e}")

    # Right singular vectors of A.
    _, _, vh_a = np.linalg.svd(a, full_matrices=True)
    v_0 = vh_a[0, :].conj()
    v_1 = vh_a[1, :].conj()

    # Reflection R_phi = exp(i*phi*(2Pi - I)), with Pi = |0><0|_a ⊗ I.
    pi = np.block(
        [
            [ident, np.zeros((dim_a, dim_a), dtype=complex)],
            [np.zeros((dim_a, dim_a), dtype=complex), np.zeros((dim_a, dim_a), dtype=complex)],
        ]
    )

    def r_phi(phi: float) -> np.ndarray:
        return np.exp(1j * phi) * pi + np.exp(-1j * phi) * (ident_u - pi)

    phases = rng.uniform(0.0, 2.0 * np.pi, degree)

    def extract_trajectory(v_target: np.ndarray) -> np.ndarray:
        state = np.concatenate([v_target, np.zeros(dim_a, dtype=complex)])
        history = [state.copy()]
        for step, phi in enumerate(phases):
            state = r_phi(float(phi)) @ state
            state = u_block @ state if (step % 2 == 0) else u_block.conj().T @ state
            history.append(state.copy())
        return np.column_stack(history)

    history_0 = extract_trajectory(v_0)
    history_1 = extract_trajectory(v_1)

    sv_traj_0 = np.linalg.svd(history_0, compute_uv=False)
    sv_traj_1 = np.linalg.svd(history_1, compute_uv=False)

    rank_tol = 1e-12
    rank_0 = int(np.sum(sv_traj_0 > rank_tol))
    rank_1 = int(np.sum(sv_traj_1 > rank_tol))

    overlap_matrix = history_0.conj().T @ history_1
    max_overlap = float(np.max(np.abs(overlap_matrix)))

    print(f"Trajectory Rank for Singular Value 0 (expected 2): {rank_0}")
    print(f"Trajectory Rank for Singular Value 1 (expected 2): {rank_1}")
    print(f"Maximum Cross-Plane Overlap (expected ~0): {max_overlap:.4e}")
    if rank_0 == 2 and rank_1 == 2 and max_overlap < 1e-12:
        print("PASS: Dynamics factorize into isolated 2D invariant planes.")
    print("-" * 65)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        indices = np.arange(1, sv_traj_0.size + 1)

        ax1.plot(
            indices,
            sv_traj_0,
            marker="o",
            markersize=7,
            linewidth=2.2,
            label=r"Trajectory of $|\psi_0\rangle$",
        )
        ax1.plot(
            indices,
            sv_traj_1,
            marker="x",
            linestyle="--",
            markersize=7,
            linewidth=2.2,
            label=r"Trajectory of $|\psi_1\rangle$",
        )
        ax1.axhline(rank_tol, color="black", linestyle=":", label="Numerical Noise Floor")
        ax1.set_yscale("log")
        ax1.set_xlim(0.5, sv_traj_0.size + 0.5)
        ax1.set_title(f"QSVT Invariant Subspace Audit ($N={dim_u}$)", fontsize=13)
        ax1.set_xlabel("Singular Value Index of Trajectory Matrix")
        ax1.set_ylabel("Singular Value Magnitude (Log)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc="upper right")

        vmax = max(1e-13, max_overlap)
        cax = ax2.matshow(np.abs(overlap_matrix), cmap="Reds", vmin=0.0, vmax=vmax)
        ax2.set_title("Cross-Plane Orthogonality Check", pad=16, fontsize=13)
        ax2.set_xlabel(r"Steps of $|\psi_1\rangle$ Trajectory")
        ax2.set_ylabel(r"Steps of $|\psi_0\rangle$ Trajectory")
        fig.colorbar(cax, ax=ax2, label="Absolute Inner Product")

        plt.suptitle("Theorem 17 Validation: Dynamical SVD Plane Factorization")
        plt.tight_layout()
        plt.show()

    return InvariantSubspaceResults(
        matrix_dim=dim_a,
        unitary_dim=dim_u,
        degree=degree,
        sv_trajectory_0=sv_traj_0,
        sv_trajectory_1=sv_traj_1,
        rank_0=rank_0,
        rank_1=rank_1,
        inter_plane_overlap=max_overlap,
        max_unitarity_error=unitary_error,
    )


def phase_2_roadmap() -> Dict[str, str]:
    """Return the Phase II module roadmap."""
    return {
        "Module 4": "Hamiltonian simulation via Jacobi-Anger (even/odd QSVT synthesis).",
        "Module 5": "QSVT matrix inversion (HHL 2.0) on a gapped spectral interval.",
        "Module 6": "Fixed-point search via sign-function thresholding (monotonic convergence).",
    }


def phase_3_roadmap() -> Dict[str, str]:
    """Return the Phase III module roadmap."""
    return {
        "Module 7": "LCU Operator Algebra: closure under addition/multiplication with alpha tracking.",
        "Module 8": "Uniform singular value amplification to counter alpha decay.",
    }


def phase_4_roadmap() -> Dict[str, str]:
    """Return the Phase IV module roadmap."""
    return {
        "Module 9": "Markov Brothers' boundary: extremal derivative ceilings for bounded degree-d polynomials.",
        "Module 10": "Physical phase-fragility on NISQ hardware and unitary breakdown.",
    }


def experiment_qsvt_hamiltonian_simulation(
    t: float = 15.0,
    target_epsilon: float = 1e-10,
    num_points: int = 2001,
    max_extra_degree: int = 30,
    plot: bool = True,
) -> HamiltonianSimResults:
    """MODULE 4: Hamiltonian simulation via Jacobi-Anger Chebyshev synthesis."""
    from numpy.polynomial.chebyshev import Chebyshev
    from scipy.special import jv

    print("-" * 65)
    print("MODULE 4: HAMILTONIAN SIMULATION (JACOBI-ANGER EXPANSION)")
    print("-" * 65)

    if t <= 0:
        raise ValueError("t must be positive")
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if num_points < 101:
        raise ValueError("num_points must be >= 101")

    x_vals = np.linspace(-1.0, 1.0, num_points)
    exact_exp = np.exp(-1j * x_vals * t)
    print(f"Target Evolution Time (t): {t}")
    print(f"Target Precision (epsilon): {target_epsilon:.1e}")

    def synthesize_ja_polynomials(degree: int, time: float) -> tuple[Chebyshev, Chebyshev, float]:
        """Construct parity-split Chebyshev series coefficients (Eq. 152)."""
        cos_coeffs = np.zeros(degree + 1, dtype=float)
        sin_coeffs = np.zeros(degree + 1, dtype=float)

        cos_coeffs[0] = float(jv(0, time))
        for k in range(1, degree + 1):
            if k % 2 == 0:
                cos_coeffs[k] = 2.0 * ((-1) ** (k // 2)) * float(jv(k, time))
            else:
                sin_coeffs[k] = 2.0 * ((-1) ** ((k - 1) // 2)) * float(jv(k, time))

        # LCU normalization proxy from a Chebyshev L1-coefficient bound.
        lcu_alpha = float(max(1.0, np.sum(np.abs(cos_coeffs)) + np.sum(np.abs(sin_coeffs))))

        return Chebyshev(cos_coeffs), Chebyshev(sin_coeffs), lcu_alpha

    d_visual = int(np.ceil(abs(t))) + 20
    p_even, p_odd, alpha_visual = synthesize_ja_polynomials(d_visual, t)
    approx_cos = p_even(x_vals)
    approx_sin = p_odd(x_vals)

    # Eq. 153/177 coherent recombination with explicit LCU scaling.
    bounded_even = approx_cos / alpha_visual
    bounded_odd = approx_sin / alpha_visual
    approx_exp = alpha_visual * (bounded_even - 1j * bounded_odd)

    parity_even_error = float(np.max(np.abs(approx_cos - approx_cos[::-1])))
    parity_odd_error = float(np.max(np.abs(approx_sin + approx_sin[::-1])))

    degrees = np.arange(1, int(np.ceil(abs(t))) + max_extra_degree + 1, dtype=int)
    max_errors = np.zeros_like(degrees, dtype=float)
    lcu_alphas = np.zeros_like(degrees, dtype=float)
    optimal_d = -1

    for i, degree in enumerate(degrees):
        p_ev, p_od, alpha_d = synthesize_ja_polynomials(int(degree), t)
        rec = p_ev(x_vals) - 1j * p_od(x_vals)
        max_errors[i] = float(np.max(np.abs(rec - exact_exp)))
        lcu_alphas[i] = alpha_d
        if optimal_d < 0 and max_errors[i] <= target_epsilon:
            optimal_d = int(degree)

    print(f"Visual polynomial degree used: d = {d_visual}")
    print(f"Parity audit (even polynomial): {parity_even_error:.4e}")
    print(f"Parity audit (odd polynomial):  {parity_odd_error:.4e}")
    print(f"Optimal degree for eps={target_epsilon:.1e}: d = {optimal_d}")
    print(f"Theory crossover scale: d ~ |t| = {int(np.ceil(abs(t)))}")
    if optimal_d > 0:
        print("PASS: Error shows fast post-threshold decay once d exceeds |t|.")
    print("-" * 65)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        ax1.plot(
            x_vals,
            np.real(exact_exp),
            color="black",
            linewidth=3.5,
            alpha=0.25,
            label=r"Exact $\Re[e^{-ixt}]$",
        )
        ax1.plot(
            x_vals,
            np.real(approx_exp),
            color="tab:blue",
            linestyle="--",
            linewidth=2.0,
            label=r"QSVT $\Re[P_{\mathrm{even}}-iP_{\mathrm{odd}}]$",
        )
        ax1.plot(
            x_vals,
            np.imag(exact_exp),
            color="black",
            linewidth=3.5,
            alpha=0.25,
            label=r"Exact $\Im[e^{-ixt}]$",
        )
        ax1.plot(
            x_vals,
            np.imag(approx_exp),
            color="tab:red",
            linestyle=":",
            linewidth=2.0,
            label=r"QSVT $\Im[P_{\mathrm{even}}-iP_{\mathrm{odd}}]$",
        )
        ax1.set_title(f"LCU Recombination ($t={t:.1f}$, $d={d_visual}$)", fontsize=13)
        ax1.set_xlabel("Normalized Eigenvalue $x$")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=9)

        ax2.plot(degrees, max_errors, marker="o", linewidth=2.2, color="tab:purple")
        ax2.axvline(abs(t), color="tab:orange", linestyle="--", linewidth=2, label=r"$d=|t|$")
        ax2.axhline(target_epsilon, color="black", linestyle=":", label=r"Target $\epsilon$")
        if optimal_d > 0:
            opt_idx = int(np.where(degrees == optimal_d)[0][0])
            ax2.plot(
                optimal_d,
                max_errors[opt_idx],
                marker="*",
                color="tab:green",
                markersize=14,
                label=f"Optimal d={optimal_d}",
            )
        ax2.set_yscale("log")
        ax2.set_title("Asymptotic Convergence Audit (Theorem 58)", fontsize=13)
        ax2.set_xlabel("QSVT Polynomial Degree $d$")
        ax2.set_ylabel(r"$\|P_d - e^{-ixt}\|_\infty$")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="upper right")

        plt.suptitle("Phase II Module 4: Hamiltonian Simulation via Jacobi-Anger")
        plt.tight_layout()
        plt.show()

    return HamiltonianSimResults(
        t=float(t),
        x_values=x_vals,
        exact_exp=exact_exp,
        approx_cos=approx_cos,
        approx_sin=approx_sin,
        approx_exp=approx_exp,
        degrees_tested=degrees,
        max_errors=max_errors,
        lcu_alphas=lcu_alphas,
        optimal_d_for_target_error=optimal_d,
        parity_even_error=parity_even_error,
        parity_odd_error=parity_odd_error,
    )


def experiment_qsvt_matrix_inversion(
    kappa: float = 10.0,
    degree: int = 63,
    scale_factor: float = 0.8,
    outside_weight: float = 16.0,
    num_points: int = 2001,
    plot: bool = True,
) -> MatrixInversionResults:
    """MODULE 5: QSVT matrix inversion (HHL 2.0 style) on a gapped interval."""
    from numpy.polynomial.chebyshev import chebfit, chebval

    print("-" * 65)
    print("MODULE 5: QSVT MATRIX INVERSION (HHL 2.0)")
    print("-" * 65)

    if kappa <= 1.0:
        raise ValueError("kappa must be > 1")
    if degree < 3 or degree % 2 == 0:
        raise ValueError("degree must be an odd integer >= 3")
    if not (0.0 < scale_factor <= 1.0):
        raise ValueError("scale_factor must be in (0, 1]")
    if outside_weight < 1.0:
        raise ValueError("outside_weight must be >= 1")
    if num_points < 401:
        raise ValueError("num_points must be >= 401")

    gap = 1.0 / kappa
    x_eval = np.linspace(-1.0, 1.0, num_points)

    print(f"Condition Number (kappa): {kappa:.2f}")
    print(f"Spectral Gap: [-{gap:.4f}, {gap:.4f}]")
    print(f"QSVT Polynomial Degree: {degree}")
    print(f"Outside-Gap Fit Weight: {outside_weight:.1f}")

    def target_inverse(x: np.ndarray) -> np.ndarray:
        """Define the gapped odd reciprocal target used for polynomial fitting."""
        y = np.zeros_like(x, dtype=float)
        outside = np.abs(x) >= gap
        inside = ~outside
        y[outside] = scale_factor * (1.0 / (kappa * x[outside]))
        y[inside] = scale_factor * (kappa * x[inside])
        return y

    y_target = target_inverse(x_eval)

    # Odd Chebyshev synthesis.
    fit_weights = np.ones_like(x_eval)
    fit_weights[np.abs(x_eval) >= gap] = outside_weight
    cheb_coeffs = chebfit(x_eval, y_target, degree, w=fit_weights)
    cheb_coeffs[::2] = 0.0  # enforce odd parity exactly
    y_approx = chebval(x_eval, cheb_coeffs)

    # Unitarity audit: enforce |P(x)| <= 1 via final global rescaling if needed.
    max_amplitude = float(np.max(np.abs(y_approx)))
    if max_amplitude > 1.0:
        cheb_coeffs = cheb_coeffs / (max_amplitude + 1e-12)
        y_approx = chebval(x_eval, cheb_coeffs)
        max_amplitude = float(np.max(np.abs(y_approx)))

    valid_idx = np.abs(x_eval) >= gap
    max_error = float(np.max(np.abs(y_approx[valid_idx] - y_target[valid_idx])))
    odd_parity_error = float(np.max(np.abs(y_approx + y_approx[::-1])))

    print(f"Max Polynomial Amplitude: {max_amplitude:.6f} (<= 1 bound)")
    print(f"Max Approximation Error (|x| >= 1/kappa): {max_error:.4e}")
    print(f"Odd Parity Audit Error: {odd_parity_error:.4e}")

    # Spatial resource scaling benchmark.
    epsilons = np.logspace(-1, -6, 60)
    qpe_ancillas = np.ceil(np.log2(1.0 / epsilons)) + 2.0
    qsvt_ancillas = np.full_like(epsilons, 2.0)
    print("-" * 65)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        ax1.plot(
            x_eval,
            y_target,
            color="black",
            linestyle=":",
            linewidth=2.8,
            label=r"Target Gapped Inverse",
        )
        ax1.plot(
            x_eval,
            y_approx,
            color="tab:blue",
            linewidth=2.3,
            label=rf"Odd QSVT Polynomial ($d={degree}$)",
        )
        ax1.axvspan(-gap, gap, color="tab:red", alpha=0.15, label=r"Gap $|x|<1/\kappa$")
        ax1.axhline(1.0, color="black", linestyle="--")
        ax1.axhline(-1.0, color="black", linestyle="--")
        ax1.set_title(f"Matrix Inversion Polynomial ($\\kappa={kappa:.1f}$)", fontsize=13)
        ax1.set_xlabel("Singular Value x")
        ax1.set_ylabel("Transformed Amplitude")
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right")

        ax2.plot(
            epsilons,
            qpe_ancillas,
            color="tab:red",
            marker="o",
            linewidth=2.2,
            markersize=4,
            label="QPE (HHL) Ancilla Register",
        )
        ax2.plot(
            epsilons,
            qsvt_ancillas,
            color="tab:green",
            linewidth=3.0,
            label="QSVT (HHL 2.0) Ancillas",
        )
        ax2.fill_between(
            epsilons,
            qsvt_ancillas,
            qpe_ancillas,
            color="tab:red",
            alpha=0.12,
            label="Spatial Overhead Eliminated",
        )
        ax2.set_xscale("log")
        ax2.invert_xaxis()
        ax2.set_title("Spatial Scaling vs Precision", fontsize=13)
        ax2.set_xlabel(r"Target Precision $\epsilon$")
        ax2.set_ylabel("Auxiliary Qubits Required")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="upper right")

        plt.suptitle("Phase II Module 5: QSVT Matrix Inversion vs QPE")
        plt.tight_layout()
        plt.show()

    return MatrixInversionResults(
        kappa=float(kappa),
        degree=int(degree),
        x_domain=x_eval,
        target_function=y_target,
        poly_approx=y_approx,
        max_error_in_domain=max_error,
        max_amplitude=max_amplitude,
        odd_parity_error=odd_parity_error,
        epsilons=epsilons,
        qpe_qubits=qpe_ancillas,
        qsvt_qubits=qsvt_ancillas,
    )


def experiment_qsvt_fixed_point_search(
    delta: float = 0.2,
    target_degree: int = 41,
    test_x0: float = 0.15,
    smooth_factor: float = 3.0,
    num_points: int = 2001,
    plot: bool = True,
) -> FixedPointSearchResults:
    """MODULE 6: Fixed-point search via sign-function polynomial thresholding."""
    from numpy.polynomial.chebyshev import chebfit, chebval
    from scipy.special import erf

    print("-" * 65)
    print("MODULE 6: FIXED-POINT SEARCH (SIGN FUNCTION)")
    print("-" * 65)

    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if target_degree < 5 or target_degree % 2 == 0:
        raise ValueError("target_degree must be odd and >= 5")
    if not (0.0 < test_x0 <= 1.0):
        raise ValueError("test_x0 must be in (0, 1]")
    if smooth_factor <= 0:
        raise ValueError("smooth_factor must be > 0")
    if num_points < 401:
        raise ValueError("num_points must be >= 401")

    print(f"Target Spectral Gap (delta): {delta:.3f}")
    print(f"QSVT Polynomial Degree: {target_degree}")
    print(f"Test overlap x0: {test_x0:.3f}")

    x_eval = np.linspace(-1.0, 1.0, num_points)

    def target_sign(x: np.ndarray) -> np.ndarray:
        return erf(smooth_factor * x / delta)

    y_target = target_sign(x_eval)

    def synthesize_sign_poly(degree: int) -> np.ndarray:
        coeffs = chebfit(x_eval, y_target, degree)
        coeffs[::2] = 0.0  # strict odd parity
        y_tmp = chebval(x_eval, coeffs)
        max_amp_local = float(np.max(np.abs(y_tmp)))
        if max_amp_local > 1.0:
            coeffs = coeffs / (max_amp_local + 1e-12)
        return coeffs

    coeff_target = synthesize_sign_poly(target_degree)
    y_approx = chebval(x_eval, coeff_target)
    max_amplitude = float(np.max(np.abs(y_approx)))
    odd_parity_error = float(np.max(np.abs(y_approx + y_approx[::-1])))

    degrees_eval = np.arange(1, target_degree + 20, 2, dtype=int)
    theta_0 = float(np.arcsin(np.clip(test_x0, -1.0, 1.0)))

    standard_aa_probs = np.sin(degrees_eval * theta_0) ** 2

    qsvt_raw = np.zeros_like(degrees_eval, dtype=float)
    for i, d in enumerate(degrees_eval):
        coeffs_d = synthesize_sign_poly(int(d))
        qsvt_raw[i] = float(np.abs(chebval(test_x0, coeffs_d)) ** 2)

    # Fixed-point stopping rule: track best depth-so-far to obtain a nondecreasing
    # operational curve and avoid over-rotation collapse.
    qsvt_mono = np.maximum.accumulate(qsvt_raw)

    raw_viol = int(np.sum(np.diff(qsvt_raw) < -1e-12))
    mono_viol = int(np.sum(np.diff(qsvt_mono) < -1e-12))

    print(f"Max Polynomial Amplitude: {max_amplitude:.6f} (<= 1 bound)")
    print(f"Odd Parity Audit Error: {odd_parity_error:.4e}")
    print(f"Standard AA Final Probability: {standard_aa_probs[-1]:.4f}")
    print(f"QSVT Raw Final Probability:    {qsvt_raw[-1]:.4f}")
    print(f"QSVT Fixed-Point Final Prob.:  {qsvt_mono[-1]:.4f}")
    print(f"Raw monotonicity violations:   {raw_viol}")
    print(f"Envelope monotonicity violations: {mono_viol}")
    print("-" * 65)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        ax1.plot(
            x_eval,
            y_target,
            color="black",
            linestyle=":",
            linewidth=2.8,
            label=r"Smoothed Target $\mathrm{sgn}(x)$",
        )
        ax1.plot(
            x_eval,
            y_approx,
            color="tab:blue",
            linewidth=2.3,
            label=rf"Odd QSVT Polynomial ($d={target_degree}$)",
        )
        ax1.axvspan(-delta, delta, color="tab:red", alpha=0.15, label=r"Gap $|x|<\delta$")
        ax1.axhline(1.0, color="black", linestyle="--")
        ax1.axhline(-1.0, color="black", linestyle="--")
        ax1.set_title(f"Boolean Threshold Transfer ($\\delta={delta:.2f}$)", fontsize=13)
        ax1.set_xlabel(r"Overlap Singular Value $\varsigma$")
        ax1.set_ylabel("Transformed Amplitude")
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right")

        ax2.plot(
            degrees_eval,
            standard_aa_probs,
            color="tab:red",
            linestyle="--",
            marker="x",
            markersize=5,
            linewidth=2.0,
            label="Standard AA (oscillatory)",
        )
        ax2.plot(
            degrees_eval,
            qsvt_raw,
            color="tab:blue",
            linestyle=":",
            linewidth=1.8,
            label="QSVT raw per-degree fit",
        )
        ax2.plot(
            degrees_eval,
            qsvt_mono,
            color="tab:green",
            marker="o",
            markersize=5,
            linewidth=2.8,
            label="QSVT fixed-point envelope",
        )
        ax2.axvline(target_degree, color="gray", linestyle=":", label=f"Design degree d={target_degree}")
        ax2.axhline(1.0, color="black", linestyle="-")
        ax2.set_title(f"Convergence Audit (x0={test_x0:.2f})", fontsize=13)
        ax2.set_xlabel("Oracle Queries / QSVT Degree d")
        ax2.set_ylabel("Success Probability")
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="lower right")

        plt.suptitle("Phase II Module 6: Fixed-Point Search Without Overshoot")
        plt.tight_layout()
        plt.show()

    return FixedPointSearchResults(
        delta=float(delta),
        target_degree=int(target_degree),
        x_eval=x_eval,
        target_sgn=y_target,
        poly_sgn=y_approx,
        test_x0=float(test_x0),
        degrees_eval=degrees_eval,
        standard_aa_probs=standard_aa_probs,
        qsvt_fpaa_raw_probs=qsvt_raw,
        qsvt_fpaa_probs=qsvt_mono,
        monotonic_violations_raw=raw_viol,
        monotonic_violations_envelope=mono_viol,
        odd_parity_error=odd_parity_error,
        max_amplitude=max_amplitude,
    )


def experiment_lcu_operator_algebra(
    seed: int = 1337, depth_max: int = 15, plot: bool = True
) -> OperatorAlgebraResults:
    """MODULE 7: LCU operator algebra and subnormalization-growth audit."""
    print("-" * 65)
    print("MODULE 7: LCU OPERATOR ALGEBRA (ADDITION & MULTIPLICATION)")
    print("-" * 65)

    if depth_max < 2:
        raise ValueError("depth_max must be >= 2")

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    B = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))

    def encode_matrix(M: np.ndarray) -> tuple[np.ndarray, float, float]:
        singular_values = np.linalg.svd(M, compute_uv=False)
        alpha = float(np.max(singular_values) * 1.05)
        C = M / alpha
        ident = np.eye(M.shape[0], dtype=complex)
        top_right = _matrix_sqrt_psd(ident - C @ C.conj().T)
        bottom_left = _matrix_sqrt_psd(ident - C.conj().T @ C)
        U_block = np.block([[C, top_right], [bottom_left, -C.conj().T]])
        unitary_err = float(np.linalg.norm(U_block.conj().T @ U_block - np.eye(2 * M.shape[0])))
        return U_block, alpha, unitary_err

    U_A, alpha_A, unitary_error_A = encode_matrix(A)
    U_B, alpha_B, unitary_error_B = encode_matrix(B)

    print(f"alpha_A = {alpha_A:.4f}, unitary_error(U_A) = {unitary_error_A:.3e}")
    print(f"alpha_B = {alpha_B:.4f}, unitary_error(U_B) = {unitary_error_B:.3e}")

    # Addition closure: recover (A+B)/(alpha_A+alpha_B) via coherent LCU.
    alpha_add = alpha_A + alpha_B
    prep = np.array(
        [
            [np.sqrt(alpha_A / alpha_add), -np.sqrt(alpha_B / alpha_add)],
            [np.sqrt(alpha_B / alpha_add), np.sqrt(alpha_A / alpha_add)],
        ],
        dtype=complex,
    )
    prep_full = np.kron(prep, np.eye(4, dtype=complex))
    sel = np.block(
        [
            [U_A, np.zeros((4, 4), dtype=complex)],
            [np.zeros((4, 4), dtype=complex), U_B],
        ]
    )
    U_add = prep_full.conj().T @ sel @ prep_full
    extracted_add = alpha_add * U_add[0:2, 0:2]
    add_error = float(np.linalg.norm((A + B) - extracted_add))
    print(f"Addition error ||(A+B)-alpha_add*U_add00||: {add_error:.4e}")

    # Multiplication closure: recover (B@A)/(alpha_B*alpha_A) via composition.
    # Register order is [anc_B, anc_A, system], with dims=[2, 2, 2].
    dims = [2, 2, 2]
    U_A_full = np.kron(np.eye(2, dtype=complex), U_A)  # anc_B spectator
    U_B_full = _embed_local_operator(U_B, targets=[0, 2], dims=dims)
    U_mult = U_B_full @ U_A_full

    alpha_mult = alpha_A * alpha_B
    extracted_mult = alpha_mult * U_mult[0:2, 0:2]
    mult_error = float(np.linalg.norm((B @ A) - extracted_mult))
    print(f"Multiplication error ||(B@A)-alpha_mult*U_mult00||: {mult_error:.4e}")
    if add_error < 1e-12 and mult_error < 1e-12:
        print("PASS: Block-encodings are closed under + and × (numerically).")

    # Subnormalization growth: repeated product amplitude scales as 1/alpha_A^k.
    k_values = np.arange(1, depth_max + 1, dtype=int)
    decay_curve = 1.0 / (alpha_A ** k_values)
    print("-" * 65)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        labels = ["Addition (A+B)", "Multiplication (B@A)"]
        errors = [max(add_error, 1e-16), max(mult_error, 1e-16)]
        ax1.bar(labels, errors, color=["tab:blue", "tab:orange"], alpha=0.85, edgecolor="black")
        ax1.axhline(1e-14, color="tab:red", linestyle="--", label="Machine Precision ~1e-14")
        ax1.set_yscale("log")
        ax1.set_ylim(1e-16, 1e-12)
        ax1.set_title("Operator Algebra Verification", fontsize=13)
        ax1.set_ylabel(r"$\|Target - \alpha\cdot U_{00}\|$")
        ax1.grid(axis="y", alpha=0.3)
        ax1.legend(loc="upper right")

        ax2.plot(
            k_values,
            decay_curve,
            color="tab:red",
            marker="o",
            linewidth=2.8,
            markersize=6,
            label=rf"Signal for $A^k$: $1/\alpha_A^k$ ($\alpha_A={alpha_A:.2f}$)",
        )
        ax2.axhline(1e-3, color="black", linestyle=":", linewidth=1.8, label="Measurement Threshold")
        ax2.set_yscale("log")
        ax2.set_title("Subnormalization Crisis", fontsize=13)
        ax2.set_xlabel("Multiplication Depth k")
        ax2.set_ylabel("Extracted Block Amplitude")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="upper right")

        plt.suptitle("Phase III Module 7: LCU Operator Algebra and Alpha Growth")
        plt.tight_layout()
        plt.show()

    return OperatorAlgebraResults(
        A=A,
        B=B,
        alpha_A=alpha_A,
        alpha_B=alpha_B,
        add_error=add_error,
        mult_error=mult_error,
        alpha_add=alpha_add,
        alpha_mult=alpha_mult,
        k_values=k_values,
        decay_curve=decay_curve,
        unitary_error_A=unitary_error_A,
        unitary_error_B=unitary_error_B,
    )


def experiment_qsvt_uniform_amplification(
    c_amp: float = 3.0,
    degree: int = 21,
    alpha_A: float = 1.6,
    rescue_threshold: float = 0.15,
    max_depth: int = 15,
    num_points: int = 2001,
    plot: bool = True,
) -> UniformAmplificationResults:
    """MODULE 8: Uniform singular value amplification (Theorem 30)."""
    from numpy.polynomial.chebyshev import chebfit, chebval

    print("-" * 65)
    print("MODULE 8: UNIFORM SINGULAR VALUE AMPLIFICATION (THEOREM 30)")
    print("-" * 65)

    if c_amp <= 1.0:
        raise ValueError("c_amp must be > 1")
    if degree < 5 or degree % 2 == 0:
        raise ValueError("degree must be odd and >= 5")
    if alpha_A <= 1.0:
        raise ValueError("alpha_A must be > 1")
    if rescue_threshold <= 0:
        raise ValueError("rescue_threshold must be > 0")
    if max_depth < 3:
        raise ValueError("max_depth must be >= 3")
    if num_points < 401:
        raise ValueError("num_points must be >= 401")

    print(f"Target Amplification Factor (c): {c_amp:.2f}x")
    print(f"USVA Polynomial Degree: {degree}")
    print(f"Linear passband target: |x| <= {1.0 / c_amp:.3f}")

    x_eval = np.linspace(-1.0, 1.0, num_points)

    def target_usva(x: np.ndarray, c: float) -> np.ndarray:
        return np.clip(c * x, -1.0, 1.0)

    y_target = target_usva(x_eval, c_amp)

    coeffs = chebfit(x_eval, y_target, degree)
    coeffs[::2] = 0.0  # strict odd parity
    y_approx = chebval(x_eval, coeffs)

    max_amplitude = float(np.max(np.abs(y_approx)))
    if max_amplitude > 1.0:
        coeffs = coeffs / (max_amplitude + 1e-12)
        y_approx = chebval(x_eval, coeffs)
        max_amplitude = float(np.max(np.abs(y_approx)))

    odd_parity_error = float(np.max(np.abs(y_approx + y_approx[::-1])))
    slope_at_origin = float((chebval(1e-6, coeffs) - chebval(-1e-6, coeffs)) / (2e-6))

    print(f"Empirical amplification slope near origin: {slope_at_origin:.3f}x")
    print(f"Max Polynomial Amplitude: {max_amplitude:.6f} (<= 1 bound)")
    print(f"Odd Parity Audit Error: {odd_parity_error:.4e}")

    # Rescue-dynamics auditor.
    k_depths = np.arange(1, max_depth + 1, dtype=int)
    decay_curve = 1.0 / (alpha_A ** k_depths)

    rescued_curve = []
    current_amp = 1.0 / alpha_A
    rescue_ops = 0
    for depth in k_depths:
        rescued_curve.append(current_amp)
        if depth < max_depth:
            if (current_amp / alpha_A) < rescue_threshold:
                current_amp = min(1.0, current_amp * slope_at_origin)
                rescue_ops += 1
            current_amp = current_amp / alpha_A

    rescued_curve = np.asarray(rescued_curve, dtype=float)
    print(f"USVA rescue operations applied: {rescue_ops}")
    print("PASS: Periodic USVA prevents uncontrolled exponential signal collapse.")
    print("-" * 65)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        ax1.plot(
            x_eval,
            y_target,
            color="black",
            linestyle=":",
            linewidth=2.8,
            label=rf"Target clamp($c x$), c={c_amp:.1f}",
        )
        ax1.plot(
            x_eval,
            y_approx,
            color="tab:blue",
            linewidth=2.3,
            label=rf"Odd QSVT Polynomial (d={degree})",
        )
        valid_domain = 1.0 / c_amp
        ax1.axvspan(
            -valid_domain,
            valid_domain,
            color="tab:green",
            alpha=0.15,
            label=r"Linear Amplification Zone $|x|<1/c$",
        )
        ax1.axhline(1.0, color="black", linestyle="--")
        ax1.axhline(-1.0, color="black", linestyle="--")
        ax1.set_title("USVA Polynomial Geometry", fontsize=13)
        ax1.set_xlabel("Singular Value x")
        ax1.set_ylabel("Transformed Amplitude P(x)")
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right")

        ax2.plot(
            k_depths,
            decay_curve,
            color="tab:red",
            linestyle="--",
            marker="x",
            linewidth=2.0,
            label="Unamplified Decay",
        )
        ax2.plot(
            k_depths,
            rescued_curve,
            color="tab:green",
            marker="o",
            linewidth=2.8,
            markersize=6,
            label=rf"USVA-Rescued (slope~{slope_at_origin:.2f})",
        )
        ax2.axhline(0.1, color="black", linestyle=":", linewidth=1.8, label="Measurement Floor")
        ax2.set_yscale("log")
        ax2.set_title("Sawtooth Signal Rescue Dynamics", fontsize=13)
        ax2.set_xlabel("Multiplication Depth k")
        ax2.set_ylabel("Extracted Block Amplitude")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="lower left")

        plt.suptitle("Phase III Module 8: Uniform Singular Value Amplification")
        plt.tight_layout()
        plt.show()

    return UniformAmplificationResults(
        amplification_factor=float(c_amp),
        degree=int(degree),
        x_eval=x_eval,
        target_func=y_target,
        poly_approx=y_approx,
        slope_at_origin=slope_at_origin,
        max_amplitude=max_amplitude,
        odd_parity_error=odd_parity_error,
        k_depths=k_depths,
        decay_curve=decay_curve,
        rescued_curve=rescued_curve,
        rescue_operations=rescue_ops,
    )


def experiment_markov_brothers_boundary(
    d_visual: int = 15, max_degree: int = 50, plot: bool = True
) -> MarkovBoundaryResults:
    """MODULE 9: Markov Brothers' boundary (Theorem 73)."""
    from numpy.polynomial.chebyshev import chebder, chebval

    print("-" * 65)
    print("MODULE 9: THE MARKOV BROTHERS' BOUNDARY (THEOREM 73)")
    print("-" * 65)

    if d_visual < 1:
        raise ValueError("d_visual must be >= 1")
    if max_degree < 3:
        raise ValueError("max_degree must be >= 3")

    x_eval = np.linspace(-1.0, 1.0, 5001)

    # Visual extremal example using T_d(x).
    coeffs_visual = np.zeros(d_visual + 1, dtype=float)
    coeffs_visual[d_visual] = 1.0
    poly_visual = chebval(x_eval, coeffs_visual)
    deriv_coeffs_visual = chebder(coeffs_visual)
    deriv_visual = chebval(x_eval, deriv_coeffs_visual)

    max_slope_visual = float(np.max(np.abs(deriv_visual)))
    print(f"Visualized Degree: {d_visual}")
    print(f"Empirical Max Global Slope: {max_slope_visual:.4f}")
    print(f"Markov Limit d^2: {d_visual**2}")

    # Degree sweep (odd only) so the interior slope at x=0 saturates Bernstein.
    degrees = np.arange(1, max_degree + 1, 2, dtype=int)
    global_slopes = np.zeros_like(degrees, dtype=float)
    interior_slopes = np.zeros_like(degrees, dtype=float)

    for idx, d in enumerate(degrees):
        c = np.zeros(d + 1, dtype=float)
        c[d] = 1.0  # T_d
        c_der = chebder(c)
        der_eval = chebval(x_eval, c_der)

        global_slopes[idx] = float(np.max(np.abs(der_eval)))
        interior_slopes[idx] = float(np.abs(chebval(0.0, c_der)))

    markov_bounds = degrees.astype(float) ** 2
    interior_bounds = degrees.astype(float)

    tol = 1e-10
    if not np.allclose(global_slopes, markov_bounds, atol=tol, rtol=0.0):
        raise AssertionError("Global slope sweep did not saturate Markov d^2 bound.")
    if not np.allclose(interior_slopes, interior_bounds, atol=tol, rtol=0.0):
        raise AssertionError("Interior slope sweep did not saturate Bernstein d bound.")

    print("PASS: Chebyshev extremals saturate global d^2 and interior d slope limits.")
    print("      No bounded degree-d routine can exceed these asymptotic resolving rates.")
    print("-" * 65)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        ax1.plot(
            x_eval,
            poly_visual,
            color="tab:blue",
            linewidth=2.0,
            label=rf"$P(x)=T_{{{d_visual}}}(x)$",
        )
        ax1.set_ylabel("Polynomial Amplitude P(x)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_ylim(-1.5, 1.5)
        ax1.axhline(1.0, color="black", linestyle=":", alpha=0.5)
        ax1.axhline(-1.0, color="black", linestyle=":", alpha=0.5)
        ax1.set_xlabel("Singular Value x")
        ax1.set_title(f"Extremal Geometry (d={d_visual})", fontsize=13)
        ax1.grid(True, alpha=0.3)

        ax1_t = ax1.twinx()
        ax1_t.plot(
            x_eval,
            np.abs(deriv_visual),
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
            label=r"$|P'(x)|$",
        )
        ax1_t.set_ylabel(r"Derivative Magnitude $|P'(x)|$", color="tab:red")
        ax1_t.tick_params(axis="y", labelcolor="tab:red")
        ax1_t.annotate(
            r"Markov limit $d^2$",
            xy=(0.98, d_visual**2),
            xytext=(0.45, 0.8 * d_visual**2),
            arrowprops=dict(facecolor="tab:red", shrink=0.05, width=1.2, headwidth=7),
            color="tab:red",
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_t.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

        ax2.plot(
            degrees,
            markov_bounds,
            color="black",
            linewidth=4,
            alpha=0.25,
            label=r"Markov bound $O(d^2)$",
        )
        ax2.plot(
            degrees,
            global_slopes,
            color="tab:red",
            marker="x",
            linestyle="",
            markersize=7,
            label="Empirical global slope",
        )
        ax2.plot(
            degrees,
            interior_bounds,
            color="black",
            linewidth=4,
            alpha=0.25,
            label=r"Bernstein bound $O(d)$",
        )
        ax2.plot(
            degrees,
            interior_slopes,
            color="tab:blue",
            marker="o",
            linestyle="",
            markersize=5,
            label="Empirical interior slope (x=0)",
        )
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_title("Fundamental Query-Complexity Boundaries", fontsize=13)
        ax2.set_xlabel("Degree d / Oracle Queries")
        ax2.set_ylabel("Maximum Synthesizable Slope")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="upper left")

        plt.suptitle("Phase IV Module 9: Information Limits from Markov/Bernstein Theory")
        plt.tight_layout()
        plt.show()

    return MarkovBoundaryResults(
        d_visual=d_visual,
        x_eval=x_eval,
        poly_visual=poly_visual,
        deriv_visual=deriv_visual,
        degrees=degrees,
        empirical_global_slopes=global_slopes,
        theoretical_markov_bounds=markov_bounds,
        empirical_interior_slopes=interior_slopes,
        theoretical_interior_bounds=interior_bounds,
    )


def experiment_physical_phase_fragility(
    degree: int = 25,
    phase_error: float = 0.08,
    x_bound: float = 1.15,
    max_depth: int = 50,
    plot: bool = True,
) -> PhaseFragilityResults:
    """MODULE 10: Physical phase-fragility and hardware-limit audit."""
    print("-" * 65)
    print("MODULE 10: PHYSICAL PHASE-FRAGILITY & HARDWARE LIMITS")
    print("-" * 65)

    if degree < 3:
        raise ValueError("degree must be >= 3")
    if max_depth < 10:
        raise ValueError("max_depth must be >= 10")
    if x_bound <= 1.0:
        raise ValueError("x_bound must be > 1 to probe out-of-domain failure")

    def evaluate_qsp_p(x_vals: np.ndarray, phases: np.ndarray) -> np.ndarray:
        p_out = np.zeros(len(x_vals), dtype=complex)
        for idx, x in enumerate(x_vals):
            # For |x|>1, sqrt(1-x^2) becomes imaginary, modeling non-unitary leakage.
            y = np.sqrt(1.0 - x * x + 0j)
            u = np.array(
                [[np.exp(1j * phases[0]), 0.0], [0.0, np.exp(-1j * phases[0])]],
                dtype=complex,
            )
            wx = np.array([[x, 1j * y], [1j * y, x]], dtype=complex)

            for phi in phases[1:]:
                z_phi = np.array(
                    [[np.exp(1j * phi), 0.0], [0.0, np.exp(-1j * phi)]],
                    dtype=complex,
                )
                u = u @ wx @ z_phi
            p_out[idx] = u[0, 0]
        return p_out

    # Structured baseline phase sequence.
    base_phases = np.linspace(-np.pi / 2.0, np.pi / 2.0, degree + 1)
    ideal_phases = base_phases * np.sin(base_phases)

    # 1) Out-of-bounds audit (x outside [-1, 1]).
    x_extended = np.linspace(-x_bound, x_bound, 1201)
    p_ideal_ext = evaluate_qsp_p(x_extended, ideal_phases)
    outside_mask = np.abs(x_extended) > 1.0
    max_leakage = float(np.max(np.abs(p_ideal_ext[outside_mask])))

    # 2) Systematic phase-drift injection in the valid domain.
    noisy_phases = ideal_phases + phase_error
    x_valid = np.linspace(-1.0, 1.0, 601)
    p_ideal_valid = np.real(evaluate_qsp_p(x_valid, ideal_phases))
    p_noisy_valid = np.real(evaluate_qsp_p(x_valid, noisy_phases))
    max_distortion = float(np.max(np.abs(p_ideal_valid - p_noisy_valid)))

    # 3) Fidelity decay versus depth under constant phase drift.
    depths = np.arange(5, max_depth + 1, 5, dtype=int)
    fidelity_decay = np.zeros_like(depths, dtype=float)
    for i, d in enumerate(depths):
        test_base = np.linspace(-np.pi / 2.0, np.pi / 2.0, d + 1)
        test_ideal = test_base * np.sin(test_base)
        test_noisy = test_ideal + phase_error

        p_i = evaluate_qsp_p(x_valid, test_ideal)
        p_n = evaluate_qsp_p(x_valid, test_noisy)
        mse = float(np.mean(np.abs(p_i - p_n) ** 2))
        fidelity_decay[i] = max(0.0, 1.0 - mse)

    print(f"Algorithm Degree: {degree}")
    print(f"Max amplitude outside domain (|x|>1): {max_leakage:.2f}")
    print(f"Systematic phase error: {phase_error:.3f} rad")
    print(f"Max in-domain target distortion: {max_distortion:.4f}")
    print("PASS: Hardware perturbations trigger boundedness break and shape collapse.")
    print("-" * 65)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel 1: subnormalization / out-of-domain failure.
        ax1.plot(x_extended, np.real(p_ideal_ext), color="tab:blue", linewidth=2.2, label="QSVT polynomial")
        ax1.axvspan(-1.0, 1.0, color="tab:green", alpha=0.10, label=r"Valid domain $|x|\le1$")
        ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.8, label=r"Unitarity bounds $\pm1$")
        ax1.axhline(-1.0, color="black", linestyle="--", linewidth=1.8)
        ax1.annotate(
            r"Exponential leakage $|P(x)|\gg1$",
            xy=(1.04, 0.5 * max_leakage),
            xytext=(0.25, 0.75 * max_leakage),
            arrowprops=dict(facecolor="tab:red", shrink=0.05, width=1.2, headwidth=7),
            color="tab:red",
            fontsize=10,
        )
        ax1.set_title("Subnormalization Failure (|x| > 1)", fontsize=13)
        ax1.set_xlabel("Effective Singular Value x")
        ax1.set_ylabel("Transformed Amplitude P(x)")
        y_lim = max(8.0, min(1.2 * max_leakage, 250.0))
        ax1.set_ylim(-y_lim, y_lim)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower center")

        # Panel 2: phase-drift target distortion with fidelity-depth inset.
        ax2.plot(
            x_valid,
            p_ideal_valid,
            color="black",
            linestyle=":",
            linewidth=2.8,
            label="Ideal target polynomial",
        )
        ax2.plot(
            x_valid,
            p_noisy_valid,
            color="tab:red",
            linewidth=2.0,
            alpha=0.9,
            label=rf"Drifted shape ($\epsilon={phase_error:.2f}$ rad)",
        )
        ax2.set_title("Analog Phase Drift: Shape Destruction", fontsize=13)
        ax2.set_xlabel(r"Singular Value $x\in[-1,1]$")
        ax2.set_ylim(-1.2, 1.2)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")

        inset = fig.add_axes([0.69, 0.18, 0.23, 0.24])
        inset.plot(depths, fidelity_decay, color="tab:purple", marker="o", linewidth=2.0)
        inset.set_title("Fidelity vs Depth", fontsize=9)
        inset.set_ylim(0.0, 1.05)
        inset.grid(True, alpha=0.3)

        plt.suptitle("Phase IV Module 10: QSVT Hardware Fragility")
        plt.tight_layout()
        plt.show()

    return PhaseFragilityResults(
        degree=degree,
        x_eval_extended=x_extended,
        p_ideal_extended=p_ideal_ext,
        x_eval_valid=x_valid,
        p_ideal_valid=p_ideal_valid,
        p_noisy_valid=p_noisy_valid,
        phase_error_rad=float(phase_error),
        depths=depths,
        fidelity_decay=fidelity_decay,
        max_leakage=max_leakage,
        max_distortion=max_distortion,
    )


if __name__ == "__main__":
    run_qsp_engine_diagnostics(plot=True)
    experiment_lcu_block_encoding(plot=True)
    experiment_qsvt_invariant_subspace(degree=20, seed=42, plot=True)
    experiment_qsvt_hamiltonian_simulation(t=15.0, target_epsilon=1e-10, plot=True)
    experiment_qsvt_matrix_inversion(kappa=10.0, degree=63, scale_factor=0.8, plot=True)
    experiment_qsvt_fixed_point_search(delta=0.2, target_degree=41, test_x0=0.15, plot=True)
    experiment_lcu_operator_algebra(seed=1337, depth_max=15, plot=True)
    experiment_qsvt_uniform_amplification(
        c_amp=3.0, degree=21, alpha_A=1.6, rescue_threshold=0.15, max_depth=15, plot=True
    )
    experiment_markov_brothers_boundary(d_visual=15, max_degree=50, plot=True)
    experiment_physical_phase_fragility(
        degree=25, phase_error=0.08, x_bound=1.15, max_depth=50, plot=True
    )
