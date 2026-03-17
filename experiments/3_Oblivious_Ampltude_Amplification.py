"""Simple implementation of oblivious amplitude amplification
using Qiskit.

Defines :class:`ObliviousAmplificationLab` which builds circuits and
runs basic tests for the block encoding, iteration operator, and fidelity
checks. Intended for use from a script or notebook.

Standing notation aligned with final.tex:
- H_Good / H_Bad: target and non-target sectors
- Pi_Good / Pi_Bad: corresponding projectors
- p = ||Pi_Good |All>||^2 is the success parameter
- sin^2(theta0)=p for the canonical two-dimensional geometry
- complexity is discussed in oracle/query calls to the OAA iterate Q
"""

from __future__ import annotations

import logging
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator, Statevector, random_unitary

from one_click_utils import start_one_click_session

# configure logger for informative output
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class AmplificationResults:
    """Container for results from the acid tests."""

    success_probabilities: np.ndarray
    msd: float
    fidelity_curve: np.ndarray


@dataclass
class SubnormalizationRescueResults:
    """Container for circuit-level OAA rescue experiment outputs."""

    p: float
    theta: float
    k_opt: int
    k_values: np.ndarray
    success_probabilities: np.ndarray


@dataclass
class GeometricObstructionResults:
    """Container for valid-vs-violated purified-setting comparison."""

    k_values: np.ndarray
    valid_fidelity: np.ndarray
    invalid_fidelity: np.ndarray
    valid_success_probability: np.ndarray
    invalid_success_probability: np.ndarray


@dataclass
class ExplicitLCUBlockEncodingResults:
    """Container for explicit LCU construction and resource metrics."""

    alpha: float
    matrix_distance: float
    native_gate_counts: dict[str, dict[str, int]]
    ft_t_counts: dict[str, int]
    ft_depths: dict[str, int]


@dataclass
class IdentityBlockAudit:
    """Numerical audit for the clean-ancilla block of M = A^\\dagger P1 A."""

    p_reference: float
    distance_to_p_identity: float
    max_off_diagonal: float
    diagonal_entries: np.ndarray
    m_tl: np.ndarray


@dataclass
class IdentityBlockExtractorResults:
    """Results for purified-setting proof and LCU cross-check."""

    purified: IdentityBlockAudit
    lcu: IdentityBlockAudit
    lcu_distance_to_h2_over_alpha2: float


class ObliviousAmplificationLab:
    """Implements the laboratory for oblivious amplitude amplification.

    The class is parameterised by ``m`` (number of data qubits) and ``l``
    (number of ancilla qubits used for the block encoding).  For the purposes
    of the paper we always use ``l = 1``; the code is written generically so
    that more ancilla can be added if necessary.

    The central objects are::

        U    : random m-qubit unitary
        p    : success probability parameter (a float in (0,1])
        B    : \\sqrt{p} U (the operator we wish to block encode)
        A    : unitary acting on m + l qubits with
               P_1 A P_1 = B (= top-left block) where
               P_1 = |0^l><0^l|_anc is the clean-ancilla projector.
        R_0  : reflection about the clean ancilla subspace
        R_b  : phase flip on the non-target ancilla sector (orthogonal to |0^l>)
        Q    : full iteration operator A R_0 A^{-1} R_b

    The public methods mirror these definitions.  In the comments below we
    refer repeatedly to Theorem 1 (``Geometric Core'') and the Equivalence
    Theorem; sections of the paper are marked with ``[G]'' or ``[E]''.
    """

    def __init__(self, m: int, l: int = 1, seed: Optional[int] = None):
        """Initialise the lab.

        Parameters
        ----------
        m : int
            Number of data qubits (dimension of the space on which ``U`` acts).
        l : int
            Number of ancilla qubits used for the block encoding.  We default to
            one because the proofs in the paper are easiest to follow in that
            case.
        seed : Optional[int]
            Random seed passed to :func:`qiskit.quantum_info.random_unitary`.
        """
        self.m = m
        self.l = l
        self.dim_data = 2 ** m
        self.dim_anc = 2 ** l
        self.dim_total = self.dim_anc * self.dim_data

        # generate the random unitary U used in the definition of B.  The
        # distribution is Haar random.  We cache both the matrix and the
        # corresponding gate so that it can be inserted into circuits.
        self.U = random_unitary(self.dim_data, seed=seed).data
        self.U_gate = Operator(self.U).data  # just a convenient wrapper

        # choose a success probability p and compute B = sqrt(p) U.  In
        # principle p could be learned from data; in our experiments we pick a
        # value and then verify that all inputs give the same value (test 1).
        self.p = np.random.random() * 0.8 + 0.1  # avoid p too close to 0 or 1
        self.B = np.sqrt(self.p) * self.U

        logger.info("Lab initialised with m=%d, l=%d, p=%.4f", m, l, self.p)

    # ------------------------------------------------------------------
    # Block encoding machinery (Section G of the paper)
    # ------------------------------------------------------------------

    def _ancilla_rotation_gate(self) -> QuantumCircuit:
        """Circuit preparing the ancilla in
        ``sqrt(p)|0> + sqrt(1-p)|1>`` (for ``l == 1``),
        consistent with ``sin^2(theta0)=p``.
        """
        circ = QuantumCircuit(self.l)
        if self.l == 1:
            theta = np.arccos(np.sqrt(self.p))
            circ.ry(2 * theta, 0)
        else:
            # for l>1 we simply rotate the first ancilla and leave the others
            # untouched.  A more general preparation could be done if needed.
            theta = np.arccos(np.sqrt(self.p))
            circ.ry(2 * theta, 0)
        return circ

    def block_encode(self) -> np.ndarray:
        """Construct the block-encoding unitary A [G].

        Returns
        -------
        A : ndarray
            A unitary matrix of size 2^{m+l} whose top-left block is
            sqrt(p) U.  The matrix is obtained by building a quantum
            circuit and extracting its operator representation.
        """
        # build the circuit: ancilla register followed by data register ensures
        # the conventional Qiskit tensor-product ordering.
        anc = QuantumRegister(self.l, "anc")
        data = QuantumRegister(self.m, "data")
        circ = QuantumCircuit(anc, data)

        # prepare ancilla and conditionally apply U when ancilla=0
        circ.compose(self._ancilla_rotation_gate(), inplace=True)

        # convert control-0 into control-1 using X gates
        if self.l != 1:
            raise NotImplementedError("Only l=1 is supported in this demo.")

        # add an X on the control to convert control-0 to control-1
        circ.x(anc[0])
        # append the controlled-U.  ``Operator`` objects don't have a
        # ``control`` method directly, so convert to an instruction first.
        U_gate = Operator(self.U).to_instruction()
        circ.append(U_gate.control(1), [anc[0]] + data[:])
        circ.x(anc[0])

        # convert to matrix and sanity-check unitarity
        A = Operator(circ).data
        assert np.allclose(A.conj().T @ A, np.eye(self.dim_total), atol=1e-8)
        self.A = A
        return A

    # ------------------------------------------------------------------
    # Reflection operators (Equivalence theorem mapping)
    # ------------------------------------------------------------------

    def R_zero(self) -> np.ndarray:
        """Reflection about the all-zero ancilla state.        """
        P0 = np.zeros((self.dim_anc, self.dim_anc))
        P0[0, 0] = 1
        return 2 * np.kron(P0, np.eye(self.dim_data)) - np.eye(self.dim_total)

    def R_bad(self) -> np.ndarray:
        """Phase flip on non-zero ancilla states (effective H_Bad sector)."""
        P0 = np.zeros((self.dim_anc, self.dim_anc))
        P0[0, 0] = 1
        return np.kron(2 * P0 - np.eye(self.dim_anc), np.eye(self.dim_data))

    def build_Q(self) -> np.ndarray:
        """Compute the iteration operator Q = A R_zero A^{-1} R_bad.
        """
        if not hasattr(self, "A"):
            self.block_encode()
        R0 = self.R_zero()
        Rb = self.R_bad()
        A = self.A
        Q = A @ R0 @ A.conj().T @ Rb
        # sanity: Q should be unitary
        assert np.allclose(Q.conj().T @ Q, np.eye(self.dim_total), atol=1e-8)
        self.Q = Q
        return Q

    # ------------------------------------------------------------------
    # Acid tests (numerical verification of the theoretical statements)
    # ------------------------------------------------------------------

    def _random_state(self) -> np.ndarray:
        """Random normalized m-qubit state."""
        vec = np.random.randn(self.dim_data) + 1j * np.random.randn(self.dim_data)
        vec /= np.linalg.norm(vec)
        return vec

    def run_acid_tests(self, num_states: int = 20, max_k: int = 20) -> AmplificationResults:
        """Run input-independence and fidelity tests on random states.

        Args:
            num_states: number of random inputs
            max_k: max iterations
        Returns:
            AmplificationResults with probabilities, msd, and fidelity curve.
        """
        if not hasattr(self, "Q"):
            self.build_Q()

        # arrays to gather data
        probs = np.zeros(num_states)
        msd_accum = 0.0  # mean square deviation accumulator
        fidelity_curve = np.zeros((num_states, max_k + 1))

        # precompute somewhat optimal iteration count for amplitude
        # amplification; this is the integer closest to pi/(4 arcsin(sqrt(p))).
        k_opt = int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(self.p)))))
        logger.info("estimated optimal k = %d", k_opt)

        # projector onto |0>_ancilla \otimes I_data.  we will use it often
        P0 = np.zeros((self.dim_anc, self.dim_anc), dtype=complex)
        P0[0, 0] = 1
        P0_full = np.kron(P0, np.eye(self.dim_data))

        for i in range(num_states):
            phi = self._random_state()

            # --- test 1: input independence of probability p
            init = np.kron(np.array([1] + [0] * (self.dim_anc - 1)), phi)
            psi = self.A @ init
            probs[i] = np.real(psi.conj().T @ P0_full @ psi)

            # prepare vectors G and B spanning the invariant subspace for this phi
            # |G> = |0>_anc ⊗ U|phi>
            G = np.kron(np.array([1] + [0] * (self.dim_anc - 1)), self.U @ phi)
            G /= np.linalg.norm(G)
            # compute |B> by orthogonalising psi against G
            comp = psi - np.vdot(G, psi) * G
            if np.linalg.norm(comp) < 1e-12:
                B = np.zeros_like(comp)
            else:
                B = comp / np.linalg.norm(comp)

            # verify rotations up to max_k
            v = psi.copy()
            for k in range(max_k + 1):
                # projection onto {G,B}
                proj_length = abs(np.vdot(G, v)) ** 2 + abs(np.vdot(B, v)) ** 2
                msd_accum += (1 - proj_length) ** 2  # squared deviation
                # fidelity with U applied directly to phi (only data register)
                # extract data part by tracing out ancilla (for pure state with 1
                # ancilla this is equivalent to selecting the appropriate
                # subvector since our ancilla is always |0> or |1>).
                data_state = v[: self.dim_data]  # amplitudes when ancilla=0
                data_state /= np.linalg.norm(data_state)
                fidelity_curve[i, k] = abs(np.vdot(self.U @ phi, data_state)) ** 2
                # iterate if not at end
                if k < max_k:
                    v = self.Q @ v

        mean_prob = probs.mean()
        var_prob = probs.var()
        msd = msd_accum / (num_states * (max_k + 1))
        logger.info("success probability p: mean=%.6f", mean_prob)
        logger.info("success probability p variance across inputs: %.6e", var_prob)
        logger.info("mean square deviation over trajectories: %.3e", msd)
        if k_opt <= max_k:
            fid_at_opt = fidelity_curve[:, k_opt]
            logger.info("average fidelity at k_opt=%d: %.6f", k_opt, fid_at_opt.mean())

        self.last_results = AmplificationResults(
            success_probabilities=probs, msd=msd, fidelity_curve=fidelity_curve
        )
        return self.last_results

    # ------------------------------------------------------------------
    # No-cloning comparison script
    # ------------------------------------------------------------------

    def no_cloning_comparison(self) -> None:
        """Demonstrate the failure of naive AA with one copy.

        Shows that guessing the input destroys the fidelity.
        """
        # select an example state and attempt to build a naive reflection
        psi = self._random_state()
        # naive reflection operator would be R_psi = I - 2 |psi><psi|.
        R_psi = np.eye(self.dim_data) - 2 * np.outer(psi, psi.conj())

        logger.info("attempting naive AA with unknown state")
        logger.info("R_psi depends explicitly on psi and hence cannot be built"
                    " when only one copy is available (no-cloning).")
        logger.info("by contrast, the OAA construction uses operators on the"
                    " ancilla alone and needs no knowledge of psi.")

        # small numerical experiment: suppose we are forced to guess psi by
        # measuring the lone copy in the computational basis.  The guessed
        # state will typically be orthogonal to the real state, and any
        # resulting amplitude amplification will fail spectacularly.
        basis_index = np.random.choice(self.dim_data, p=np.abs(psi) ** 2)
        guess = np.zeros(self.dim_data, dtype=complex)
        guess[basis_index] = 1
        R_guess = np.eye(self.dim_data) - 2 * np.outer(guess, guess.conj())
        # start from psi and apply the "naive" reflection
        psi_after = R_guess @ psi
        fidelity = abs(np.vdot(psi_after, psi)) ** 2
        logger.info("single-shot measurement guessed basis |%d>, fidelity=%.3e", basis_index, fidelity)
        logger.info("this illustrates that with only one copy your reflection is"
                    " essentially random and the probability of success is no better"
                    " than chance.")


def experiment_subnormalization_rescue(
    p: float = 0.005,
    data_gate: str = "h",
    overshoot_factor: float = 1.5,
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> SubnormalizationRescueResults:
    """Circuit-level demonstration that OAA rescues heavily subnormalized blocks.

    The experiment prepares a block-encoding-like operator ``A`` with ancilla
    success probability ``p`` and applies

        Q = A R_0 A^{-1} R_bad

    for increasing iteration counts, while tracking the probability of the clean
    ancilla ``|0>`` (the effective ``Pi_Good`` event) through exact statevector simulation.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1).")
    if overshoot_factor <= 0.0:
        raise ValueError("overshoot_factor must be positive.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting in experiment_subnormalization_rescue."
        ) from exc

    anc = QuantumRegister(1, "anc")
    data = QuantumRegister(1, "data")

    # Build A: ancilla amplitudes set P(anc=0)=p, then apply a controlled data
    # operation conditioned on ancilla=0.
    A_qc = QuantumCircuit(anc, data, name="A")
    theta_prep = np.arccos(np.sqrt(p))
    A_qc.ry(2.0 * theta_prep, anc[0])
    A_qc.x(anc[0])
    if data_gate.lower() == "h":
        A_qc.ch(anc[0], data[0])
    elif data_gate.lower() == "x":
        A_qc.cx(anc[0], data[0])
    else:
        raise ValueError("data_gate must be one of {'h', 'x'}.")
    A_qc.x(anc[0])
    A_inv = A_qc.inverse()
    A_inv.name = "A_dagger"

    # Reflection on |0> ancilla: I - 2|0><0| = XZX (up to global phase
    # conventions this works as the required oracle/diffusion reflection here).
    R_qc = QuantumCircuit(anc, data, name="R0/R_H_Bad")
    R_qc.x(anc[0])
    R_qc.z(anc[0])
    R_qc.x(anc[0])

    theta0 = np.arcsin(np.sqrt(p))
    theta = theta0  # kept for backward-compatible result field name
    k_opt = int(np.floor(np.pi / (4.0 * theta0)))
    max_k = max(1, int(np.floor(overshoot_factor * k_opt)))

    k_values = np.arange(max_k + 1, dtype=int)
    success_probs = np.zeros(max_k + 1, dtype=float)

    for k in k_values:
        qc = QuantumCircuit(anc, data)
        # Start from |0>_anc|0>_data and prepare the weak block-encoded state A|0,0>.
        qc.append(A_qc, [anc[0], data[0]])
        for _ in range(k):
            # One oblivious iteration Q = A R0 A^{-1} Rbad.
            qc.append(R_qc, [anc[0], data[0]])   # R_bad
            qc.append(A_inv, [anc[0], data[0]])  # A^{-1}
            qc.append(R_qc, [anc[0], data[0]])   # R_0
            qc.append(A_qc, [anc[0], data[0]])   # A

        state = Statevector.from_instruction(qc)
        # For |data, anc> indexing, ancilla is the least-significant bit.
        success_probs[k] = float(
            np.sum(np.abs(state.data[0::2]) ** 2)
        )

    print(f"Initial success probability p: {p:.6f}")
    print(f"Theoretical OAA optimum k*: {k_opt}")
    print(f"Best simulated success probability p_k: {success_probs.max():.6f} at k={int(np.argmax(success_probs))}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        k_values,
        success_probs,
        color="tab:blue",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="OAA circuit simulation",
    )
    plt.axvline(
        x=k_opt,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"Estimated optimum k*={k_opt}",
    )
    plt.axhline(
        y=p,
        color="0.35",
        linestyle=":",
        linewidth=1.3,
        label=f"Initial p={p:.3f}",
    )
    plt.title("Subnormalization Rescue via OAA (Circuit-Level)")
    plt.xlabel("Number of OAA iterations (k)")
    plt.ylabel("Success probability p_k (clean ancilla)")
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return SubnormalizationRescueResults(
        p=p,
        theta=float(theta),
        k_opt=k_opt,
        k_values=k_values,
        success_probabilities=success_probs,
    )


def experiment_geometric_obstruction(
    p_valid: float = 0.05,
    p_invalid_zero: float = 0.05,
    p_invalid_one: float = 0.20,
    max_k: int = 15,
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> GeometricObstructionResults:
    """Negative proof: violate purified setting and observe OAA failure.

    Compares two constructions:
    1) Valid purified setting: ancilla success weight independent of data.
    2) Violated setting: ancilla rotation depends on data basis component.

    The data qubit starts in |+>, and the target is U|phi>=|+> (U=I demo).
    We track both:
    - success probability p = P(ancilla=|0>) after k OAA iterations
    - Fidelity of conditional data state (given ancilla=|0>) with |+>
    """
    if not (0.0 < p_valid < 1.0):
        raise ValueError("p_valid must be in (0, 1).")
    if not (0.0 < p_invalid_zero < 1.0):
        raise ValueError("p_invalid_zero must be in (0, 1).")
    if not (0.0 < p_invalid_one < 1.0):
        raise ValueError("p_invalid_one must be in (0, 1).")
    if max_k < 0:
        raise ValueError("max_k must be non-negative.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting in experiment_geometric_obstruction."
        ) from exc

    anc = QuantumRegister(1, "anc")
    data = QuantumRegister(1, "data")

    # Target data state is |+> for this U=I demonstration.
    plus = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=complex)

    # Valid purified setting: ancilla preparation independent of data.
    theta_valid = np.arccos(np.sqrt(p_valid))
    A_valid = QuantumCircuit(anc, data, name="A_valid")
    A_valid.ry(2.0 * theta_valid, anc[0])
    A_valid_inv = A_valid.inverse()
    A_valid_inv.name = "A_valid_dagger"

    # Violated setting: ancilla rotation conditioned on data basis component.
    theta_0 = np.arccos(np.sqrt(p_invalid_zero))
    theta_1 = np.arccos(np.sqrt(p_invalid_one))
    A_invalid = QuantumCircuit(anc, data, name="A_invalid")
    A_invalid.x(data[0])
    A_invalid.cry(2.0 * theta_0, data[0], anc[0])  # branch for data=|0>
    A_invalid.x(data[0])
    A_invalid.cry(2.0 * theta_1, data[0], anc[0])  # branch for data=|1>
    A_invalid_inv = A_invalid.inverse()
    A_invalid_inv.name = "A_invalid_dagger"

    # Common reflection I - 2|0><0| on ancilla.
    R_qc = QuantumCircuit(anc, data, name="R0/R_H_Bad")
    R_qc.x(anc[0])
    R_qc.z(anc[0])
    R_qc.x(anc[0])

    k_values = np.arange(max_k + 1, dtype=int)

    def _run_loop(A_qc: QuantumCircuit, A_inv_qc: QuantumCircuit) -> tuple[np.ndarray, np.ndarray]:
        fidelities = np.zeros(max_k + 1, dtype=float)
        probs = np.zeros(max_k + 1, dtype=float)

        for k in k_values:
            qc = QuantumCircuit(anc, data)
            qc.h(data[0])  # |phi> = |+>
            qc.append(A_qc, [anc[0], data[0]])

            for _ in range(k):
                # Apply Q repeatedly while keeping the same data input state.
                qc.append(R_qc, [anc[0], data[0]])    # R_bad
                qc.append(A_inv_qc, [anc[0], data[0]])  # A^{-1}
                qc.append(R_qc, [anc[0], data[0]])    # R_0
                qc.append(A_qc, [anc[0], data[0]])    # A

            state = Statevector.from_instruction(qc).data

            # ancilla=0 corresponds to even indices for register order (anc, data).
            amp_a0 = state[0::2]
            prob0 = float(np.sum(np.abs(amp_a0) ** 2))
            probs[k] = prob0

            if prob0 > 1e-12:
                # Post-select ancilla=0 and renormalize to get the conditional data state.
                cond_data = amp_a0 / np.sqrt(prob0)
                fidelities[k] = float(np.abs(np.vdot(plus, cond_data)) ** 2)
            else:
                fidelities[k] = 0.0

        return fidelities, probs

    valid_fid, valid_prob = _run_loop(A_valid, A_valid_inv)
    invalid_fid, invalid_prob = _run_loop(A_invalid, A_invalid_inv)

    print("Geometric Obstruction Experiment")
    print(f"Valid setting p={p_valid:.3f}")
    print(f"Invalid setting p(|0>)={p_invalid_zero:.3f}, p(|1>)={p_invalid_one:.3f}")
    print(f"Valid fidelity range: [{valid_fid.min():.6f}, {valid_fid.max():.6f}]")
    print(f"Invalid fidelity range: [{invalid_fid.min():.6f}, {invalid_fid.max():.6f}]")
    print(f"Valid max success probability p_k: {valid_prob.max():.6f}")
    print(f"Invalid max success probability p_k: {invalid_prob.max():.6f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(k_values, valid_fid, color="tab:blue", marker="o", label="Purified setting (valid)")
    ax1.plot(
        k_values,
        invalid_fid,
        color="tab:red",
        marker="x",
        linestyle="--",
        label="Violated setting (invalid)",
    )
    ax1.set_title("OAA Geometric Obstruction: Conditional Data Fidelity")
    ax1.set_ylabel("Fidelity with |+>")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="lower left")

    ax2.plot(k_values, valid_prob, color="tab:blue", marker="o", label="Purified setting (valid)")
    ax2.plot(
        k_values,
        invalid_prob,
        color="tab:red",
        marker="x",
        linestyle="--",
        label="Violated setting (invalid)",
    )
    ax2.set_title("OAA Geometric Obstruction: Success Probability p")
    ax2.set_xlabel("OAA iterations (k)")
    ax2.set_ylabel("Success probability p_k (clean ancilla)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return GeometricObstructionResults(
        k_values=k_values,
        valid_fidelity=valid_fid,
        invalid_fidelity=invalid_fid,
        valid_success_probability=valid_prob,
        invalid_success_probability=invalid_prob,
    )


def experiment_explicit_lcu_block_encoding(
    c0: float = 0.6,
    c1: float = 0.4,
    ft_basis: Optional[list[str]] = None,
    optimization_level: int = 0,
    save_csv_path: Optional[str] = None,
) -> ExplicitLCUBlockEncodingResults:
    """Construct explicit LCU block encoding for H = c0 X_0 + c1 Z_1.

    Builds PREP, SELECT, PREP^\\dagger as explicit circuits, constructs
    A = PREP^\\dagger * SELECT * PREP, verifies the top-left block
    M_TL = H / alpha, and reports native and Clifford+T proxy costs.
    """
    if c0 == 0.0 and c1 == 0.0:
        raise ValueError("At least one coefficient must be non-zero.")
    if c0 < 0.0 or c1 < 0.0:
        raise ValueError(
            "This demo expects non-negative coefficients. For signed LCUs,"
            " include sign handling in SELECT."
        )
    if optimization_level not in (0, 1, 2, 3):
        raise ValueError("optimization_level must be one of 0,1,2,3.")

    alpha = abs(c0) + abs(c1)
    w0 = c0 / alpha
    w1 = c1 / alpha

    data = QuantumRegister(2, "data")
    anc = QuantumRegister(1, "anc")

    # PREP: sqrt(w0)|0> + sqrt(w1)|1>
    theta = np.arccos(np.sqrt(w0))
    prep = QuantumCircuit(data, anc, name="PREP")
    prep.ry(2.0 * theta, anc[0])

    # SELECT: |0><0|_anc ⊗ X_0 + |1><1|_anc ⊗ Z_1
    select = QuantumCircuit(data, anc, name="SELECT")
    select.x(anc[0])
    select.cx(anc[0], data[0])  # anc=0 branch -> X on qubit 0
    select.x(anc[0])
    select.cz(anc[0], data[1])  # anc=1 branch -> Z on qubit 1

    prep_dag = prep.inverse()
    prep_dag.name = "PREP_dagger"

    # Compose full LCU block-encoding unitary.
    A_qc = QuantumCircuit(data, anc, name="A_LCU")
    A_qc.compose(prep, inplace=True)
    A_qc.compose(select, inplace=True)
    A_qc.compose(prep_dag, inplace=True)

    # Numerical verification M_TL = H/alpha.
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    i2 = np.eye(2, dtype=complex)
    x0 = np.kron(i2, x)   # data ordering |q1 q0>
    z1 = np.kron(z, i2)
    H_target = c0 * x0 + c1 * z1

    A_matrix = Operator(A_qc).data
    # ancilla register is created after data, so anc is MSB; top-left 4x4 is anc=0 block.
    m_tl = A_matrix[:4, :4]
    h_norm = H_target / alpha
    distance = float(np.linalg.norm(m_tl - h_norm))

    print("-" * 62)
    print("EXPLICIT LCU BLOCK ENCODING")
    print("-" * 62)
    print(f"Hamiltonian: H = {c0:.3f} X_0 + {c1:.3f} Z_1")
    print(f"alpha = |c0| + |c1| = {alpha:.6f}")
    print(f"||M_TL - H/alpha||_F = {distance:.3e}")
    print("Verification:", "PASS" if distance < 1e-10 else "CHECK NUMERICS")

    native_gate_counts = {
        "PREP": dict(prep.count_ops()),
        "SELECT": dict(select.count_ops()),
        "PREP_dagger": dict(prep_dag.count_ops()),
        "A_LCU_total": dict(A_qc.count_ops()),
    }

    if ft_basis is None:
        # Treat these as a lightweight Clifford+T proxy basis for resource reporting.
        ft_basis = ["cx", "h", "s", "sdg", "t", "tdg", "x", "z"]
    logging.getLogger("qiskit").setLevel(logging.WARNING)

    def _compile_and_metrics(circ: QuantumCircuit) -> tuple[int, int]:
        # Compile each module separately so PREP/SELECT/PREP^\dagger costs are visible.
        ft_circ = transpile(
            circ,
            basis_gates=ft_basis,
            optimization_level=optimization_level,
        )
        ops = ft_circ.count_ops()
        t_count = int(ops.get("t", 0) + ops.get("tdg", 0))
        return t_count, int(ft_circ.depth())

    prep_t, prep_depth = _compile_and_metrics(prep)
    select_t, select_depth = _compile_and_metrics(select)
    prep_dag_t, prep_dag_depth = _compile_and_metrics(prep_dag)
    total_t, total_depth = _compile_and_metrics(A_qc)

    ft_t_counts = {
        "PREP": prep_t,
        "SELECT": select_t,
        "PREP_dagger": prep_dag_t,
        "A_LCU_total": total_t,
    }
    ft_depths = {
        "PREP": prep_depth,
        "SELECT": select_depth,
        "PREP_dagger": prep_dag_depth,
        "A_LCU_total": total_depth,
    }

    print("\n" + "-" * 62)
    print("RESOURCE SUMMARY (Native + Clifford+T Proxy)")
    print("-" * 62)
    print(f"{'Module':<14} {'Native Ops':<34} {'T-count':>8} {'Depth':>8}")
    for key in ("PREP", "SELECT", "PREP_dagger", "A_LCU_total"):
        print(
            f"{key:<14} {str(native_gate_counts[key]):<34} "
            f"{ft_t_counts[key]:>8} {ft_depths[key]:>8}"
        )

    if save_csv_path is not None:
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["module", "native_ops", "t_count", "depth"])
            for key in ("PREP", "SELECT", "PREP_dagger", "A_LCU_total"):
                writer.writerow([key, native_gate_counts[key], ft_t_counts[key], ft_depths[key]])

    return ExplicitLCUBlockEncodingResults(
        alpha=float(alpha),
        matrix_distance=distance,
        native_gate_counts=native_gate_counts,
        ft_t_counts=ft_t_counts,
        ft_depths=ft_depths,
    )


def _clean_ancilla_subspace_indices(num_qubits: int, ancilla_positions: list[int]) -> np.ndarray:
    """Basis indices where all ancilla-position bits are 0."""
    out: list[int] = []
    for basis_index in range(2**num_qubits):
        # Keep computational basis states that lie in the clean-ancilla subspace.
        if all(((basis_index >> pos) & 1) == 0 for pos in ancilla_positions):
            out.append(basis_index)
    return np.array(out, dtype=int)


def extract_identity_block_metrics(
    A_qc: QuantumCircuit,
    ancilla_qubits: list,
    p_reference: Optional[float] = None,
) -> IdentityBlockAudit:
    """Compute M=A^\\dagger P1 A and audit clean-ancilla block against pI.

    This helper works for any qubit order by deriving ancilla bit positions from
    ``A_qc.qubits`` and selecting the corresponding clean-ancilla computational
    basis subspace.
    """
    if not ancilla_qubits:
        raise ValueError("ancilla_qubits must contain at least one qubit.")

    num_qubits = A_qc.num_qubits
    full_dim = 2**num_qubits
    # Map each Qiskit qubit object to its bit position in statevector indexing.
    qubit_to_pos = {qb: idx for idx, qb in enumerate(A_qc.qubits)}
    anc_positions = [qubit_to_pos[qb] for qb in ancilla_qubits]
    clean_idx = _clean_ancilla_subspace_indices(num_qubits, anc_positions)
    clean_dim = len(clean_idx)

    A = Operator(A_qc).data
    A_dag = A.conj().T

    # Build P1 directly in the computational basis using clean_idx, so this stays
    # correct even when ancilla is not the first/last register.
    P1 = np.zeros((full_dim, full_dim), dtype=complex)
    P1[np.ix_(clean_idx, clean_idx)] = np.eye(clean_dim, dtype=complex)

    M = A_dag @ P1 @ A
    # "Top-left block" in theory corresponds here to the clean_idx principal block.
    m_tl = M[np.ix_(clean_idx, clean_idx)]

    if p_reference is None:
        p_reference = float(np.real(np.trace(m_tl)) / clean_dim)
    target = p_reference * np.eye(clean_dim, dtype=complex)

    distance = float(np.linalg.norm(m_tl - target))
    off_diag = m_tl - np.diag(np.diag(m_tl))
    max_off_diag = float(np.max(np.abs(off_diag))) if clean_dim > 1 else 0.0

    return IdentityBlockAudit(
        p_reference=float(p_reference),
        distance_to_p_identity=distance,
        max_off_diagonal=max_off_diag,
        diagonal_entries=np.diag(m_tl),
        m_tl=m_tl,
    )


def experiment_identity_block_extractor(
    m: int = 2,
    l: int = 1,
    p: float = 0.15,
    seed: int = 42,
    c0: float = 0.6,
    c1: float = 0.4,
    save_csv_path: Optional[str] = None,
) -> IdentityBlockExtractorResults:
    """Numerically verify Proposition-2 style identity block in purified setting.

    Also performs an LCU cross-check for H=c0*X_0 + c1*Z_1. For this Hamiltonian
    block encoding, M_TL is generally not pI; the expected structure is
    M_TL = (H/alpha)^\\dagger (H/alpha).
    """
    if m < 1:
        raise ValueError("m must be >= 1.")
    if l != 1:
        raise NotImplementedError("This experiment currently supports l=1.")
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1).")
    if c0 == 0.0 and c1 == 0.0:
        raise ValueError("At least one of c0 or c1 must be non-zero.")
    if c0 < 0.0 or c1 < 0.0:
        raise ValueError("Use non-negative c0,c1 in this demo.")

    # ------------------------------------------------------------------
    # Case 1: Purified setting (independent initial weight)
    # ------------------------------------------------------------------
    data = QuantumRegister(m, "data")
    anc = QuantumRegister(l, "anc")
    A_purified = QuantumCircuit(data, anc, name="A_purified")
    A_purified.ry(2.0 * np.arccos(np.sqrt(p)), anc[0])

    U = random_unitary(2**m, seed=seed).to_instruction()
    A_purified.x(anc[0])
    A_purified.append(U.control(1), [anc[0]] + data[:])
    A_purified.x(anc[0])

    purified_audit = extract_identity_block_metrics(
        A_qc=A_purified,
        ancilla_qubits=[anc[0]],
        p_reference=p,
    )

    # ------------------------------------------------------------------
    # Case 2: Explicit LCU block encoding for H = c0 X_0 + c1 Z_1 (2 data qubits)
    # ------------------------------------------------------------------
    if m != 2:
        # This demo Hamiltonian is defined on two data qubits only.
        raise ValueError("LCU cross-check is defined for m=2 in this implementation.")

    data_lcu = QuantumRegister(2, "data")
    anc_lcu = QuantumRegister(1, "anc")
    A_lcu = QuantumCircuit(data_lcu, anc_lcu, name="A_LCU")
    alpha = abs(c0) + abs(c1)
    theta = np.arccos(np.sqrt(c0 / alpha))
    A_lcu.ry(2.0 * theta, anc_lcu[0])
    A_lcu.x(anc_lcu[0])
    A_lcu.cx(anc_lcu[0], data_lcu[0])  # anc=0 branch -> X_0
    A_lcu.x(anc_lcu[0])
    A_lcu.cz(anc_lcu[0], data_lcu[1])  # anc=1 branch -> Z_1
    A_lcu.ry(-2.0 * theta, anc_lcu[0])

    # For LCU, we infer p_est from trace(M_TL)/dim to test "distance to pI" objectively.
    lcu_audit = extract_identity_block_metrics(
        A_qc=A_lcu,
        ancilla_qubits=[anc_lcu[0]],
        p_reference=None,
    )

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    i2 = np.eye(2, dtype=complex)
    H = c0 * np.kron(i2, x) + c1 * np.kron(z, i2)
    h_norm = H / alpha
    # Expected LCU invariant for this setting is (H/alpha)^\dagger(H/alpha), not pI.
    lcu_target = h_norm.conj().T @ h_norm
    lcu_distance_to_h2 = float(np.linalg.norm(lcu_audit.m_tl - lcu_target))

    print("-" * 72)
    print("IDENTITY BLOCK EXTRACTOR: NUMERICAL AUDIT")
    print("-" * 72)
    print("Purified setting (Proposition-2 check)")
    print(f"p (input): {p:.6f}")
    print(f"diag(M_TL): {np.round(purified_audit.diagonal_entries, 6)}")
    print(f"max |offdiag(M_TL)|: {purified_audit.max_off_diagonal:.3e}")
    print(f"||M_TL - pI||_F: {purified_audit.distance_to_p_identity:.3e}")
    print("status:", "PASS" if purified_audit.distance_to_p_identity < 1e-14 else "CHECK")

    print("\nLCU cross-check (H = c0 X_0 + c1 Z_1)")
    print(f"p_est (trace/dim): {lcu_audit.p_reference:.6f}")
    print(f"diag(M_TL): {np.round(lcu_audit.diagonal_entries, 6)}")
    print(f"max |offdiag(M_TL)|: {lcu_audit.max_off_diagonal:.3e}")
    print(f"||M_TL - p_est I||_F: {lcu_audit.distance_to_p_identity:.3e}")
    print(f"||M_TL - (H/alpha)^dagger(H/alpha)||_F: {lcu_distance_to_h2:.3e}")
    print("-" * 72)

    if save_csv_path is not None:
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "case",
                    "p_reference",
                    "distance_to_pI",
                    "max_offdiag",
                    "distance_to_expected_lcu_target",
                ]
            )
            writer.writerow(
                [
                    "purified_setting",
                    purified_audit.p_reference,
                    purified_audit.distance_to_p_identity,
                    purified_audit.max_off_diagonal,
                    "",
                ]
            )
            writer.writerow(
                [
                    "explicit_lcu",
                    lcu_audit.p_reference,
                    lcu_audit.distance_to_p_identity,
                    lcu_audit.max_off_diagonal,
                    lcu_distance_to_h2,
                ]
            )

    return IdentityBlockExtractorResults(
        purified=purified_audit,
        lcu=lcu_audit,
        lcu_distance_to_h2_over_alpha2=lcu_distance_to_h2,
    )


# end of module


def run_all_one_click(show_plot: bool = False) -> None:
    """Run all OAA experiments and persist artifacts in the current directory."""
    stem = Path(__file__).stem
    experiment_subnormalization_rescue(
        p=0.005,
        data_gate="h",
        overshoot_factor=1.5,
        show_plot=show_plot,
        save_path=f"{stem}_module1_subnormalization_rescue.png",
    )
    experiment_geometric_obstruction(
        p_valid=0.05,
        p_invalid_zero=0.05,
        p_invalid_one=0.20,
        max_k=15,
        show_plot=show_plot,
        save_path=f"{stem}_module2_geometric_obstruction.png",
    )
    experiment_explicit_lcu_block_encoding(
        c0=0.6,
        c1=0.4,
        optimization_level=0,
        save_csv_path=f"{stem}_module3_explicit_lcu_resources.csv",
    )
    experiment_identity_block_extractor(
        m=2,
        l=1,
        p=0.15,
        seed=42,
        c0=0.6,
        c1=0.4,
        save_csv_path=f"{stem}_module4_identity_block_audit.csv",
    )
    print("One-click OAA run complete.")


if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="oaa")
    # Direct "Run" in IDE executes the full OAA pipeline.
    if len(sys.argv) == 1:
        run_all_one_click(show_plot=False)
    else:
        run_all_one_click(show_plot=False)
