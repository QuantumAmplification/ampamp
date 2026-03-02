"""Simple implementation of oblivious amplitude amplification
using Qiskit.

Defines :class:`ObliviousAmplificationLab` which builds circuits and
runs basic tests for the block encoding, iteration operator, and fidelity
checks. Intended for use from a script or notebook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, random_unitary

# configure logger for informative output
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class AmplificationResults:
    """Container for results from the acid tests."""

    success_probabilities: np.ndarray
    msd: float
    fidelity_curve: np.ndarray


class ObliviousAmplificationLab:
    """Implements the laboratory for oblivious amplitude amplification.

    The class is parameterised by ``m`` (number of data qubits) and ``l``
    (number of ancilla qubits used for the block encoding).  For the purposes
    of the paper we always use ``l = 1``; the code is written generically so
    that more ancilla can be added if necessary.

    The central objects are::

        U    : random m-qubit unitary
        p    : success probability (a float in (0,1])
        B    : \sqrt{p} U (the operator we wish to block encode)
        A    : unitary acting on m + l qubits with
               P_1 A P_1 = B (= top-left block) where P_1 = |0^l><0^l|_anc.
        R_0  : reflection about the clean ancilla subspace
        R_b  : phase flip on all bad ancilla states (orthogonal to |0^l>)
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
        ``sqrt(p)|0> + sqrt(1-p)|1>`` (for ``l == 1``).
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
        """Phase flip on non-zero ancilla states.        """
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
        logger.info("success probability: mean=%.6f", mean_prob)
        logger.info("success probability variance across inputs: %.6e", var_prob)
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
        

# end of module
