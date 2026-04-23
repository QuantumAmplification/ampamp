from __future__ import annotations

"""Foundations for Quantum Amplitude Amplification.

This module provides basic classes and fundamental operations
for standard Grover search and amplitude amplification algorithms.
"""

import numpy as np
from qiskit import QuantumCircuit

class GroverEngine:
    """The core algebraic and circuit engine for Grover's Search.

    Focuses on state synthesis and geometric calculations to amplify
    amplitudes of marked solution states optimally.
    """
    def __init__(self, n_qubits: int, marked_indices: list[int]):
        """Initializes the GroverEngine.

        Args:
            n_qubits (int): The number of qubits representing the problem space ($N = 2^n$).
            marked_indices (list[int]): List indicating the integers representing 
                the marked states in the problem subspace.
        """
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
            
        self.n = n_qubits
        self.N = 2**n_qubits
        
        # Remove duplicates
        unique_marked = list(set(marked_indices))
        if any(m < 0 or m >= self.N for m in unique_marked):
            raise ValueError(f"Marked indices must be between 0 and {self.N - 1}")
        if len(unique_marked) == 0:
            raise ValueError("Must provide at least one marked state.")
            
        self.marked = unique_marked
        self.M = len(self.marked)
        
        # Fundamental Geometric properties
        self.solution_density = self.M / self.N
        self.theta = 2 * np.arcsin(np.sqrt(self.solution_density))
        self.k_optimal = int(np.floor(np.pi / (2 * self.theta) - 0.5))

    @staticmethod
    def compute_success_prob(lambda_val: float, k: int) -> float:
        """Analytic success probability calculation.

        Computes the success probability $P(k)$ after $k$ Grover iterations
        given the solution density $\\lambda$. The probability formula is defined geometrically as 
        $P(k) = \\sin^2((2k + 1) \\theta / 2)$, where $\\theta = 2 \\arcsin(\\sqrt{\\lambda})$.

        Args:
            lambda_val (float): The solution density fraction, $M/N$.
            k (int): The number of Grover iterations applied.

        Returns:
            float: The probability of measuring a marked state.

        Raises:
            ValueError: If lambda_val is not within the inclusive bound [0, 1].
        """
        if not (0 <= lambda_val <= 1):
            raise ValueError("lambda_val must be in [0, 1]")
        theta = 2.0 * np.arcsin(np.sqrt(lambda_val))
        return float(np.sin((2 * k + 1) * theta / 2.0) ** 2)

    def get_oracle(self) -> QuantumCircuit:
        """Generates a Phase Oracle for the marked states.

        Constructs a standard quantum phase oracle gate circuit that flips the sign 
        of the marked states while leaving orthogonal unmarked states alone.

        Returns:
            QuantumCircuit: An n-qubit oracle circuit flipping the phase of marked states.
        """
        qc = QuantumCircuit(self.n)
        for index in self.marked:
            target_bin = format(index, f'0{self.n}b')[::-1]
            for i, bit in enumerate(target_bin):
                if bit == '0': qc.x(i)
            if self.n == 1:
                qc.z(0)
            else:
                qc.h(self.n - 1)
                qc.mcx(list(range(self.n - 1)), self.n - 1)
                qc.h(self.n - 1)
            
            for i, bit in enumerate(target_bin):
                if bit == '0': qc.x(i)
        return qc

    def get_diffusion(self) -> QuantumCircuit:
        """Generates the Grover Diffusion operator (Inversion about mean).

        Returns:
            QuantumCircuit: An n-qubit circuit corresponding to $2 |s\\rangle\\langle s| - I$, 
            where $|s\\rangle$ is the equal superposition state.
        """
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        qc.x(range(self.n))
        if self.n == 1:
            qc.z(0)
        else:
            qc.h(self.n - 1)
            qc.mcx(list(range(self.n - 1)), self.n - 1)
            qc.h(self.n - 1)
        qc.x(range(self.n))
        qc.h(range(self.n))
        qc.global_phase += np.pi
        return qc

    def construct_circuit(self, iterations: int) -> QuantumCircuit:
        """Synthesizes the full Grover circuit.

        Links the oracle operator and diffusion operator for $k$ iterations 
        starting from the full uniformly mixed amplitude distribution $|s\\rangle$.

        Args:
            iterations (int): The amount of generalized Grover operator iterations 
                to sequentially run $Q = -A S_0 A^{-1} S_f$.

        Returns:
            QuantumCircuit: The synthesized Grover standard algorithm sequence.
        """
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        
        oracle = self.get_oracle()
        diffusion = self.get_diffusion()
        
        for _ in range(iterations):
            qc.append(oracle, range(self.n))
            qc.append(diffusion, range(self.n))
        
        return qc


# ----------------------------------------------------------------------------

