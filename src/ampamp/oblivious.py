"""Oblivious Amplitude Amplification module.

Provides the `ObliviousEngine` to handle oblivious operations such as block encoding.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, random_unitary

class ObliviousEngine:
    """Engine for Oblivious Amplitude Amplification (OAA).

    Handles block-encoding unitaries and 'purified' reflection sequences.
    """
    def __init__(self, m_data_qubits: int, l_ancilla_qubits: int = 1, p: float = 0.5):
        """Initializes the ObliviousEngine.

        Args:
            m_data_qubits (int): The number of data qubits.
            l_ancilla_qubits (int): The number of ancilla qubits. Defaults to 1.
            p (float): The success probability of the block encoding. Defaults to 0.5.
        """
        if m_data_qubits < 1:
            raise ValueError("m_data_qubits must be >= 1")
        if l_ancilla_qubits < 1:
            raise ValueError("l_ancilla_qubits must be >= 1")
        if not (0.0 <= p <= 1.0):
            raise ValueError("Success probability p must be in [0, 1]")
            
        self.m = m_data_qubits
        self.l = l_ancilla_qubits
        self.p = p # Success probability of the block encoding
        
    def get_ancilla_rotation(self) -> QuantumCircuit:
        """Prepares the ancilla in the $\\sqrt{p}|0\\rangle + \\sqrt{1-p}|1\\rangle$ state.

        Returns:
            QuantumCircuit: The rotation circuit for the ancilla.
        """
        qc = QuantumCircuit(self.l)
        theta = np.arccos(np.sqrt(self.p))
        qc.ry(2 * theta, 0)
        return qc

    def construct_block_encoding(self, unitary_matrix: np.ndarray) -> QuantumCircuit:
        """Embeds a unitary $U$ into a larger operator $A$.

        The construction ensures that the top-left block of $A$ behaves as $\\sqrt{p} U$.

        Args:
            unitary_matrix (np.ndarray): The unitary matrix $U$ to encode.

        Returns:
            QuantumCircuit: The constructed quantum circuit implementing the block encoding.
        """
        expected_shape = (2**self.m, 2**self.m)
        if unitary_matrix.shape != expected_shape:
            raise ValueError(f"unitary_matrix must have shape {expected_shape}")
            
        anc = QuantumRegister(self.l, "anc")
        data = QuantumRegister(self.m, "data")
        qc = QuantumCircuit(anc, data)

        # 1. Rotate ancilla
        qc.compose(self.get_ancilla_rotation(), inplace=True)

        # 2. Controlled-U conditioned on ancilla=0
        qc.x(anc[0])
        u_instr = Operator(unitary_matrix).to_instruction()
        qc.append(u_instr.control(1), [anc[0]] + data[:])
        qc.x(anc[0])
        
        return qc

    def get_reflections(self) -> QuantumCircuit:
        """Constructs the joint $R_0$ (ancilla) and $R_{bad}$ (data) reflections.

        Returns:
            QuantumCircuit: The circuit that conditionally applies phase reflections.
        """
        anc = QuantumRegister(self.l)
        data = QuantumRegister(self.m)
        qc = QuantumCircuit(anc, data)
        # Reflection about |0> on ancilla: I - 2|0><0|
        qc.x(0)
        qc.z(0)
        qc.x(0)
        return qc