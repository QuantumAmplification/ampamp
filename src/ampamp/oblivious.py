import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, random_unitary

class ObliviousEngine:
    """
    Engine for Oblivious Amplitude Amplification (OAA).
    Handles block-encoding unitaries and 'purified' reflection sequences.
    """
    def __init__(self, m_data_qubits: int, l_ancilla_qubits: int = 1, p: float = 0.5):
        self.m = m_data_qubits
        self.l = l_ancilla_qubits
        self.p = p # Success probability of the block encoding
        
    def get_ancilla_rotation(self) -> QuantumCircuit:
        """Prepares the ancilla in the sqrt(p)|0> + sqrt(1-p)|1> state."""
        qc = QuantumCircuit(self.l)
        theta = np.arccos(np.sqrt(self.p))
        qc.ry(2 * theta, 0)
        return qc

    def construct_block_encoding(self, unitary_matrix: np.ndarray) -> QuantumCircuit:
        """
        Embeds a unitary U into a larger operator A such that the 
        top-left block is sqrt(p) * U.
        """
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
        """Constructs the joint R0 (ancilla) and Rbad (data) reflections."""
        anc = QuantumRegister(self.l)
        data = QuantumRegister(self.m)
        qc = QuantumCircuit(anc, data)
        # Reflection about |0> on ancilla: I - 2|0><0|
        qc.x(0)
        qc.z(0)
        qc.x(0)
        return qc