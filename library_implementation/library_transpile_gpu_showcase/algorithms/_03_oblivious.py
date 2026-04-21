import numpy as np
from qiskit import QuantumCircuit
from ampamp.oblivious import ObliviousEngine


def build_circuit() -> QuantumCircuit:
    eng = ObliviousEngine(2, 1, 0.6)
    block = eng.construct_block_encoding(np.eye(4, dtype=complex))
    refl = eng.get_reflections()
    qc = QuantumCircuit(3)
    for _ in range(8):
        qc.compose(block, inplace=True)
        qc.compose(refl, inplace=True)
        qc.compose(block.inverse(), inplace=True)
        qc.compose(refl, inplace=True)
    return qc
