import numpy as np
from qiskit import QuantumCircuit
from ampamp.qsvt import QSVTSynthesizer


def build_circuit() -> QuantumCircuit:
    degree = 41
    coeffs = QSVTSynthesizer.synthesize_matrix_inverse(degree=degree, kappa=6.0)
    qc = QuantumCircuit(1)
    for idx, c in enumerate(coeffs[: degree + 1]):
        qc.rz(float(np.clip(c, -1.0, 1.0)), 0)
        qc.rx(np.pi / 9.0 if idx % 2 == 0 else np.pi / 13.0, 0)
    return qc
