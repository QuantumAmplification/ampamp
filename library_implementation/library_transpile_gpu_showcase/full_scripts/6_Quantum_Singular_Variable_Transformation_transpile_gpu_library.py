import os, sys
import numpy as np
from qiskit import QuantumCircuit, transpile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ampamp.qsvt import QSVTSynthesizer


def build_circuit():
    degree = 41
    coeffs = QSVTSynthesizer.synthesize_matrix_inverse(degree=degree, kappa=6.0)
    qc = QuantumCircuit(1)
    for i,c in enumerate(coeffs[:degree+1]):
        qc.rz(float(np.clip(c,-1.0,1.0)),0)
        qc.rx(np.pi/9.0 if i % 2 == 0 else np.pi/13.0,0)
    return qc


def run_scenario_a():
    t = transpile(build_circuit(), basis_gates=['cx','id','rz','sx','x'], optimization_level=3)
    print('QSVT A depth=', t.depth(), 'cx=', t.count_ops().get('cx',0))


if __name__ == '__main__':
    run_scenario_a()
