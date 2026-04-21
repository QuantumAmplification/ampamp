from qiskit import QuantumCircuit
from ampamp.grover import GroverEngine


def build_circuit() -> QuantumCircuit:
    eng = GroverEngine(5, [3])
    qc = QuantumCircuit(5)
    qc.h(range(5))
    qc.append(eng.get_oracle(), range(5))
    qc.rx(0.35, range(5))
    qc.append(eng.get_diffusion(), range(5))
    return qc
