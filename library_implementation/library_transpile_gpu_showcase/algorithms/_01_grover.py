from qiskit import QuantumCircuit
from ampamp.grover import GroverEngine


def build_circuit() -> QuantumCircuit:
    return GroverEngine(6, [10, 25]).construct_circuit(1)
