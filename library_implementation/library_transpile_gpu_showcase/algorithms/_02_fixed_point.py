from qiskit import QuantumCircuit
from ampamp.fixed_point import FixedPointEngine


def build_circuit() -> QuantumCircuit:
    return FixedPointEngine(3, 0.1).build_fixed_point_circuit(6, [0])
