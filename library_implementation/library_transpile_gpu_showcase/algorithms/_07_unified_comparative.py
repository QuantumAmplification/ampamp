from qiskit import QuantumCircuit
from ._02_fixed_point import build_circuit as build_fixed_point


def build_circuit() -> QuantumCircuit:
    return build_fixed_point()
