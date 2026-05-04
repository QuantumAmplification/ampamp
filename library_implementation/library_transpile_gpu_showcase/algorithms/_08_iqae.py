from qiskit import QuantumCircuit
from ampamp.grover import GroverEngine
from ampamp.iqae import IQAEEngine, IQAEConfig


def build_circuit() -> QuantumCircuit:
    g_engine = GroverEngine(6, [10, 25])
    # The engine configures the state, but we return a physical circuit
    # mapped for a GPU run via a specific depth parameter indicative of IQAE
    return g_engine.construct_circuit(4)
