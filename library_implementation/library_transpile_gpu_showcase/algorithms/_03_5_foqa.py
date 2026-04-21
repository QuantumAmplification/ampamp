from qiskit import QuantumCircuit
from ampamp.foqa import FOQAEngine


def build_circuit() -> QuantumCircuit:
    return FOQAEngine(0.5).build_proxy_sequence(n_steps=32, mizel_c=1.4, m_content=1)
