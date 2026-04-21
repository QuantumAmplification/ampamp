from qiskit import QuantumCircuit


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    for _ in range(6):
        qc.ch(0, 1)
        qc.ccx(0, 1, 2)
        qc.ch(0, 2)
        qc.cx(0, 1)
        qc.rz(0.2, 0)
    return qc
