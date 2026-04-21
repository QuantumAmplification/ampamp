from qiskit import QuantumCircuit
from ampamp.variable_time import VTAAEngine, VariableTimeBranch


def build_circuit() -> QuantumCircuit:
    _ = VTAAEngine([
        VariableTimeBranch(1.0, 0.35, 0.75),
        VariableTimeBranch(2.0, 0.40, 0.85),
        VariableTimeBranch(3.0, 0.25, 0.92),
    ])
    base = VTAAEngine.build_staged_state_circuit(0.2, 0.7)
    qc = QuantumCircuit(base.num_qubits)
    for _ in range(4):
        qc.compose(base, inplace=True)
    return qc
