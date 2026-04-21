import os, sys
from qiskit import QuantumCircuit, transpile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ampamp.variable_time import VTAAEngine, VariableTimeBranch


def build_circuit():
    _ = VTAAEngine([VariableTimeBranch(1.0,0.35,0.75), VariableTimeBranch(2.0,0.4,0.85), VariableTimeBranch(3.0,0.25,0.92)])
    base = VTAAEngine.build_staged_state_circuit(0.2,0.7)
    qc = QuantumCircuit(base.num_qubits)
    for _ in range(4):
        qc.compose(base, inplace=True)
    return qc


def run_scenario_a():
    t = transpile(build_circuit(), basis_gates=['cx','id','rz','sx','x'], optimization_level=3)
    print('VTAA A depth=', t.depth(), 'cx=', t.count_ops().get('cx',0))


if __name__ == '__main__':
    run_scenario_a()
