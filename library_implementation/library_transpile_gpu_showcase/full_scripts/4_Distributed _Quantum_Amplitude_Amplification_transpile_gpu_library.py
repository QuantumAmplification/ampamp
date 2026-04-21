import os, sys
import numpy as np
from qiskit import QuantumCircuit, transpile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ampamp.distributed import DQAAEngine, OracleSynthesizer


def build_circuit():
    eng = DQAAEngine(8,2)
    parts = eng.partition_targets(['01010101','11001100','00111100'])
    prefix = next((k for k,v in parts.items() if v), '00')
    local = eng.build_node_circuit(np.array([0.3,0.2,0.17]), np.array([0.4,0.2,0.15]), parts.get(prefix, []))
    oracle = OracleSynthesizer(8,2,'(v0 & v1 & v2) | (~v0 & v3)').compile_node_formula(prefix)
    qc = QuantumCircuit(eng.local_n)
    qc.compose(local, inplace=True)
    qc.compose(oracle, inplace=True)
    return qc


def run_scenario_a():
    t = transpile(build_circuit(), basis_gates=['cx','id','rz','sx','x'], optimization_level=3)
    print('DQAA A depth=', t.depth(), 'cx=', t.count_ops().get('cx',0))


if __name__ == '__main__':
    run_scenario_a()
