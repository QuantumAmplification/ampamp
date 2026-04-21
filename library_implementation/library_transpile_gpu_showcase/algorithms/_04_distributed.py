import numpy as np
from qiskit import QuantumCircuit
from ampamp.distributed import DQAAEngine, OracleSynthesizer


def build_circuit() -> QuantumCircuit:
    eng = DQAAEngine(global_n=8, j_prefixes=2)
    parts = eng.partition_targets(["01010101", "11001100", "00111100"])
    prefix = next((k for k, v in parts.items() if v), "00")
    local = eng.build_node_circuit(
        alphas=np.array([0.3, 0.2, 0.17, 0.15, 0.12, 0.1]),
        betas=np.array([0.4, 0.1, 0.2, 0.15, 0.11, 0.08]),
        local_targets=parts.get(prefix, []),
    )
    oracle = OracleSynthesizer(8, 2, "(v0 & v1 & v2) | (~v0 & v3) | (v1 & ~v2 & v4)").compile_node_formula(prefix)
    qc = QuantumCircuit(eng.local_n)
    qc.compose(local, inplace=True)
    qc.compose(oracle, inplace=True)
    return qc
