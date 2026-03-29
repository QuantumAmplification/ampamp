from __future__ import annotations

from typing import Any, Dict

import numpy as np
from qiskit import QuantumCircuit

from ampamp.distributed import DQAAEngine, OracleSynthesizer

from library_transpile_showcase.common import default_profiler, parity_report


def run() -> Dict[str, Any]:
    engine = DQAAEngine(global_n=8, j_prefixes=2)
    partitions = engine.partition_targets(["01010101", "11001100", "00111100"])
    prefix = next((k for k, v in partitions.items() if v), "00")

    local = engine.build_node_circuit(
        alphas=np.array([0.3, 0.2, 0.17, 0.15, 0.12, 0.1]),
        betas=np.array([0.4, 0.1, 0.2, 0.15, 0.11, 0.08]),
        local_targets=partitions.get(prefix, []),
    )

    oracle = OracleSynthesizer(
        global_n=8,
        j=2,
        formula_text="(v0 & v1 & v2) | (~v0 & v3) | (v1 & ~v2 & v4)",
    ).compile_node_formula(prefix)

    qc = QuantumCircuit(engine.local_n)
    qc.compose(local, inplace=True)
    qc.compose(oracle, inplace=True)

    profile = default_profiler(qc.num_qubits).profile_circuit(qc)
    return {
        "algorithm": "distributed",
        "library_modules": ["ampamp.distributed", "ampamp.transpilation"],
        "engine": {
            "global_n": engine.global_n,
            "j_prefixes": engine.j,
            "local_n": engine.local_n,
            "selected_prefix": prefix,
        },
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), []),
    }
