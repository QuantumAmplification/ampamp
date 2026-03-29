from __future__ import annotations

from typing import Any, Dict

import numpy as np
from qiskit import QuantumCircuit

from ampamp.oblivious import ObliviousEngine

from library_transpile_showcase.common import default_profiler, parity_report


def run() -> Dict[str, Any]:
    engine = ObliviousEngine(m_data_qubits=2, l_ancilla_qubits=1, p=0.6)
    block = engine.construct_block_encoding(np.eye(4, dtype=complex))
    refl = engine.get_reflections()

    # OAA-style proxy iterate: A R A^dag R repeated
    qc = QuantumCircuit(engine.l + engine.m)
    for _ in range(8):
        qc.compose(block, inplace=True)
        qc.compose(refl, inplace=True)
        qc.compose(block.inverse(), inplace=True)
        qc.compose(refl, inplace=True)

    profile = default_profiler(qc.num_qubits).profile_circuit(qc)

    return {
        "algorithm": "oblivious",
        "library_modules": ["ampamp.oblivious", "ampamp.transpilation"],
        "engine": {"m": engine.m, "l": engine.l, "p": engine.p, "iterations": 8},
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), []),
    }
