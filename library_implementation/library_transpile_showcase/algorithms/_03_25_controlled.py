from __future__ import annotations

from typing import Any, Dict

from qiskit import QuantumCircuit

from ampamp.grover import GroverEngine

from library_transpile_showcase.common import default_profiler, parity_report


def run() -> Dict[str, Any]:
    # Controlled proxy with repeated controlled Grover-like skeleton.
    _ = GroverEngine(n_qubits=2, marked_indices=[1])
    qc = QuantumCircuit(3)
    qc.h(0)
    for _ in range(6):
        qc.ch(0, 1)
        qc.ccx(0, 1, 2)
        qc.ch(0, 2)
        qc.cx(0, 1)
        qc.rz(0.2, 0)

    profile = default_profiler(qc.num_qubits).profile_circuit(qc)
    return {
        "algorithm": "controlled_quantum_amplification_proxy",
        "library_modules": ["ampamp.grover", "ampamp.transpilation"],
        "engine": {"repetitions": 6},
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), []),
    }
