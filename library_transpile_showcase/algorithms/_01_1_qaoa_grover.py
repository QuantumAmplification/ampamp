from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from qiskit import QuantumCircuit

from ampamp.grover import GroverEngine

from library_transpile_showcase.common import default_profiler, load_reference_depths, parity_report


def run() -> Dict[str, Any]:
    engine = GroverEngine(n_qubits=5, marked_indices=[3])
    qc = QuantumCircuit(5)
    qc.h(range(5))
    qc.append(engine.get_oracle(), range(5))
    qc.rx(0.35, range(5))
    qc.append(engine.get_diffusion(), range(5))

    profile = default_profiler(qc.num_qubits).profile_circuit(qc)
    refs = load_reference_depths(Path("transpile/[RESULT]1.1_QAOA_Modular_Transpilation/hardware_profiling.jsonl"))

    return {
        "algorithm": "qaoa_grover_proxy",
        "library_modules": ["ampamp.grover", "ampamp.transpilation"],
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), refs),
    }
