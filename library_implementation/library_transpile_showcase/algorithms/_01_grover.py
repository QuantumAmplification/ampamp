from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ampamp.grover import GroverEngine

from library_transpile_showcase.common import default_profiler, load_reference_depths, parity_report


def run() -> Dict[str, Any]:
    engine = GroverEngine(n_qubits=6, marked_indices=[10, 25])
    qc = engine.construct_circuit(1)
    profile = default_profiler(qc.num_qubits).profile_circuit(qc)

    success_curve = [
        GroverEngine.compute_success_prob(engine.solution_density, k)
        for k in range(4)
    ]
    refs = load_reference_depths(Path("transpile/[RESULT]1_Grover_Modular_Transpilation/hardware_profiling.jsonl"))

    return {
        "algorithm": "grover",
        "library_modules": ["ampamp.grover", "ampamp.transpilation"],
        "engine": {
            "n_qubits": engine.n,
            "marked": engine.marked,
            "k_optimal": engine.k_optimal,
        },
        "diagnostics": {
            "solution_density": float(engine.solution_density),
            "theta": float(engine.theta),
            "success_curve_k0_k3": [float(x) for x in success_curve],
        },
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), refs),
    }
