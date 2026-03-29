from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ampamp.fixed_point import FixedPointEngine

from library_transpile_showcase.common import default_profiler, load_reference_depths, parity_report


def run() -> Dict[str, Any]:
    engine = FixedPointEngine(L=3, delta=0.1)
    qc = engine.build_fixed_point_circuit(num_qubits=6, marked_indices=[0])
    profile = default_profiler(qc.num_qubits).profile_circuit(qc)
    refs = load_reference_depths(Path("transpile/[RESULT]2_Fixed_Point_Modular/hardware_profiling.jsonl"))

    return {
        "algorithm": "fixed_point",
        "library_modules": ["ampamp.fixed_point", "ampamp.transpilation"],
        "engine": {
            "L": engine.L,
            "delta": engine.delta,
            "alphas": [float(x) for x in engine.alphas],
            "betas": [float(x) for x in engine.betas],
        },
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), refs),
    }
