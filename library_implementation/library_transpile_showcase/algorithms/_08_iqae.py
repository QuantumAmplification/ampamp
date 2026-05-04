from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ampamp.grover import GroverEngine
from ampamp.iqae import IQAEEngine, IQAEConfig

from library_transpile_showcase.common import default_profiler, load_reference_depths, parity_report


def run() -> Dict[str, Any]:
    g_engine = GroverEngine(n_qubits=6, marked_indices=[10, 25])
    iqae_cfg = IQAEConfig(epsilon=0.01)
    engine = IQAEEngine(g_engine, iqae_cfg)

    # Profiling a deep IQAE query
    qc = g_engine.construct_circuit(4)
    profile = default_profiler(qc.num_qubits).profile_circuit(qc)

    refs = load_reference_depths(Path("transpile/[RESULT]7_Iterative_Quantum_Amplitude_Estimation_transpile/hardware_profiling.jsonl"))

    return {
        "algorithm": "iqae",
        "library_modules": ["ampamp.iqae", "ampamp.transpilation"],
        "engine": {
            "n_qubits": g_engine.n,
            "epsilon": iqae_cfg.epsilon,
        },
        "diagnostics": {
            "k_sampled": 4
        },
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), refs),
    }
