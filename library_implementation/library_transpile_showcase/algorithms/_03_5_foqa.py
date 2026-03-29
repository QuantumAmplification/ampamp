from __future__ import annotations

from typing import Any, Dict

from ampamp.foqa import FOQAEngine

from library_transpile_showcase.common import default_profiler, parity_report


def run() -> Dict[str, Any]:
    engine = FOQAEngine(theta=0.5)
    sched = FOQAEngine.generate_mizel_schedule(c=1.4, iterations=32)
    probs = engine.simulate_recurrence(sched)
    qc = engine.build_proxy_sequence(n_steps=32, mizel_c=1.4, m_content=1)

    profile = default_profiler(qc.num_qubits).profile_circuit(qc)
    return {
        "algorithm": "foqa",
        "library_modules": ["ampamp.foqa", "ampamp.transpilation"],
        "engine": {
            "theta": engine.theta,
            "n_steps": 32,
            "success_first": float(probs[0]),
            "success_last": float(probs[-1]),
        },
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), []),
    }
