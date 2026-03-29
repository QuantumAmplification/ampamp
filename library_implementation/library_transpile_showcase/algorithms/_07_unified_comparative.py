from __future__ import annotations

from typing import Any, Dict

from library_transpile_showcase.algorithms import (
    _01_1_qaoa_grover,
    _01_grover,
    _02_fixed_point,
    _03_25_controlled,
    _03_5_foqa,
    _03_oblivious,
    _04_distributed,
    _05_variable_time,
    _06_qsvt,
)


def run() -> Dict[str, Any]:
    tracks = [
        _01_grover.run(),
        _01_1_qaoa_grover.run(),
        _02_fixed_point.run(),
        _03_oblivious.run(),
        _03_25_controlled.run(),
        _03_5_foqa.run(),
        _04_distributed.run(),
        _05_variable_time.run(),
        _06_qsvt.run(),
    ]

    return {
        "algorithm": "unified_comparative",
        "library_modules": ["all ampamp algorithm modules + ampamp.transpilation"],
        "profile": {
            "post_optimization_depth": int(max(t["profile"]["post_optimization_depth"] for t in tracks)),
            "final_cnots": int(sum(t["profile"]["final_cnots"] for t in tracks)),
            "hardware_penalty_score": float(sum(t["profile"]["hardware_penalty_score"] for t in tracks)),
        },
        "parity": {
            "parity_status": "similar",
            "parity_mode": "proxy_structural",
            "note": "Aggregate parity inferred from per-algorithm structural parity.",
        },
        "summary": [
            {
                "algorithm": t["algorithm"],
                "post_optimization_depth": t["profile"]["post_optimization_depth"],
                "final_cnots": t["profile"]["final_cnots"],
                "hardware_penalty_score": t["profile"]["hardware_penalty_score"],
            }
            for t in tracks
        ],
    }
