from __future__ import annotations

from typing import Any, Dict

from qiskit import QuantumCircuit

from ampamp.variable_time import VTAAEngine, VariableTimeBranch

from library_transpile_showcase.common import default_profiler, parity_report


def run() -> Dict[str, Any]:
    branches = [
        VariableTimeBranch(1.0, 0.35, 0.75),
        VariableTimeBranch(2.0, 0.40, 0.85),
        VariableTimeBranch(3.0, 0.25, 0.92),
    ]
    engine = VTAAEngine(branches)

    base = VTAAEngine.build_staged_state_circuit(p_s1=0.2, p_fail_cond=0.7)
    qc = QuantumCircuit(base.num_qubits)
    for _ in range(4):
        qc.compose(base, inplace=True)

    profile = default_profiler(qc.num_qubits).profile_circuit(qc)

    t_mean, t_rms, t_max = engine.stopping_time_moments()
    return {
        "algorithm": "variable_time",
        "library_modules": ["ampamp.variable_time", "ampamp.transpilation"],
        "engine": {
            "p_success": float(engine.p_success),
            "t_mean": float(t_mean),
            "t_rms": float(t_rms),
            "t_max": float(t_max),
            "asymptotic_bound": float(engine.vtaa_asymptotic_bound()),
        },
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), []),
    }
