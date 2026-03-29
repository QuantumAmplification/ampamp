from __future__ import annotations

from typing import Any, Dict

import numpy as np
from qiskit import QuantumCircuit

from ampamp.diagnostics import QSVTAuditor
from ampamp.qsvt import QSVTSynthesizer, SU2QSPEngine

from library_transpile_showcase.common import default_profiler, parity_report


def run() -> Dict[str, Any]:
    engine = SU2QSPEngine()
    degree = 41
    coeffs = QSVTSynthesizer.synthesize_matrix_inverse(degree=degree, kappa=6.0)

    qc = QuantumCircuit(1)
    for idx, c in enumerate(coeffs[: degree + 1]):
        qc.rz(float(np.clip(c, -1.0, 1.0)), 0)
        qc.rx(np.pi / 9.0 if idx % 2 == 0 else np.pi / 13.0, 0)

    profile = default_profiler(qc.num_qubits).profile_circuit(qc)
    phases = np.linspace(0.0, np.pi / 2.0, 10)
    unitary_ok, parity_ok = QSVTAuditor(engine).audit_unitarity_and_parity(phases)

    return {
        "algorithm": "qsvt",
        "library_modules": ["ampamp.qsvt", "ampamp.diagnostics", "ampamp.transpilation"],
        "diagnostics": {
            "unitarity_ok": bool(unitary_ok),
            "parity_ok": bool(parity_ok),
            "degree": degree,
        },
        "profile": profile,
        "parity": parity_report(int(profile["post_optimization_depth"]), []),
    }
