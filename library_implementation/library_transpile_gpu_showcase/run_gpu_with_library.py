from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from qiskit import QuantumCircuit, transpile

from ampamp.distributed import DQAAEngine, OracleSynthesizer
from ampamp.fixed_point import FixedPointEngine
from ampamp.foqa import FOQAEngine
from ampamp.grover import GroverEngine
from ampamp.oblivious import ObliviousEngine
from ampamp.qsvt import QSVTSynthesizer
from ampamp.variable_time import VTAAEngine


@dataclass
class CircuitSpec:
    name: str
    circuit: QuantumCircuit
    modules: list[str]


def build_specs() -> Dict[str, CircuitSpec]:
    specs: Dict[str, CircuitSpec] = {}

    grover = GroverEngine(n_qubits=6, marked_indices=[10, 25])
    specs["grover"] = CircuitSpec(
        "grover",
        grover.construct_circuit(1),
        ["ampamp.grover"],
    )

    fp = FixedPointEngine(L=3, delta=0.1)
    specs["fixed_point"] = CircuitSpec(
        "fixed_point",
        fp.build_fixed_point_circuit(num_qubits=6, marked_indices=[0]),
        ["ampamp.fixed_point", "ampamp.grover"],
    )

    ob = ObliviousEngine(m_data_qubits=2, l_ancilla_qubits=1, p=0.6)
    ob_qc = QuantumCircuit(3)
    block = ob.construct_block_encoding(np.eye(4, dtype=complex))
    refl = ob.get_reflections()
    for _ in range(4):
        ob_qc.compose(block, inplace=True)
        ob_qc.compose(refl, inplace=True)
        ob_qc.compose(block.inverse(), inplace=True)
        ob_qc.compose(refl, inplace=True)
    specs["oblivious"] = CircuitSpec(
        "oblivious",
        ob_qc,
        ["ampamp.oblivious"],
    )

    foqa = FOQAEngine(theta=0.5)
    specs["foqa"] = CircuitSpec(
        "foqa",
        foqa.build_proxy_sequence(n_steps=20, mizel_c=1.4, m_content=1),
        ["ampamp.foqa"],
    )

    dqaa = DQAAEngine(global_n=8, j_prefixes=2)
    parts = dqaa.partition_targets(["01010101", "11001100", "00111100"])
    prefix = next((k for k, v in parts.items() if v), "00")
    local = dqaa.build_node_circuit(
        alphas=np.array([0.3, 0.2, 0.17, 0.15]),
        betas=np.array([0.4, 0.1, 0.2, 0.15]),
        local_targets=parts.get(prefix, []),
    )
    oracle = OracleSynthesizer(
        global_n=8,
        j=2,
        formula_text="(v0 & v1 & v2) | (~v0 & v3) | (v1 & ~v2 & v4)",
    ).compile_node_formula(prefix)
    dqaa_qc = QuantumCircuit(dqaa.local_n)
    dqaa_qc.compose(local, inplace=True)
    dqaa_qc.compose(oracle, inplace=True)
    specs["distributed"] = CircuitSpec(
        "distributed",
        dqaa_qc,
        ["ampamp.distributed"],
    )

    vtaa_base = VTAAEngine.build_staged_state_circuit(p_s1=0.2, p_fail_cond=0.7)
    vtaa_qc = QuantumCircuit(vtaa_base.num_qubits)
    for _ in range(3):
        vtaa_qc.compose(vtaa_base, inplace=True)
    specs["variable_time"] = CircuitSpec(
        "variable_time",
        vtaa_qc,
        ["ampamp.variable_time"],
    )

    coeffs = QSVTSynthesizer.synthesize_matrix_inverse(degree=21, kappa=6.0)
    qsvt_qc = QuantumCircuit(1)
    for idx, c in enumerate(coeffs[:22]):
        qsvt_qc.rz(float(np.clip(c, -1.0, 1.0)), 0)
        qsvt_qc.rx(np.pi / 9.0 if idx % 2 == 0 else np.pi / 13.0, 0)
    specs["qsvt"] = CircuitSpec(
        "qsvt",
        qsvt_qc,
        ["ampamp.qsvt"],
    )

    return specs


def _get_backend(prefer_gpu: bool = True):
    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:
        return None, f"qiskit_aer unavailable: {exc}"

    if prefer_gpu:
        try:
            backend = AerSimulator(method="statevector", device="GPU")
            return backend, "GPU"
        except Exception:
            pass

    try:
        backend = AerSimulator(method="statevector")
        return backend, "CPU"
    except Exception as exc:
        return None, f"AerSimulator init failed: {exc}"


def transpile_and_run(backend, spec: CircuitSpec, shots: int = 1024) -> Dict[str, Any]:
    basis_gates = ["cx", "id", "rz", "sx", "x"]
    transpiled = transpile(spec.circuit, backend=backend, basis_gates=basis_gates, optimization_level=3)

    profile = {
        "logical_depth": int(spec.circuit.depth()),
        "logical_gates": int(sum(spec.circuit.count_ops().values())),
        "post_optimization_depth": int(transpiled.depth()),
        "final_cnots": int(transpiled.count_ops().get("cx", 0)),
        "final_gates": int(sum(transpiled.count_ops().values())),
    }

    # execute if possible
    exec_info: Dict[str, Any] = {"executed": False}
    try:
        qc_run = transpiled.copy()
        if not any(inst.operation.name == "measure" for inst in qc_run.data):
            qc_run.measure_all()
        result = backend.run(qc_run, shots=shots).result()
        counts = result.get_counts(qc_run)
        exec_info = {
            "executed": True,
            "shots": int(shots),
            "support_size": int(len(counts)),
            "dominant_count": int(max(counts.values()) if counts else 0),
        }
    except Exception as exc:
        exec_info = {
            "executed": False,
            "reason": str(exc),
        }

    return {
        "algorithm": spec.name,
        "library_modules": spec.modules,
        "profile": profile,
        "execution": exec_info,
    }


def main() -> None:
    out_dir = ROOT / "library_implementation" / "library_transpile_gpu_showcase" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    backend, mode = _get_backend(prefer_gpu=True)
    if backend is None:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "reason": mode,
        }
        (out_dir / "gpu_transpile_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(payload)
        return

    specs = build_specs()
    results = {name: transpile_and_run(backend, spec) for name, spec in specs.items()}

    bundle = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend_mode": mode,
        "results": results,
    }

    json_path = out_dir / "gpu_transpile_results.json"
    json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / "gpu_transpile_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "backend_mode",
                "post_optimization_depth",
                "final_cnots",
                "final_gates",
                "executed",
                "support_size",
            ],
        )
        writer.writeheader()
        for name, item in results.items():
            profile = item["profile"]
            execution = item["execution"]
            writer.writerow(
                {
                    "algorithm": name,
                    "backend_mode": mode,
                    "post_optimization_depth": profile.get("post_optimization_depth"),
                    "final_cnots": profile.get("final_cnots"),
                    "final_gates": profile.get("final_gates"),
                    "executed": execution.get("executed"),
                    "support_size": execution.get("support_size", ""),
                }
            )

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
