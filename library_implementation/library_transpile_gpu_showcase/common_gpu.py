from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit, transpile

try:
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover
    AerSimulator = None  # type: ignore[assignment]

from ampamp.transpilation import TranspilationProfileConfig, TranspilationProfiler


@dataclass
class GpuRunInfo:
    available: bool
    mode: str
    reason: str = ""


def detect_gpu_backend() -> tuple[Optional[Any], GpuRunInfo]:
    if AerSimulator is None:
        return None, GpuRunInfo(False, "cpu", "qiskit-aer unavailable")
    try:
        backend = AerSimulator(method="statevector", device="GPU")
        return backend, GpuRunInfo(True, "gpu")
    except Exception as exc:
        return AerSimulator(method="statevector"), GpuRunInfo(False, "cpu", f"GPU unavailable: {exc}")


def default_profiler(n_qubits: int) -> TranspilationProfiler:
    edges = [[i, j] for i in range(n_qubits) for j in range(n_qubits) if i != j]
    return TranspilationProfiler(
        TranspilationProfileConfig(
            coupling_map_edges=edges,
            optimize_optimization_level=3,
        )
    )


def gpu_transpile_metrics(circuit: QuantumCircuit, backend: Any) -> Dict[str, Any]:
    t_qc = transpile(circuit, backend=backend, optimization_level=3)
    ops = t_qc.count_ops()
    return {
        "gpu_transpiled_depth": int(t_qc.depth()),
        "gpu_transpiled_gates": int(sum(ops.values())),
        "gpu_transpiled_cnots": int(ops.get("cx", 0)),
    }
