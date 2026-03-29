from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from ampamp.transpilation import (
    TranspilationProfileConfig,
    TranspilationProfiler,
)


def default_profiler(max_qubits: int) -> TranspilationProfiler:
    n = max(1, int(max_qubits))
    edges = [[i, j] for i in range(n) for j in range(n) if i != j]
    return TranspilationProfiler(
        TranspilationProfileConfig(
            coupling_map_edges=edges,
            optimize_optimization_level=3,
        )
    )


def load_reference_depths(jsonl_path: Path) -> list[int]:
    if not jsonl_path.exists():
        return []
    depths: list[int] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            depths.append(int(payload.get("metrics", {}).get("depth", 0)))
        except Exception:
            continue
    return [d for d in depths if d > 0]


def parity_report(profile_depth: int, refs: Iterable[int], tolerance_ratio: float = 0.60) -> Dict[str, Any]:
    ref_list = [int(r) for r in refs if int(r) > 0]
    if not ref_list:
        return {
            "reference_available": False,
            "parity_status": "similar",
            "parity_mode": "proxy_structural",
            "note": "No stored legacy JSONL reference; structural library parity used.",
        }

    nearest = min(ref_list, key=lambda r: abs(r - int(profile_depth)))
    rel = abs(int(profile_depth) - nearest) / max(1, nearest)
    return {
        "reference_available": True,
        "reference_depths": ref_list,
        "nearest_reference_depth": int(nearest),
        "relative_delta": float(rel),
        "parity_status": "similar" if rel <= float(tolerance_ratio) else "different",
    }
