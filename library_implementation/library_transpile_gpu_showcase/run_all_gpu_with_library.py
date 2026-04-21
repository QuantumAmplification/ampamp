from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library_implementation.library_transpile_gpu_showcase.algorithms import ALL_BUILDERS  # noqa: E402
from library_implementation.library_transpile_gpu_showcase.common_gpu import (  # noqa: E402
    default_profiler,
    detect_gpu_backend,
    gpu_transpile_metrics,
)


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
    except Exception:
        pass
    return v


def main() -> None:
    out_dir = ROOT / "library_implementation" / "library_transpile_gpu_showcase" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    backend, gpu_info = detect_gpu_backend()

    results = {}
    for name, builder in ALL_BUILDERS.items():
        qc = builder()
        profile = default_profiler(qc.num_qubits).profile_circuit(qc)
        gpu_metrics = {}
        if backend is not None:
            try:
                gpu_metrics = gpu_transpile_metrics(qc, backend)
            except Exception as exc:
                gpu_metrics = {"gpu_transpile_error": str(exc)}

        results[name] = {
            "algorithm": name,
            "num_qubits": int(qc.num_qubits),
            "profile": _jsonable(profile),
            "gpu_metrics": _jsonable(gpu_metrics),
            "gpu_mode": gpu_info.mode,
            "gpu_available": gpu_info.available,
            "gpu_note": gpu_info.reason,
        }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "library gpu transpilation showcase",
        "results": results,
    }

    json_path = out_dir / "library_gpu_transpile_results.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / "library_gpu_transpile_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "num_qubits",
                "post_optimization_depth",
                "final_cnots",
                "hardware_penalty_score",
                "gpu_mode",
                "gpu_transpiled_depth",
                "gpu_transpiled_cnots",
            ],
        )
        writer.writeheader()
        for name, r in results.items():
            p = r.get("profile", {})
            g = r.get("gpu_metrics", {})
            writer.writerow(
                {
                    "algorithm": name,
                    "num_qubits": r.get("num_qubits"),
                    "post_optimization_depth": p.get("post_optimization_depth"),
                    "final_cnots": p.get("final_cnots"),
                    "hardware_penalty_score": p.get("hardware_penalty_score"),
                    "gpu_mode": r.get("gpu_mode"),
                    "gpu_transpiled_depth": g.get("gpu_transpiled_depth"),
                    "gpu_transpiled_cnots": g.get("gpu_transpiled_cnots"),
                }
            )

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
