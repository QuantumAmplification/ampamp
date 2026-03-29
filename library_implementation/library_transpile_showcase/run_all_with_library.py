from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from library_transpile_showcase.algorithms import ALL_ALGORITHM_RUNNERS  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass
    return value


def main() -> None:
    out_dir = ROOT / "library_transpile_showcase" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {name: _jsonable(run()) for name, run in ALL_ALGORITHM_RUNNERS.items()}

    bundle = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "per-algorithm library transpilation showcase",
        "algorithms": results,
    }

    json_path = out_dir / "library_transpile_results.json"
    json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / "library_transpile_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "post_optimization_depth",
                "final_cnots",
                "hardware_penalty_score",
                "parity_status",
            ],
        )
        writer.writeheader()
        for name, payload in results.items():
            profile = payload.get("profile", {})
            parity = payload.get("parity", {})
            writer.writerow(
                {
                    "algorithm": name,
                    "post_optimization_depth": profile.get("post_optimization_depth"),
                    "final_cnots": profile.get("final_cnots"),
                    "hardware_penalty_score": profile.get("hardware_penalty_score"),
                    "parity_status": parity.get("parity_status"),
                }
            )

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
