import json
import os
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from .core import CircuitMetrics

class HardwareReport:
    """Handles the persistence and summary generation of hardware profiling runs."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "hardware_profiling.jsonl")
        
    def log_result(self, scenario_label: str, metrics: CircuitMetrics, extra: Optional[Dict[str, Any]] = None):
        """Saves a profiling result to a JSON Lines file."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record = {
            "timestamp": timestamp,
            "scenario": scenario_label,
            "metrics": asdict(metrics),
            "extra": extra or {}
        }
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
            
    def generate_summary_text(self) -> str:
        """Reads back the logs and generates a textual summary table."""
        if not os.path.exists(self.log_path):
             return "No profiling logs found."
             
        summary = ["HARDWARE PROFILING SUMMARY", "=" * 25]
        with open(self.log_path, 'r', encoding='utf-8') as f:
             for line in f:
                 data = json.loads(line)
                 sc = data['scenario']
                 mt = data['metrics']
                 summary.append(f"Scenario {sc}:")
                 summary.append(f"  - Depth: {mt['depth']}")
                 summary.append(f"  - CX Count: {mt['gate_counts'].get('cx', 0)}")
                 summary.append(f"  - Blowup: {mt['blowup_factor']:.2f}x")
                 
        return "\n".join(summary)

    def save_plot(self, plt_func: Any, filename: str = "profiling_plot.png"):
        """Saves a plot using a provided matplotlib-compatible function."""
        # This is a stub for potential plotting integration
        path = os.path.join(self.output_dir, filename)
        # plt_func(path)
        return path
