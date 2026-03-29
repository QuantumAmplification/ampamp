import numpy as np
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from qiskit import QuantumCircuit
from .core import transpile_for_hardware, CircuitMetrics

@dataclass
class ScenarioResult:
    """The result of a single hardware profiling scenario."""
    label: str
    metrics: CircuitMetrics
    extra: Dict[str, Any] = None

class AlgorithmProfiler:
    """Profiles quantum algorithms against diverse physical constraints."""
    
    def __init__(self, name: str):
        self.name = name
        self.scenarios: Dict[str, Callable] = {}
        
    def add_scenario(self, label: str, func: Callable):
        """Adds a named scenario to the profiler."""
        self.scenarios[label.upper()] = func
        
    def run_scenario(self, label: str, **kwargs: Any) -> ScenarioResult:
        """Executes a specific scenario with custom parameters."""
        label = label.upper()
        if label not in self.scenarios:
            raise ValueError(f"Unknown scenario '{label}'.")
        
        # Scenario functions should return (QuantumCircuit, extra_metrics)
        qc, extra = self.scenarios[label](**kwargs)
        
        # Default hardware transpile if not already done in scenario
        if isinstance(qc, tuple):
             t_qc, metrics = qc
        else:
            # Generic catch-all transpile
            t_qc, metrics = transpile_for_hardware(qc)
            
        return ScenarioResult(label=label, metrics=metrics, extra=extra)

    @staticmethod
    def calculate_swap_overhead(base_metrics: CircuitMetrics, constrained_metrics: CircuitMetrics) -> int:
        """Estimates the number of SWAP gates added by connectivity constraints."""
        # Assuming each SWAP decomposes into 3 CX gates
        extra_cx = constrained_metrics.cx_count - base_metrics.cx_count
        return max(0, extra_cx // 3)

    @staticmethod
    def calculate_depth_penalty(base_metrics: CircuitMetrics, constrained_metrics: CircuitMetrics) -> float:
        """Calculates the multiplicative depth penalty of hardware constraints."""
        if base_metrics.depth == 0:
            return 1.0
        return constrained_metrics.depth / base_metrics.depth
