"""Amplitude Amplification Library.

This package provides a production-ready, pip-installable library for 
research-grade quantum computing amplitude amplification algorithms. 
It includes various engines for Grover, Fixed-Point, Oblivious, FOQA, 
Distributed Quantum Amplitude Amplification, Variable-Time, and QSVT.
"""

# src/ampamp/__init__.py
from .grover import GroverEngine
from .oracles import (
    OracleBuilder,
    OracleSpec,
    build_bit_flip_oracle,
    build_phase_oracle,
    build_unitary_oracle,
    marked_bitstrings_from_formula,
)
from .entanglement import EntanglementCountConfig, profile_entanglement_counts
from .fixed_point import FixedPointEngine
from .oblivious import ObliviousEngine
from .foqa import FOQAEngine
from .distributed import DQAAEngine, OracleSynthesizer
from .variable_time import VTAAEngine, VariableTimeBranch
from .qsvt import SU2QSPEngine, QSVTSynthesizer  
from .iqae import IQAEEngine, IQAEResult, IQAEConfig
from .transpilation import TranspilationProfiler, TranspilationBatchProfiler, TranspilationProfileConfig, HardwareCostWeights
from .transpilation_validation import (
    BackendValidationRunner,
    BackendValidationConfig,
    ValidationNoiseConfig,
    ValidationLogConfig,
)
from .diagnostics import (
    GroverAuditor, 
    FPAAAuditor, 
    ObliviousAuditor,
    FOQAAuditor,
    DistributedAuditor,
    VTAAAuditor,
    FundamentalLimitsAuditor,
    QSVTAuditor  
)

__all__ = [
    "GroverEngine",
    "OracleBuilder",
    "OracleSpec",
    "build_phase_oracle",
    "build_bit_flip_oracle",
    "build_unitary_oracle",
    "marked_bitstrings_from_formula",
    "EntanglementCountConfig",
    "profile_entanglement_counts",
    "FixedPointEngine",
    "ObliviousEngine",
    "FOQAEngine",
    "DQAAEngine", "OracleSynthesizer", "VTAAEngine", "VariableTimeBranch",
    "SU2QSPEngine", "QSVTSynthesizer",
    "IQAEEngine", "IQAEResult", "IQAEConfig",
    "TranspilationProfiler", "TranspilationBatchProfiler", "TranspilationProfileConfig", "HardwareCostWeights",
    "BackendValidationRunner", "BackendValidationConfig", "ValidationNoiseConfig", "ValidationLogConfig",
    "GroverAuditor", "FPAAAuditor", "ObliviousAuditor", "FOQAAuditor", 
    "DistributedAuditor", "VTAAAuditor", "FundamentalLimitsAuditor", "QSVTAuditor"
]
