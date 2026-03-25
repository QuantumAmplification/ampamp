"""Amplitude Amplification Library.

This package provides a production-ready, pip-installable library for 
research-grade quantum computing amplitude amplification algorithms. 
It includes various engines for Grover, Fixed-Point, Oblivious, FOQA, 
Distributed Quantum Amplitude Amplification, Variable-Time, and QSVT.
"""

# src/ampamp/__init__.py
from .grover import GroverEngine
from .fixed_point import FixedPointEngine
from .oblivious import ObliviousEngine
from .foqa import FOQAEngine
from .distributed import DQAAEngine, OracleSynthesizer
from .variable_time import VTAAEngine, VariableTimeBranch
from .qsvt import SU2QSPEngine, QSVTSynthesizer  
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
    "GroverEngine", "FixedPointEngine", "ObliviousEngine", "FOQAEngine", 
    "DQAAEngine", "OracleSynthesizer", "VTAAEngine", "VariableTimeBranch",
    "SU2QSPEngine", "QSVTSynthesizer",
    "GroverAuditor", "FPAAAuditor", "ObliviousAuditor", "FOQAAuditor", 
    "DistributedAuditor", "VTAAAuditor", "FundamentalLimitsAuditor", "QSVTAuditor"
]