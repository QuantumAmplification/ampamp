import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ampamp  # noqa: E402


def test_public_api_exports_are_importable_and_unique():
    assert len(ampamp.__all__) == len(set(ampamp.__all__))
    for name in ampamp.__all__:
        assert hasattr(ampamp, name), f"ampamp.__all__ exposes missing name {name}"


def test_public_api_includes_framework_level_utilities():
    expected = {
        "GroverEngine",
        "FixedPointEngine",
        "ObliviousEngine",
        "FOQAEngine",
        "DQAAEngine",
        "OracleSynthesizer",
        "VTAAEngine",
        "VariableTimeBranch",
        "SU2QSPEngine",
        "QSVTSynthesizer",
        "IQAEEngine",
        "IQAEResult",
        "IQAEConfig",
        "OracleBuilder",
        "OracleSpec",
        "build_phase_oracle",
        "build_bit_flip_oracle",
        "marked_bitstrings_from_formula",
        "EntanglementCountConfig",
        "profile_entanglement_counts",
        "TranspilationProfiler",
        "TranspilationBatchProfiler",
        "TranspilationProfileConfig",
        "HardwareCostWeights",
        "BackendValidationRunner",
        "BackendValidationConfig",
        "ValidationNoiseConfig",
        "ValidationLogConfig",
        "GroverAuditor",
        "FPAAAuditor",
        "ObliviousAuditor",
        "FOQAAuditor",
        "DistributedAuditor",
        "VTAAAuditor",
        "FundamentalLimitsAuditor",
        "QSVTAuditor",
    }

    assert expected.issubset(set(ampamp.__all__))
