import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ampamp import (  # noqa: E402
    DQAAEngine,
    DistributedAuditor,
    FOQAEngine,
    FOQAAuditor,
    FPAAAuditor,
    FixedPointEngine,
    FundamentalLimitsAuditor,
    GroverAuditor,
    GroverEngine,
    ObliviousAuditor,
    ObliviousEngine,
    QSVTAuditor,
    SU2QSPEngine,
    VTAAAuditor,
    VTAAEngine,
    VariableTimeBranch,
)


def test_grover_auditor_tracks_subspace_and_souffle_heatmap():
    auditor = GroverAuditor(GroverEngine(2, [1]))
    subspace = auditor.verify_subspace_rotation(max_k=1)
    clone_purity_original, clone_purity_copy = auditor.run_cloning_test(max_k=0)
    k_mesh, lambda_mesh, success = auditor.generate_souffle_heatmap(k_max=3, lambda_max=0.5, res=5)

    assert set(subspace) == {"a_k", "b_k", "purity", "success"}
    assert len(subspace["success"]) == 2
    assert np.allclose(subspace["purity"], 1.0, atol=1e-8)
    assert len(clone_purity_original) == 1
    assert len(clone_purity_copy) == 1
    assert k_mesh.shape == lambda_mesh.shape == success.shape == (5, 3)


def test_fpaa_auditor_reports_passband_and_ftqc_cost():
    auditor = FPAAAuditor(FixedPointEngine(L=3, delta=0.2))
    passband = auditor.audit_passband(lambda_range=np.array([0.1, 0.5, 1.0]))
    cost = auditor.estimate_ftqc_cost()

    assert passband["success_probability"].shape == (3,)
    assert passband["min_success"] >= 0.0
    assert passband["max_success"] <= 1.0
    assert passband["passband_threshold"] == pytest.approx(1.0 - 0.2**2)
    assert cost["phase_count"] == 2
    assert cost["estimated_t_count"] >= 0


def test_oblivious_auditor_distance_and_acid_test():
    auditor = ObliviousAuditor(ObliviousEngine(m_data_qubits=1))
    distance = auditor.verify_lcu_distance(
        actual_matrix=np.eye(2),
        target_hamiltonian=np.eye(2),
        alpha=1.0,
    )

    assert distance == pytest.approx(0.0)
    acid = auditor.run_acid_test(num_states=4)
    assert acid["success_probabilities"].shape == (4,)
    assert acid["input_state_independent"] is True


def test_foqa_auditor_runs_recursions_and_complexity_audit():
    auditor = FOQAAuditor(FOQAEngine(theta=0.2))
    damping = auditor.audit_damping_regimes(iterations=4, mizel_c=1.4)
    empty = auditor.audit_empty_database_paradox(iterations=4)
    complexity = auditor.audit_asymptotic_complexity(target_success=0.9)

    assert set(damping) == {"underdamped", "overdamped", "critical"}
    assert all(values.shape == (4,) for values in damping.values())
    assert empty.shape == (4,)
    assert complexity["iterations_to_target"].shape == complexity["lambda_values"].shape
    assert complexity["log_log_slope"] < 0.0


def test_distributed_auditor_reports_network_diagnostics():
    auditor = DistributedAuditor(DQAAEngine(global_n=4, j_prefixes=2))
    lucky = auditor.verify_lucky_node_theorem(num_marked=2, trials=10)
    obstruction = auditor.audit_entanglement_obstruction(target_global="1010")
    noise = auditor.benchmark_nisq_noise(noise_model=object(), shots=100)
    sifting = auditor.simulate_network_sifting(shots_per_node=100)

    assert lucky["violations"] == 0
    assert obstruction["obstruction_detected"] is True
    assert noise["local_width"] == 2
    assert sifting["detection_threshold"] > sifting["null_mean"]


def test_vtaa_and_fundamental_limit_auditors_report_metrics():
    vtaa = VTAAAuditor(VTAAEngine([VariableTimeBranch(1.0, 1.0, 0.5)]))
    ratios = vtaa.sweep_cost_ratios(total_ps=0.5, t1=1.0, t2=2.0, t3=3.0)
    assert ratios["cost_ratio"].shape == (51,)
    assert ratios["best_ratio"] <= ratios["worst_ratio"]

    limits = FundamentalLimitsAuditor()
    svd = limits.audit_subspace_svd(n=2, k_max=3)
    open_system = limits.audit_open_system_trajectory(n=2, k_max=3, phase_damp_1q=0.01, phase_damp_2q=0.02)
    ftqc = limits.audit_ftqc_diffusion_scaling(n_min=2, n_max=3)
    leakage = limits.audit_phase_leakage(eps_oracle_deg=1.0, eps_diff_deg=1.0)

    assert svd["rank"] <= 2
    assert open_system["trace_distance_proxy"].shape == (4,)
    assert len(ftqc["rows"]) == 2
    assert leakage["max_leakage_proxy"] >= 0.0


def test_qsvt_auditor_reports_hardware_limit_diagnostics():
    auditor = QSVTAuditor(SU2QSPEngine())
    is_unitary, has_parity = auditor.audit_unitarity_and_parity(np.array([0.0]), tolerance=1e-10)
    gibbs = auditor.audit_gibbs_catastrophe(degree=3)
    subnorm = auditor.audit_subnormalization_hubris(dim=2)
    quantized = auditor.audit_phase_quantization(degree=3, bit_depth=8)
    parity = auditor.audit_parity_scramble(dim=3)

    assert is_unitary is True
    assert has_parity is True
    assert gibbs["coefficients"].shape == (4,)
    assert subnorm["unsafe_defect_is_psd"] is False
    assert quantized["max_sequence_error"] >= 0.0
    assert parity["mixed_parity_detected"] is True
