import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ampamp import (  # noqa: E402
    DQAAEngine,
    FOQAEngine,
    FixedPointEngine,
    GroverEngine,
    IQAEConfig,
    IQAEEngine,
    IQAEResult,
    ObliviousEngine,
    OracleSynthesizer,
    QSVTSynthesizer,
    SU2QSPEngine,
    VTAAEngine,
    VariableTimeBranch,
)


def _is_unitary(matrix: np.ndarray, atol: float = 1e-8) -> bool:
    ident = np.eye(matrix.shape[0], dtype=complex)
    return bool(np.allclose(matrix.conj().T @ matrix, ident, atol=atol))


def test_grover_engine_constructs_oracle_diffusion_and_circuit():
    engine = GroverEngine(n_qubits=3, marked_indices=[5, 5])

    assert engine.N == 8
    assert engine.M == 1
    assert engine.solution_density == pytest.approx(1 / 8)
    assert GroverEngine.compute_success_prob(engine.solution_density, 0) == pytest.approx(1 / 8)

    oracle = engine.get_oracle()
    diffusion = engine.get_diffusion()
    circuit = engine.construct_circuit(iterations=2)

    assert oracle.num_qubits == 3
    assert diffusion.num_qubits == 3
    assert circuit.num_qubits == 3
    assert _is_unitary(Operator(oracle).data)
    assert _is_unitary(Operator(diffusion).data)


@pytest.mark.parametrize(
    "args, message",
    [
        ((0, [0]), "n_qubits"),
        ((2, []), "at least one"),
        ((2, [4]), "between 0 and 3"),
    ],
)
def test_grover_engine_rejects_invalid_problem_specs(args, message):
    with pytest.raises(ValueError, match=message):
        GroverEngine(*args)


def test_fixed_point_engine_generates_phases_and_circuit():
    engine = FixedPointEngine(L=5, delta=0.1)
    circuit = engine.build_fixed_point_circuit(num_qubits=3, marked_indices=[1, 6])
    one_qubit_circuit = engine.build_fixed_point_circuit(num_qubits=1, marked_indices=[1])
    one_qubit_state = Statevector.from_instruction(one_qubit_circuit)

    assert engine.num_grover_iterates == 2
    assert len(engine.alphas) == 2
    assert len(engine.betas) == 2
    assert np.allclose(engine.betas, -engine.alphas[::-1])
    assert len(engine.zetas) == 5
    assert np.allclose(engine.zetas, engine.zetas[::-1])
    assert engine.gamma == pytest.approx(1.0 / np.cosh(np.arccosh(10.0) / 5.0))
    assert engine.lambda_min == pytest.approx(1.0 - engine.gamma**2)
    assert engine.success_probability(engine.lambda_min) == pytest.approx(1.0 - engine.delta**2)
    assert engine.success_probability(1.0) == pytest.approx(1.0)
    assert one_qubit_state.probabilities()[1] == pytest.approx(engine.success_probability(0.5))
    assert circuit.num_qubits == 3
    assert circuit.count_ops()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"L": 0, "delta": 0.1}, "L must be"),
        ({"L": 4, "delta": 0.1}, "odd"),
        ({"L": 3, "delta": 0.0}, "delta"),
    ],
)
def test_fixed_point_engine_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FixedPointEngine(**kwargs)


def test_oblivious_engine_prepares_ancilla_and_block_encoding():
    engine = ObliviousEngine(m_data_qubits=1, l_ancilla_qubits=1, p=0.25)
    rotation = engine.get_ancilla_rotation()
    state = Statevector.from_instruction(rotation)
    block = engine.construct_block_encoding(np.eye(2, dtype=complex))
    reflection = engine.get_reflections()

    assert state.probabilities()[0] == pytest.approx(0.25)
    assert block.num_qubits == 2
    assert "c-unitary" in block.count_ops()
    assert reflection.global_phase == pytest.approx(np.pi)


def test_oblivious_engine_validation():
    with pytest.raises(ValueError, match="m_data_qubits"):
        ObliviousEngine(0)
    with pytest.raises(ValueError, match="l_ancilla"):
        ObliviousEngine(1, 0)
    with pytest.raises(ValueError, match="p"):
        ObliviousEngine(1, 1, 1.5)
    with pytest.raises(ValueError, match="shape"):
        ObliviousEngine(2).construct_block_encoding(np.eye(2))


def test_foqa_engine_schedules_recurrence_and_proxy_circuit():
    engine = FOQAEngine(theta=0.3)
    mizel = engine.generate_mizel_schedule(c=1.5, iterations=4)
    constant = engine.generate_constant_schedule(alpha=0.2, iterations=4)
    split = engine.build_lcu_split_operator(alpha_n=0.4)
    recurrence = engine.simulate_recurrence(mizel)
    proxy = engine.build_proxy_sequence(n_steps=3, m_content=2)

    assert mizel.shape == (4,)
    assert np.allclose(constant, np.full(4, 0.2))
    assert split.shape == (8, 8)
    assert _is_unitary(split)
    assert np.all((0.0 <= recurrence) & (recurrence <= 1.0))
    assert proxy.num_qubits == 4


def test_foqa_engine_validation():
    with pytest.raises(ValueError, match="Theta"):
        FOQAEngine(np.pi / 2)
    with pytest.raises(ValueError, match="iterations"):
        FOQAEngine.generate_mizel_schedule(1.0, 0)
    with pytest.raises(ValueError, match="n_steps"):
        FOQAEngine(0.2).build_proxy_sequence(0)


def test_distributed_engine_partitions_and_synthesizes_local_oracle():
    engine = DQAAEngine(global_n=4, j_prefixes=2)
    partitions = engine.partition_targets(["0011", "1001", "1010"])

    assert partitions["00"] == ["11"]
    assert partitions["10"] == ["01", "10"]
    assert partitions["01"] == []

    synth = OracleSynthesizer(global_n=3, j=1, formula_text="v0 & v1")
    false_oracle = synth.compile_node_formula(prefix="0")
    true_oracle = OracleSynthesizer(global_n=3, j=1, formula_text="v0 | ~v0").compile_node_formula("1")
    local_oracle = synth.compile_node_formula(prefix="1")

    assert false_oracle.num_qubits == 2
    assert false_oracle.global_phase == pytest.approx(0.0)
    assert true_oracle.global_phase == pytest.approx(np.pi)
    assert local_oracle.num_qubits == 2
    assert local_oracle.count_ops()

    node_circuit = engine.build_node_circuit(np.array([0.3]), np.array([-0.3]), ["11"])
    assert node_circuit.num_qubits == engine.local_n
    assert node_circuit.count_ops()


def test_distributed_engine_rejects_invalid_prefix_sizes():
    with pytest.raises(ValueError, match="Prefix count"):
        DQAAEngine(global_n=4, j_prefixes=0)
    with pytest.raises(ValueError, match="Prefix count"):
        OracleSynthesizer(global_n=4, j=4, formula_text="v0")


def test_vtaa_engine_normalizes_branches_and_builds_state_circuit():
    legacy_branch = VariableTimeBranch(stop_time=2.0, weight=1.5, success_given_branch=0.5)
    branches = [
        VariableTimeBranch(stopping_time=3.0, weight=2.0, p_success=0.25),
        VariableTimeBranch(stopping_time=1.0, weight=1.0, p_success=1.0),
    ]
    engine = VTAAEngine(branches)
    t_mean, t_rms, t_max = engine.stopping_time_moments()
    staged = VTAAEngine.build_staged_state_circuit(p_s1=0.2, p_fail_cond=0.3)

    assert branches[0].stop_time == pytest.approx(3.0)
    assert branches[0].success_given_branch == pytest.approx(0.25)
    assert legacy_branch.stopping_time == pytest.approx(2.0)
    assert legacy_branch.p_success == pytest.approx(0.5)
    assert np.allclose(engine.stopping_times, [1.0, 3.0])
    assert np.allclose(engine.stop_times, [1.0, 3.0])
    assert np.sum(engine.weights) == pytest.approx(1.0)
    assert engine.p_success == pytest.approx((1.0 / 3.0) + (2.0 / 3.0) * 0.25)
    assert t_mean == pytest.approx((1.0 / 3.0) * 1.0 + (2.0 / 3.0) * 3.0)
    assert t_rms > 0.0 and t_max == pytest.approx(3.0)
    assert engine.vtaa_asymptotic_bound() > 0.0
    assert staged.num_qubits == 4


def test_vtaa_engine_validation_and_zero_success_bound():
    with pytest.raises(ValueError, match="At least one"):
        VTAAEngine([])
    with pytest.raises(ValueError, match="negative"):
        VTAAEngine([VariableTimeBranch(1.0, -1.0, 0.0)])
    with pytest.raises(ValueError, match="positive"):
        VTAAEngine([VariableTimeBranch(1.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        VTAAEngine([VariableTimeBranch(1.0, 1.0, 2.0)])

    zero_success = VTAAEngine([VariableTimeBranch(1.0, 1.0, 0.0)])
    assert zero_success.vtaa_asymptotic_bound() == float("inf")
    with pytest.raises(ValueError, match="Probabilities"):
        VTAAEngine.build_staged_state_circuit(-0.1, 0.5)


def test_qsvt_engine_and_synthesizer_outputs_are_structured():
    w = SU2QSPEngine.w_signal(0.25)
    z = SU2QSPEngine.z_rotation(0.7)
    p_vals, q_vals = SU2QSPEngine.evaluate_sequence(np.array([0.1, 0.2]), np.array([-1.0, 0.0, 1.0]))
    even, odd, alpha = QSVTSynthesizer.synthesize_jacobi_anger(degree=6, time=1.2)
    inverse_coeffs = QSVTSynthesizer.synthesize_matrix_inverse(degree=5, kappa=3.0)
    public_inverse_coeffs = QSVTSynthesizer.get_inverse_polynomial(degree=5, kappa=3.0)

    assert _is_unitary(w)
    assert _is_unitary(z)
    assert p_vals.shape == (3,)
    assert q_vals.shape == (3,)
    assert alpha >= 1.0
    assert even.coef.shape == (7,)
    assert odd.coef.shape == (7,)
    assert np.allclose(inverse_coeffs[::2], 0.0)
    assert np.allclose(inverse_coeffs, public_inverse_coeffs)


def test_qsvt_synthesizer_rejects_invalid_matrix_inverse_specs():
    with pytest.raises(ValueError, match="Degree"):
        QSVTSynthesizer.synthesize_matrix_inverse(0, 2.0)
    with pytest.raises(ValueError, match="odd"):
        QSVTSynthesizer.synthesize_matrix_inverse(4, 2.0)
    with pytest.raises(ValueError, match="kappa"):
        QSVTSynthesizer.synthesize_matrix_inverse(3, 1.0)


def test_iqae_config_result_and_estimators_return_results():
    result = IQAEResult(
        a_hat=0.25,
        theta_hat=0.5,
        confidence_interval=(0.2, 0.3),
        num_oracle_queries=12,
        rounds=3,
        estimator="iqae",
    )
    assert result.a_hat == pytest.approx(0.25)

    engine = IQAEEngine(GroverEngine(2, [1]), IQAEConfig(epsilon=0.1, alpha=0.1, n_shots=20, max_rounds=3))
    iqae = engine.estimate_iterative()
    mlae = engine.estimate_mle(max_k=2)
    esprit = engine.estimate_esprit(max_k=3)

    for estimate, estimator in [(iqae, "iqae"), (mlae, "mlae"), (esprit, "esprit")]:
        assert estimate.estimator == estimator
        assert 0.0 <= estimate.a_hat <= 1.0
        assert 0.0 <= estimate.theta_hat <= np.pi / 2.0
        assert estimate.confidence_interval[0] <= estimate.a_hat <= estimate.confidence_interval[1]
        assert estimate.num_oracle_queries >= 0


def test_iqae_engine_validates_inputs():
    with pytest.raises(ValueError, match="GroverEngine"):
        IQAEEngine(object())
    with pytest.raises(ValueError, match="epsilon"):
        IQAEEngine(GroverEngine(2, [1]), IQAEConfig(epsilon=0.5))
    with pytest.raises(ValueError, match="alpha"):
        IQAEEngine(GroverEngine(2, [1]), IQAEConfig(alpha=1.0))
    with pytest.raises(ValueError, match="n_shots"):
        IQAEEngine(GroverEngine(2, [1]), IQAEConfig(n_shots=0))
    with pytest.raises(ValueError, match="max_rounds"):
        IQAEEngine(GroverEngine(2, [1]), IQAEConfig(max_rounds=0))
