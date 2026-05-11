import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Operator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ampamp import (  # noqa: E402
    GroverEngine,
    OracleBuilder,
    build_bit_flip_oracle,
    build_phase_oracle,
    build_unitary_oracle,
    marked_bitstrings_from_formula,
)


def test_phase_oracle_marks_indices_with_standard_sign_flip():
    oracle = build_phase_oracle(num_qubits=2, marked_indices=[2])
    diagonal = np.diag(Operator(oracle).data)

    assert np.allclose(diagonal, np.array([1.0, 1.0, -1.0, 1.0]))


def test_phase_oracle_supports_arbitrary_phase_with_diagonal_synthesis():
    oracle = build_phase_oracle(
        num_qubits=2,
        marked_bitstrings=["01"],
        phase=np.pi / 3.0,
        synthesis="diagonal",
    )
    diagonal = np.diag(Operator(oracle).data)

    assert np.isclose(diagonal[1], np.exp(1j * np.pi / 3.0))
    assert np.allclose([diagonal[0], diagonal[2], diagonal[3]], [1.0, 1.0, 1.0])


def test_bit_flip_oracle_flips_output_qubit_for_marked_input():
    oracle = build_bit_flip_oracle(num_qubits=2, marked_indices=[2])
    matrix = Operator(oracle).data

    input_index = 2
    flipped_output_index = input_index + 2**2

    assert np.isclose(matrix[flipped_output_index, input_index], 1.0)
    assert np.isclose(matrix[input_index, flipped_output_index], 1.0)


def test_formula_source_enumerates_satisfying_bitstrings():
    marked = marked_bitstrings_from_formula(3, "(v0 & ~v1) | v2")

    assert marked == ("001", "011", "100", "101", "111")


def test_oracle_builder_exposes_common_marked_state_views():
    builder = OracleBuilder.from_formula(3, "v0 & ~v1")

    assert builder.marked_bitstrings() == ("100", "101")
    assert builder.marked_indices() == (4, 5)


def test_unitary_oracle_accepts_user_supplied_matrix():
    matrix = np.diag([1.0, -1.0])
    oracle = build_unitary_oracle(matrix)

    assert oracle.num_qubits == 1
    assert np.allclose(Operator(oracle).data, matrix)


def test_oracle_builder_constructs_unitary_matrix_oracle():
    matrix = np.diag([1.0, 1.0, -1.0, 1.0])
    oracle = OracleBuilder.from_unitary_matrix(matrix).unitary_oracle()

    assert oracle.num_qubits == 2
    assert np.allclose(Operator(oracle).data, matrix)


def test_unitary_oracle_rejects_non_unitary_matrix():
    matrix = np.array([[1.0, 0.0], [0.0, 2.0]])

    try:
        build_unitary_oracle(matrix)
    except ValueError as exc:
        assert "unitary_matrix must satisfy" in str(exc)
    else:
        raise AssertionError("Expected non-unitary matrix to be rejected")


def test_grover_engine_uses_general_phase_oracle_behavior():
    engine_oracle = GroverEngine(n_qubits=3, marked_indices=[5]).get_oracle()
    helper_oracle = build_phase_oracle(num_qubits=3, marked_indices=[5])

    assert np.allclose(Operator(engine_oracle).data, Operator(helper_oracle).data)
