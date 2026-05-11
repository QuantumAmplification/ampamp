from __future__ import annotations

"""General oracle construction utilities for amplitude-amplification workflows.

This module provides a small public framework for constructing Qiskit-native
phase and bit-flip oracles from marked indices, marked bitstrings, or Boolean
formulae over variables ``v0, v1, ...``.  It also accepts a user-supplied
unitary matrix when the oracle operator is already known.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
import sympy as sp
from qiskit import QuantumCircuit


def _unique_preserving_order(values: Sequence[int] | Sequence[str]) -> tuple:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _validate_num_qubits(num_qubits: int) -> int:
    resolved = int(num_qubits)
    if resolved < 1:
        raise ValueError("num_qubits must be >= 1")
    return resolved


def _normalize_indices(num_qubits: int, marked_indices: Sequence[int]) -> tuple[int, ...]:
    limit = 2 ** num_qubits
    values = tuple(int(index) for index in marked_indices)
    if not values:
        raise ValueError("marked_indices must contain at least one marked state")
    if any(index < 0 or index >= limit for index in values):
        raise ValueError(f"marked_indices must be between 0 and {limit - 1}")
    return _unique_preserving_order(values)


def _normalize_bitstrings(num_qubits: int, marked_bitstrings: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(bitstring).strip() for bitstring in marked_bitstrings)
    if not values:
        raise ValueError("marked_bitstrings must contain at least one marked state")
    for bitstring in values:
        if len(bitstring) != num_qubits:
            raise ValueError(f"marked bitstring {bitstring!r} must have length {num_qubits}")
        if any(bit not in {"0", "1"} for bit in bitstring):
            raise ValueError(f"marked bitstring {bitstring!r} must contain only 0 and 1")
    return _unique_preserving_order(values)


def _indices_to_bitstrings(num_qubits: int, marked_indices: Sequence[int]) -> tuple[str, ...]:
    return tuple(format(index, f"0{num_qubits}b") for index in marked_indices)


def _bitstrings_to_indices(marked_bitstrings: Sequence[str]) -> tuple[int, ...]:
    return tuple(int(bitstring, 2) for bitstring in marked_bitstrings)


def _infer_num_qubits_from_dimension(dimension: int) -> int:
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("unitary_matrix dimension must be a power of two")
    return int(np.log2(dimension))


def _validate_unitary_matrix(
    unitary_matrix: ArrayLike,
    *,
    num_qubits: int | None = None,
    atol: float = 1e-8,
) -> tuple[int, np.ndarray]:
    matrix = np.asarray(unitary_matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary_matrix must be a square 2D array")

    inferred_qubits = _infer_num_qubits_from_dimension(matrix.shape[0])
    if num_qubits is not None and _validate_num_qubits(num_qubits) != inferred_qubits:
        raise ValueError(
            "num_qubits does not match unitary_matrix dimension: "
            f"expected {2 ** int(num_qubits)}x{2 ** int(num_qubits)}, got {matrix.shape}"
        )

    identity = np.eye(matrix.shape[0], dtype=complex)
    if not np.allclose(matrix.conj().T @ matrix, identity, atol=float(atol)):
        raise ValueError("unitary_matrix must satisfy U.conj().T @ U = I")
    return inferred_qubits, matrix


@dataclass(frozen=True)
class OracleSpec:
    """Validated oracle source specification.

    Exactly one source must be supplied: ``marked_indices``,
    ``marked_bitstrings``, ``formula_text``, or ``unitary_matrix``.  Bitstrings
    use the usual most-significant-bit-left convention.  For formulae, ``v0``
    denotes the leftmost bit, ``v1`` the next bit, and so on.  Matrix oracles
    are checked as square power-of-two-dimensional unitary arrays.
    """

    num_qubits: int | None = None
    marked_indices: Sequence[int] | None = None
    marked_bitstrings: Sequence[str] | None = None
    formula_text: str | None = None
    unitary_matrix: ArrayLike | None = None
    phase: float = np.pi
    variable_prefix: str = "v"
    max_formula_qubits: int = 16
    unitary_atol: float = 1e-8

    def __post_init__(self) -> None:
        sources = (
            self.marked_indices is not None,
            self.marked_bitstrings is not None,
            self.formula_text is not None,
            self.unitary_matrix is not None,
        )
        if sum(bool(value) for value in sources) != 1:
            raise ValueError(
                "Provide exactly one of marked_indices, marked_bitstrings, "
                "formula_text, or unitary_matrix"
            )

        if self.unitary_matrix is not None:
            num_qubits, unitary_matrix = _validate_unitary_matrix(
                self.unitary_matrix,
                num_qubits=self.num_qubits,
                atol=self.unitary_atol,
            )
            object.__setattr__(self, "num_qubits", num_qubits)
            object.__setattr__(self, "unitary_matrix", unitary_matrix)
        else:
            if self.num_qubits is None:
                raise ValueError("num_qubits is required for marked-state and formula oracle sources")
            num_qubits = _validate_num_qubits(self.num_qubits)
            object.__setattr__(self, "num_qubits", num_qubits)

        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(self, "variable_prefix", str(self.variable_prefix))
        object.__setattr__(self, "max_formula_qubits", int(self.max_formula_qubits))
        object.__setattr__(self, "unitary_atol", float(self.unitary_atol))

        if self.marked_indices is not None:
            object.__setattr__(
                self,
                "marked_indices",
                _normalize_indices(num_qubits, self.marked_indices),
            )
        if self.marked_bitstrings is not None:
            bitstrings = _normalize_bitstrings(num_qubits, self.marked_bitstrings)
            object.__setattr__(self, "marked_bitstrings", bitstrings)
        if self.formula_text is not None:
            formula_text = str(self.formula_text).strip()
            if not formula_text:
                raise ValueError("formula_text must be nonempty")
            object.__setattr__(self, "formula_text", formula_text)


def marked_bitstrings_from_formula(
    num_qubits: int,
    formula_text: str,
    *,
    variable_prefix: str = "v",
    max_formula_qubits: int = 16,
) -> tuple[str, ...]:
    """Enumerate satisfying bitstrings for a Boolean formula.

    The formula is parsed by SymPy and may use variables such as ``v0``,
    ``v1`` and standard Boolean operators ``&``, ``|`` and ``~``.  The
    implementation intentionally uses truth-table enumeration so that phase
    and bit-flip oracle constructors share the same variable-order semantics.
    """

    num_qubits = _validate_num_qubits(num_qubits)
    if num_qubits > int(max_formula_qubits):
        raise ValueError(
            "Formula truth-table synthesis would require "
            f"2^{num_qubits} evaluations; raise max_formula_qubits to opt in."
        )

    variable_prefix = str(variable_prefix)
    expression = sp.sympify(str(formula_text).strip(), evaluate=False)
    symbols = [sp.Symbol(f"{variable_prefix}{idx}") for idx in range(num_qubits)]
    allowed_names = {symbol.name for symbol in symbols}
    unknown = sorted(str(symbol) for symbol in expression.free_symbols if str(symbol) not in allowed_names)
    if unknown:
        raise ValueError(f"Formula references variables outside the {variable_prefix}0..{variable_prefix}{num_qubits - 1} range: {unknown}")

    marked: list[str] = []
    for index in range(2 ** num_qubits):
        bitstring = format(index, f"0{num_qubits}b")
        substitutions = {
            symbols[bit_index]: bitstring[bit_index] == "1"
            for bit_index in range(num_qubits)
        }
        value = sp.simplify_logic(expression.subs(substitutions), force=True)
        if value == sp.true:
            marked.append(bitstring)
    return tuple(marked)


def _diagonal_gate_class():
    try:
        from qiskit.circuit.library import DiagonalGate as Diagonal
    except ImportError:  # pragma: no cover - compatibility with older Qiskit
        from qiskit.circuit.library import Diagonal
    return Diagonal


def _append_phase_pi_for_bitstring(qc: QuantumCircuit, bitstring: str) -> None:
    little_endian_bits = bitstring[::-1]
    for qubit, bit in enumerate(little_endian_bits):
        if bit == "0":
            qc.x(qubit)

    if qc.num_qubits == 1:
        qc.z(0)
    else:
        target = qc.num_qubits - 1
        qc.h(target)
        qc.mcx(list(range(target)), target)
        qc.h(target)

    for qubit, bit in enumerate(little_endian_bits):
        if bit == "0":
            qc.x(qubit)


def _append_bit_flip_for_bitstring(qc: QuantumCircuit, bitstring: str, output_qubit: int) -> None:
    input_qubits = list(range(len(bitstring)))
    little_endian_bits = bitstring[::-1]
    for qubit, bit in zip(input_qubits, little_endian_bits):
        if bit == "0":
            qc.x(qubit)

    if len(input_qubits) == 1:
        qc.cx(input_qubits[0], output_qubit)
    else:
        qc.mcx(input_qubits, output_qubit)

    for qubit, bit in zip(input_qubits, little_endian_bits):
        if bit == "0":
            qc.x(qubit)


class OracleBuilder:
    """Build phase and bit-flip oracles from one validated oracle source."""

    def __init__(self, spec: OracleSpec):
        self.spec = spec

    @classmethod
    def from_marked_indices(
        cls,
        num_qubits: int,
        marked_indices: Sequence[int],
        *,
        phase: float = np.pi,
    ) -> "OracleBuilder":
        return cls(OracleSpec(num_qubits=num_qubits, marked_indices=marked_indices, phase=phase))

    @classmethod
    def from_bitstrings(
        cls,
        marked_bitstrings: Sequence[str],
        *,
        num_qubits: int | None = None,
        phase: float = np.pi,
    ) -> "OracleBuilder":
        bitstrings = tuple(str(bitstring).strip() for bitstring in marked_bitstrings)
        if num_qubits is None:
            if not bitstrings:
                raise ValueError("marked_bitstrings must contain at least one marked state")
            num_qubits = len(bitstrings[0])
        return cls(OracleSpec(num_qubits=num_qubits, marked_bitstrings=bitstrings, phase=phase))

    @classmethod
    def from_formula(
        cls,
        num_qubits: int,
        formula_text: str,
        *,
        phase: float = np.pi,
        variable_prefix: str = "v",
        max_formula_qubits: int = 16,
    ) -> "OracleBuilder":
        return cls(
            OracleSpec(
                num_qubits=num_qubits,
                formula_text=formula_text,
                phase=phase,
                variable_prefix=variable_prefix,
                max_formula_qubits=max_formula_qubits,
            )
        )

    @classmethod
    def from_unitary_matrix(
        cls,
        unitary_matrix: ArrayLike,
        *,
        num_qubits: int | None = None,
        atol: float = 1e-8,
    ) -> "OracleBuilder":
        return cls(
            OracleSpec(
                num_qubits=num_qubits,
                unitary_matrix=unitary_matrix,
                unitary_atol=atol,
            )
        )

    def marked_bitstrings(self) -> tuple[str, ...]:
        if self.spec.unitary_matrix is not None:
            raise ValueError("marked_bitstrings are not defined for an arbitrary unitary oracle")
        if self.spec.marked_bitstrings is not None:
            return tuple(self.spec.marked_bitstrings)
        if self.spec.marked_indices is not None:
            return _indices_to_bitstrings(self.spec.num_qubits, self.spec.marked_indices)
        return marked_bitstrings_from_formula(
            self.spec.num_qubits,
            str(self.spec.formula_text),
            variable_prefix=self.spec.variable_prefix,
            max_formula_qubits=self.spec.max_formula_qubits,
        )

    def marked_indices(self) -> tuple[int, ...]:
        return _bitstrings_to_indices(self.marked_bitstrings())

    def phase_oracle(self, *, synthesis: str = "auto", name: str | None = None) -> QuantumCircuit:
        """Return a phase oracle over the input register.

        ``synthesis`` may be ``"auto"``, ``"mcx"``, or ``"diagonal"``.  MCX
        synthesis is available for the standard sign-flip phase
        ``phase = pi``.  Diagonal synthesis supports arbitrary marked-state
        phases.
        """

        synthesis = str(synthesis).lower()
        if synthesis not in {"auto", "mcx", "diagonal"}:
            raise ValueError("synthesis must be one of: auto, mcx, diagonal")

        bitstrings = self.marked_bitstrings()
        qc = QuantumCircuit(self.spec.num_qubits, name=name or "phase_oracle")
        phase_is_pi = bool(np.isclose(np.mod(self.spec.phase, 2.0 * np.pi), np.pi))

        if synthesis == "auto":
            synthesis = "mcx" if phase_is_pi else "diagonal"
        if synthesis == "mcx" and not phase_is_pi:
            raise ValueError("MCX phase-oracle synthesis only supports phase=pi; use synthesis='diagonal'")

        if synthesis == "mcx":
            for bitstring in bitstrings:
                _append_phase_pi_for_bitstring(qc, bitstring)
            return qc

        diagonal = np.ones(2 ** self.spec.num_qubits, dtype=complex)
        marked_phase = np.exp(1j * float(self.spec.phase))
        for index in _bitstrings_to_indices(bitstrings):
            diagonal[index] = marked_phase
        Diagonal = _diagonal_gate_class()
        qc.append(Diagonal(diagonal), range(self.spec.num_qubits))
        return qc

    def bit_flip_oracle(self, *, name: str | None = None) -> QuantumCircuit:
        """Return ``|x>|y> -> |x>|y xor f(x)>`` with the output qubit last."""

        if self.spec.unitary_matrix is not None:
            raise ValueError("bit_flip_oracle is not defined for an arbitrary unitary oracle")
        output_qubit = self.spec.num_qubits
        qc = QuantumCircuit(self.spec.num_qubits + 1, name=name or "bit_flip_oracle")
        for bitstring in self.marked_bitstrings():
            _append_bit_flip_for_bitstring(qc, bitstring, output_qubit)
        return qc

    def unitary_oracle(self, *, name: str | None = None) -> QuantumCircuit:
        """Return a circuit that applies the supplied oracle unitary matrix."""

        if self.spec.unitary_matrix is None:
            return self.phase_oracle(name=name)

        try:
            from qiskit.circuit.library import UnitaryGate
        except ImportError:  # pragma: no cover - compatibility with older Qiskit
            from qiskit.extensions import UnitaryGate

        gate = UnitaryGate(self.spec.unitary_matrix, label=name or "unitary_oracle")
        qc = QuantumCircuit(self.spec.num_qubits, name=name or "unitary_oracle")
        qc.append(gate, range(self.spec.num_qubits))
        return qc


def _build_spec_from_optional_sources(
    num_qubits: int,
    *,
    marked_indices: Sequence[int] | None = None,
    marked_bitstrings: Sequence[str] | None = None,
    formula_text: str | None = None,
    phase: float = np.pi,
    variable_prefix: str = "v",
    max_formula_qubits: int = 16,
) -> OracleSpec:
    return OracleSpec(
        num_qubits=num_qubits,
        marked_indices=marked_indices,
        marked_bitstrings=marked_bitstrings,
        formula_text=formula_text,
        phase=phase,
        variable_prefix=variable_prefix,
        max_formula_qubits=max_formula_qubits,
    )


def build_phase_oracle(
    num_qubits: int,
    *,
    marked_indices: Sequence[int] | None = None,
    marked_bitstrings: Sequence[str] | None = None,
    formula_text: str | None = None,
    phase: float = np.pi,
    synthesis: str = "auto",
    variable_prefix: str = "v",
    max_formula_qubits: int = 16,
    name: str | None = None,
) -> QuantumCircuit:
    """Convenience wrapper for constructing a phase oracle."""

    spec = _build_spec_from_optional_sources(
        num_qubits,
        marked_indices=marked_indices,
        marked_bitstrings=marked_bitstrings,
        formula_text=formula_text,
        phase=phase,
        variable_prefix=variable_prefix,
        max_formula_qubits=max_formula_qubits,
    )
    return OracleBuilder(spec).phase_oracle(synthesis=synthesis, name=name)


def build_bit_flip_oracle(
    num_qubits: int,
    *,
    marked_indices: Sequence[int] | None = None,
    marked_bitstrings: Sequence[str] | None = None,
    formula_text: str | None = None,
    variable_prefix: str = "v",
    max_formula_qubits: int = 16,
    name: str | None = None,
) -> QuantumCircuit:
    """Convenience wrapper for constructing a bit-flip oracle."""

    spec = _build_spec_from_optional_sources(
        num_qubits,
        marked_indices=marked_indices,
        marked_bitstrings=marked_bitstrings,
        formula_text=formula_text,
        variable_prefix=variable_prefix,
        max_formula_qubits=max_formula_qubits,
    )
    return OracleBuilder(spec).bit_flip_oracle(name=name)


def build_unitary_oracle(
    unitary_matrix: ArrayLike,
    *,
    num_qubits: int | None = None,
    atol: float = 1e-8,
    name: str | None = None,
) -> QuantumCircuit:
    """Convenience wrapper for constructing an oracle from a unitary matrix."""

    return OracleBuilder.from_unitary_matrix(
        unitary_matrix,
        num_qubits=num_qubits,
        atol=atol,
    ).unitary_oracle(name=name)
