# Oracle Construction

`ampamp.oracles` is the general oracle-construction layer for amplitude-amplification workflows.

It supports four source types:

- marked basis-state indices
- marked bitstrings
- Boolean formula strings
- user-supplied unitary matrices

Exactly one source is supplied for a single oracle specification.

## Boolean Formula To Circuit

Use `formula_text` when the user gives a Boolean function or expression. The expression is parsed by SymPy and evaluated over variables `v0`, `v1`, ..., where `v0` denotes the leftmost bit in the bitstring convention.

Supported operator examples include:

- `&` for AND
- `|` for OR
- `~` for NOT
- parentheses for grouping

```python
from ampamp import build_phase_oracle, build_bit_flip_oracle

phase_oracle = build_phase_oracle(
    num_qubits=4,
    formula_text="v0 & (v2 | v3)",
)

bit_flip_oracle = build_bit_flip_oracle(
    num_qubits=4,
    formula_text="v0 & ~v1",
)
```

For formula sources, synthesis enumerates satisfying assignments. To keep accidental truth-table blowups explicit, formula synthesis defaults to `max_formula_qubits=16`; raise this value only when that cost is intended.

```python
from ampamp import marked_bitstrings_from_formula

marked = marked_bitstrings_from_formula(
    num_qubits=3,
    formula_text="(v0 & ~v1) | v2",
)

print(marked)
```

## Direct Unitary Matrix Oracle

Use `build_unitary_oracle` when the oracle is already available as a matrix. The matrix must be:

- square
- power-of-two dimensional
- unitary within the requested tolerance

The number of qubits is inferred from the matrix dimension unless `num_qubits` is supplied for an extra consistency check.

```python
import numpy as np

from ampamp import build_unitary_oracle

unitary_matrix = np.diag([1, 1, -1, 1])
oracle = build_unitary_oracle(unitary_matrix)

print(oracle.num_qubits)  # 2
```

For lower-level construction, use `OracleBuilder.from_unitary_matrix(...)`:

```python
import numpy as np

from ampamp import OracleBuilder

oracle = OracleBuilder.from_unitary_matrix(
    np.diag([1, -1]),
).unitary_oracle()
```

## Marked-State Oracles

Marked indices and bitstrings remain the simplest path when the target states are already known.

```python
from ampamp import build_phase_oracle, build_bit_flip_oracle

phase_oracle = build_phase_oracle(
    num_qubits=3,
    marked_indices=[5],
)

bit_flip_oracle = build_bit_flip_oracle(
    num_qubits=3,
    marked_bitstrings=["101"],
)
```

## Builder Pattern

`OracleBuilder` is useful when the same validated oracle source should expose multiple views:

```python
from ampamp import OracleBuilder

builder = OracleBuilder.from_formula(3, "v0 & ~v1")

print(builder.marked_bitstrings())
print(builder.marked_indices())

phase = builder.phase_oracle()
bit_flip = builder.bit_flip_oracle()
```

## Generated API Reference

::: ampamp.oracles
