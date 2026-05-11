<section class="ampamp-hero">
  <div class="ampamp-hero__inner">
    <div>
      <p class="ampamp-eyebrow">Amplitude amplification tooling for Python</p>
      <h1>Build, inspect, profile, and validate quantum amplification workflows.</h1>
      <p class="ampamp-lede">
        <code>ampamp</code> packages Grover search, fixed-point schedules, oracle construction,
        variable-time models, QSVT/QSP helpers, diagnostics, transpilation profiling, and
        backend validation behind a compact research API.
      </p>
      <div class="ampamp-actions">
        <a class="ampamp-button ampamp-button--primary" href="#quick-start">Quick start</a>
        <a class="ampamp-button" href="api/oracles/">Oracle guide</a>
        <a class="ampamp-button" href="https://github.com/QuantumAmplification/ampamp">GitHub</a>
      </div>
      <span class="ampamp-version">Current PyPI release: 0.1.4</span>
    </div>
    <div class="ampamp-terminal" aria-label="Installation example">
      <div class="ampamp-terminal__bar">
        <span class="ampamp-dot"></span>
        <span class="ampamp-dot"></span>
        <span class="ampamp-dot"></span>
      </div>
      <pre><code>python3 -m pip install ampamp

from ampamp import GroverEngine

engine = GroverEngine(
    n_qubits=6,
    marked_indices=[10, 25],
)
qc = engine.construct_circuit(
    iterations=engine.k_optimal,
)</code></pre>
    </div>
  </div>
</section>

## What Is In The Library {: .ampamp-section-title }

<div class="grid cards" markdown>

-   **Grover Search**

    ---
    Standard amplitude-amplification circuits with phase oracles, diffusion operators, optimal-iteration estimates, and analytic success probabilities.

    [API Reference](api/grover.md)

-   **Oracle Construction**

    ---
    Build phase, bit-flip, or direct unitary oracle circuits from marked states, Boolean expression strings, or user-supplied unitary matrices.

    [API Reference](api/oracles.md)

-   **Fixed-Point AA**

    ---
    Chebyshev-style fixed-point schedules and circuit synthesis for monotone amplification behavior.

    [API Reference](api/fixed_point.md)

-   **Oblivious / FOQA**

    ---
    Ancilla preparation, block-encoding scaffolds, reflection helpers, fixed-point oblivious schedules, and recurrence simulation.

    [Oblivious](api/oblivious.md) · [FOQA](api/foqa.md)

-   **Distributed AA**

    ---
    Prefix/suffix partitioning for distributed search targets and symbolic local-oracle synthesis for node-local subproblems.

    [API Reference](api/distributed.md)

-   **Variable-Time AA**

    ---
    Branch records, stopping-time statistics, success-mass calculations, asymptotic estimates, and staged-state examples.

    [API Reference](api/variable_time.md)

-   **QSVT / QSP**

    ---
    SU(2) QSP sequence evaluation and Chebyshev helpers for Jacobi-Anger and matrix-inverse polynomial synthesis.

    [API Reference](api/qsvt.md)

-   **Diagnostics**

    ---
    Auditors for subspace rotation, fixed-point schedules, block-encoding structure, distributed partitions, VTAA branches, and QSVT parity checks.

    [API Reference](api/diagnostics.md)

-   **Transpilation**

    ---
    Staged compile metrics, routing statistics, basis translation, timing models, batch profiling, and hardware-cost scoring.

    [API Reference](api/transpilation.md)

-   **Backend Validation**

    ---
    Ideal/noisy simulator comparisons, total-variation-distance checks, noise presets, and optional JSONL validation logs.

    [API Reference](api/transpilation_validation.md)

-   **Entanglement Count**

    ---
    Light and hard active-entangled-qubit profiling for Qiskit circuits.

    [API Reference](api/entanglement.md)

</div>

## Quick Start

Build and profile a Grover circuit:

```python
from ampamp import GroverEngine, TranspilationProfiler, TranspilationProfileConfig

engine = GroverEngine(n_qubits=6, marked_indices=[10, 25])
qc = engine.construct_circuit(iterations=engine.k_optimal)

config = TranspilationProfileConfig(
    coupling_map_edges=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
)
metrics = TranspilationProfiler(config).profile_circuit(qc)

print(engine.k_optimal)
print(metrics["post_optimization_depth"])
print(metrics["final_cnots"])
```

## Oracle Quick Start

`ampamp` supports the two oracle-entry modes used by most amplitude-amplification workflows:

1. The user supplies a Boolean function or expression string, and the library constructs the oracle circuit.
2. The user supplies a unitary matrix directly, and the library validates and wraps it as an oracle circuit.

```python
import numpy as np

from ampamp import OracleBuilder, build_phase_oracle, build_unitary_oracle

formula_oracle = build_phase_oracle(
    num_qubits=4,
    formula_text="v0 & (v2 | v3)",
)

builder_oracle = OracleBuilder.from_formula(
    num_qubits=4,
    formula_text="v0 & (v2 | v3)",
).phase_oracle()

unitary_matrix = np.diag([1, 1, -1, 1])
matrix_oracle = build_unitary_oracle(unitary_matrix)
```

## Repository Scope

This repository now focuses on the installable `ampamp` package, tests, and documentation. The previous implementation-comparison folders were moved out because they are scenario workflows rather than core library code:

```text
https://github.com/QuantumAmplification/Implementation
```

## Testing

```bash
PYTHONPATH=src pytest
```

Focused checks:

```bash
PYTHONPATH=src pytest tests/test_oracles.py
PYTHONPATH=src pytest tests/test_transpilation_module.py tests/test_transpilation_validation.py
```

## Documentation

Serve this site locally:

```bash
mkdocs serve
```

Build it:

```bash
mkdocs build
```
