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

## Canonical Paper Trail {: .ampamp-section-title }

`ampamp` is a practical library, but each module sits on a well-established algorithmic line. The documentation below names the main papers behind the current implementation surface, from Grover search and amplitude estimation through fixed-point schedules, block encodings, variable-time amplification, and QSVT.

<div class="grid cards" markdown>

-   **Survey and Taxonomy**

    ---
    Start with the companion survey [Kumar, Tahir, and Daiya (2026)](https://doi.org/10.5281/zenodo.20054981), which gives the broader amplitude-amplification landscape that this package turns into executable tools.

    Used across the docs and SciPost manuscript.

-   **Grover Search and Limits**

    ---
    The search core follows [Grover (1996)](https://doi.org/10.1145/237814.237866), with tight-search and optimality context from [Boyer, Brassard, Hoyer, and Tapp (1998)](https://doi.org/10.1002/(SICI)1521-3978(199806)46:4/5%3C493::AID-PROP493%3E3.0.CO;2-P), [Bennett, Bernstein, Brassard, and Vazirani (1997)](https://doi.org/10.1137/S0097539796300933), and [Zalka (1999)](https://doi.org/10.1103/PhysRevA.60.2746).

    Maps to `GroverEngine`, phase oracles, diffusion, and success-probability utilities.

-   **Amplitude Amplification and Estimation**

    ---
    The general amplification and estimation interface is grounded in [Brassard, Hoyer, Mosca, and Tapp (2002)](https://doi.org/10.1090/conm/305/05215), arbitrary-phase amplification in [Hoyer (2000)](https://doi.org/10.1103/PhysRevA.62.052304), and QPE-free estimation in [Grinko, Gacon, Zoufal, and Woerner (2021)](https://doi.org/10.1038/s41534-021-00379-1).

    Maps to analytic probability helpers and the `IQAEConfig` / `IQAEResult` scaffolding.

-   **Fixed-Point and Damped Search**

    ---
    Fixed-point behavior is connected to [Grover (2005)](https://doi.org/10.1103/PhysRevLett.95.150501), [Mizel (2009)](https://doi.org/10.1103/PhysRevLett.102.150501), [Yoder, Low, and Chuang (2014)](https://doi.org/10.1103/PhysRevLett.113.210501), and [Low, Yoder, and Chuang (2016)](https://doi.org/10.1103/PhysRevX.6.041067).

    Maps to `FixedPointEngine`, `FOQAEngine`, phase schedules, and recurrence simulation.

-   **Oracle, LCU, and Block-Encoding Methods**

    ---
    The oblivious-amplification and block-encoding parts reflect the LCU and qubitization paper trail, including [Berry, Childs, Cleve, Kothari, and Somma (2015)](https://doi.org/10.1103/PhysRevLett.114.090502), [Berry, Childs, and Kothari (2015)](https://doi.org/10.1109/FOCS.2015.54), [Low and Chuang (2019)](https://doi.org/10.22331/q-2019-07-12-163), and [Babbush et al. (2018)](https://doi.org/10.1103/PhysRevX.8.041015).

    Maps to `ObliviousEngine`, direct unitary oracle wrapping, and block-encoding scaffolds.

-   **Variable-Time and Linear Algebra Algorithms**

    ---
    Variable-time amplification follows [Ambainis (2012)](https://doi.org/10.4230/LIPIcs.STACS.2012.636), with the linear-system motivation from [Harrow, Hassidim, and Lloyd (2009)](https://doi.org/10.1103/PhysRevLett.103.150502) and improved precision dependence in [Childs, Kothari, and Somma (2017)](https://doi.org/10.1137/16M1087072).

    Maps to `VTAAEngine`, branch records, stopping-time moments, and asymptotic estimates.

-   **QSP and QSVT**

    ---
    The QSP/QSVT helpers sit on [Low and Chuang (2017)](https://doi.org/10.1103/PhysRevLett.118.010501), [Low and Chuang (2019)](https://doi.org/10.22331/q-2019-07-12-163), [Gilyen, Su, Low, and Wiebe (2019)](https://doi.org/10.1145/3313276.3316366), and [Martyn, Rossi, Tan, and Chuang (2021)](https://doi.org/10.1103/PRXQuantum.2.040203).

    Maps to `SU2QSPEngine`, `QSVTSynthesizer`, Chebyshev helpers, and parity diagnostics.

-   **Quantum Walks and Partitioned Search**

    ---
    The distributed and partitioned-search tools are software bookkeeping layers, but their motivation is close to quantum-walk search and walk/simulation relationships such as [Childs (2010)](https://doi.org/10.1007/s00220-009-0930-1).

    Maps to `DQAAEngine`, prefix/suffix partitions, and symbolic local-oracle synthesis.

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
