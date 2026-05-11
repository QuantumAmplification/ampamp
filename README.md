# AmplitudeAmplification (`ampamp`)

`ampamp` is a Python library for quantum amplitude-amplification, QSVT/QSP, circuit diagnostics, transpilation profiling, and backend-validation workflows.

The current PyPI release is `0.1.4`:

```bash
python3 -m pip install ampamp
```

For local development from this repository:

```bash
python3 -m pip install -e .
```

Minimum Python: `>=3.9`

## What The Library Provides

`ampamp` exposes small engine classes and utilities for:

- Grover-style amplitude amplification
- Fixed-point amplitude amplification (FPAA)
- Oblivious amplitude amplification (OAA)
- Fixed-point oblivious amplitude amplification (FOQA)
- Distributed amplitude amplification (DQAA)
- Variable-time amplitude amplification (VTAA)
- Quantum singular value transformation and quantum signal processing (QSVT/QSP)
- Oracle construction from marked states, Boolean formula strings, or user-supplied unitary matrices
- Entanglement-count profiling
- Transpilation profiling and hardware-cost scoring
- Ideal/noisy backend validation

Scenario scripts and implementation-comparison workflows are maintained separately at:

```text
https://github.com/QuantumAmplification/Implementation
```

This package repository now focuses on the installable library, docs, tests, and release artifacts.

## Repository Layout

- `src/ampamp/`: installable library source code
- `docs/`: MkDocs documentation and API pages
- `tests/`: unit tests and regression checks
- `dist/`: local build artifacts when releases are prepared
- `implementation_folders.zip`: local archive of the moved implementation-comparison folders, if present

## Public API Overview

Common imports:

```python
from ampamp import (
    GroverEngine,
    FixedPointEngine,
    ObliviousEngine,
    FOQAEngine,
    DQAAEngine,
    OracleBuilder,
    OracleSpec,
    OracleSynthesizer,
    build_phase_oracle,
    build_bit_flip_oracle,
    build_unitary_oracle,
    marked_bitstrings_from_formula,
    EntanglementCountConfig,
    profile_entanglement_counts,
    VTAAEngine,
    VariableTimeBranch,
    SU2QSPEngine,
    QSVTSynthesizer,
    IQAEEngine,
    IQAEConfig,
    IQAEResult,
    GroverAuditor,
    FPAAAuditor,
    ObliviousAuditor,
    FOQAAuditor,
    DistributedAuditor,
    VTAAAuditor,
    FundamentalLimitsAuditor,
    QSVTAuditor,
    HardwareCostWeights,
    TranspilationProfiler,
    TranspilationBatchProfiler,
    TranspilationProfileConfig,
    BackendValidationRunner,
    BackendValidationConfig,
    ValidationNoiseConfig,
    ValidationLogConfig,
)
```

## Quick Start

### Grover circuit and transpilation profile

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

### Oracle construction

`ampamp` supports two main oracle-entry modes:

1. A user supplies a Boolean function or expression, often as a string, and the library constructs an oracle circuit.
2. A user supplies a unitary matrix directly, and the library wraps it as an oracle circuit after validation.

```python
import numpy as np

from ampamp import (
    OracleBuilder,
    build_bit_flip_oracle,
    build_phase_oracle,
    build_unitary_oracle,
)

phase_from_indices = build_phase_oracle(
    num_qubits=4,
    marked_indices=[3, 11],
)

phase_from_formula = build_phase_oracle(
    num_qubits=4,
    formula_text="v0 & (~v1 | v3)",
)

bit_flip_from_formula = build_bit_flip_oracle(
    num_qubits=4,
    formula_text="v0 & v2",
)

builder_oracle = OracleBuilder.from_formula(
    num_qubits=4,
    formula_text="v0 & (v2 | v3)",
).phase_oracle()

unitary_matrix = np.diag([1, 1, -1, 1])
matrix_oracle = build_unitary_oracle(unitary_matrix)
```

For distributed symbolic oracle work, `OracleSynthesizer` can substitute fixed node prefixes into a global Boolean formula and synthesize node-local oracles.

### Fixed-point amplitude amplification

```python
from ampamp import FixedPointEngine

fp = FixedPointEngine(L=5, delta=0.1)
qc = fp.build_fixed_point_circuit(num_qubits=4, marked_indices=[3, 11])

print(fp.alphas)
print(fp.betas)
print(qc.depth())
```

### Entanglement count profiling

```python
from ampamp import EntanglementCountConfig, profile_entanglement_counts

light = profile_entanglement_counts(
    qc,
    EntanglementCountConfig.light(max_qubits=12),
)

print(light["peak_active_entangled_qubits"])
```

### Backend validation

```python
from ampamp import (
    BackendValidationRunner,
    BackendValidationConfig,
    ValidationNoiseConfig,
    ValidationLogConfig,
)

runner = BackendValidationRunner(
    BackendValidationConfig(
        shots=1024,
        seed=42,
        noise=ValidationNoiseConfig(noise_level="light"),
        logging=ValidationLogConfig(enabled=True, log_dir="./validation_logs"),
    )
)

result = runner.validate_circuit("example", qc)
print(result["status"], result["metrics"]["tvd"])
```

## Engine Summary

- `GroverEngine`: standard oracle, diffusion, full Grover circuit, and success-probability utilities.
- `FixedPointEngine`: Chebyshev-style phase schedules and fixed-point circuit synthesis.
- `ObliviousEngine`: ancilla preparation, block-encoding scaffolds, and reflection construction.
- `FOQAEngine`: fixed-point oblivious schedules, recurrence simulation, and split-operator helpers.
- `DQAAEngine`: prefix/suffix partitioning for distributed search targets.
- `OracleBuilder`: phase, bit-flip, and unitary oracle construction.
- `OracleSynthesizer`: SymPy-backed local oracle synthesis from global formulas.
- `VTAAEngine`: variable-time branch moments, success mass, asymptotic estimates, and staged-state examples.
- `SU2QSPEngine`: QSP sequence evaluation.
- `QSVTSynthesizer`: Chebyshev helpers for Jacobi-Anger and matrix-inverse approximations.
- `IQAEEngine`, `IQAEConfig`, `IQAEResult`: QPE-free amplitude-estimation API scaffolding.

## Diagnostics

Auditors inspect configured engines and derived objects:

- `GroverAuditor`
- `FPAAAuditor`
- `ObliviousAuditor`
- `FOQAAuditor`
- `DistributedAuditor`
- `VTAAAuditor`
- `FundamentalLimitsAuditor`
- `QSVTAuditor`

These are intended as executable documentation and sanity checks before heavier simulation or transpilation runs.

## Transpilation And Backend Validation

The transpilation module provides:

- `TranspilationProfiler`: staged single-circuit profiling
- `TranspilationBatchProfiler`: batch profiling under a shared configuration
- `TranspilationProfileConfig`: basis, coupling map, timing, and optimization settings
- `HardwareCostWeights`: weighted hardware-cost scoring

The backend validation module provides:

- `BackendValidationRunner`: ideal/noisy simulator comparison
- `BackendValidationConfig`: shots, seed, thresholds, optimization level, and qubit cap
- `ValidationNoiseConfig`: ideal, preset, or custom noise models
- `ValidationLogConfig`: optional JSONL validation records

## Documentation

Serve the docs locally:

```bash
mkdocs serve
```

Main docs entry:

```text
docs/index.md
```

API pages include Grover, fixed-point, oblivious, FOQA, distributed, variable-time, QSVT, diagnostics, transpilation, transpilation validation, entanglement, and oracle construction.

## Testing

Run the full test suite:

```bash
PYTHONPATH=src pytest
```

Focused checks:

```bash
PYTHONPATH=src pytest tests/test_oracles.py
PYTHONPATH=src pytest tests/test_transpilation_module.py tests/test_transpilation_validation.py
```

## Implementation Workflows

The previous `library_implementation/` and `non_library_implementation/` folders were moved out of this package repository because they are comparison and scenario workflows rather than core library code.

They are available here:

```text
https://github.com/QuantumAmplification/Implementation
```

A local archive may also exist as:

```text
implementation_folders.zip
```

## Notes

- Some simulator-heavy workflows can be slow or environment-sensitive.
- If Matplotlib cache warnings appear, set a writable `MPLCONFIGDIR`.
- For PyPI releases, bump `pyproject.toml`, run tests, build with `python3 -m build`, validate with `twine check`, then upload with `twine upload`.
