# AmplitudeAmplification (`ampamp`)

A quantum algorithm engineering repository focused on amplitude amplification and its modern polynomial form (QSVT), with both:

- reusable **library APIs** (`ampamp` package), and
- scenario-driven **non-library research/transpile workflows**.

This README is intentionally detailed so you can get productive without opening docs first.

## What This Repo Contains

`ampamp` provides engines and utilities for:

- Grover-style amplitude amplification
- Fixed-point amplitude amplification (FPAA)
- Oblivious amplitude amplification (OAA)
- Fixed-point oblivious amplitude amplification (FOQA)
- Distributed amplitude amplification (DQAA)
- Variable-time amplitude amplification (VTAA)
- Quantum singular value transformation (QSVT / QSP)
- Transpilation profiling and backend validation workflows

## Repository Layout

Top-level structure is now intentionally split:

- `library_implementation/`
  Contains workflows implemented with library abstractions.
  - `library_transpile_showcase/`: per-algorithm library-only transpilation showcase

- `non_library_implementation/`
  Contains scenario-style and script-first workflows.
  - `experiments/`
  - `transpile/`
  - `Transpile Algorithms GPU Parallelization/`

Other important folders:

- `src/ampamp/`: installable library source code
- `docs/`: MkDocs site and API pages
- `tests/`: tests
- `dist/`: built distribution artifacts

## Library API Overview

Import surface from `ampamp`:

```python
from ampamp import (
    GroverEngine,
    OracleBuilder,
    OracleSpec,
    build_phase_oracle,
    build_bit_flip_oracle,
    marked_bitstrings_from_formula,
    EntanglementCountConfig,
    profile_entanglement_counts,
    FixedPointEngine,
    ObliviousEngine,
    FOQAEngine,
    DQAAEngine,
    OracleSynthesizer,
    VTAAEngine,
    VariableTimeBranch,
    SU2QSPEngine,
    QSVTSynthesizer,
    TranspilationProfiler,
    TranspilationBatchProfiler,
    TranspilationProfileConfig,
    HardwareCostWeights,
    BackendValidationRunner,
    BackendValidationConfig,
    ValidationNoiseConfig,
    ValidationLogConfig,
)
```

### Core Engines

- `GroverEngine`: oracle/diffusion builders, Grover circuit synthesis, success-probability utilities.
- `OracleBuilder` + helpers: general phase and bit-flip oracle construction from marked indices, bitstrings, or Boolean formulae.
- `profile_entanglement_counts`: light/hard active-entanglement count profiling for Qiskit circuits.
- `FixedPointEngine`: Chebyshev-derived phase schedule generation + fixed-point circuit synthesis.
- `ObliviousEngine`: ancilla prep, block-encoding circuit construction, reflection construction.
- `FOQAEngine`: Mizel/constant schedules, recurrence simulation, FOQA proxy circuit sequence builder.
- `DQAAEngine` + `OracleSynthesizer`: distributed partitioning and local-oracle synthesis.
- `VTAAEngine`: variable-time branch statistics and staged-state circuit synthesis.
- `SU2QSPEngine` + `QSVTSynthesizer`: QSP evaluation, Jacobi-Anger synthesis, matrix-inverse polynomial synthesis.

### Diagnostics

Available auditors include:

- `GroverAuditor`, `FPAAAuditor`, `ObliviousAuditor`, `FOQAAuditor`
- `DistributedAuditor`, `VTAAAuditor`, `FundamentalLimitsAuditor`, `QSVTAuditor`

Use these to inspect subspace behavior, recurrence behavior, and structural validity checks.

### Transpilation Utilities (Library Native)

- `TranspilationProfiler`: staged compile metrics (logical/routing/optimization/timing)
- `TranspilationBatchProfiler`: profile multiple circuits together
- `TranspilationProfileConfig`: basis gates, coupling map, duration model, optimization levels
- `HardwareCostWeights`: weighted hardware penalty terms

### Backend Validation Utilities

- `BackendValidationRunner`: ideal-vs-noisy validation
- `BackendValidationConfig`: shots, seed, thresholds, basis config
- `ValidationNoiseConfig`: preset/custom noise settings
- `ValidationLogConfig`: JSONL structured logging

## Installation

Recommended (editable install while developing):

```bash
pip install -e .
```

Minimum Python: `>=3.9`

Key dependencies:

- `qiskit>=1.0`
- `qiskit-aer>=0.14`
- `numpy`, `scipy`, `sympy`, `matplotlib`

## Quick Start (Library)

### 1) Build and profile a Grover circuit

```python
from ampamp import GroverEngine, TranspilationProfiler, TranspilationProfileConfig

engine = GroverEngine(n_qubits=6, marked_indices=[10, 25])
qc = engine.construct_circuit(iterations=1)

profiler = TranspilationProfiler(
    TranspilationProfileConfig(
        coupling_map_edges=[[i, j] for i in range(6) for j in range(6) if i != j],
        optimize_optimization_level=3,
    )
)

metrics = profiler.profile_circuit(qc)
print(metrics["post_optimization_depth"], metrics["final_cnots"])
```

### 2) Fixed-point schedule + circuit

```python
from ampamp import FixedPointEngine

fp = FixedPointEngine(L=3, delta=0.1)
qc = fp.build_fixed_point_circuit(num_qubits=6, marked_indices=[0])
print(len(fp.alphas), len(fp.betas), qc.num_qubits)
```

### 3) General oracle construction

```python
from ampamp import build_bit_flip_oracle, build_phase_oracle

phase_oracle = build_phase_oracle(num_qubits=4, marked_indices=[3, 11])
formula_oracle = build_phase_oracle(num_qubits=4, formula_text="v0 & (~v1 | v3)")
bit_flip_oracle = build_bit_flip_oracle(num_qubits=4, marked_bitstrings=["0011", "1011"])
```

### 4) Light/hard entanglement count

```python
from ampamp import EntanglementCountConfig, profile_entanglement_counts

light = profile_entanglement_counts(qc, EntanglementCountConfig.light(max_qubits=12))
hard = profile_entanglement_counts(qc, EntanglementCountConfig.hard(max_qubits=8))

print(light["peak_active_entangled_qubits"])
```

### 5) Backend validation with structured logs

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

# result = runner.validate_circuit("my_circuit", qc)
```

## Library Transpilation Showcase

`library_implementation/library_transpile_showcase/` demonstrates per-algorithm transpilation using library APIs.

It includes one file per algorithm track plus an orchestrator:

- `_01_grover.py`
- `_01_1_qaoa_grover.py`
- `_02_fixed_point.py`
- `_03_oblivious.py`
- `_03_25_controlled.py`
- `_03_5_foqa.py`
- `_04_distributed.py`
- `_05_variable_time.py`
- `_06_qsvt.py`
- `_07_unified_comparative.py`

Run it:

```bash
PYTHONPATH=src:. python3 library_implementation/library_transpile_showcase/run_all_with_library.py
```

Outputs:

- `library_implementation/library_transpile_showcase/results/library_transpile_results.json`
- `library_implementation/library_transpile_showcase/results/library_transpile_summary.csv`

## Non-Library Workflows

`non_library_implementation/` preserves script-first experimental and transpilation workflows:

- `experiments/`: algorithm experiments and exploration scripts
- `transpile/`: full transpilation scenario suites and result logs
- `Transpile Algorithms GPU Parallelization/`: GPU-focused transpilation workflows

These are valuable for reproducing scenario-heavy studies and script-level comparisons.

## Documentation

If you want API docs with generated signatures:

```bash
mkdocs serve
```

Main docs entry: `docs/index.md`

API pages include:

- `grover`, `fixed_point`, `oblivious`, `foqa`, `distributed`, `variable_time`, `qsvt`, `diagnostics`
- `transpilation`, `transpilation_validation`

## Testing

Run tests:

```bash
pytest -q
```

For focused transpilation tests:

```bash
pytest -q tests/test_transpilation_module.py tests/test_transpilation_validation.py
```

## Current Status

- Library APIs are active and usable for all core algorithm families.
- Library-side transpilation workflow is demonstrated in per-algorithm form.
- Non-library scenario suites remain available for research-style execution and comparison.

## Notes

- Some advanced scenario scripts may be significantly heavier than quick API examples.
- Depending on environment, Aer/OpenMP runtime constraints can affect simulator-heavy flows.
- If Matplotlib cache warnings appear, set a writable `MPLCONFIGDIR`.
