# ampamp Robustness Test Report

Date: 2026-05-05

This report summarizes the library-hardening pass for `ampamp` as a general
framework for amplitude amplification. The goal was to ensure every public
class has a real behavior test path and that the previously unfinished public
surfaces now return structured, deterministic library results.

## Verification Commands

```bash
PYTHONPATH=/Users/varundaiya/ampamp/src python3 -m pytest
python3 -m py_compile src/ampamp/*.py
PYTHONPATH=/Users/varundaiya/ampamp/src python3 test_ampamp.py
```

Results:

```text
43 passed in 1.82s
py_compile passed
test_ampamp.py passed
public imports ok
```

## Test Files Added

```text
tests/test_core_engines.py
tests/test_diagnostics.py
tests/test_public_api.py
```

These complement the existing/new tests:

```text
tests/test_entanglement.py
tests/test_oracles.py
tests/test_transpilation_module.py
tests/test_transpilation_validation.py
```

## Public API Coverage

| Class or API | Tested behavior |
| --- | --- |
| `GroverEngine` | Initialization, duplicate marked-state handling, success probability, oracle construction, diffusion construction, circuit construction, invalid inputs. |
| `FixedPointEngine` | Odd-degree Yoder-Low-Chuang schedule, exact `gamma`, palindromic `zetas`, matched `alphas`/`betas`, exact fixed-point threshold probability, circuit construction, invalid `L` and `delta`. |
| `ObliviousEngine` | Ancilla preparation probability, block encoding, reflection circuit, invalid qubit counts, invalid probability, invalid matrix shape. |
| `FOQAEngine` | Mizel schedule, constant schedule, LCU split operator unitarity, recurrence bounds, proxy sequence, invalid theta and step counts. |
| `DQAAEngine` | Prefix partitioning, local FPAA node-circuit construction, invalid prefix sizes. |
| `OracleSynthesizer` | False, true, and nontrivial node-local formula synthesis. |
| `VariableTimeBranch` | Public `stopping_time` and `p_success` names, plus backward-compatible legacy aliases. |
| `VTAAEngine` | Branch sorting, weight normalization, stopping-time moments, asymptotic bound, staged circuit construction, invalid branch inputs, zero-success infinite bound. |
| `SU2QSPEngine` | Signal unitary, Z rotation unitary, QSP sequence evaluation shape. |
| `QSVTSynthesizer` | Jacobi-Anger outputs, LCU norm bound, public inverse-polynomial synthesis, odd matrix-inverse polynomial, invalid degree/kappa. |
| `IQAEConfig` | Valid and invalid estimator configuration. |
| `IQAEResult` | Result dataclass fields. |
| `IQAEEngine` | Requires `GroverEngine`, validates config, returns structured IQAE, MLAE, and ESPRIT-style estimates. |
| `OracleSpec` | Source validation through oracle tests. |
| `OracleBuilder` | Marked index/bitstring views, formula source, phase oracle and bit-flip oracle construction. |
| `build_phase_oracle` | Standard sign-flip phase oracle and arbitrary diagonal phase. |
| `build_bit_flip_oracle` | Output-qubit flip semantics for marked input states. |
| `marked_bitstrings_from_formula` | Formula truth-table enumeration semantics. |
| `EntanglementCountConfig` | Light/hard construction, hardware limits. |
| `profile_entanglement_counts` | Sampled light mode, every-step hard mode, peak active-entangled-qubit count, skipped status for too many qubits. |
| `HardwareCostWeights` | Used in transpilation cost calculation. |
| `TranspilationProfileConfig` | Basis, coupling, and cost configuration. |
| `TranspilationProfiler` | Staged metrics, CNOT counts, routing/optimization depths, hardware penalty. |
| `TranspilationBatchProfiler` | Named multi-circuit profiling. |
| `ValidationNoiseConfig` | Ideal and custom noise resolution. |
| `ValidationLogConfig` | JSONL log path behavior through validation test. |
| `BackendValidationConfig` | Validation limits, shots, seed, logging, thresholds. |
| `BackendValidationRunner` | Ideal/noisy validation, TVD metrics, structured logging, too-many-qubits rejection. |
| `GroverAuditor` | Subspace rotation purity, cloning-test output shape, souffle heatmap shape. |
| `FPAAAuditor` | Exact passband sweep from the YLC success polynomial and FTQC T-count/depth estimates. |
| `ObliviousAuditor` | LCU distance calculation and input-state independence acid test. |
| `FOQAAuditor` | Damping-regime recurrence, empty-database paradox, and log-log complexity audit. |
| `DistributedAuditor` | Lucky-node Monte Carlo check, entanglement obstruction metric, NISQ noise proxy, and network sifting thresholds. |
| `VTAAAuditor` | Early-success cost-ratio sweep against worst-case amplitude amplification. |
| `FundamentalLimitsAuditor` | Subspace SVD, open-system trajectory proxy, diffusion scaling, and phase-leakage audit. |
| `QSVTAuditor` | Unitarity/parity check, Gibbs ringing, subnormalization, phase quantization, and parity-scramble diagnostics. |

## What The Tests Prove

The suite now checks the framework at four levels.

1. Public import surface

The `ampamp.__all__` surface is checked for uniqueness and missing names.
This prevents the package from advertising objects that cannot be imported.

2. Algorithmic engines

Each engine is exercised with valid inputs, invalid inputs, and at least one
semantic property. Examples include Grover phase/diffusion unitarity, VTAA
weight normalization, FOQA LCU unitarity, and QSVT parity-enforced inverse
coefficients.

3. Framework utilities

The new oracle framework is tested independently from Grover. The new
entanglement-count profiler is also tested independently, including both
hardware-aware modes:

- `light`: sampled checkpoints, lower cost.
- `hard`: every quantum instruction, higher cost.

4. Diagnostics and estimator surfaces

Auditor methods now return structured metrics instead of placeholder
boundaries. IQAE, MLAE, and ESPRIT-style estimators are exercised for valid
result ranges, confidence intervals, and query accounting.

## Residual Risks

- Several diagnostics are intentionally lightweight analytical or proxy audits
  so the default suite remains fast. They should be supplemented by slow
  numerical sweeps for paper-grade benchmarks.
- IQAE estimators are deterministic, shot-count based implementations over the
  `GroverEngine` success model. Hardware-backed sampling can be added later
  without changing the result dataclass.
- Tests are deterministic and small enough for CI. They are not a replacement
  for large-scale numerical validation sweeps across high qubit counts.

## Recommended Next Steps

1. Add a CI job that runs:

```bash
PYTHONPATH=src python3 -m pytest
python3 -m py_compile src/ampamp/*.py
```

2. Add optional slow tests for larger circuits behind a marker such as
`pytest -m slow`, so the normal suite stays fast while deeper validation is
still available.
