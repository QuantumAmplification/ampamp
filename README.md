<div align="center">

# ampamp

### Amplitude amplification, QSVT, and circuit-analysis workflows for Python

`ampamp` is a research library for building quantum amplification circuits,
inspecting their mathematical behavior, profiling compilation cost, and
validating ideal/noisy backend behavior from one compact API.

[![PyPI](https://img.shields.io/pypi/v/ampamp?color=1f6feb)](https://pypi.org/project/ampamp/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-3776AB)](https://pypi.org/project/ampamp/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-2ea44f)](https://quantumamplification.github.io/ampamp/)
[![Survey](https://img.shields.io/badge/survey-Zenodo-1682d4)](https://doi.org/10.5281/zenodo.20054981)
[![Repository](https://img.shields.io/badge/GitHub-QuantumAmplification%2Fampamp-24292f)](https://github.com/QuantumAmplification/ampamp)

[Docs](https://quantumamplification.github.io/ampamp/) ·
[PyPI](https://pypi.org/project/ampamp/) ·
[Survey paper](https://doi.org/10.5281/zenodo.20054981) ·
[Source](https://github.com/QuantumAmplification/ampamp)

</div>

---

## Why ampamp?

Amplitude amplification is the shared engine behind Grover search, amplitude
estimation, fixed-point schedules, oblivious amplification, variable-time
algorithms, and QSVT-style workflows. In papers these often appear as separate
constructions. In experiments and software, the recurring need is simpler:

1. describe the search, oracle, block encoding, branch model, or polynomial;
2. build an inspectable circuit or classical schedule;
3. profile the circuit under a hardware model;
4. validate the behavior under ideal or noisy simulation;
5. keep the whole workflow reproducible.

`ampamp` is built for that practical loop.

Requires Python `>=3.9`.

```bash
python3 -m pip install ampamp
```

```python
from ampamp import GroverEngine, TranspilationProfiler, TranspilationProfileConfig

engine = GroverEngine(n_qubits=6, marked_indices=[10, 25])
qc = engine.construct_circuit(iterations=engine.k_optimal)

metrics = TranspilationProfiler(
    TranspilationProfileConfig(
        coupling_map_edges=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
    )
).profile_circuit(qc)

print(engine.k_optimal)
print(metrics["post_optimization_depth"])
print(metrics["final_cnots"])
```

## At A Glance

| Area | What it gives you | Main entry points |
| --- | --- | --- |
| Grover search | Phase oracles, diffusion operators, optimal-iteration estimates, success probabilities, full circuits | `GroverEngine` |
| Oracle construction | Marked-state, Boolean-formula, bit-flip, phase, and direct unitary oracle circuits | `OracleBuilder`, `build_phase_oracle`, `build_bit_flip_oracle`, `build_unitary_oracle` |
| Fixed-point AA | Chebyshev-style phase schedules and monotone amplification circuits | `FixedPointEngine` |
| Oblivious / FOQA | Ancilla preparation, block-encoding scaffolds, reflections, damping schedules, recurrence simulation | `ObliviousEngine`, `FOQAEngine` |
| Distributed AA | Prefix/suffix target partitioning and local symbolic oracle synthesis | `DQAAEngine`, `OracleSynthesizer` |
| Variable-time AA | Branch models, stopping-time moments, success mass, asymptotic estimates | `VTAAEngine`, `VariableTimeBranch` |
| QSVT / QSP | SU(2) QSP sequence evaluation and Chebyshev polynomial helpers | `SU2QSPEngine`, `QSVTSynthesizer` |
| Diagnostics | Executable checks for rotations, schedules, partitions, branches, and polynomial parity | `GroverAuditor`, `FPAAAuditor`, `QSVTAuditor`, ... |
| Transpilation | Staged compile metrics, routing counts, timing models, batch profiling, hardware-cost scores | `TranspilationProfiler`, `TranspilationBatchProfiler` |
| Backend validation | Ideal/noisy simulator comparisons, total-variation distance, noise presets, JSONL logs | `BackendValidationRunner` |
| Entanglement count | Light and hard active-entangled-qubit profiling for Qiskit circuits | `profile_entanglement_counts` |

## Workflow Examples

### Oracle construction

Use marked indices, Boolean formulas, or direct unitary matrices.

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

For distributed symbolic work, `OracleSynthesizer` substitutes fixed node
prefixes into a global Boolean formula and emits node-local phase oracles.

### Fixed-point amplification

```python
from ampamp import FixedPointEngine

fp = FixedPointEngine(L=5, delta=0.1)
qc = fp.build_fixed_point_circuit(num_qubits=4, marked_indices=[3, 11])

print(fp.alphas)
print(fp.betas)
print(qc.depth())
```

### Variable-time branch analysis

```python
from ampamp import VTAAEngine, VariableTimeBranch

branches = [
    VariableTimeBranch(stop_time=1.0, weight=0.4, success_given_branch=0.8),
    VariableTimeBranch(stop_time=3.0, weight=0.6, success_given_branch=0.5),
]

vtaa = VTAAEngine(branches)
print(vtaa.p_success)
print(vtaa.stopping_time_moments())
print(vtaa.vtaa_asymptotic_bound())
```

### Backend validation

```python
from ampamp import GroverEngine
from ampamp import (
    BackendValidationRunner,
    BackendValidationConfig,
    ValidationNoiseConfig,
    ValidationLogConfig,
)

qc = GroverEngine(n_qubits=4, marked_indices=[3, 11]).construct_circuit(
    iterations=1,
)

runner = BackendValidationRunner(
    BackendValidationConfig(
        shots=1024,
        seed=42,
        noise=ValidationNoiseConfig(noise_level="light"),
        logging=ValidationLogConfig(enabled=True, log_dir="./validation_logs"),
    )
)

result = runner.validate_circuit("grover_demo", qc)
print(result["status"], result["metrics"]["tvd"])
```

### Entanglement count profiling

```python
from ampamp import EntanglementCountConfig, GroverEngine, profile_entanglement_counts

qc = GroverEngine(n_qubits=5, marked_indices=[0]).construct_circuit(
    iterations=1,
)

light = profile_entanglement_counts(
    qc,
    EntanglementCountConfig.light(max_qubits=12),
)

print(light["peak_active_entangled_qubits"])
```

## Public Import Surface

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

## Documentation

| Resource | Link |
| --- | --- |
| Documentation site | https://quantumamplification.github.io/ampamp/ |
| PyPI package | https://pypi.org/project/ampamp/ |
| Source repository | https://github.com/QuantumAmplification/ampamp |
| Implementation comparison workflows | https://github.com/QuantumAmplification/Implementation |

Serve the docs locally:

```bash
mkdocs serve
```

Build the docs:

```bash
mkdocs build --strict
```

## Research Lineage

`ampamp` is a software layer over a long amplitude-amplification paper trail:
Grover search, polynomial lower bounds, amplitude estimation, arbitrary-phase
amplification, fixed-point search, controlled amplification, oblivious
amplification, LCU/block encodings, variable-time algorithms, Hamiltonian
simulation, QSP, and QSVT.

The companion survey for the project is:

```text
M. Kumar, Y. Tahir, and V. Daiya,
"Amplitude Amplification Algorithms",
Zenodo, 2026.
https://doi.org/10.5281/zenodo.20054981
```

The SciPost manuscript in this repository contains the fuller bibliography,
including Grover, Bennett-Bernstein-Brassard-Vazirani, Beals-Buhrman-Cleve-
Mosca-de Wolf, Brassard-Hoyer-Mosca-Tapp, Yoder-Low-Chuang, Low-Chuang,
Ambainis, Harrow-Hassidim-Lloyd, Gilyen-Su-Low-Wiebe, Martyn-Rossi-Tan-Chuang,
and the distributed/FOQA papers used by the package.

## Repository Layout

```text
src/ampamp/       installable library source
docs/             MkDocs documentation and API pages
tests/            unit tests and regression checks
dist/             local release artifacts, when built
fd.tex            SciPost manuscript draft
main.tex          alternate SciPost manuscript draft
CITATION.cff      citation metadata
```

Scenario scripts and implementation-comparison workflows live outside the
package repository:

```text
https://github.com/QuantumAmplification/Implementation
```

## Development

Install locally:

```bash
python3 -m pip install -e .
```

Run the test suite:

```bash
PYTHONPATH=src pytest
```

Focused checks:

```bash
PYTHONPATH=src pytest tests/test_oracles.py
PYTHONPATH=src pytest tests/test_transpilation_module.py tests/test_transpilation_validation.py
```

Build a release artifact locally:

```bash
python3 -m build
python3 -m twine check dist/*
```

## Reproducibility Notes

For papers, notebooks, and benchmark runs, record:

- `ampamp` version;
- Python version;
- Qiskit and Qiskit Aer versions;
- backend or simulator settings;
- transpilation basis gates, coupling map, and optimization levels;
- random seeds and shot counts;
- validation noise preset or custom noise parameters.

Some simulator-heavy workflows can be environment-sensitive. If Matplotlib cache
warnings appear, set a writable `MPLCONFIGDIR`.

## Citation

Until an archival software DOI is attached to a release, cite the repository and
the companion survey:

```bibtex
@software{Kumar2026QuantumAmplificationLibrary,
  author = {Kumar, Mithilesh and Tahir, Yusuf and Daiya, Varun},
  title = {Quantum Amplification Library},
  year = {2026},
  month = apr,
  url = {https://github.com/QuantumAmplification/ampamp}
}

@misc{Kumar2026AmplitudeAmplification,
  author = {Kumar, Mithilesh and Tahir, Yusuf and Daiya, Varun},
  title = {Amplitude Amplification Algorithms},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.20054981},
  url = {https://doi.org/10.5281/zenodo.20054981}
}
```
