# Library-Only Transpilation Showcase

This folder demonstrates that each algorithm can be transpiled using the `ampamp` library modules directly.
Each algorithm has its own file under `library_transpile_showcase/algorithms/`.

## Per-algorithm files

- `_01_grover.py` (`ampamp.grover` + `ampamp.diagnostics` + `ampamp.transpilation`)
- `_01_1_qaoa_grover.py` (`ampamp.grover` + `ampamp.transpilation`)
- `_02_fixed_point.py` (`ampamp.fixed_point` + `ampamp.transpilation`)
- `_03_oblivious.py` (`ampamp.oblivious` + `ampamp.transpilation`)
- `_03_25_controlled.py` (`ampamp.grover` + `ampamp.transpilation`)
- `_03_5_foqa.py` (`ampamp.foqa` + `ampamp.transpilation`)
- `_04_distributed.py` (`ampamp.distributed` + `ampamp.transpilation`)
- `_05_variable_time.py` (`ampamp.variable_time` + `ampamp.transpilation`)
- `_06_qsvt.py` (`ampamp.qsvt` + `ampamp.diagnostics` + `ampamp.transpilation`)
- `_07_unified_comparative.py` (aggregates all algorithm files)

## Parity with existing transpile outputs

Where reference logs exist (`[RESULT]` JSONL files), per-algorithm parity reports compare optimized depth against reference depth values.

## Run all

```bash
PYTHONPATH=src python3 library_transpile_showcase/run_all_with_library.py
```

## Output

- `library_transpile_showcase/results/library_transpile_results.json`
- `library_transpile_showcase/results/library_transpile_summary.csv`
