# Library Transpile GPU Showcase

Library-based GPU transpilation suite for all algorithm tracks, mirroring the non-library GPU workflow style.

## Included algorithm tracks

- 1_grover
- 1.1_qaoa_grover
- 2_fixed_point
- 3_oblivious
- 3.25_controlled
- 3.5_foqa
- 4_distributed
- 5_variable_time
- 6_qsvt
- 7_unified_comparative

## Run

```bash
PYTHONPATH=src:. python3 library_implementation/library_transpile_gpu_showcase/run_all_gpu_with_library.py
```

## Output

- `library_implementation/library_transpile_gpu_showcase/results/library_gpu_transpile_results.json`
- `library_implementation/library_transpile_gpu_showcase/results/library_gpu_transpile_summary.csv`

## Notes

- The runner tries `AerSimulator(device="GPU")` first.
- If GPU is unavailable, it falls back to CPU automatically and records the reason in output metadata.
