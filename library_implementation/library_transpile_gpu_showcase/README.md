# Library Transpile GPU Showcase

This folder demonstrates GPU-oriented transpilation and execution using only library APIs (`ampamp`).

## What it does

- Builds representative circuits from library modules:
  - Grover
  - Fixed-Point
  - Oblivious
  - FOQA
  - Distributed
  - Variable-Time
  - QSVT
- Tries to initialize Aer with `device="GPU"`.
- Falls back to CPU automatically if GPU backend is unavailable.
- Transpiles circuits with a native gate basis and optimization level 3.
- Optionally executes transpiled circuits and records basic execution stats.

## Run

```bash
PYTHONPATH=src:. python3 library_implementation/library_transpile_gpu_showcase/run_gpu_with_library.py
```

## Output

- `results/gpu_transpile_results.json`
- `results/gpu_transpile_summary.csv`

## Notes

- If GPU is not available, the runner reports `backend_mode=CPU` and still produces results.
- This is a library-driven workflow; no non-library scenario scripts are required.
