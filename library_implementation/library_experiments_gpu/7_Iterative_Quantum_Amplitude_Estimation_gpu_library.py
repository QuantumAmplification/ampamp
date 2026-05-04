"""Theory-side statistical analysis of Iterative Quantum Amplitude Estimation (IQAE) and MLAE.

This is the LIBRARY implementation equivalent, utilizing the ampamp.grover module
to avoid rewriting standard logical components. This script is GPU accelerated.
"""

from __future__ import annotations

import ast
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))

_AER_GPU_HINT = (
    "This script requires qiskit-aer-gpu on CUDA-capable Linux/x86_64 "
    "(or qiskit-aer-gpu-cu11 for CUDA 11)."
)

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from ampamp.grover import GroverEngine

try:
    # Use the non-library one-click-utils if we don't have a library one available
    sys.path.insert(0, os.path.join(ROOT, "non_library_implementation", "experiments"))
    from one_click_utils import start_one_click_session
except Exception:
    def start_one_click_session(script_file, *, figure_prefix=None, log_name="terminal_output.log"):
        import atexit
        import io
        import os
        from pathlib import Path
        script_path = Path(script_file).resolve()
        result_dir = script_path.parent / f"[RESULT]{script_path.stem}"
        result_dir.mkdir(parents=True, exist_ok=True)
        old_stdout, old_stderr, old_cwd = sys.stdout, sys.stderr, Path.cwd()
        log_handle = open(result_dir / log_name, "w", encoding="utf-8")
        class _Tee(io.TextIOBase):
            def __init__(self, *streams): self._streams = streams
            def write(self, data): [s.write(data) or s.flush() for s in self._streams]; return len(data)
            def flush(self): [s.flush() for s in self._streams]
        sys.stdout = _Tee(old_stdout, log_handle)
        sys.stderr = _Tee(old_stderr, log_handle)
        os.chdir(result_dir)
        try:
            import matplotlib.pyplot as plt
            old_show = plt.show
            prefix = figure_prefix or script_path.stem
            counter = {"n": 0}
            def _save_show(*args, **kwargs):
                del args, kwargs
                for fig_id in list(plt.get_fignums()):
                    counter["n"] += 1
                    plt.figure(fig_id).savefig(result_dir / f"{prefix}_figure_{counter['n']:03d}.png", dpi=220, bbox_inches="tight")
                plt.close("all")
            plt.show = _save_show
        except Exception:
            old_show = None
        def _cleanup():
            try:
                if old_show is not None:
                    import matplotlib.pyplot as plt
                    plt.show = old_show
            except Exception:
                pass
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.chdir(old_cwd)
            log_handle.close()
        atexit.register(_cleanup)
        return result_dir


class IQAELab:
    """
    Simulates Quantum Amplitude Estimation algorithms (MLAE and IQAE concepts).
    Wrapped around the ampamp library using GPU-accelerated backend.
    """
    def __init__(self, n_qubits, good_indices):
        self.engine = GroverEngine(n_qubits=n_qubits, marked_indices=good_indices)
        self.n = self.engine.n
        self.good_indices = self.engine.marked
        self.p_true = float(self.engine.solution_density)
        self.theta0_true = float(self.engine.theta)
        try:
            self.backend = AerSimulator(device="GPU")
        except Exception as exc:
            print("Missing/incompatible qiskit_aer. Falling back to CPU.")
            print(f"{_AER_GPU_HINT} Original error: {type(exc).__name__}: {exc}")
            self.backend = AerSimulator()

    def simulate_circuit(self, k: int, shots: int = 100) -> int:
        qc = self.engine.construct_circuit(k)
        qc.measure_all()
        t_qc = transpile(qc, self.backend)
        result = self.backend.run(t_qc, shots=shots).result()
        counts = result.get_counts()
        
        successes = 0
        for index in self.good_indices:
            target_bin = format(index, f'0{self.n}b')
            successes += counts.get(target_bin, 0)
            
        return successes

    def analyze_mlae(self, max_k: int, shots: int):
        k_schedule = [0]
        curr_k = 1
        while curr_k <= max_k:
            k_schedule.append(curr_k)
            curr_k *= 2
            
        results = []
        for k in k_schedule:
            h_k = self.simulate_circuit(k, shots)
            results.append((k, h_k, shots))
            
        theta_grid = np.linspace(0, np.pi/2, 1000)
        likelihood = np.ones_like(theta_grid, dtype=float)
        
        for k, h, n in results:
            P_k = np.sin((2*k + 1) * theta_grid)**2
            P_k = np.clip(P_k, 1e-10, 1 - 1e-10)
            log_L = h * np.log(P_k) + (n - h) * np.log(1 - P_k)
            likelihood += log_L
            
        likelihood = np.exp(likelihood - np.max(likelihood))
        likelihood /= np.trapz(likelihood, theta_grid)
        
        theta_hat = theta_grid[np.argmax(likelihood)]
        p_hat = np.sin(theta_hat)**2
        
        return k_schedule, results, theta_grid, likelihood, p_hat

def run_full_analysis(
    n_qubits: int = 6,
    good_indices: str = "10,25",
    max_k: int = 16,
    shots: int = 100,
    save_plot: str = "iqae_geometric_evidence_gpu_library.png",
    show_plot: bool = False
):
    print("\n" + "=" * 70)
    print("ITERATIVE QUANTUM AMPLITUDE ESTIMATION (IQAE/MLAE) LABORATORY (GPU LIBRARY BACKED)")
    print("=" * 70)
    
    indices = [int(i.strip()) for i in good_indices.split(',')]
    lab = IQAELab(n_qubits=n_qubits, good_indices=indices)
    
    print(f"Target Qubits: {n_qubits}")
    print(f"Marked States: {len(indices)} (p = {lab.p_true:.6f})")
    print(f"True Angle (theta0): {lab.theta0_true:.6f}")
    
    print(f"\nRunning Maximum Likelihood Estimation Schedule up to max_k={max_k}")
    k_schedule, results, theta_grid, likelihood, p_hat = lab.analyze_mlae(max_k=max_k, shots=shots)
    
    print(f"\nResults Collected (shots={shots}):")
    for k, h, n in results:
        print(f"  k={k:<3} -> {h:<4} successes (Empirical P_k = {h/n:.4f})")
        
    print(f"\nEstimated Amplitude (p_hat): {p_hat:.6f}")
    print(f"Absolute Error: {abs(p_hat - lab.p_true):.6e}")
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(theta_grid, likelihood, label="Likelihood Function L(theta)", color='blue')
    plt.axvline(lab.theta0_true, color='green', linestyle='--', label=f"True theta0 = {lab.theta0_true:.4f}")
    plt.axvline(np.arcsin(np.sqrt(p_hat)), color='red', linestyle=':', label=f"Estimated theta_hat = {np.arcsin(np.sqrt(p_hat)):.4f}")
    plt.title("MLAE Likelihood Landscape (GPU Library Backed)")
    plt.xlabel("Theta")
    plt.ylabel("Likelihood (normalized)")
    plt.legend()
    plt.grid(True)
    
    out = Path(save_plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"\nSaved figure: {out}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def _parse_cli_value(raw: str):
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return raw

def _parse_command_line(argv: Sequence[str]) -> Optional[dict]:
    if not argv:
        return {}
    if any(token.startswith("-") for token in argv):
        return None
    kwargs = {}
    for token in argv:
        if "=" in token:
            k, v = token.split("=", 1)
            kwargs[k.strip()] = _parse_cli_value(v.strip())
    return kwargs

if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="iqae_gpu_library")
    cli_kwargs = _parse_command_line(sys.argv[1:])
    if cli_kwargs is None:
        run_full_analysis()
    elif cli_kwargs:
        run_full_analysis(**cli_kwargs)
    else:
        run_full_analysis()
