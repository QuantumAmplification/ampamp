"""Theory-side geometric analysis of standard Grover search.

Standing notation aligned with final.tex:
- H_Good / H_Bad: marked and unmarked subspaces.
- |All> = H^{otimes n}|0>^{otimes n}: prepared uniform superposition.
- p = ||Pi_Good |All>||^2 = M/N: initial success probability.
- sin^2(theta0) = p and one Grover iterate rotates by theta = 2 theta0.
- P_k = sin^2((2k + 1) theta0): ideal success probability after k iterations.
"""

from __future__ import annotations

import ast
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace

try:
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

DEFAULT_TEST_CASES = (
    "Case 1 (M << N)|1024|3;"
    "Case 2 (M = N/2)|1024|512;"
    "Case 3 (M > N/2)|1024|768"
)
DEFAULT_PLOT_PATH = "grover_geometric_evidence.png"


def grover_base_angle(p_value: float) -> float:
    """Return theta0, defined by sin^2(theta0) = p."""
    if not 0.0 <= p_value <= 1.0:
        raise ValueError("p_value must lie in [0, 1]")
    return float(np.arcsin(np.sqrt(p_value)))


def grover_rotation_angle(p_value: float) -> float:
    """Return the Grover rotation angle theta = 2 theta0."""
    return 2.0 * grover_base_angle(p_value)


def grover_optimal_iterations(p_value: float) -> int:
    """Return the standard near-optimal Grover iteration count."""
    theta0 = grover_base_angle(p_value)
    if theta0 == 0.0 or theta0 == np.pi / 2:
        return 0
    return int(np.floor(np.pi / (4.0 * theta0) - 0.5))


def grover_success_probability(p_value: float, k: int) -> float:
    """Return the ideal Grover success probability after k iterations."""
    if k < 0:
        raise ValueError("k must be non-negative")
    theta0 = grover_base_angle(p_value)
    return float(np.sin((2 * k + 1) * theta0) ** 2)


def _parse_index_list(raw: str) -> list[int]:
    text = raw.strip()
    if not text:
        return []
    return [int(chunk.strip(), 0) for chunk in text.split(",") if chunk.strip()]


def _parse_test_cases(raw: str) -> list[dict[str, int | str]]:
    cases: list[dict[str, int | str]] = []
    for case_index, chunk in enumerate(raw.split(";"), start=1):
        item = chunk.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) == 2:
            label = f"Case {case_index}"
            n_value, m_value = parts
        elif len(parts) == 3:
            label, n_value, m_value = parts
        else:
            raise ValueError(
                "Each test case must use 'label|N|M' or 'N|M' format, "
                f"received '{item}'."
            )
        cases.append({"label": label, "N": int(n_value, 0), "M": int(m_value, 0)})
    return cases


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


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote_char = ""
    escaped = False

    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if quote_char:
            current.append(ch)
            if ch == "\\":
                escaped = True
            elif ch == quote_char:
                quote_char = ""
            continue
        if ch in ("'", '"'):
            quote_char = ch
            current.append(ch)
            continue
        if ch in "([{":
            depth += 1
            current.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
            continue
        if ch == "," and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue
        current.append(ch)

    piece = "".join(current).strip()
    if piece:
        parts.append(piece)
    return parts


def _parse_kwargs_text(raw: str) -> dict:
    kwargs = {}
    for chunk in _split_top_level_commas(raw.strip()):
        if "=" not in chunk:
            raise ValueError(f"Expected key=value pair, got '{chunk}'")
        key, value = chunk.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _parse_kwargs_tokens(tokens: Sequence[str]) -> dict:
    kwargs = {}
    for token in tokens:
        piece = token.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Expected key=value pair, got '{piece}'")
        key, value = piece.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _parse_command_line(argv: Sequence[str]) -> Optional[dict]:
    if not argv:
        return {}
    if any(token.startswith("-") for token in argv):
        return None
    if len(argv) == 1:
        return _parse_kwargs_text(argv[0])
    return _parse_kwargs_tokens(argv)


def _normalize_good_indices(value) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") or text.startswith("("):
            parsed = ast.literal_eval(text)
            return [int(item) for item in parsed]
        if "|" in text and "," not in text:
            return [int(chunk.strip(), 0) for chunk in text.split("|") if chunk.strip()]
        return _parse_index_list(text)
    return [int(item) for item in value]


def _normalize_test_cases(value) -> list[dict[str, int | str]]:
    if isinstance(value, str):
        return _parse_test_cases(value)
    return list(value)


def print_math_overview() -> None:
    print("\n" + "=" * 70)
    print("GROVER MATHEMATICAL OVERVIEW")
    print("=" * 70)
    print("Initial state decomposition:")
    print("  |psi_0> = sqrt(p)|Good> + sqrt(1 - p)|Bad>, with p = M/N.")
    print("Angular parametrization:")
    print("  sin^2(theta0) = p and one Grover iterate rotates by theta = 2 theta0.")
    print("Ideal success law:")
    print("  P_k = sin^2((2k + 1) theta0).")
    print("Near-optimal iteration count:")
    print("  k* = floor(pi / (4 theta0) - 1/2).")
    print("Invariant-subspace statement:")
    print("  Ideal Grover dynamics remain in span{|Good>, |Bad>}.\n")


class GroverGeometricAnalysis:
    """
    Theory-side numerical analysis of Grover's two-dimensional geometry.

    Standing notation aligned with final.tex:
    - H_Good / H_Bad: target and non-target subspaces
    - |All> = H^{⊗n}|0>^{⊗n}: prepared input state
    - p = ||Pi_Good |All>||^2 = M/N: initial success probability
    - sin^2(theta0) = p and Grover step angle theta = 2*theta0
    """
    def __init__(self, n_qubits, good_indices):
        """
        Initialize the ideal Grover instance.

        Args:
            n_qubits (int): Total number of qubits in the quantum register.
            good_indices (list of int): Computational-basis indices spanning H_Good.
        """
        self.n = int(n_qubits)
        if self.n <= 0:
            raise ValueError("n_qubits must be positive")
        self.N = 2 ** self.n

        self.good_indices = sorted({int(idx) for idx in good_indices})
        for idx in self.good_indices:
            if idx < 0 or idx >= self.N:
                raise ValueError(f"good index {idx} is outside [0, {self.N - 1}]")
        self.good_index_set = set(self.good_indices)
        self.M = len(self.good_indices)

        self.p = self.M / self.N
        self.p_init = self.p
        self.theta0 = grover_base_angle(self.p)
        self.theta = grover_rotation_angle(self.p)
        self.k_optimal = grover_optimal_iterations(self.p)

        self.backend = AerSimulator()

        self.good_vec = np.zeros(self.N, dtype=complex)
        if self.M > 0:
            amp_good = 1.0 / np.sqrt(self.M)
            for idx in self.good_indices:
                self.good_vec[idx] = amp_good

        self.bad_vec = np.zeros(self.N, dtype=complex)
        bad_count = self.N - self.M
        if bad_count > 0:
            amp_bad = 1.0 / np.sqrt(bad_count)
            for idx in range(self.N):
                if idx not in self.good_index_set:
                    self.bad_vec[idx] = amp_bad

    def grover_success_prob(self, p_value: float, k: int) -> float:
        """Standard Grover success probability in terms of p (with p=M/N in search)."""
        return grover_success_probability(p_value, k)

    def get_oracle(self):
        """
        Standard Phase Oracle for Section II.
        
        Constructs a phase oracle O = I - 2*Pi_Good that flips phase on H_Good.
        This is accomplished by flipping each marked basis state to all 1s, applying a multi-controlled Z gate 
        (simulated using H, Multi-Controlled X, and H), and uncomputing the bit flips.
        
        Returns:
            QuantumCircuit: The synthesized phase oracle circuit.
        """
        qc = QuantumCircuit(self.n)
        for index in self.good_indices:
            # Convert the good index to its binary string representation (little-endian for Qiskit)
            good_bin = format(index, f'0{self.n}b')[::-1]
            
            # Apply X-gates to transform the zero-bits of the good state into ones
            for i, bit in enumerate(good_bin):
                if bit == '0': qc.x(i)
                
            # Apply a multi-controlled Z gate by wrapping a multi-controlled X (MCX) in Hadamard gates
            qc.h(self.n - 1)
            qc.mcx(list(range(self.n - 1)), self.n - 1)
            qc.h(self.n - 1)
            
            # Uncompute the initial X-gates to restore the state basis
            for i, bit in enumerate(good_bin):
                if bit == '0': qc.x(i)
        return qc

    def get_diffusion(self):
        """
        Standard Diffusion Reflection.
        
        Constructs the Grover diffusion operator (inversion about the mean).
        This operator reflects about |All> to amplify overlap with H_Good.
        
        Returns:
            QuantumCircuit: The synthesized diffusion operator circuit.
        """
        qc = QuantumCircuit(self.n)
        
        # Apply Hadamard gates to transform back to the computational basis
        qc.h(range(self.n))
        
        # Apply X gates to effectively map the zero state |00...0> to |11...1>
        qc.x(range(self.n))
        
        # Apply a multi-controlled Z gate to shift the phase of the |11...1> state
        qc.h(self.n - 1)
        qc.mcx(list(range(self.n - 1)), self.n - 1)
        qc.h(self.n - 1)
        
        # Uncompute the X gates map back from |11...1> to |00...0>
        qc.x(range(self.n))
        
        # Apply Hadamard gates to return to the superposition basis
        qc.h(range(self.n))
        
        return qc

    def analyze_geometry(self, max_k):
        """
        Module 1 & 2: Track two-dimensional rotation and overshoot instability.
        
        Observes the precise quantum state vector at each step of Grover's iteration to 
        empirically verify that the operation constitutes a rotation in a 2D invariant subspace 
        spanned by |Good> and |Bad>.
        
        Args:
            max_k (int): Maximum number of iterations to simulate.
            
        Returns:
            tuple: Arrays containing |Good> and |Bad> amplitudes, total subspace probability, 
                   and the empirical success probabilities.
        """
        a_k_vals = []
        b_k_vals = []
        p_total_vals = []
        success_probs = []
        
        # Iterate over increasing numbers of Grover operator applications
        for k in range(max_k + 1):
            qc = QuantumCircuit(self.n)
            
            # Initialize to the uniform superposition state
            qc.h(range(self.n))
            
            if k > 0:
                # Apply the Oracle and Diffusion operators k times
                oracle = self.get_oracle()
                diff = self.get_diffusion()
                for _ in range(k):
                    qc.append(oracle, range(self.n))
                    qc.append(diff, range(self.n))
            
            # Step B: Iterative Statevector Extraction
            # Capture the full statevector to analyze its overlap with the invariant subspace
            qc.save_statevector()
            result = self.backend.run(transpile(qc, self.backend)).result()
            state_k = np.array(result.get_statevector())
            
            # Step C: The Projection Logic
            # Projection coefficients in the {|Good>,|Bad>} plane
            ak = np.dot(self.good_vec.conj(), state_k)
            bk = np.dot(self.bad_vec.conj(), state_k)
            
            # Theoretical invariant subspace conservation check
            prob_in_subspace = np.abs(ak)**2 + np.abs(bk)**2
            
            a_k_vals.append(ak)
            b_k_vals.append(bk)
            p_total_vals.append(prob_in_subspace)
            
            # Success probability is |ak|^2 = ||Pi_Good|psi_k>||^2
            success_probs.append(np.abs(ak)**2)
            
        return np.array(a_k_vals), np.array(b_k_vals), np.array(p_total_vals), success_probs

    def simulate_cloning_barrier(self, max_k):
        """
        Module 3: Proves that naive CNOT cloning fails due to entanglement.
        
        Provides an empirical demonstration of the No-Cloning Theorem by showing how naive 
        attempt at copying via CNOT gates unavoidably destroys the purity of the state 
        due to unwanted system-environment entanglement.
        
        Args:
            max_k (int): Maximum number of iterations to simulate.
            
        Returns:
            tuple: Lists defining the state purity before and after attempting to copy the quantum register.
        """
        purity_original = []
        purity_after_copy = []
        
        for k in range(max_k + 1):
            # 1. Ideal System A Evaluation (Before Copying)
            qc_ideal = QuantumCircuit(self.n)
            qc_ideal.h(range(self.n))
            if k > 0:
                oracle = self.get_oracle()
                diff = self.get_diffusion()
                for _ in range(k):
                    qc_ideal.append(oracle, range(self.n))
                    qc_ideal.append(diff, range(self.n))
            
            qc_ideal.save_statevector()
            state_ideal = self.backend.run(transpile(qc_ideal, self.backend)).result().get_statevector()
            rho_ideal = partial_trace(state_ideal, []) # Full trace returns original
            pur_ideal = np.real(np.trace(np.dot(rho_ideal.data, rho_ideal.data)))
            purity_original.append(pur_ideal)
            
            # 2. Naive Copy System (System A + System B)
            qc = QuantumCircuit(self.n * 2) 
            qc.h(range(self.n))
            
            if k > 0:
                for _ in range(k):
                    qc.append(oracle, range(self.n))
                    qc.append(diff, range(self.n))
                
            # Naive CNOT 'Copy' Phase
            for i in range(self.n):
                qc.cx(i, i + self.n)
                
            qc.save_statevector()
            state_cloned = self.backend.run(transpile(qc, self.backend)).result().get_statevector()
            
            # 3. Analyze Original Register Purity After Copy
            rho_A_after = partial_trace(state_cloned, range(self.n, 2 * self.n))
            pur_collapse = np.real(np.trace(np.dot(rho_A_after.data, rho_A_after.data)))
            purity_after_copy.append(pur_collapse)
            
        return purity_original, purity_after_copy

    def sensitivity_heatmap(self, k_max, p_max, resolution=500):
        """
        Module 5: Generate the two-parameter Grover success heatmap.
        
        Calculates a vectorized surface of Grover success probabilities across varying 
        iteration counts and solution densities to highlight its extreme sensitivity.
        
        Args:
            k_max (int): The upper limit for the number of Grover iterations.
            p_max (float): Upper bound for p = M/N.
            resolution (int): Resolution of the discretized test grid.
            
        Returns:
            tuple: Coordinates and probabilistic success rates defining the instability heatmap.
        """
        # 1. Create the Meshgrid for extensive parametric sampling
        k_range = np.arange(0, k_max)
        p_range = np.linspace(0.001, p_max, resolution)
        K, L = np.meshgrid(k_range, p_range)
        
        # 2. Vectorized Analytic Calculation
        theta = 2 * np.arcsin(np.sqrt(L))
        success_heatmap = np.sin((2*K + 1) * theta / 2)**2
        
        return K, L, success_heatmap

    def recursive_nesting_analysis(self, k1, k2, max_p):
        """
        Module 4: Demonstrates self-similar scaling and extreme gate depth in nested Amplitude Amplification (AA).
        
        Examines the theoretical implications of executing nested iterations of standard Grover 
        logic—comparing multi-level success amplifications against corresponding exponential circuit depth.
        
        Args:
            k1 (int): Application count for an inner-layer Grover operator.
            k2 (int): Application count for an outer-layer meta-Grover operator.
            max_p (float): Maximum initial success probability in the sweep.
            
        Returns:
            tuple: Solution density arrays, nested level probability curves, and gate depth analysis mappings.
        """
        p_vals = np.linspace(0.0001, max_p, 500)
        
        # Level 1 probability P1
        theta_1 = 2 * np.arcsin(np.sqrt(p_vals))
        p1_vals = np.sin((2 * k1 + 1) * theta_1 / 2)**2
        
        # Level 2 Probability P2 (Treating P1 as the initial state)
        # Using exact rotation mechanics P_nested = sin^2((2k2 + 1) * arcsin(sqrt(P1)))
        theta_2 = 2 * np.arcsin(np.sqrt(p1_vals))
        p2_vals = np.sin((2 * k2 + 1) * theta_2 / 2)**2
        
        # 2. Gate Depth Analysis (Assuming standard decomposition overhead)
        oracle = self.get_oracle()
        diff = self.get_diffusion()
        
        oracle_depth = transpile(oracle, basis_gates=['u3', 'cx']).depth()
        diff_depth = transpile(diff, basis_gates=['u3', 'cx']).depth()
        
        # Standard Grover Depth for roughly equivalent success
        # Exact equivalent single-layer iteration count from angle matching:
        # 2*k_eq + 1 = (2*k1 + 1)(2*k2 + 1)
        k_equiv = 2 * k1 * k2 + k1 + k2
        std_depth = k_equiv * (oracle_depth + diff_depth)
        
        # Nested Grover Depth
        l1_depth = k1 * (oracle_depth + diff_depth)
        # Level 2 requires L1, L1_dagger, plus a phase flip
        l2_diff_overhead = l1_depth * 2 + 10 # approximate cost of L1 uncomputation
        nested_depth = k2 * (oracle_depth + l2_diff_overhead)
        
        depth_data = {
            'std_depth': std_depth,
            'nested_depth': nested_depth,
            'k_equiv': k_equiv
        }
        
        return p_vals, p1_vals, p2_vals, depth_data

GroverGeometricLab = GroverGeometricAnalysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Theory-side geometric analysis of standard Grover search.")
    parser.add_argument("--n-qubits", type=int, default=6, help="Number of qubits in the ideal Grover instance.")
    parser.add_argument(
        "--good-indices",
        type=str,
        default="10,25",
        help="Comma-separated marked basis indices, e.g. 10,25 or 0b1010,0b11001.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Maximum Grover iteration count to simulate. Defaults to ceil(k* * k-multiplier).",
    )
    parser.add_argument(
        "--k-multiplier",
        type=float,
        default=3.0,
        help="Multiplier used to set max-k from the near-optimal k*.",
    )
    parser.add_argument("--nesting-k1", type=int, default=3, help="Inner iteration count for the nested-amplification study.")
    parser.add_argument("--nesting-k2", type=int, default=3, help="Outer iteration count for the nested-amplification study.")
    parser.add_argument(
        "--nesting-p-max",
        type=float,
        default=0.15,
        help="Maximum initial success probability p used in the nested-amplification sweep.",
    )
    parser.add_argument("--heatmap-k-max", type=int, default=50, help="Maximum iteration count for the sensitivity heatmap.")
    parser.add_argument(
        "--heatmap-p-max",
        type=float,
        default=0.5,
        help="Maximum initial success probability p used in the sensitivity heatmap.",
    )
    parser.add_argument(
        "--heatmap-resolution",
        type=int,
        default=500,
        help="Grid resolution for the sensitivity heatmap.",
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        default=DEFAULT_TEST_CASES,
        help="Semicolon-separated case entries using 'label|N|M' or 'N|M'.",
    )
    parser.add_argument("--skip-test-cases", action="store_true", help="Skip the M-versus-N summary table.")
    parser.add_argument("--no-math-overview", action="store_true", help="Suppress the concise mathematical overview.")
    parser.add_argument("--run-qiskit-demo", action="store_true", help="Run the optional 3-qubit sampled-circuit demonstration.")
    parser.add_argument("--save-plot", type=str, default=DEFAULT_PLOT_PATH, help="Path for the composite figure.")
    parser.add_argument("--show-plot", action="store_true", help="Display the figure interactively after saving.")
    return parser


def _resolve_max_k(analysis: GroverGeometricAnalysis, requested_max_k: Optional[int], k_multiplier: float) -> int:
    if requested_max_k is not None:
        if requested_max_k < 0:
            raise ValueError("max-k must be non-negative")
        return int(requested_max_k)
    if k_multiplier < 0.0:
        raise ValueError("k-multiplier must be non-negative")
    return int(np.ceil(analysis.k_optimal * k_multiplier))


def run_case_table(test_cases: Sequence[dict[str, int | str]]) -> None:
    print("\n" + "=" * 70)
    print("GROVER TEST CASES: M vs N")
    print("=" * 70)

    for case in test_cases:
        label = str(case["label"])
        n_value = int(case["N"])
        m_value = int(case["M"])

        print(f"\n{label}:")
        print(f"  N = {n_value}, M = {m_value}")

        if n_value <= 0:
            print("  Skipped: N must be positive.")
            continue
        if m_value < 0 or m_value > n_value:
            print("  Skipped: require 0 <= M <= N.")
            continue

        p_value = m_value / n_value
        k_star = grover_optimal_iterations(p_value)
        success_prob = grover_success_probability(p_value, k_star)

        print(f"  p = M/N = {p_value:.6f}")
        print(f"  k* (near-optimal Grover iterations) = {k_star}")
        print(f"  Grover success probability at k* = {success_prob:.6f}")


def render_analysis_figure(
    analysis: GroverGeometricAnalysis,
    max_k: int,
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    p_total: np.ndarray,
    probs: Sequence[float],
    pur_orig: Sequence[float],
    pur_copy: Sequence[float],
    p_vals_nesting: np.ndarray,
    p1_curve: np.ndarray,
    p2_curve: np.ndarray,
    heatmap_k: np.ndarray,
    heatmap_p: np.ndarray,
    success_heatmap: np.ndarray,
    *,
    nesting_k1: int,
    nesting_k2: int,
    save_plot_path: str,
    show_plot: bool,
) -> None:
    fig = plt.figure(figsize=(24, 12))

    ax1 = plt.subplot(2, 3, 1)
    k_continuous = np.linspace(0, max_k, 300)
    p_theoretical = np.sin((2 * k_continuous + 1) * analysis.theta0) ** 2
    ax1.plot(k_continuous, p_theoretical, color="black", linestyle="--", alpha=0.5, label="Theoretical curve")
    ax1.plot(range(max_k + 1), probs, color="red", marker="o", linestyle="", label="Statevector simulation")
    if analysis.k_optimal > 0:
        ax1.axvline(analysis.k_optimal, color="green", linestyle=":", label="Near-optimal k*")
        ax1.text(analysis.k_optimal + 0.1, 0.1, "Near-optimal\niteration", color="green", ha="left")
        crash_k = 2 * analysis.k_optimal
        if crash_k <= max_k:
            ax1.axvline(crash_k, color="purple", linestyle=":", label="2k* overshoot point")
            ax1.text(crash_k + 0.1, 0.1, "Overshoot\npoint", color="purple", ha="left")
    ax1.axhline(0.5, color="orange", linestyle="--", label="Reference threshold (P=0.5)")
    ax1.set_title("Overshoot Instability in Standard Grover Search")
    ax1.set_xlabel("Iterations (k)")
    ax1.set_ylabel("Success probability P_k")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(
        range(max_k + 1),
        p_total,
        color="blue",
        marker="x",
        label=r"$|\langle H_{\mathrm{Good}}|\psi_k\rangle|^2 + |\langle H_{\mathrm{Bad}}|\psi_k\rangle|^2$",
    )
    ax2.set_title("Invariant-Subspace Conservation")
    ax2.set_xlabel("Iterations (k)")
    ax2.set_ylabel("Total probability in span{|Good>, |Bad>}")
    ax2.set_ylim(0.9, 1.1)
    ax2.legend()

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(np.real(b_vals), np.real(a_vals), marker="o", color="purple", linestyle="-", label=r"$|\psi_k\rangle$ path")
    theta_ideal = np.linspace(0, np.pi / 2, 100)
    ax3.plot(np.cos(theta_ideal), np.sin(theta_ideal), color="gray", linestyle="--", alpha=0.5, label="Unit-circle reference")
    ax3.set_title("Geometric Rotation in the Invariant Plane")
    ax3.set_xlabel(r"$H_{\mathrm{Bad}}$ amplitude ($b_k$)")
    ax3.set_ylabel(r"$H_{\mathrm{Good}}$ amplitude ($a_k$)")
    ax3.set_aspect("equal")
    ax3.grid(True)
    ax3.legend()

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(p_vals_nesting, p1_curve, color="blue", linestyle="--", label=f"Level 1 (k={nesting_k1})")
    ax4.plot(
        p_vals_nesting,
        p2_curve,
        color="red",
        linestyle="-",
        label=f"Nested composition (k1={nesting_k1}, k2={nesting_k2})",
    )
    ax4.set_title("Nested Amplitude-Amplification Composition")
    ax4.set_xlabel("Initial success probability p = M/N")
    ax4.set_ylabel("Success probability P(p)")
    ax4.grid(True)
    ax4.legend()

    ax5 = plt.subplot(2, 3, 5)
    x = np.arange(len(pur_orig))
    width = 0.35
    ax5.bar(x - width / 2, pur_orig, width, label="Original-register purity", color="lightgreen")
    ax5.bar(x + width / 2, pur_copy, width, label="Purity after naive copy", color="salmon")
    ax5.axhline(0.833, color="red", linestyle="--", label="UQCM reference (0.833)")
    ax5.set_title("No-Cloning Diagnostic: Reduced-State Purity")
    ax5.set_xlabel("Iterations (k)")
    ax5.set_ylabel("Purity Tr(rho^2)")
    ax5.set_ylim(0, 1.1)
    ax5.legend()

    ax6 = plt.subplot(2, 3, 6)
    contour = ax6.pcolormesh(heatmap_p, heatmap_k, success_heatmap, shading="auto", cmap="inferno")
    fig.colorbar(contour, ax=ax6, label="Success probability P_k")
    ax6.set_title("Grover Success Landscape over (p, k)")
    ax6.set_xlabel("Initial success probability p = M/N")
    ax6.set_ylabel("Iteration count k")

    plt.tight_layout()
    out = Path(save_plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved figure: {out}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_optional_qiskit_demo() -> None:
    """Optional 3-qubit circuit demonstration for H_Good state '101'."""
    try:
        from qiskit import ClassicalRegister, QuantumRegister, transpile
        from qiskit.visualization import plot_histogram
        from qiskit_aer import AerSimulator

        data = QuantumRegister(3, "data")
        anc = QuantumRegister(1, "anc")
        creg = ClassicalRegister(3, "c")
        qc = QuantumCircuit(data, anc, creg)

        def apply_phase_oracle_good_101(qcircuit: QuantumCircuit) -> None:
            qcircuit.x(data[1])
            qcircuit.mcp(np.pi, [data[0], data[1], data[2]], anc[0])
            qcircuit.x(data[1])

        def apply_diffusion_operator_3q(qcircuit: QuantumCircuit) -> None:
            qcircuit.h(data)
            qcircuit.x(data)
            qcircuit.h(data[2])
            qcircuit.ccx(data[0], data[1], data[2])
            qcircuit.h(data[2])
            qcircuit.x(data)
            qcircuit.h(data)

        qc.h(data)
        qc.x(anc[0])
        apply_phase_oracle_good_101(qc)
        apply_diffusion_operator_3q(qc)
        qc.measure(data, creg)

        backend = AerSimulator()
        compiled = transpile(qc, backend, optimization_level=1)
        shots = 4096
        result = backend.run(compiled, shots=shots).result()
        counts = result.get_counts()

        good_state = "101"
        good_probability = counts.get(good_state, 0) / shots
        print("\n" + "=" * 70)
        print("Qiskit circuit demonstration (H_Good = 101)")
        print("Counts:", counts)
        print(f"p(H_Good = {good_state}) = {good_probability:.4f}")

        plot_histogram(counts, title="Grover search (3-bit, 1 iteration, H_Good = 101)")
        plt.tight_layout()
        plt.savefig("grover_circuit_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved figure: grover_circuit_histogram.png")
        print("=" * 70)
    except Exception as exc:
        print("\nQiskit demonstration skipped (runtime failed):", exc)


def run_full_analysis(
    n_qubits: int = 6,
    good_indices="10,25",
    max_k: Optional[int] = None,
    k_multiplier: float = 3.0,
    nesting_k1: int = 3,
    nesting_k2: int = 3,
    nesting_p_max: float = 0.15,
    heatmap_k_max: int = 50,
    heatmap_p_max: float = 0.5,
    heatmap_resolution: int = 500,
    test_cases=DEFAULT_TEST_CASES,
    skip_test_cases: bool = False,
    no_math_overview: bool = False,
    run_qiskit_demo: bool = False,
    save_plot: str = DEFAULT_PLOT_PATH,
    show_plot: bool = False,
) -> None:
    if heatmap_k_max <= 0:
        raise ValueError("heatmap_k_max must be positive")
    if not 0.0 < heatmap_p_max <= 1.0:
        raise ValueError("heatmap_p_max must lie in (0, 1]")
    if heatmap_resolution <= 1:
        raise ValueError("heatmap_resolution must exceed 1")
    if not 0.0 < nesting_p_max <= 1.0:
        raise ValueError("nesting_p_max must lie in (0, 1]")

    normalized_good_indices = _normalize_good_indices(good_indices)
    analysis = GroverGeometricAnalysis(n_qubits=n_qubits, good_indices=normalized_good_indices)
    resolved_max_k = _resolve_max_k(analysis, max_k, k_multiplier)

    if not no_math_overview:
        print_math_overview()

    if not skip_test_cases:
        run_case_table(_normalize_test_cases(test_cases))

    print("\n" + "=" * 70)
    print("GROVER GEOMETRIC ANALYSIS")
    print("=" * 70)
    print(f"n = {analysis.n}, N = {analysis.N}, M = {analysis.M}, p = {analysis.p:.6f}")
    print(f"theta0 = {analysis.theta0:.6f} rad, theta = {analysis.theta:.6f} rad")
    print(f"Near-optimal iteration count k* = {analysis.k_optimal}")
    print(f"Analysis range: k = 0, ..., {resolved_max_k}")

    a_vals, b_vals, p_total, probs = analysis.analyze_geometry(resolved_max_k)
    pur_orig, pur_copy = analysis.simulate_cloning_barrier(resolved_max_k)
    p_vals_nesting, p1_curve, p2_curve, gate_depths = analysis.recursive_nesting_analysis(
        k1=nesting_k1,
        k2=nesting_k2,
        max_p=nesting_p_max,
    )
    heatmap_k, heatmap_p, success_heatmap = analysis.sensitivity_heatmap(
        k_max=heatmap_k_max,
        p_max=heatmap_p_max,
        resolution=heatmap_resolution,
    )

    mean_sq_dev = np.mean((1.0 - p_total) ** 2)
    print(f"Mean squared deviation from invariant subspace: {mean_sq_dev:.2e}")
    if mean_sq_dev < 1e-15:
        print("Invariant-subspace conservation confirmed within double precision.")

    print(
        "Gate-depth comparison for matched amplification: "
        f"standard={gate_depths['std_depth']} gates, "
        f"nested={gate_depths['nested_depth']} gates."
    )

    render_analysis_figure(
        analysis,
        resolved_max_k,
        a_vals,
        b_vals,
        p_total,
        probs,
        pur_orig,
        pur_copy,
        p_vals_nesting,
        p1_curve,
        p2_curve,
        heatmap_k,
        heatmap_p,
        success_heatmap,
        nesting_k1=nesting_k1,
        nesting_k2=nesting_k2,
        save_plot_path=save_plot,
        show_plot=bool(show_plot),
    )

    if run_qiskit_demo:
        run_optional_qiskit_demo()


def _interactive_rerun_prompt() -> None:
    if not sys.stdin.isatty():
        return

    print("\n" + "=" * 70)
    print("INTERACTIVE RE-RUN MODE")
    print("=" * 70)
    print("Press Enter to finish, or enter custom key=value pairs to rerun.")
    print("Example: n_qubits=5, good_indices=[3, 7], max_k=4")
    print("Example: test_cases=Sparse|1024|3;Dense|1024|64, skip_test_cases=False")

    try:
        raw = input("Custom parameters: ").strip()
    except EOFError:
        print("\nInteractive mode closed.")
        return

    if not raw:
        print("Interactive mode finished.")
        return
    if "=" not in raw:
        print("No key=value parameters detected. Interactive mode finished.")
        return

    try:
        kwargs = _parse_kwargs_text(raw)
    except Exception as exc:
        print(f"Could not parse custom parameters: {exc}")
        print("Interactive mode finished without rerun.")
        return

    allowed = {
        "n_qubits",
        "good_indices",
        "max_k",
        "k_multiplier",
        "nesting_k1",
        "nesting_k2",
        "nesting_p_max",
        "heatmap_k_max",
        "heatmap_p_max",
        "heatmap_resolution",
        "test_cases",
        "skip_test_cases",
        "no_math_overview",
        "run_qiskit_demo",
        "save_plot",
        "show_plot",
    }
    unknown = set(kwargs) - allowed
    if unknown:
        print(f"Unknown argument(s): {', '.join(sorted(unknown))}")
        print("Interactive mode finished without rerun.")
        return

    print(f"\nRe-running with parameters: {kwargs}")
    run_full_analysis(**kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    run_full_analysis(
        n_qubits=args.n_qubits,
        good_indices=args.good_indices,
        max_k=args.max_k,
        k_multiplier=args.k_multiplier,
        nesting_k1=args.nesting_k1,
        nesting_k2=args.nesting_k2,
        nesting_p_max=args.nesting_p_max,
        heatmap_k_max=args.heatmap_k_max,
        heatmap_p_max=args.heatmap_p_max,
        heatmap_resolution=args.heatmap_resolution,
        test_cases=args.test_cases,
        skip_test_cases=args.skip_test_cases,
        no_math_overview=args.no_math_overview,
        run_qiskit_demo=args.run_qiskit_demo,
        save_plot=args.save_plot,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="grover")
    cli_kwargs = _parse_command_line(sys.argv[1:])
    if cli_kwargs is None:
        main()
    elif cli_kwargs:
        allowed = {
            "n_qubits",
            "good_indices",
            "max_k",
            "k_multiplier",
            "nesting_k1",
            "nesting_k2",
            "nesting_p_max",
            "heatmap_k_max",
            "heatmap_p_max",
            "heatmap_resolution",
            "test_cases",
            "skip_test_cases",
            "no_math_overview",
            "run_qiskit_demo",
            "save_plot",
            "show_plot",
        }
        unknown = set(cli_kwargs) - allowed
        if unknown:
            raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
        run_full_analysis(**cli_kwargs)
    else:
        run_full_analysis()
        _interactive_rerun_prompt()

