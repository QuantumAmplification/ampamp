"""
VTAA Transpilation Master Suite — Scenarios A through O
========================================================
15 irreducible physical realities of Variable-Time Amplitude Amplification.

Part 1 (A-F): Algorithmic Geometry & Subspace Physics
Part 2 (G-I): Fault-Tolerant (FTQC) Bottlenecks
Part 3 (J-O): System Architecture & Hardware Realities

Imports the experiment functions from:
    5_Variable_Time_Amplitude_Amplification.py
"""

from __future__ import annotations
import math, os, sys, time, importlib.util
import ast
import inspect
import traceback
import numpy as np

_BOOTSTRAP_HERE = os.path.dirname(os.path.abspath(__file__))
if _BOOTSTRAP_HERE not in sys.path:
    sys.path.insert(0, _BOOTSTRAP_HERE)

from aer_publishability import (
    parse_publishability_cli,
    prepare_backend_validation_artifacts,
    render_backend_validation_summary,
    run_cli_scenario,
    save_figure_with_metadata,
    wrap_scenarios,
)
from transpile_path_utils import ensure_directory_on_syspath, resolve_project_file

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, fn):
        self.terminal = sys.stdout
        self.log = open(fn, "w", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m); self.log.write(m)
    def flush(self):
        self.terminal.flush(); self.log.flush()
    def close(self):
        self.log.close()

# ---------------------------------------------------------------------------
# Import VTAA module
# ---------------------------------------------------------------------------
_HERE = os.fspath(ensure_directory_on_syspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
_VTAA_PATH = os.fspath(
    resolve_project_file(__file__, "5_Variable_Time_Amplitude_Amplification.py", preferred_dirs=("Theory Algorithms",))
)


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _import_vtaa():
    spec = importlib.util.spec_from_file_location("vtaa_module", _VTAA_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vtaa_module"] = mod
    spec.loader.exec_module(mod)
    return mod

vtaa = _import_vtaa()

SEP = "=" * 70
BASIS_CLIFFORD_T = ["cx", "h", "s", "sdg", "t", "tdg", "x", "z"]
BASIS_NISQ = ["cx", "id", "rz", "sx", "x"]


def _parse_cli_value(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return raw


def _parse_kwargs_text(raw):
    kwargs = {}
    text = raw.strip()
    if not text:
        return kwargs
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected key=value pair, got '{item}'")
        key, value = item.split("=", 1)
        kwargs[key.strip()] = _parse_cli_value(value.strip())
    return kwargs


def _format_signature_help(fn):
    sig = inspect.signature(fn)
    parts = []
    for name, param in sig.parameters.items():
        if param.default is inspect._empty:
            parts.append(name)
        else:
            parts.append(f"{name}={param.default!r}")
    return ", ".join(parts) if parts else "(no parameters)"


def run_interactive_scenario_repl(scenarios, *, sep):
    if not sys.stdin.isatty():
        return
    scenario_pairs = list(scenarios)
    scenario_map = {label.upper(): fn for label, fn in scenario_pairs}
    print(f"\n{sep}")
    print("INTERACTIVE SCENARIO RE-RUN MODE")
    print(sep)
    print("You can now rerun any scenario with custom inputs.")
    print(f"Available labels: {', '.join(label for label, _ in scenario_pairs)}")
    print("Enter a label like A or O, or press Enter to finish.")
    while True:
        try:
            choice = input("\nScenario label to rerun: ").strip().upper()
        except EOFError:
            print("\nInteractive mode closed.")
            return
        if not choice:
            print("Interactive mode finished.")
            return
        if choice not in scenario_map:
            print(f"Unknown scenario '{choice}'. Available: {', '.join(scenario_map)}")
            continue
        fn = scenario_map[choice]
        print(f"Selected {choice}: {fn.__name__}")
        print(f"Parameters: {_format_signature_help(fn)}")
        print("Enter overrides as comma-separated key=value pairs.")
        print("Example: n=16, k_max=20")
        print("Press Enter to use defaults.")
        try:
            raw_kwargs = input("Custom parameters: ")
            kwargs = _parse_kwargs_text(raw_kwargs)
            print(f"\nRe-running {choice} with parameters: {kwargs if kwargs else 'defaults'}")
            fn(**kwargs)
        except Exception as exc:
            print(f"\nScenario {choice} failed with custom parameters.")
            print(f"Error: {exc}")
            traceback.print_exc()

# ╔═════════════════════════════════════════════════════════════════════╗
# ║  PART 1: ALGORITHMIC GEOMETRY & SUBSPACE PHYSICS (A–F)            ║
# ╚═════════════════════════════════════════════════════════════════════╝

def run_scenario_a(n: int = 8, guessed_m: int = 1, actual_m: int = 5, k_scan_factor: float = 1.5):
    """A. OVER-ROTATION OVERHEAD ANALYSIS — gate depth expended beyond the optimal peak."""
    print(f"\n{SEP}")
    print("SCENARIO A: OVER-ROTATION OVERHEAD ANALYSIS")
    print(SEP)
    print(f"Guessing M={guessed_m} when true M={actual_m} in n={n} (N={2**n}). Quantifying wasted depth.\n")
    res = vtaa.experiment_souffle_catastrophe(n=n, guessed_m=guessed_m, actual_m=actual_m, k_scan_factor=k_scan_factor)

    # Find the exact optimal k for the actual value of M.
    theta_actual = float(np.arcsin(np.sqrt(res.actual_m / res.N)))
    k_true_opt = max(0, int(np.floor(np.pi / (4.0 * theta_actual) - 0.5)))
    p_true_peak = float(np.sin((2 * k_true_opt + 1) * theta_actual) ** 2)

    # Gate cost model: each Grover iterate ~ 2*n CX gates on linear topology
    cx_per_iter = 2 * res.n
    wasted_iters = max(0, res.k_opt_guess - k_true_opt)
    wasted_cx = wasted_iters * cx_per_iter
    wasted_ns = wasted_cx * 100  # 100 ns per CX

    print(f"{'Metric':<40} | {'Value'}")
    print("-" * 60)
    print(f"{'N = 2^n':<40} | {res.N}")
    print(f"{'Guessed M':<40} | {res.guessed_m}")
    print(f"{'Actual M':<40} | {res.actual_m}")
    print(f"{'k* (guessed, from M=1)':<40} | {res.k_opt_guess}")
    print(f"{'k* (true, from M=5)':<40} | {k_true_opt}")
    print(f"{'P at guessed k* (actual circuit)':<40} | {res.prob_at_halt:.6f}")
    print(f"{'P at true k* (actual circuit)':<40} | {p_true_peak:.6f}")
    print(f"{'Peak P before guessed halt':<40} | {res.peak_prob_before_halt:.6f}")
    print(f"{'Reduction from peak':<40} | {res.collapse_from_peak:.6f}")
    print(f"{'Wasted iterations past true peak':<40} | {wasted_iters}")
    print(f"{'Wasted CX gates (est.)':<40} | {wasted_cx}")
    print(f"{'Wasted limit-clock time (est.)':<40} | {wasted_ns} ns")

    print(f"\n-> The algorithm executed {wasted_iters} iterations beyond the true peak.")
    print(f"   Success probability decreased from {res.peak_prob_before_halt:.4f} to {res.prob_at_halt:.4f}.")
    print(f"   This incurred {wasted_cx} CX gates ({wasted_ns} ns) with no net gain.")
    print("-> CONCLUSION: Without knowing M, over-rotation consumes physical")
    print("   coherence time. VTAA is motivated in part by reducing this failure mode.")


def run_scenario_b(
    ideal_n: int = 8,
    ideal_k_max: int = 20,
    noisy_n: int = 4,
    noisy_k_max: int = 12,
    phase_damp_1q: float = 0.03,
    phase_damp_2q: float = 0.10,
):
    """B. THE GEOMETRIC PHASE STAIRCASE DISTORTION — noise warps angular progression."""
    print(f"\n{SEP}")
    print("SCENARIO B: THE GEOMETRIC PHASE STAIRCASE DISTORTION")
    print(SEP)
    print("Ideal: theta_k = (2k+1)*theta_0. Noise distorts the linear angular staircase.\n")

    # Ideal staircase first
    res_ideal = vtaa.experiment_geometric_phase_staircase(n=ideal_n, k_max=ideal_k_max)

    # Now simulate noisy staircase via open-system (dephasing destroys phase linearity)
    try:
        res_noisy = vtaa.experiment_open_system_trajectory(
            n=noisy_n, k_max=noisy_k_max, phase_damp_1q=phase_damp_1q, phase_damp_2q=phase_damp_2q
        )
        has_noisy = True
    except Exception:
        has_noisy = False

    print(f"Ideal staircase (n={res_ideal.n}, k_max={res_ideal.k_max}):")
    print(f"  theta_0 = {res_ideal.theta_0:.8f} rad")
    print(f"  Max angular error (ideal vs theory): {res_ideal.max_abs_error:.2e} rad")
    print(f"  Staircase linearity: {'PERFECT' if res_ideal.max_abs_error < 1e-10 else 'BROKEN'}")

    if has_noisy:
        # The noisy trajectory spirals inward; extract effective angles
        noisy_angles = np.arctan2(res_noisy.noisy_x, res_noisy.noisy_z)
        ideal_angles = np.arctan2(res_noisy.ideal_x, res_noisy.ideal_z)
        angle_drift = np.abs(noisy_angles - ideal_angles)
        purity_final = float(res_noisy.noisy_purity[-1])

        print(f"\nNoisy staircase (n={res_noisy.n}, dephasing 1q={res_noisy.phase_damp_1q}, 2q={res_noisy.phase_damp_2q}):")
        print(f"  Max angular drift from ideal: {float(np.max(angle_drift)):.4f} rad")
        print(f"  Final purity: {purity_final:.4f} (1.0=pure, 1/N={1/res_noisy.N:.4f}=maximally mixed)")
        print(f"  Purity at k=0: {float(res_noisy.noisy_purity[0]):.4f}")

    print("\n-> CONCLUSION: The ideal phase staircase is mathematically perfect. Under")
    print("   hardware dephasing, the phase angles drift and the amplitude decays,")
    print("   causing the algorithm to miss its probability peak.")


def run_scenario_c(n: int = 10, k_max: int = 25, rank_threshold: float = 1e-12, noisy_perturbation: float = 0.001):
    """C. THE SVD RANK EXPLOSION — noise shatters the 2D invariant plane."""
    print(f"\n{SEP}")
    print("SCENARIO C: THE SVD RANK EXPLOSION")
    print(SEP)
    print("Ideal AA trajectory is confined to a rank-2 subspace. Under noise, rank explodes.\n")

    res = vtaa.experiment_2d_subspace_extractor(n=n, k_max=k_max, rank_threshold=rank_threshold)

    print(f"History matrix: {res.history_shape[0]} x {res.history_shape[1]} (N x (k_max+1))")
    print(f"Empirical rank (threshold {res.rank_threshold:.0e}): {res.empirical_rank}")
    print(f"sigma_3 / sigma_1 ratio: {res.sigma3_to_sigma1:.2e}")
    print(f"Float64 SVD floor: {res.float64_svd_floor:.2e}")
    print(f"\nTop 6 singular values:")
    for i, sv in enumerate(res.singular_values[:6]):
        marker = " ← rank boundary" if i == 2 and sv < res.rank_threshold else ""
        print(f"  sigma_{i+1} = {sv:.6e}{marker}")

    # Now simulate noisy history matrix (add depolarizing perturbation)
    np.random.seed(42)
    N = 2 ** n
    history_ideal, _, good_idx, _ = vtaa._standard_grover_state_history(n=n, k_max=k_max)
    # Add realistic noise: each state gets ~0.1% random perturbation
    history_noisy = [s + noisy_perturbation * np.random.randn(N) for s in history_ideal]
    history_noisy = [s / np.linalg.norm(s) for s in history_noisy]
    sv_noisy = np.linalg.svd(np.column_stack(history_noisy), compute_uv=False)
    rank_noisy = int(np.count_nonzero(sv_noisy > 1e-12))

    print(f"\nNoisy simulation ({100*noisy_perturbation:.1f}% depolarizing perturbation per state):")
    print(f"  Empirical rank: {rank_noisy}  (was {res.empirical_rank} ideal)")
    print(f"  sigma_3/sigma_1: {sv_noisy[2]/sv_noisy[0]:.4e}")

    print(f"\n-> CONCLUSION: Ideal AA is mathematically confined to rank-2 (|Good>, |Bad>).")
    print(f"   With just 0.1% noise, rank explodes to {rank_noisy}, proving the state bleeds")
    print(f"   into the full 2^n Hilbert space. VTAA inherits this fragility.")


def run_scenario_d(
    n: int = 6,
    k_max: int = 20,
    eps_oracle_deg: float = -5.0,
    eps_diff_deg: float = 2.0,
    crosstalk_oracle_deg: float = 0.6,
    local_z_detune_deg: float = 0.6,
):
    """D. THE INVARIANT PLANE BREACH (Crosstalk) — analog errors destroy subspace."""
    print(f"\n{SEP}")
    print("SCENARIO D: THE INVARIANT PLANE BREACH (Crosstalk)")
    print(SEP)
    print("Phase mismatch alone stays rank-2. Crosstalk + detuning shatters it.\n")

    res = vtaa.experiment_phase_mismatch_leakage(
        n=n,
        k_max=k_max,
        eps_oracle_deg=eps_oracle_deg,
        eps_diff_deg=eps_diff_deg,
        crosstalk_oracle_deg=crosstalk_oracle_deg,
        local_z_detune_deg=local_z_detune_deg,
    )

    print(f"{'Model':<45} | {'Max Rank':<10} | {'Rank-2 preserved?'}")
    print("-" * 75)
    print(f"{'Ideal (180°, 180°)':<45} | {int(np.max(res.rank_ideal)):<10} | YES")
    print(f"{'Mismatch only (175°, 182°)':<45} | {int(np.max(res.rank_mismatch_only)):<10} | {'YES' if res.mismatch_only_rank2_all else 'NO'}")
    print(f"{'Mismatch + crosstalk + detuning':<45} | {int(np.max(res.rank_leaky)):<10} | {'NO' if res.leaky_exceeds_rank2 else 'YES'}")

    print(f"\nSingular value spectra (final iteration, k={res.k_max}):")
    for label, sv in [("Ideal", res.final_sv_ideal), ("Mismatch", res.final_sv_mismatch_only), ("Leaky", res.final_sv_leaky)]:
        top3 = sv[:min(3, len(sv))]
        print(f"  {label}: " + ", ".join(f"{v:.4e}" for v in top3))

    print(f"\n-> Mismatch-only rank-2 preserved: {res.mismatch_only_rank2_all}")
    print(f"   Leaky model exceeds rank-2: {res.leaky_exceeds_rank2}")
    print("-> CONCLUSION: Pure phase mismatch generalizes the AA rotation but remains in")
    print("   the 2D plane. Analog crosstalk and local Z-detuning break rank-2 confinement,")
    print("   showing that realistic hardware errors, not only systematic offsets, dominate this limitation.")


def run_scenario_e(n: int = 4, k_max: int = 12, phase_damp_1q: float = 0.02, phase_damp_2q: float = 0.08):
    """E. DEPHASING-INDUCED TRAJECTORY CONTRACTION — T2 decay drives the state toward a mixed limit."""
    print(f"\n{SEP}")
    print("SCENARIO E: DEPHASING-INDUCED TRAJECTORY CONTRACTION (Open-System Decay)")
    print(SEP)
    print("T2 phase damping contracts the Bloch trajectory toward the origin.\n")

    try:
        res = vtaa.experiment_open_system_trajectory(n=n, k_max=k_max, phase_damp_1q=phase_damp_1q, phase_damp_2q=phase_damp_2q)
    except Exception as exc:
        print(f"[Scenario E requires qiskit-aer. Skipped: {exc}]")
        return

    k_vals = np.arange(res.k_max + 1)
    p_ideal = (res.ideal_x ** 2 + res.ideal_z ** 2)
    p_noisy = (res.noisy_x ** 2 + res.noisy_z ** 2)
    radius_decay = np.sqrt(p_noisy) / np.maximum(np.sqrt(p_ideal), 1e-15)

    print(f"{'k':<5} | {'Purity':<10} | {'Trace Dist to Plane':<22} | {'Bloch Radius Ratio'}")
    print("-" * 60)
    sample_points = sorted(set([0, k_max // 4, k_max // 2, (3 * k_max) // 4, k_max]))
    for k in sample_points:
        if k <= res.k_max:
            print(f"{k:<5} | {res.noisy_purity[k]:<10.4f} | {res.trace_distance_to_plane[k]:<22.4f} | {radius_decay[k]:.4f}")

    print(f"\n-> Purity decays from {res.noisy_purity[0]:.4f} to {res.noisy_purity[-1]:.4f}")
    print(f"   Trace distance to ideal 2D plane grows to {res.trace_distance_to_plane[-1]:.4f}")
    print(f"   Mixed-state limit: 1/N = {1/res.N:.4f}")
    print("-> CONCLUSION: T2 dephasing causes the Bloch vector to contract toward the origin")
    print("   rather than reaching the target pole. The probability peak is reduced before arrival.")
    print("   VTAA, with its multi-stage structure, accumulates this decay at every stage.")


def run_scenario_f(n: int = 8, m_good: int = 3, coarse_grid: int = 48, dac_bits_sweep=(6, 8, 10, 12, 14, 16)):
    """F. THE EXACT-AA DAC LIMITATION — finite DAC resolution quantizes fractional phases."""
    print(f"\n{SEP}")
    print("SCENARIO F: THE EXACT-AA RESOLUTION LIMIT (The DAC Limitation)")
    print(SEP)
    print("Exact-AA's fractional final phase must be physically synthesized. DAC bits limit precision.\n")

    res = vtaa.experiment_exact_amplitude_amplification(n=n, m_good=m_good, coarse_grid=coarse_grid)

    exact_alpha_deg = float(np.degrees(res.exact_oracle_phase))
    exact_beta_deg = float(np.degrees(res.exact_diffusion_phase))

    print(f"{'Metric':<40} | {'Value'}")
    print("-" * 60)
    print(f"{'n (qubits)':<40} | {res.n}")
    print(f"{'M (marked states)':<40} | {res.m_good}")
    print(f"{'Standard AA peak k*':<40} | {res.k_peak_standard}")
    print(f"{'Standard AA peak probability':<40} | {res.standard_probs[res.k_peak_standard]:.8f}")
    print(f"{'Exact-AA oracle phase (deg)':<40} | {exact_alpha_deg:.4f}")
    print(f"{'Exact-AA diffusion phase (deg)':<40} | {exact_beta_deg:.4f}")
    print(f"{'Exact-AA probability':<40} | {res.exact_prob:.10f}")
    print(f"{'Gap from unity':<40} | {1.0 - res.exact_prob:.2e}")

    # DAC quantization sweep
    print(f"\nDAC Resolution Impact:")
    print(f"{'DAC Bits':<10} | {'Resolution (deg)':<18} | {'Quantized alpha':<18} | {'Quantized beta':<18} | {'P_success':<12} | {'|1-P|'}")
    print("-" * 95)
    for dac_bits in dac_bits_sweep:
        step_deg = 360.0 / (2 ** dac_bits)
        q_alpha = round(exact_alpha_deg / step_deg) * step_deg
        q_beta = round(exact_beta_deg / step_deg) * step_deg
        # Recompute probability with quantized phases
        N = res.N
        good_idx = np.arange(res.m_good)
        proj_good = np.zeros((N, N), dtype=complex)
        proj_good[good_idx, good_idx] = 1.0
        ket_s = np.ones(N, dtype=complex) / np.sqrt(N)
        proj_s = np.outer(ket_s, ket_s.conj())
        oracle_pi = np.eye(N, dtype=complex) - 2.0 * proj_good
        diff_pi = 2.0 * proj_s - np.eye(N, dtype=complex)
        psi = ket_s.copy()
        for _ in range(res.k_base):
            psi = diff_pi @ (oracle_pi @ psi)
        q_a_rad = np.radians(q_alpha)
        q_b_rad = np.radians(q_beta)
        oracle_q = np.eye(N, dtype=complex) + (np.exp(1j * q_a_rad) - 1.0) * proj_good
        diff_q = np.eye(N, dtype=complex) + (np.exp(1j * q_b_rad) - 1.0) * proj_s
        psi_final = diff_q @ (oracle_q @ psi)
        p_q = float(np.sum(np.abs(psi_final[good_idx]) ** 2))
        print(f"{dac_bits:<10} | {step_deg:<18.6f} | {q_alpha:<18.4f} | {q_beta:<18.4f} | {p_q:<12.8f} | {abs(1-p_q):.2e}")

    print("\n-> CONCLUSION: Exact-AA achieves P=1.0 only with infinite phase resolution.")
    print("   Physical DAC quantization (typically 8-12 bits) introduces an irreducible gap.")
    print("   At 8-bit DAC, the gap from unity can be ~1e-3, comparable to noise floors.")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  PART 2: FAULT-TOLERANT (FTQC) BOTTLENECKS (G–I)                 ║
# ╚═════════════════════════════════════════════════════════════════════╝

def run_scenario_g(n_min: int = 5, n_max: int = 30, noancilla_max: int = 8, optimization_level: int = 1):
    """G. DIFFUSION SYNTHESIS MEMORY LIMIT — T-count growth without auxiliary ancillas."""
    print(f"\n{SEP}")
    print("SCENARIO G: DIFFUSION SYNTHESIS MEMORY LIMIT (FTQC T-Gate Scaling)")
    print(SEP)
    print("Comparing v-chain (with auxiliary ancillas) vs no-ancilla MCX decomposition.\n")

    res = vtaa.experiment_ftqc_diffusion_scaling(
        n_min=n_min, n_max=n_max, noancilla_max=noancilla_max, optimization_level=optimization_level
    )

    print(f"{'n':<6} | {'T-count (v-chain)':<20} | {'Dirty Ancillas':<16} | {'T-count (no ancilla)'}")
    print("-" * 65)
    for i, n in enumerate(res.n_values_vchain):
        na_str = ""
        for j, nj in enumerate(res.n_values_noancilla):
            if nj == n:
                na_str = str(int(res.t_counts_noancilla[j]))
                break
        print(f"{int(n):<6} | {int(res.t_counts_vchain[i]):<20} | {int(res.ancilla_counts_vchain[i]):<16} | {na_str if na_str else 'N/A (too slow)'}")
        if int(n) > 15 and int(n) % 5 != 0:
            continue  # skip non-round values for brevity

    if len(res.n_values_noancilla) >= 2:
        ratio = float(res.t_counts_noancilla[-1]) / max(1, float(res.t_counts_vchain[len(res.n_values_noancilla)-1]))
        print(f"\n-> At n={int(res.n_values_noancilla[-1])}: no-ancilla T-count is {ratio:.1f}x the v-chain T-count")
    print("-> CONCLUSION: Without auxiliary ancillas, the T-count grows super-linearly,")
    print("   creating a memory limit. VTAA's diffusion operator at each stage inherits")
    print("   this cost. Memory-constrained VTAA is T-gate limited.")


def run_scenario_h(n: int = 8, m_good: int = 3):
    """H. THE EXACT-AA SYNTHESIS OVERHEAD — compiling fractional phase to Clifford+T."""
    print(f"\n{SEP}")
    print("SCENARIO H: THE EXACT-AA SYNTHESIS OVERHEAD (Clifford+T Compilation)")
    print(SEP)
    print("The fractional final step requires non-Clifford synthesis → T-gate explosion.\n")

    try:
        from qiskit import QuantumCircuit, transpile as qk_transpile
    except ImportError:
        print("[Scenario H requires Qiskit. Skipped.]")
        return

    res = vtaa.experiment_exact_amplitude_amplification(n=n, m_good=m_good)
    n = res.n

    # Standard Grover step (pi phases) → compiles to pure Clifford
    qc_std = QuantumCircuit(n)
    qc_std.h(range(n))
    good_idx_list = list(range(res.m_good))
    for gi in good_idx_list:
        bits = format(gi, f'0{n}b')[::-1]
        for q, b in enumerate(bits):
            if b == '0': qc_std.x(q)
        qc_std.mcp(np.pi, list(range(n-1)), n-1)
        for q, b in enumerate(bits):
            if b == '0': qc_std.x(q)
    qc_std.h(range(n)); qc_std.x(range(n))
    qc_std.mcp(np.pi, list(range(n-1)), n-1)
    qc_std.x(range(n)); qc_std.h(range(n))

    tc_std = qk_transpile(qc_std, basis_gates=BASIS_CLIFFORD_T, optimization_level=1, seed_transpiler=42)
    ops_std = tc_std.count_ops()
    t_std = int(ops_std.get("t", 0) + ops_std.get("tdg", 0))

    # Exact-AA fractional step (non-pi phases) → requires Solovay-Kitaev
    qc_exact = QuantumCircuit(n)
    alpha = res.exact_oracle_phase
    beta = res.exact_diffusion_phase
    for gi in good_idx_list:
        bits = format(gi, f'0{n}b')[::-1]
        for q, b in enumerate(bits):
            if b == '0': qc_exact.x(q)
        qc_exact.mcp(alpha, list(range(n-1)), n-1)
        for q, b in enumerate(bits):
            if b == '0': qc_exact.x(q)
    qc_exact.h(range(n)); qc_exact.x(range(n))
    qc_exact.mcp(beta, list(range(n-1)), n-1)
    qc_exact.x(range(n)); qc_exact.h(range(n))

    tc_exact = qk_transpile(qc_exact, basis_gates=BASIS_CLIFFORD_T, optimization_level=1, seed_transpiler=42)
    ops_exact = tc_exact.count_ops()
    t_exact = int(ops_exact.get("t", 0) + ops_exact.get("tdg", 0))

    print(f"{'Metric':<40} | {'Standard (pi phases)':<22} | {'Exact-AA (fractional)'}")
    print("-" * 85)
    print(f"{'Total gates':<40} | {sum(ops_std.values()):<22} | {sum(ops_exact.values())}")
    print(f"{'T + Tdg gates':<40} | {t_std:<22} | {t_exact}")
    print(f"{'Circuit depth':<40} | {tc_std.depth():<22} | {tc_exact.depth()}")
    print(f"{'CX gates':<40} | {int(ops_std.get('cx', 0)):<22} | {int(ops_exact.get('cx', 0))}")

    if t_std > 0:
        ratio = t_exact / t_std
        print(f"\n-> Exact-AA T-gate tax: {ratio:.2f}x the standard step T-count")
    print("-> CONCLUSION: The single fractional final step of Exact-AA re-introduces")
    print("   the Solovay-Kitaev T-gate explosion. On FTQC hardware, this ONE step")
    print("   can dominate the total T-budget of an otherwise Clifford-efficient circuit.")


def run_scenario_i(p_s1: float = 0.20, p_fail_total: float = 0.70):
    """I. THE COHERENT FLAG SYNTHESIS OVERHEAD — T-gates for VTAA branching logic."""
    print(f"\n{SEP}")
    print("SCENARIO I: THE COHERENT FLAG SYNTHESIS OVERHEAD")
    print(SEP)
    print("VTAA coherent branching (continue/success/fail flags) vs standard AA diffusion.\n")

    try:
        from qiskit import QuantumCircuit, QuantumRegister, transpile as qk_transpile
    except ImportError:
        print("[Scenario I requires Qiskit. Skipped.]")
        return

    # Build VTAA-style flag circuit (2-stage model)
    res_synth = vtaa.experiment_vtaa_state_synthesis()

    # Reconstruct the circuit for compilation analysis
    stage_reg = QuantumRegister(1, "stage_j")
    flag_reg = QuantumRegister(2, "flag")
    data_reg = QuantumRegister(1, "data")
    qc_vtaa = QuantumCircuit(data_reg, flag_reg, stage_reg, name="vtaa_flags")
    qc_vtaa.ry(2.0 * np.arcsin(np.sqrt(p_s1)), flag_reg[0])
    qc_vtaa.x(flag_reg[0])
    qc_vtaa.cx(flag_reg[0], stage_reg[0])
    qc_vtaa.x(flag_reg[0])
    p_continue = 1.0 - p_s1
    p_fail_cond = p_fail_total / p_continue
    qc_vtaa.x(flag_reg[0])
    qc_vtaa.cry(2.0 * np.arcsin(np.sqrt(p_fail_cond)), flag_reg[0], flag_reg[1])
    qc_vtaa.x(flag_reg[0])
    qc_vtaa.x(flag_reg[1])
    qc_vtaa.ccx(stage_reg[0], flag_reg[1], flag_reg[0])
    qc_vtaa.x(flag_reg[1])

    tc_vtaa = qk_transpile(qc_vtaa, basis_gates=BASIS_CLIFFORD_T, optimization_level=1, seed_transpiler=42)
    ops_vtaa = tc_vtaa.count_ops()
    t_vtaa = int(ops_vtaa.get("t", 0) + ops_vtaa.get("tdg", 0))

    # Standard AA diffusion on same data qubit count (1 data + no flags)
    qc_std = QuantumCircuit(1)
    qc_std.h(0); qc_std.x(0); qc_std.z(0); qc_std.x(0); qc_std.h(0)
    tc_std = qk_transpile(qc_std, basis_gates=BASIS_CLIFFORD_T, optimization_level=1, seed_transpiler=42)
    ops_std = tc_std.count_ops()
    t_std = int(ops_std.get("t", 0) + ops_std.get("tdg", 0))

    print(f"VTAA State Synthesis Output:")
    print(f"  P(success) = {res_synth.success_probability:.4f}")
    print(f"  P(continue) = {res_synth.continue_probability:.4f}")
    print(f"  P(fail) = {res_synth.fail_probability:.4f}")

    print(f"\n{'Metric':<35} | {'VTAA (flag logic)':<20} | {'Standard AA'}")
    print("-" * 75)
    print(f"{'Total qubits':<35} | {qc_vtaa.num_qubits:<20} | {qc_std.num_qubits}")
    print(f"{'Total gates (Clifford+T)':<35} | {sum(ops_vtaa.values()):<20} | {sum(ops_std.values())}")
    print(f"{'T + Tdg gates':<35} | {t_vtaa:<20} | {t_std}")
    print(f"{'Circuit depth':<35} | {tc_vtaa.depth():<20} | {tc_std.depth()}")

    print(f"\n-> VTAA flag logic requires {t_vtaa} T-gates for ONE 2-stage synthesis.")
    print("   Each VTAA amplification round repeats this → T-cost grows linearly with stages.")
    print("-> CONCLUSION: Coherent branching is more expensive than simple diffusion.")
    print("   The T-gate overhead of maintaining quantum flags is the hidden VTAA overhead.")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  PART 3: SYSTEM ARCHITECTURE & HARDWARE REALITIES (J–O)          ║
# ╚═════════════════════════════════════════════════════════════════════╝

def run_scenario_j(p_s1: float = 0.20, p_fail: float = 0.70):
    """J. VTAA STATE-SYNTHESIS ROUTING PENALTY — SWAP cost on restricted topology."""
    print(f"\n{SEP}")
    print("SCENARIO J: THE VTAA STATE-SYNTHESIS ROUTING PENALTY")
    print(SEP)
    print("Transpiling VTAA flag logic onto Linear-1D to expose SWAP insertion.\n")

    try:
        from qiskit import QuantumCircuit, QuantumRegister, transpile as qk_transpile
        from qiskit.transpiler import CouplingMap
    except ImportError:
        print("[Scenario J requires Qiskit. Skipped.]")
        return

    # Build VTAA flag circuit
    stage_reg = QuantumRegister(1, "stage_j")
    flag_reg = QuantumRegister(2, "flag")
    data_reg = QuantumRegister(1, "data")
    qc = QuantumCircuit(data_reg, flag_reg, stage_reg)
    qc.ry(2.0 * np.arcsin(np.sqrt(p_s1)), flag_reg[0])
    qc.x(flag_reg[0]); qc.cx(flag_reg[0], stage_reg[0]); qc.x(flag_reg[0])
    p_fail_cond = p_fail / (1.0 - p_s1)
    qc.x(flag_reg[0]); qc.cry(2.0 * np.arcsin(np.sqrt(p_fail_cond)), flag_reg[0], flag_reg[1]); qc.x(flag_reg[0])
    qc.x(flag_reg[1]); qc.ccx(stage_reg[0], flag_reg[1], flag_reg[0]); qc.x(flag_reg[1])

    n_total = qc.num_qubits
    cmap = CouplingMap.from_line(n_total)

    # All-to-all (no routing)
    tc_free = qk_transpile(qc, basis_gates=BASIS_NISQ, optimization_level=3, seed_transpiler=42)
    # Linear constrained
    tc_linear = qk_transpile(qc, basis_gates=BASIS_NISQ, coupling_map=cmap, optimization_level=3, seed_transpiler=42)
    # SWAP-view
    tc_swap = qk_transpile(qc, basis_gates=BASIS_NISQ + ["swap"], coupling_map=cmap, optimization_level=3, seed_transpiler=42)

    ops_free = tc_free.count_ops()
    ops_lin = tc_linear.count_ops()
    ops_swap = tc_swap.count_ops()
    swaps = int(ops_swap.get("swap", 0))

    print(f"{'Metric':<35} | {'All-to-All':<15} | {'Linear-1D'}")
    print("-" * 65)
    print(f"{'Total gates':<35} | {sum(ops_free.values()):<15} | {sum(ops_lin.values())}")
    print(f"{'CX gates':<35} | {int(ops_free.get('cx',0)):<15} | {int(ops_lin.get('cx',0))}")
    print(f"{'Circuit depth':<35} | {tc_free.depth():<15} | {tc_linear.depth()}")
    print(f"{'SWAP gates inserted':<35} | {'0':<15} | {swaps}")

    routing_cx = int(ops_lin.get('cx', 0)) - int(ops_free.get('cx', 0))
    print(f"\n-> Routing CX overhead: {routing_cx} extra CX gates on Linear-1D")
    print(f"   {swaps} SWAPs inserted to connect flag/stage/data registers.")
    print("-> CONCLUSION: VTAA's multi-register architecture (data + flags + clock)")
    print("   incurs SWAP penalties when mapped to restricted topologies.")


def run_scenario_k(total_ps: float = 0.05, t1: float = 100.0, t2: float = 1000.0, t3: float = 10000.0, sample_pcts=(1, 10, 25, 50, 75, 90, 99)):
    """K. ASYMPTOTIC CROSSOVER ANALYSIS — VTAA versus worst-case standard AA cost."""
    print(f"\n{SEP}")
    print("SCENARIO K: ASYMPTOTIC CROSSOVER ANALYSIS")
    print(SEP)
    print("Sweeping early-success fraction to determine when VTAA becomes lower cost than worst-case AA.\n")

    res = vtaa.experiment_vtaa_cost_sweep(total_ps=total_ps, t1=t1, t2=t2, t3=t3)

    crossover_idx = None
    for i, (v, s) in enumerate(zip(res.vtaa_costs, res.standard_costs)):
        if v < s:
            crossover_idx = i
            break

    print(f"Total p_s = {res.total_ps:.2%}, stages: t1={t1}, t2={t2}, t3={t3}")
    print(f"\n{'Early-Success %':<18} | {'Standard AA Cost':<18} | {'VTAA Cost':<18} | {'Lower-cost method'}")
    print("-" * 75)
    for pct in sample_pcts:
        idx = int(pct * len(res.early_success_ratios) / 100)
        idx = min(idx, len(res.early_success_ratios) - 1)
        s = res.standard_costs[idx]; v = res.vtaa_costs[idx]
        lower_cost_method = "VTAA" if v < s else "Standard AA"
        print(f"{100*res.early_success_ratios[idx]:<18.1f} | {s:<18.1f} | {v:<18.1f} | {lower_cost_method}")

    if crossover_idx is not None:
        xover_pct = 100.0 * res.early_success_ratios[crossover_idx]
        print(f"\n-> Crossover point: VTAA becomes lower cost when more than {xover_pct:.1f}% of success terminates at stage 1.")
    else:
        print(f"\n-> No crossover found: standard AA is always cheaper for this configuration.")

    print("-> CONCLUSION: VTAA only outperforms standard AA when a substantial fraction of")
    print("   the success amplitude terminates at the least expensive stage. If most branches")
    print("   reach T_max, VTAA remains slower.")


def run_scenario_l(t_halt: int = 100, t_max: int = 10000, cx_time_ns: int = 100, T1_ns: int = 100_000, T2_ns: int = 80_000, n_qubits_idle: int = 4):
    """L. THE IDLE-QUBIT DECOHERENCE LIMIT — early-halting branches decay while waiting."""
    print(f"\n{SEP}")
    print("SCENARIO L: THE 'IDLE QUBIT' DECOHERENCE LIMITATION")
    print(SEP)
    print("Early-halting branches wait in superposition while T_max branch finishes.\n")

    # Model: branch halts at t=100, but T_max=10000.
    # During the wait, each idle qubit accumulates T1/T2 decay.
    idle_time = t_max - t_halt  # in gate units
    idle_ns = idle_time * cx_time_ns

    # Per-qubit survival probability
    p_survive_T1 = np.exp(-idle_ns / T1_ns)
    p_survive_T2 = np.exp(-idle_ns / T2_ns)
    # Multi-qubit survival (all qubits must survive)
    p_survive_all_T1 = p_survive_T1 ** n_qubits_idle
    p_coherence = p_survive_T2 ** n_qubits_idle

    print(f"{'Parameter':<45} | {'Value'}")
    print("-" * 65)
    print(f"{'Branch halt time (t_halt)':<45} | {t_halt} gate-steps")
    print(f"{'T_max':<45} | {t_max} gate-steps")
    print(f"{'Idle duration':<45} | {idle_time} gate-steps = {idle_ns/1e6:.1f} ms")
    print(f"{'Idle data qubits':<45} | {n_qubits_idle}")
    print(f"{'T1 (relaxation)':<45} | {T1_ns/1e3:.0f} μs")
    print(f"{'T2 (dephasing)':<45} | {T2_ns/1e3:.0f} μs")
    print(f"{'Per-qubit T1 survival':<45} | {p_survive_T1:.6f}")
    print(f"{'Per-qubit T2 survival':<45} | {p_survive_T2:.6f}")
    print(f"{'4-qubit T1 survival (all survive)':<45} | {p_survive_all_T1:.6f}")
    print(f"{'4-qubit coherence survival':<45} | {p_coherence:.6f}")

    fidelity_loss = (1.0 - p_coherence) * 100
    print(f"\n-> Idle-time coherence loss: {fidelity_loss:.1f}%")
    print(f"   The early-halting branch loses {fidelity_loss:.1f}% of its amplitude fidelity")
    print(f"   while waiting for the T_max branch to finish.")
    print("-> CONCLUSION: VTAA's coherent superposition requires all branches to wait for")
    print("   the slowest branch. Early branches therefore experience severe idle-qubit decay.")
    print("   This is the fundamental physical cost of maintaining coherent parallelism.")


def run_scenario_m(zz_rate_per_cx: float = 0.002, n_data_cx_gates: int = 500, n_flag_qubits: int = 3, cx_sweep=(100, 250, 500, 1000, 2000)):
    """M. SPECTATOR QUBIT CROSSTALK — flag qubits leak while data register executes."""
    print(f"\n{SEP}")
    print("SCENARIO M: SPECTATOR QUBIT CROSSTALK")
    print(SEP)
    print("While driving deep gates on data register, idle flag qubits accumulate ZZ errors.\n")

    # Model: typical cross-resonance ZZ crosstalk

    # Accumulated phase error per flag qubit
    phase_error_per_flag = zz_rate_per_cx * n_data_cx_gates  # radians
    # Multi-qubit fidelity hit
    flag_infidelity = 1.0 - np.cos(phase_error_per_flag / 2) ** 2
    total_flag_infidelity = 1.0 - (1.0 - flag_infidelity) ** n_flag_qubits

    print(f"{'Parameter':<45} | {'Value'}")
    print("-" * 65)
    print(f"{'ZZ crosstalk per CX gate (rad)':<45} | {zz_rate_per_cx:.4f}")
    print(f"{'CX gates on data register':<45} | {n_data_cx_gates}")
    print(f"{'Flag/clock qubits (spectators)':<45} | {n_flag_qubits}")
    print(f"{'Accumulated ZZ phase per flag (rad)':<45} | {phase_error_per_flag:.4f}")
    print(f"{'Per-flag infidelity':<45} | {flag_infidelity:.6f}")
    print(f"{'Total flag-register infidelity':<45} | {total_flag_infidelity:.6f}")

    # Sweep CX gate count
    print(f"\nSpectator leakage vs data-register depth:")
    print(f"{'Data CX Gates':<15} | {'Phase Error (rad)':<20} | {'Flag Infidelity'}")
    print("-" * 55)
    for cx in cx_sweep:
        ph = zz_rate_per_cx * cx
        inf = 1.0 - (np.cos(ph / 2) ** 2) ** n_flag_qubits
        print(f"{cx:<15} | {ph:<20.4f} | {inf:.6f}")

    print("\n-> CONCLUSION: VTAA's auxiliary flag and clock qubits act as spectators during")
    print("   data-register oracle execution. ZZ crosstalk accumulates phase errors on these")
    print("   idle qubits, corrupting the branching logic that VTAA fundamentally depends on.")


def run_scenario_n(mid_meas_latency_ns: int = 1000, cx_time_ns: int = 100):
    """N. RESTART AND COHERENT-AMPLIFICATION TRADEOFF — dynamic reset versus VTAA depth."""
    print(f"\n{SEP}")
    print("SCENARIO N: RESTART AND COHERENT-AMPLIFICATION TRADEOFF")
    print(SEP)
    print("Comparing: reset-and-restart (dynamic circuits) vs deep coherent VTAA.\n")

    branches = vtaa.example_instance()
    lab = vtaa.VariableTimeAmplitudeAmplificationLab(branches)
    report = lab.build_report()

    # Restart model: geometric waiting with mid-circuit measurement + reset
    # Each restart attempt costs E[T] gate-steps
    t_mean_gs = report.t_mean
    restart_circuit_ns = t_mean_gs * cx_time_ns
    restart_total_ns = report.expected_time_until_success_restart * cx_time_ns
    restart_attempts = 1.0 / report.p_success
    restart_meas_overhead = restart_attempts * mid_meas_latency_ns

    # VTAA model: one deep coherent run
    vtaa_circuit_ns = report.vtaa_asymptotic_bound * cx_time_ns
    # Add worst-case AA for comparison
    wc_aa_ns = report.expected_time_until_success_worst_case_aa * cx_time_ns

    print(f"{'Metric':<45} | {'Value'}")
    print("-" * 70)
    print(f"{'p_success':<45} | {report.p_success:.6f}")
    print(f"{'E[T] (mean branch time)':<45} | {report.t_mean:.2f} gate-steps")
    print(f"{'T_max (worst-case branch)':<45} | {report.t_max:.2f} gate-steps")
    print(f"{'Grover k*':<45} | {report.grover_k_opt}")

    print(f"\n{'Strategy':<35} | {'Cost (gate-steps)':<20} | {'Est. Time (ns)':<18} | {'+ Meas Latency'}")
    print("-" * 95)
    print(f"{'Restart until success':<35} | {report.expected_time_until_success_restart:<20.1f} | {restart_total_ns:<18.0f} | {restart_meas_overhead:.0f} ns")
    print(f"{'Worst-case standard AA':<35} | {report.expected_time_until_success_worst_case_aa:<20.1f} | {wc_aa_ns:<18.0f} | {'N/A'}")
    print(f"{'VTAA (asymptotic)':<35} | {report.vtaa_asymptotic_bound:<20.1f} | {vtaa_circuit_ns:<18.0f} | {'N/A'}")

    preferred_strategy = "Restart" if restart_total_ns + restart_meas_overhead < min(wc_aa_ns, vtaa_circuit_ns) else (
        "VTAA" if vtaa_circuit_ns < wc_aa_ns else "Worst-case AA")
    print(f"\n-> Preferred wall-clock strategy: {preferred_strategy}")
    print("-> CONCLUSION: Dynamic circuit reset-and-restart avoids the deep coherent")
    print("   unitary cost of VTAA, but pays a measurement latency overhead per attempt.")
    print("   For small p, restart is preferable. For structured problems with early branching,")
    print("   VTAA can be preferable.")


def run_scenario_o(n: int = 4, good_bits: str = "1111"):
    """O. UNIFIED HARDWARE PROFILING COMPARATIVE EVALUATION — execution-time comparison in nanoseconds."""
    print(f"\n{SEP}")
    print("SCENARIO O: UNIFIED HARDWARE PROFILING COMPARATIVE EVALUATION")
    print(SEP)

    try:
        sys.path.insert(0, _HERE)
        from quantum_profiler import HardwareProfiler
        from qiskit import QuantumCircuit, transpile as qk_transpile
        from qiskit.transpiler import CouplingMap
    except ImportError:
        print("[Scenario O requires quantum_profiler + Qiskit. Skipped.]")
        return

    def _build_grover_step(qc, n, good_bits):
        rev = good_bits[::-1]
        for q, b in enumerate(rev):
            if b == '0': qc.x(q)
        qc.mcp(np.pi, list(range(n-1)), n-1)
        for q, b in enumerate(rev):
            if b == '0': qc.x(q)
        qc.h(range(n)); qc.x(range(n))
        qc.mcp(np.pi, list(range(n-1)), n-1)
        qc.x(range(n)); qc.h(range(n))

    # 1. Standard AA: k_opt iterations
    theta0 = np.arcsin(np.sqrt(1/2**n))
    k_opt = max(0, int(np.floor(np.pi/(4*theta0) - 0.5)))
    qc_aa = QuantumCircuit(n); qc_aa.h(range(n))
    for _ in range(k_opt):
        _build_grover_step(qc_aa, n, good_bits)

    edges = [[i, i+1] for i in range(n-1)] + [[i+1, i] for i in range(n-1)]
    edges_wc = [[i, i+1] for i in range(n+1)] + [[i+1, i] for i in range(n+1)]

    # Pre-transpile with coupling map so profiler gets layout metadata
    qc_aa_t = qk_transpile(qc_aa, basis_gates=BASIS_NISQ, optimization_level=0, seed_transpiler=42)

    # 2. VTAA-style: k_opt iterations + flag overhead
    qc_wc = QuantumCircuit(n + 2)  # +2 for flags
    qc_wc.h(range(n))
    for _ in range(k_opt):
        rev = good_bits[::-1]
        for q, b in enumerate(rev):
            if b == '0': qc_wc.x(q)
        qc_wc.mcp(np.pi, list(range(n-1)), n-1)
        for q, b in enumerate(rev):
            if b == '0': qc_wc.x(q)
        qc_wc.h(range(n)); qc_wc.x(range(n))
        qc_wc.mcp(np.pi, list(range(n-1)), n-1)
        qc_wc.x(range(n)); qc_wc.h(range(n))
        qc_wc.ccx(0, 1, n)
    qc_wc_t = qk_transpile(qc_wc, basis_gates=BASIS_NISQ, optimization_level=0, seed_transpiler=42)
    prof_aa = HardwareProfiler(coupling_map_edges=edges, basis_gates=BASIS_NISQ, single_qubit_ns=20, two_qubit_ns=100)
    prof_wc = HardwareProfiler(coupling_map_edges=edges_wc, basis_gates=BASIS_NISQ, single_qubit_ns=20, two_qubit_ns=100)

    print(f"Profiling Standard AA (k={k_opt}, n={n}) ...")
    m_aa = prof_aa.profile_circuit(qc_aa_t)
    print(f"Profiling VTAA-style (k={k_opt}, n={n}+2 flag qubits) ...")
    m_wc = prof_wc.profile_circuit(qc_wc_t)

    print(f"\n{'Metric':<30} | {'Standard AA':<20} | {'VTAA-style (+flags)'}")
    print("-" * 75)
    print(f"{'Logical Depth':<30} | {m_aa['logical_depth']:<20} | {m_wc['logical_depth']}")
    print(f"{'Post-Routing SWAPs':<30} | {m_aa['routing_swaps']:<20} | {m_wc['routing_swaps']}")
    print(f"{'Final CNOT Count':<30} | {m_aa['final_cnots']:<20} | {m_wc['final_cnots']}")
    print(f"{'Total Execution Time (ns)':<30} | {m_aa['total_time_ns']:<20.1f} | {m_wc['total_time_ns']:.1f}")
    print(f"{'Unified Hardware Penalty':<30} | {m_aa['hardware_penalty_score']:<20.1f} | {m_wc['hardware_penalty_score']:.1f}")

    overhead = m_wc['total_time_ns'] / max(1, m_aa['total_time_ns'])
    print(f"\n-> VTAA overhead: {overhead:.2f}x the standard AA execution time")
    print("-> CONCLUSION: The flag/clock register overhead of VTAA adds physical execution")
    print("   time. VTAA only outperforms standard AA when the variable-time savings (shorter expected depth)")
    print("   exceed this fixed overhead — which requires strongly non-uniform branch times.")


def _save_vtaa_algorithm_figure(
    *,
    output_name="vtaa_failure_modes_profile.png",
):
    plt = _load_pyplot()
    souffle = vtaa.experiment_souffle_catastrophe(n=8, guessed_m=1, actual_m=5, k_scan_factor=1.5)
    staircase = vtaa.experiment_geometric_phase_staircase(n=8, k_max=12)
    ftqc = vtaa.experiment_ftqc_diffusion_scaling(n_min=5, n_max=10, noancilla_max=6, optimization_level=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("VTAA Failure Modes and Resource Profile", fontsize=14, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(souffle.k_values, souffle.actual_probs, linewidth=2.0, label="actual success", color="#1f77b4")
    ax1.plot(souffle.k_values, souffle.guess_probs_theoretical, linewidth=2.0, label="guessed-theory envelope", color="#d62728")
    ax1.axvline(souffle.k_opt_guess, linestyle="--", color="black", linewidth=1.2, label="guessed k*")
    ax1.set_title("Over-Rotation Overhead")
    ax1.set_ylabel("Success probability")
    ax1.legend(fontsize=8)

    ax2.plot(np.arange(len(staircase.angle_abs_error)), staircase.angle_abs_error, marker="o", linewidth=2.0, color="#2ca02c")
    ax2.set_title("Phase Staircase Error")
    ax2.set_ylabel("Absolute angle error (rad)")
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax3.plot(ftqc.n_values_vchain, ftqc.t_counts_vchain, marker="o", linewidth=2.0, label="v-chain diffusion", color="#9467bd")
    ax3.plot(ftqc.n_values_noancilla, ftqc.t_counts_noancilla, marker="s", linewidth=2.0, label="no-ancilla diffusion", color="#ff7f0e")
    ax3.set_title("FTQC T-Count Scaling")
    ax3.set_ylabel("T count")
    ax3.legend(fontsize=8)

    ax4.plot(np.arange(len(staircase.good_fidelity)), staircase.good_fidelity, marker="o", linewidth=2.0, color="#8c564b")
    ax4.set_title("Good-State Fidelity Along Staircase")
    ax4.set_ylabel("Fidelity")
    ax4.set_ylim(0.0, 1.05)

    ax1.set_xlabel("Grover iterations k")
    ax2.set_xlabel("Step index k")
    ax3.set_xlabel("Logical qubits n")
    ax4.set_xlabel("Step index k")
    for axis in axes.flat:
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "vtaa_failure_modes_profile",
            "souffle": {
                "n": 8,
                "guessed_m": 1,
                "actual_m": 5,
                "k_scan_factor": 1.5,
            },
            "staircase": {
                "n": 8,
                "k_max": 12,
            },
            "ftqc": {
                "n_min": 5,
                "n_max": 10,
                "noancilla_max": 6,
                "optimization_level": 1,
            },
        },
    )
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


def _save_vtaa_cost_figure(
    *,
    total_ps: float = 0.05,
    output_name="vtaa_cost_sweep_profile.png",
):
    plt = _load_pyplot()
    cost = vtaa.experiment_vtaa_cost_sweep(total_ps=total_ps, t1=100.0, t2=1000.0, t3=10000.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("VTAA Cost Sweep", fontsize=14, fontweight="bold")
    ax1, ax2 = axes

    x_pct = 100.0 * cost.early_success_ratios
    ax1.plot(x_pct, cost.standard_costs, linewidth=2.2, label="standard AA baseline", color="#d62728")
    ax1.plot(x_pct, cost.vtaa_costs, linewidth=2.2, label="VTAA functional", color="#1f77b4")
    ax1.set_title("Expected Query Cost")
    ax1.set_xlabel("Early-success ratio (%)")
    ax1.set_ylabel("Cost")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.35)

    advantage = np.asarray(cost.standard_costs) / np.maximum(np.asarray(cost.vtaa_costs), 1e-12)
    ax2.plot(x_pct, advantage, linewidth=2.2, color="#2ca02c")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    ax2.set_title("VTAA Advantage Factor")
    ax2.set_xlabel("Early-success ratio (%)")
    ax2.set_ylabel("Standard / VTAA")
    ax2.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "vtaa_cost_sweep_profile",
            "total_ps": float(total_ps),
            "t1": 100.0,
            "t2": 1000.0,
            "t3": 10000.0,
            "early_success_ratios": [float(x) for x in cost.early_success_ratios],
            "standard_costs": [float(x) for x in cost.standard_costs],
            "vtaa_costs": [float(x) for x in cost.vtaa_costs],
        },
    )
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


# =============================================================================
# Main Orchestrator
# =============================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    output_filepath = os.path.join(_RESULT_DIR, "terminal_output.log")
    os.chdir(_RESULT_DIR)
    logger = Logger(output_filepath)
    sys.stdout = logger
    cli_argv, publishability = parse_publishability_cli(
        sys.argv[1:],
        default_max_qubits=20,
        default_shots=1024,
        default_log_dir=_RESULT_DIR,
    )
    prepare_backend_validation_artifacts(publishability)

    print("VTAA Transpilation Benchmark Suite — Scenarios A through O (15 total)")
    print(f"Results saved to: {output_filepath}")
    print(SEP)
    print(publishability.summary())

    raw_scenarios = [
        ("A", run_scenario_a), ("B", run_scenario_b), ("C", run_scenario_c),
        ("D", run_scenario_d), ("E", run_scenario_e), ("F", run_scenario_f),
        ("G", run_scenario_g), ("H", run_scenario_h), ("I", run_scenario_i),
        ("J", run_scenario_j), ("K", run_scenario_k), ("L", run_scenario_l),
        ("M", run_scenario_m), ("N", run_scenario_n), ("O", run_scenario_o),
    ]
    scenarios = wrap_scenarios(raw_scenarios, module_globals=globals(), extra_patch_objects=(vtaa,), config=publishability)

    cli_executed = run_cli_scenario(cli_argv, scenarios)
    if not cli_executed:
        for label, fn in scenarios:
            try:
                fn()
            except Exception:
                import traceback
                print(f"\n*** SCENARIO {label} FAILED ***")
                traceback.print_exc()

        run_interactive_scenario_repl(scenarios, sep=SEP)

    print(f"\n{SEP}")
    print(f"Benchmark suite complete. {'1 scenario executed via direct CLI.' if cli_executed else '15 scenarios executed.'}")
    render_backend_validation_summary(publishability)
    _save_vtaa_algorithm_figure()
    _save_vtaa_cost_figure()
    logger.close()
    sys.stdout = logger.terminal
    print(f"\nBenchmark suite complete. Results saved to {output_filepath}")

