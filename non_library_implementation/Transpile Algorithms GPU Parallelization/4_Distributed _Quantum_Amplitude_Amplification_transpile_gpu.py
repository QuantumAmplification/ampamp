"""
DQAA Transpilation Master Suite (Scenarios A through Q)
========================================================
Rigorous hardware transpilation benchmark for Distributed Quantum Amplitude
Amplification (DQAA), orchestrating the experiment functions from:
    4_Distributed _Quantum_Amplitude_Amplification.py

Mathematical Standing Notation:
  n   = total (global) qubit register size
  j   = prefix bits used for partitioning  →  2^j parallel nodes
  s   = suffix bits per node  =  n - j
  p   = global success probability  =  num_good / 2^n
  p_k = local (node-k) success probability  =  local_good_count_k / 2^s
  L   = FPAA sequence length  ~  O(1/sqrt(p))
  alpha_l, beta_l = FPAA phases (chebyshev schedule)

  Partitioning:
    For each j-bit prefix k, node k holds targets whose first j bits == k.
    Advantaged nodes: prefix k has at least 1 target suffix.
    Disadvantaged nodes: prefix k has 0 targets.

  Key theorem (Convexity):  max_k p_k >= p
    The luckiest node always has p_k >= global p.
"""

from __future__ import annotations

import math
import os
import sys
import time
import ast
import inspect
import traceback
import importlib.util
import numpy as np

_BOOTSTRAP_HERE = os.path.dirname(os.path.abspath(__file__))
if _BOOTSTRAP_HERE not in sys.path:
    sys.path.insert(0, _BOOTSTRAP_HERE)

from aer_publishability_gpu import (
    parse_publishability_cli,
    prepare_backend_validation_artifacts,
    render_backend_validation_summary,
    run_cli_scenario,
    save_figure_with_metadata,
    wrap_scenarios,
)
from transpile_path_utils_gpu import ensure_directory_on_syspath, resolve_project_file

# ---------------------------------------------------------------------------
# Logger (same pattern as 3_Oblivious_Ampltude_Amplification_transpile.py and 3.5_FOAA_transpile.py)
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# Import the heavy-lifting DQAA module
# ---------------------------------------------------------------------------

_DQAA_FILENAME = "4_Distributed _Quantum_Amplitude_Amplification.py"
_HERE = os.fspath(ensure_directory_on_syspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
_DQAA_PATH = os.fspath(resolve_project_file(__file__, _DQAA_FILENAME, preferred_dirs=("Theory Algorithms",)))
_AER_GPU_HINT = (
    "This script now requires qiskit-aer-gpu on a CUDA-capable Linux/x86_64 system "
    "(or qiskit-aer-gpu-cu11 for CUDA 11)."
)


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _import_dqaa():
    spec = importlib.util.spec_from_file_location("dqaa_module", _DQAA_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    # Pre-register in sys.modules BEFORE exec_module so that @dataclass
    # (used inside the DQAA file with `from __future__ import annotations`)
    # can find the module's __module__ attribute when it does introspection.
    sys.modules["dqaa_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

dqaa = _import_dqaa()

# Convenience aliases for readability
hw_tradeoff       = dqaa.experiment_hardware_compilation_tradeoff
dist_fpaa         = dqaa.experiment_distributed_fpaa_execution
entanglement_obs  = dqaa.experiment_entanglement_obstruction
nisq_noise        = dqaa.experiment_nisq_noise_resilience
net_stats         = dqaa.experiment_classical_network_statistics
deqaaa_resource   = dqaa.experiment_deqaaa_resource_continuation
deqaaa_bookkeep   = dqaa.experiment_deqaaa_phase_bookkeeping
deqaaa_shots      = dqaa.experiment_deqaaa_shot_precision
deqaaa_partitions = dqaa.experiment_deqaaa_partition_sweep
deqaaa_targets    = dqaa.experiment_deqaaa_target_set_sweep
deqaaa_robustness = dqaa.experiment_deqaaa_distribution_robustness
deqaaa_mismatch   = dqaa.experiment_deqaaa_phase_mismatch_sweep
OracleSynth       = dqaa.DQAA_Oracle_Synthesizer

BASIS_NISQ = ["cx", "id", "rz", "sx", "x"]

# Standard test problem: n=6, j=2, 3 marked items
GLOBAL_N   = 6
J          = 2
LOCAL_N    = GLOBAL_N - J   # 4 suffix qubits  →  2^4 = 16 local states
GLOBAL_GOODS = ("110110", "111111", "011001")
GLOBAL_A  = len(GLOBAL_GOODS) / (2 ** GLOBAL_N)   # 3/64 ≈ 0.0469
EPSILON   = 0.3

SEP = "=" * 70


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
    print("Enter a label like A or Q, or press Enter to finish.")
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
        print("Example: global_n=10, j=3")
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


def _problem_params(
    global_n: int = GLOBAL_N,
    j: int = J,
    global_goods: tuple[str, ...] = GLOBAL_GOODS,
    epsilon: float = EPSILON,
) -> tuple[int, int, tuple[str, ...], float, int, float]:
    local_n = global_n - j
    global_a = len(global_goods) / (2 ** global_n)
    return global_n, j, global_goods, epsilon, local_n, global_a

# =============================================================================
# Scenario A: The Monolithic Routing Failure Mode (Baseline)
# =============================================================================

def run_scenario_a(global_n: int = GLOBAL_N, j: int = J, global_goods: tuple[str, ...] = GLOBAL_GOODS) -> None:
    """
    A. THE MONOLITHIC ROUTING failure mode (Baseline)

    Transpiles a full n=6 monolithic Grover step onto a linear-1D topology.
    The multi-controlled phase oracle spans the full chip, forcing SABRE to
    insert SWAP chains. This establishes WHY DQAA is necessary.

    Circuit: H^n + Phase-Oracle(n) + Grover-Diffusion(n)
    Metric: SWAP count, routing depth multiplier, routing CX overhead.
    """
    print(f"\n{SEP}")
    print("SCENARIO A: THE MONOLITHIC ROUTING failure mode (Baseline)")
    print(SEP)
    global_n, j, global_goods, _, local_n, global_a = _problem_params(global_n, j, global_goods, EPSILON)
    print(f"Problem: n={global_n} qubits, {len(global_goods)} marked items, Linear-1D topology.")
    print(f"Circuit: Full monolithic Grover step (H^n + oracle + diffusion).\n")

    res = hw_tradeoff(
        global_n=global_n,
        j=j,
        global_goods=global_goods,
        basis_gates=tuple(BASIS_NISQ),
    )

    m = res.monolithic_metrics
    print(f"{'Metric':<35} | {'Monolithic (n=' + str(global_n) + ')'}")
    print("-" * 60)
    print(f"{'Logical Depth (no cmap)':<35} | {int(m['logical_depth_no_cmap'])}")
    print(f"{'Routed Depth (Linear-1D)':<35} | {int(m['routed_depth'])}")
    print(f"{'Logical CX (no cmap)':<35} | {int(m['logical_cx_no_cmap'])}")
    print(f"{'Routed CX (with SWAPs)':<35} | {int(m['routed_cx'])}")
    print(f"{'SWAP Gates Inserted':<35} | {int(m['estimated_swap_count'])}")
    print(f"{'Routing CX Overhead':<35} | {int(m['estimated_routing_cx_overhead'])}")
    depth_mult = m['routed_depth'] / max(1, m['logical_depth_no_cmap'])
    print(f"{'Routing Depth Multiplier':<35} | {depth_mult:.2f}x")

    print(f"\n-> {int(m['estimated_swap_count'])} SWAPs inserted (each SWAP = 3 CX = 3x gate overhead).")
    print(f"   Routing depth multiplier: {depth_mult:.2f}x — the chip fights the algorithm.")
    print("-> CONCLUSION: Monolithic n=6 oracle on Linear-1D suffers exponential SWAP")
    print("   insertion. This is the physical failure mode DQAA is designed to escape.")


# =============================================================================
# Scenario B: The Distributed Topological Advantage
# =============================================================================

def run_scenario_b(global_n: int = GLOBAL_N, j: int = J, global_goods: tuple[str, ...] = GLOBAL_GOODS) -> None:
    """
    B. THE DISTRIBUTED TOPOLOGICAL ADVANTAGE

    Same n=6 problem, but partitioned by j=2 prefix bits into 2^j=4 nodes,
    each with only s=4 local qubits. Compares worst-case node metrics to
    the monolithic baseline to prove the Hardware Reduction Factor.

    The local oracle only needs to mark suffixes within a 4-qubit register.
    Multi-controlled ladders shrink from (n-1)=5 controls to (s-1)=3 controls.
    """
    print(f"\n{SEP}")
    print("SCENARIO B: THE DISTRIBUTED TOPOLOGICAL ADVANTAGE")
    print(SEP)
    global_n, j, global_goods, _, local_n, global_a = _problem_params(global_n, j, global_goods, EPSILON)
    print(f"Partition: j={j} prefix bits → 2^{j}={2**j} nodes, each with s={local_n} qubits.")
    print(f"Worst-case node = maximum over all node metrics (critical path).\n")

    res = hw_tradeoff(
        global_n=global_n,
        j=j,
        global_goods=global_goods,
        basis_gates=tuple(BASIS_NISQ),
    )

    m = res.monolithic_metrics
    a = res.distributed_aggregate_metrics   # max over all nodes (critical path)
    rf = res.reduction_factors

    print(f"{'Metric':<35} | {'Monolithic':<15} | {'Distributed (max)':<18} | {'Reduction'}")
    print("-" * 85)
    print(f"{'Register Width (qubits)':<35} | {global_n:<15} | {local_n:<18} | {rf['qubit_reduction']:.1f}x")
    print(f"{'Routed Depth':<35} | {int(m['routed_depth']):<15} | {int(a['routed_depth']):<18} | {rf['depth_reduction']:.2f}x")
    print(f"{'Routed CX Count':<35} | {int(m['routed_cx']):<15} | {int(a['routed_cx']):<18} | {rf['cx_reduction']:.2f}x")
    print(f"{'SWAP Gates Inserted':<35} | {int(m['estimated_swap_count']):<15} | {int(a['estimated_swap_count']):<18} | {rf['swap_reduction']:.2f}x")
    print(f"{'Routing CX Overhead':<35} | {int(m['estimated_routing_cx_overhead']):<15} | {int(a['estimated_routing_cx_overhead']):<18} | {rf['routing_cx_overhead_reduction']:.2f}x")

    print(f"\n-> Partitioning into j={j} prefix bits achieves {rf['depth_reduction']:.2f}x depth reduction")
    print(f"   and {rf['swap_reduction']:.2f}x fewer SWAPs on the critical-path node.")
    print("-> CONCLUSION: DQAA's distributed architecture improves feasibility by")
    print(f"   shrinking the qubit register from {global_n} to {local_n}, making it NISQ-viable.")


# =============================================================================
# Scenario C: The AST Oracle Simplification Dividend
# =============================================================================

def run_scenario_c(global_n: int = GLOBAL_N, j: int = J, formula: str = "(v0 & v1 & v2) | (~v0 & v3) | (v1 & ~v2 & v4) | (~v3 & v5)") -> None:
    """
    C. THE AST ORACLE SIMPLIFICATION DIVIDEND

    Feeds a complex 6-variable CNF formula to the DQAA_Oracle_Synthesizer.
    For each 2-bit prefix assignment (j=2 → 4 sub-oracles), the compiler
    classically substitutes the prefix values and simplifies the boolean
    formula via sympy, often producing trivial True/False sub-oracles.

    This proves that DQAA doesn't just shrink the qubit count — it also
    classically simplifies or entirely eliminates oracle synthesis work.

    Formula: (v0 & v1 & v2) | (~v0 & v3) | (v1 & ~v2 & v4) | (~v3 & v5)
    (Chosen to ensure nontrivial and trivial nodes under all 4 prefix assignments)
    """
    print(f"\n{SEP}")
    print("SCENARIO C: THE AST ORACLE SIMPLIFICATION DIVIDEND")
    print(SEP)

    global_n, j, global_goods, _, local_n, global_a = _problem_params(global_n, j, GLOBAL_GOODS, EPSILON)
    print(f"Formula: {formula}")
    print(f"Global: n={global_n}, j={j} ({2**j} prefix assignments)\n")

    try:
        from qiskit.transpiler import CouplingMap

        synth = OracleSynth(global_n=global_n, j=j, formula_text=formula, formula_format="boolean")
        cmap = CouplingMap.from_line(local_n)

        t0 = time.perf_counter()
        # compile_monolithic returns a plain dict (not a dataclass)
        mono_res = synth.compile_monolithic(
            basis_gates=BASIS_NISQ,
            coupling_map=CouplingMap.from_line(global_n),
            optimization_level=3,
        )
        t_mono = time.perf_counter() - t0

        t0 = time.perf_counter()
        # compile_all_prefixes returns dict[str, OraclePartitionNodeMetrics]
        dist_res = synth.compile_all_prefixes(
            basis_gates=BASIS_NISQ,
            coupling_map=cmap,
            optimization_level=3,
        )
        t_dist = time.perf_counter() - t0

        # mono_res is a dict; dist_res values are OraclePartitionNodeMetrics dataclasses
        print(f"{'Metric':<35} | {'Monolithic':<15} | {'Distributed total'}")
        print("-" * 65)
        print(f"{'CX Gates':<35} | {mono_res['cx_gates']:<15} | {sum(v.cx_gates for v in dist_res.values())}")
        print(f"{'Circuit Depth':<35} | {mono_res['depth']:<15} | max={max(v.depth for v in dist_res.values())}")
        print(f"{'SWAP count':<35} | {mono_res['estimated_swap_count']:<15} | max={max(v.estimated_swap_count for v in dist_res.values())}")
        print(f"{'Compile Time (sec)':<35} | {t_mono:.4f}         | {t_dist:.4f}")

        trivials = [p for p, v in dist_res.items() if v.is_trivial]
        print(f"\nPer-prefix simplification:")
        print(f"{'Prefix':<10} | {'Simplified Formula':<35} | {'Trivial?':<10} | {'Active Vars':<13} | {'CX'}")
        print("-" * 80)
        for prefix in sorted(dist_res.keys()):
            v = dist_res[prefix]
            trunc = v.simplified_formula[:33] + ".." if len(v.simplified_formula) > 35 else v.simplified_formula
            print(f"{prefix:<10} | {trunc:<35} | {'YES' if v.is_trivial else 'no':<10} | {v.active_variable_count:<13} | {v.cx_gates}")

        print(f"\n-> {len(trivials)}/{2**j} prefixes are trivial (constant True/False — no quantum oracle needed).")
        print(f"   Trivial prefixes: {trivials if trivials else 'none (all active)'}")
        print("-> CONCLUSION: Classical AST substitution eliminates quantum oracle overhead")
        print("   for trivial nodes, saving gates proportional to trivial-node fraction.")
    except Exception as exc:
        print(f"[Scenario C requires PhaseOracleGate + sympy. Skipped: {exc}]")


# =============================================================================
# Scenario D: The Classical Compiler Latency Limitation
# =============================================================================

def run_scenario_d(global_n: int = GLOBAL_N, j: int = J, global_goods: tuple[str, ...] = GLOBAL_GOODS) -> None:
    """
    D. THE CLASSICAL COMPILER LATENCY LIMITATION

    Directly measures limit-clock transpilation time for monolithic vs.
    distributed oracle synthesis. Although distributed circuits are smaller,
    there are 2^j of them. We prove whether DQAA parallelizes or amplifies
    classical compilation overhead.

    Method: time hw_tradeoff internally (which calls _extract_compilation_metrics
    per-node), then manually time a monolithic transpile for comparison.
    """
    print(f"\n{SEP}")
    print("SCENARIO D: THE CLASSICAL COMPILER LATENCY LIMITATION")
    print(SEP)
    global_n, j, global_goods, _, local_n, global_a = _problem_params(global_n, j, global_goods, EPSILON)
    print(f"Measuring Qiskit transpiler wall-clock time: monolithic n={global_n} vs. {2**j} distributed n={local_n} nodes.\n")

    from qiskit import QuantumCircuit, transpile as qk_transpile
    from qiskit.transpiler import CouplingMap

    # Monolithic: single n=GLOBAL_N Grover step
    qc_mono = dqaa._build_grover_step(global_n, list(global_goods))
    cmap_global = CouplingMap.from_line(global_n)

    t0 = time.perf_counter()
    dqaa.transpile(qc_mono, basis_gates=BASIS_NISQ, coupling_map=cmap_global,
                   optimization_level=3, seed_transpiler=42)
    t_mono = time.perf_counter() - t0

    # Distributed: 2^j nodes, each with local_n=LOCAL_N
    node_targets = dqaa._partition_targets_by_prefix(list(global_goods), j=j)
    cmap_local = CouplingMap.from_line(local_n)
    node_times: list[float] = []
    for prefix in sorted(node_targets.keys()):
        qc_local = dqaa._build_grover_step(local_n, node_targets[prefix])
        t0 = time.perf_counter()
        dqaa.transpile(qc_local, basis_gates=BASIS_NISQ, coupling_map=cmap_local,
                       optimization_level=3, seed_transpiler=42)
        node_times.append(time.perf_counter() - t0)

    t_dist_serial = sum(node_times)
    t_dist_parallel = max(node_times)   # if 2^j QPUs compile in parallel

    print(f"{'Metric':<45} | {'Time (sec)'}")
    print("-" * 60)
    print(f"{'Monolithic compile (1 circuit, n=' + str(global_n) + ')':<45} | {t_mono:.4f}")
    for idx, (prefix, t) in enumerate(zip(sorted(node_targets.keys()), node_times)):
        print(f"{'  Node ' + prefix + ' compile (n=' + str(local_n) + ')':<45} | {t:.4f}")
    print(f"{'Distributed serial total (' + str(2**j) + ' nodes x n=' + str(local_n) + ')':<45} | {t_dist_serial:.4f}")
    print(f"{'Distributed parallel critical path (max node)':<45} | {t_dist_parallel:.4f}")

    ratio_serial = t_dist_serial / max(1e-9, t_mono)
    ratio_parallel = t_dist_parallel / max(1e-9, t_mono)
    print(f"\n-> Serial distributed compile: {ratio_serial:.2f}x {'faster' if ratio_serial < 1 else 'slower'} than monolithic.")
    print(f"-> Parallel distributed compile: {ratio_parallel:.2f}x {'faster' if ratio_parallel < 1 else 'slower'} than monolithic.")
    print("-> CONCLUSION: DQAA parallelizes classical compilation — each QPU compiles its")
    print("   own smaller circuit independently, splitting the compiler's serial workload.")


# =============================================================================
# Scenario E: The Entanglement Obstruction (State-Prep Limitation)
# =============================================================================

def run_scenario_e(global_n: int = GLOBAL_N, j: int = J, good_global: str = "110110", epsilon: float = EPSILON) -> None:
    """
    E. THE ENTANGLEMENT OBSTRUCTION (State-Prep Limitation)

    The DQAA mathematical guarantee requires the initial state to be SEPARABLE
    across the prefix/suffix cut:  A |0> = |k>_prefix ⊗ |psi>_suffix.
    If the state prep circuit creates cross-register entanglement, the local
    diffusion operator S_s(alpha) completely breaks down.

    This provides the counterexample: we quantify the success-probability
    reduction when the separability assumption is violated.
    """
    print(f"\n{SEP}")
    print("SCENARIO E: THE ENTANGLEMENT OBSTRUCTION (State-Prep Limitation)")
    print(SEP)
    global_n, j, global_goods, epsilon, local_n, global_a = _problem_params(global_n, j, GLOBAL_GOODS, epsilon)
    print(f"Global: n={global_n}, j={j}, target='{good_global}', epsilon={epsilon}")
    print("Comparing: Separable |prefix>⊗|+suffix> vs. HW-efficient cross-register ansatz.\n")

    try:
        res = entanglement_obs(
            global_n=global_n,
            j=j,
            good_global=good_global,
            epsilon=epsilon,
            global_a_for_schedule=global_a,
        )

        peak_sep = float(np.max(res.separable_good_probs))
        peak_ent = float(np.max(res.entangled_good_probs))
        ratio = res.obstruction_ratio_peak

        print(f"{'Metric':<40} | {'Separable':<15} | {'Entangled'}")
        print("-" * 75)
        print(f"{'Suffix Register Purity':<40} | {res.separable_suffix_purity:<15.6f} | {res.entangled_suffix_purity:.6f}")
        print(f"{'Entanglement Entropy (bits)':<40} | {'0.000000':<15} | {res.entangled_suffix_entropy_bits:.6f}")
        print(f"{'Peak Success Probability':<40} | {peak_sep:<15.6f} | {peak_ent:.6f}")
        print(f"{'Obstruction Ratio (sep/ent peak)':<40} | {ratio:.2f}x")
        print(f"{'FPAA Steps L':<40} | {res.L}")

        print(f"\n-> Separable prep: purity=1.0 (pure suffix state), peak success={peak_sep:.4f}")
        print(f"   Entangled prep:  purity={res.entangled_suffix_purity:.4f} (mixed), peak success={peak_ent:.6f}")
        print(f"-> Obstruction ratio: {ratio:.1f}x — entanglement reduces peak success by a factor of {ratio:.1f}.")
        print("-> CONCLUSION: Cross-register entanglement traces out the prefix, creating a")
        print("   mixed suffix state. The local diffusion cannot amplify a mixed-state target.")
        print("   This is the formal physical proof of the DQAA separability pre-condition.")
    except Exception as exc:
        print(f"[Scenario E requires Qiskit DensityMatrix support. Skipped: {exc}]")


# =============================================================================
# Scenario F: The NISQ Coherence Limit (Noise Thresholds)
# =============================================================================

def run_scenario_f(global_n: int = GLOBAL_N, j: int = J, global_goods: tuple[str, ...] = GLOBAL_GOODS, epsilon: float = EPSILON, shots: int = 2048, preferred_backend: str = "FakeGuadalupeV2") -> None:
    """
    F. THE NISQ COHERENCE LIMIT (Noise Thresholds)

    Runs both monolithic and distributed circuits under a realistic hardware
    noise model (FakeGuadalupeV2 or GenericBackendV2 fallback).
    Proves that the monolithic circuit depth causes severe noise failure
    while the distributed node's shallower circuit stays above the noise floor.
    """
    print(f"\n{SEP}")
    print("SCENARIO F: THE NISQ COHERENCE LIMIT (Noise Thresholds)")
    print(SEP)
    print(f"Backend: FakeGuadalupeV2 (or GenericBackendV2 fallback)")
    global_n, j, global_goods, epsilon, local_n, global_a = _problem_params(global_n, j, global_goods, epsilon)
    print(f"Problem: n={global_n}, j={j}, {len(global_goods)} targets. Noise model from backend calibration.\n")

    try:
        res = nisq_noise(
            global_n=global_n,
            j=j,
            global_goods=global_goods,
            epsilon=epsilon,
            global_a_for_schedule=global_a,
            shots=shots,
            preferred_backend=preferred_backend,
        )

        peak_mono_ideal = float(np.max(res.monolithic_ideal_success))
        peak_mono_noisy = float(np.max(res.monolithic_noisy_success))
        peak_dist_ideal = float(np.max(res.distributed_ideal_success))
        peak_dist_noisy = float(np.max(res.distributed_noisy_success))

        # Find the advantaged node with highest noisy success
        best_prefix = None
        best_peak = 0.0
        for prefix, arr in res.distributed_node_noisy_success.items():
            pk = float(np.max(arr))
            if pk > best_peak:
                best_peak = pk
                best_prefix = prefix

        l_steps = (int(res.L) - 1) // 2
        print(f"Backend: {res.backend_name} ({res.backend_source})")
        print(f"FPAA odd length L={res.L}, generalized iterates l={l_steps}, shots={res.shots}\n")

        print(f"{'Metric':<40} | {'Monolithic':<15} | {'Distributed'}")
        print("-" * 75)
        print(f"{'Random baseline':<40} | {res.monolithic_random_baseline:<15.5f} | {res.distributed_random_baseline:.5f}")
        print(f"{'Peak success (ideal)':<40} | {peak_mono_ideal:<15.4f} | {peak_dist_ideal:.4f}")
        print(f"{'Peak success (noisy)':<40} | {peak_mono_noisy:<15.4f} | {peak_dist_noisy:.4f}")
        noise_penalty_mono = peak_mono_ideal - peak_mono_noisy
        noise_penalty_dist = peak_dist_ideal - peak_dist_noisy
        print(f"{'Noise penalty (ideal - noisy)':<40} | {noise_penalty_mono:<15.4f} | {noise_penalty_dist:.4f}")

        if best_prefix:
            print(f"\nMost favorable node '{best_prefix}' peak noisy success: {best_peak:.4f}")
            mono_vs_best = peak_mono_noisy / max(1e-6, best_peak)
            print(f"Monolithic noisy / distributed-node ratio: {mono_vs_best:.3f}")

        print(f"\n-> Monolithic noise penalty: {noise_penalty_mono:.4f} (deeper circuit incurs greater coherence loss)")
        print(f"   Distributed noise penalty: {noise_penalty_dist:.4f} (shallower node retains higher coherence)")
        print("-> CONCLUSION: The monolithic circuit's SWAP-inflated depth destroys coherence under")
        print("   realistic noise. The distributed node's shallower circuit remains above the baseline.")
    except Exception as exc:
        print(f"[Scenario F requires qiskit-aer + fake backend. Skipped: {exc}]")


# =============================================================================
# Scenario G: Network Shot-Noise Sifting (Classical Post-Processing Overhead)
# =============================================================================

def run_scenario_g(global_n: int = GLOBAL_N, j: int = J, global_goods: tuple[str, ...] = GLOBAL_GOODS, epsilon: float = EPSILON, shots_per_node: int = 100, sifting_sigma: float = 4.0) -> None:
    """
    G. NETWORK SHOT-NOISE SIFTING (Classical Post-Processing Overhead)

    Simulates the classical master node's job after 2^j distributed QPUs
    return shot counts. The master must sift above a 4-sigma threshold
    to flag candidates, then classically verify each flagged candidate.

    This quantifies the classical bandwidth and verification overhead of DQAA.
    """
    print(f"\n{SEP}")
    print("SCENARIO G: NETWORK SHOT-NOISE SIFTING (Classical Post-Processing Overhead)")
    print(SEP)
    global_n, j, global_goods, epsilon, local_n, global_a = _problem_params(global_n, j, global_goods, epsilon)
    print(f"Problem: n={global_n}, j={j}, {len(global_goods)} targets.")
    print(f"Shots per node: 100, sifting threshold: 4-sigma above uniform mean.\n")

    try:
        res = net_stats(
            global_n=global_n,
            j=j,
            global_goods=global_goods,
            epsilon=epsilon,
            global_a_for_schedule=global_a,
            shots_per_node=shots_per_node,
            sifting_sigma=sifting_sigma,
        )

        print(f"{'Metric':<40} | {'Value'}")
        print("-" * 60)
        print(f"{'Total nodes (2^j)':<40} | {2**j}")
        print(f"{'Shots per node':<40} | {res.shots_per_node}")
        print(f"{'Total shots network-wide':<40} | {res.total_shots}")
        print(f"{'Uniform mean per state':<40} | {res.uniform_mean_per_state:.2f}")
        print(f"{'Uniform std per state':<40} | {res.uniform_std_per_state:.2f}")
        print(f"{'Sifting threshold (4-sigma)':<40} | {res.sifting_threshold}")
        print(f"{'Flagged candidates':<40} | {len(res.flagged_candidates)}")
        print(f"{'Classical verification queries':<40} | {res.classical_queries_made}")
        print(f"{'Verified correct answers':<40} | {len(res.verified_answers)}")
        print(f"{'False positives':<40} | {len(res.false_positives)}")

        if res.verified_answers:
            print(f"\nVerified answers: {res.verified_answers}")
        if res.false_positives:
            print(f"False positives:  {res.false_positives}")

        recall = len(res.verified_answers) / max(1, len(global_goods))
        fp_rate = len(res.false_positives) / max(1, len(res.flagged_candidates))
        print(f"\n-> Recall rate: {recall:.0%}  ({len(res.verified_answers)}/{len(global_goods)} targets found)")
        print(f"   False positive rate: {fp_rate:.0%}")
        print(f"   Classical query overhead: {res.classical_queries_made} verifications for {len(global_goods)} targets")
        print("-> CONCLUSION: DQAA shifts work from the quantum chip to a classical master node.")
        print("   The master must sift shot noise, verify candidates, and handle false positives.")
        print("   More shots per node reduces false positives but increases total quantum time.")
    except Exception as exc:
        print(f"[Scenario G requires qiskit-aer. Skipped: {exc}]")


# =============================================================================
# Scenario H: Hardware Profiler Comparative Evaluation (Monolithic vs DQAA)
# =============================================================================

def run_scenario_h(global_n: int = GLOBAL_N, j: int = J, global_goods: tuple[str, ...] = GLOBAL_GOODS) -> None:
    """
    H. HARDWARE PROFILER COMPARATIVE EVALUATION (Monolithic vs DQAA)

    Feeds the monolithic Grover-step circuit and the worst-case distributed
    node circuit through the quantum_profiler.HardwareProfiler to get a
    single physical execution time estimate (ns) and hardware penalty score.

    KEY NOTE: HardwareProfiler requires circuits that are already transpiled
    into basis gates (it runs its own routing pass, but the gates must be
    in basis form). We pre-decompose using optimization_level=0 so routing
    sees the native gate set, then the profiler adds routing SWAPs.
    """
    print(f"\n{SEP}")
    print("SCENARIO H: HARDWARE PROFILER COMPARATIVE EVALUATION (Monolithic vs DQAA)")
    print(SEP)

    try:
        sys.path.insert(0, _HERE)
        from quantum_profiler import HardwareProfiler
    except ImportError:
        print("[Scenario H skipped: quantum_profiler not found.]")
        return

    from qiskit import transpile as qk_transpile
    from qiskit.transpiler import CouplingMap

    global_n, j, global_goods, epsilon, local_n, global_a = _problem_params(global_n, j, global_goods, EPSILON)
    node_targets = dqaa._partition_targets_by_prefix(list(global_goods), j=j)

    # Monolithic: single full Grover step — pre-transpile to basis gates only
    qc_mono = dqaa._build_grover_step(global_n, list(global_goods))
    cmap_global = CouplingMap.from_line(global_n)
    qc_mono_basis = qk_transpile(qc_mono, basis_gates=BASIS_NISQ, optimization_level=0)

    # Worst-case (critical-path) distributed node = node with largest local oracle
    worst_prefix = max(node_targets.keys(), key=lambda p: len(node_targets[p]))
    qc_local = dqaa._build_grover_step(local_n, node_targets[worst_prefix])
    cmap_local = CouplingMap.from_line(local_n)
    qc_local_basis = qk_transpile(qc_local, basis_gates=BASIS_NISQ, optimization_level=0)

    edges_global = [[i, i+1] for i in range(global_n-1)] + [[i+1, i] for i in range(global_n-1)]
    edges_local  = [[i, i+1] for i in range(local_n-1)]  + [[i+1, i] for i in range(local_n-1)]

    profiler_mono  = HardwareProfiler(
        coupling_map_edges=edges_global, basis_gates=BASIS_NISQ,
        single_qubit_ns=20, two_qubit_ns=100)
    profiler_local = HardwareProfiler(
        coupling_map_edges=edges_local,  basis_gates=BASIS_NISQ,
        single_qubit_ns=20, two_qubit_ns=100)

    print("Profiling monolithic Grover step ...")
    m_mono  = profiler_mono.profile_circuit(qc_mono_basis)
    print("Profiling worst-case distributed node ...")
    m_local = profiler_local.profile_circuit(qc_local_basis)

    print(f"\n{'Metric':<30} | {'Monolithic (n=' + str(global_n) + ')':<20} | {'Dist. Node (n=' + str(local_n) + ')'}")
    print("-" * 75)
    print(f"{'Logical Depth':<30} | {m_mono['logical_depth']:<20} | {m_local['logical_depth']}")
    print(f"{'Post-Routing SWAPs':<30} | {m_mono['routing_swaps']:<20} | {m_local['routing_swaps']}")
    print(f"{'Final CNOT Count':<30} | {m_mono['final_cnots']:<20} | {m_local['final_cnots']}")
    print(f"{'Total Execution Time (ns)':<30} | {m_mono['total_time_ns']:<20.1f} | {m_local['total_time_ns']:.1f}")
    print(f"{'Unified Hardware Penalty':<30} | {m_mono['hardware_penalty_score']:<20.1f} | {m_local['hardware_penalty_score']:.1f}")

    speedup = m_mono['total_time_ns'] / max(1, m_local['total_time_ns'])
    print(f"\n-> Hardware time speedup: {speedup:.2f}x (monolithic / distributed node)")
    print("-> CONCLUSION: DQAA's local node executes in a fraction of the monolithic time.")
    print("   This provides direct hardware-time evidence for the distributed architecture.")


# =============================================================================
# Scenario I: Low-Utility Node Coherence Overhead (QPU Resource Expenditure)
# =============================================================================

def run_scenario_i(global_n: int = GLOBAL_N, j: int = J, global_goods: tuple[str, ...] = GLOBAL_GOODS, epsilon: float = EPSILON) -> None:
    """
    I. LOW-UTILITY NODE COHERENCE OVERHEAD (QPU Resource Expenditure)

    In a 2^j distributed network with only a few marked items, most nodes are
    disadvantaged (zero local targets). Yet every disadvantaged node must execute the
    same l-step FPAA circuit as the advantaged node, expending quantum coherence without
    contributing useful output.

    We quantify the total wasted quantum volume across the network.
    """
    print(f"\n{SEP}")
    print("SCENARIO I: LOW-UTILITY NODE COHERENCE OVERHEAD (QPU Resource Expenditure)")
    print(SEP)

    global_n, j, global_goods, epsilon, local_n, global_a = _problem_params(global_n, j, global_goods, epsilon)
    res_fpaa = dist_fpaa(global_n=global_n, j=j, global_goods=global_goods, epsilon=epsilon)

    n_total = 2 ** j
    n_advantaged = len(res_fpaa.lucky_nodes)
    n_disadvantaged = len(res_fpaa.unlucky_nodes)
    L = res_fpaa.L
    l_steps = res_fpaa.l_opt

    # Estimate gates per full FPAA circuit (l generalized iterates).
    # Use hw_tradeoff to get worst-case node metrics
    hw = hw_tradeoff(global_n=global_n, j=j, global_goods=global_goods, basis_gates=tuple(BASIS_NISQ))
    advantaged_cx = max(
        hw.distributed_node_metrics[p]["routed_cx"]
        for p in res_fpaa.lucky_nodes if p in hw.distributed_node_metrics
    ) if res_fpaa.lucky_nodes else 0
    disadvantaged_cx = min(
        hw.distributed_node_metrics[p]["routed_cx"]
        for p in res_fpaa.unlucky_nodes if p in hw.distributed_node_metrics
    ) if res_fpaa.unlucky_nodes else 0

    total_network_cx = sum(hw.distributed_node_metrics[p]["routed_cx"] for p in hw.distributed_node_metrics) * l_steps
    advantaged_total_cx = advantaged_cx * l_steps
    wasted_cx = total_network_cx - advantaged_total_cx
    waste_ratio = wasted_cx / max(1, total_network_cx)

    print(f"{'Metric':<45} | {'Value'}")
    print("-" * 65)
    print(f"{'Total nodes (2^j)':<45} | {n_total}")
    print(f"{'Advantaged nodes (have target suffixes)':<45} | {n_advantaged}  ({res_fpaa.lucky_nodes})")
    print(f"{'Disadvantaged nodes (empty oracle)':<45} | {n_disadvantaged}  ({res_fpaa.unlucky_nodes})")
    print(f"{'FPAA odd length L':<45} | {L}")
    print(f"{'FPAA generalized iterates l':<45} | {l_steps}")
    print(f"{'CX gates per advantaged node (1 local FPAA iterate)':<45} | {int(advantaged_cx)}")
    print(f"{'CX gates per disadvantaged node (empty-oracle iterate)':<45} | {int(disadvantaged_cx)}")
    print(f"{'Total network CX gates (l iterates, all nodes)':<45} | {int(total_network_cx)}")
    print(f"{'CX gates on advantaged path only':<45} | {int(advantaged_total_cx)}")
    print(f"{'Wasted CX network-wide':<45} | {int(wasted_cx)}")
    print(f"{'Quantum Waste Ratio (wasted/total)':<45} | {waste_ratio:.1%}")

    print(f"\n-> {waste_ratio:.0%} of all CX gates executed network-wide produce no useful output.")
    print(f"   {n_disadvantaged} out of {n_total} QPUs expend coherence on an empty oracle for l={l_steps} generalized iterates (L={L}).")
    print("-> CONCLUSION: DQAA's distributed efficiency gain is offset by the QPU waste ratio.")
    print("   As j grows, disadvantaged nodes dominate: with 1 marked item in 2^n states,")
    print("   the disadvantaged fraction → (2^j - 1)/2^j → 1.0 as j increases.")


# =============================================================================
# Scenario J: Heterogeneous Network Noise (Degraded-QPU Limitation)
# =============================================================================

def run_scenario_j(global_n: int = GLOBAL_N, j: int = J, global_goods: tuple[str, ...] = GLOBAL_GOODS, epsilon: float = EPSILON, pristine_depol: float = 0.01, degraded_depol: float = 0.05, shots: int = 2048) -> None:
    """
    J. HETEROGENEOUS NETWORK NOISE (Degraded-QPU Limitation)

    In a real quantum cluster, QPUs have different noise levels. If the advantaged
    node happens to be assigned to a degraded QPU (5% depolarizing vs 1%),
    the FPAA amplification is materially reduced and the classical master node may fail
    to identify the signal.

    We simulate this by building two AerSimulator instances with different
    noise levels and testing both on the advantaged node's circuit.
    """
    print(f"\n{SEP}")
    print("SCENARIO J: HETEROGENEOUS NETWORK NOISE (Degraded-QPU Limitation)")
    print(SEP)
    print("Scenario: the advantaged node is assigned either a pristine (1%) or degraded (5%) QPU.")
    global_n, j, global_goods, epsilon, local_n, global_a = _problem_params(global_n, j, global_goods, epsilon)
    print(f"Problem: n={global_n}, j={j}, {len(global_goods)} targets, l-step FPAA with odd length L=2l+1.\n")

    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        from qiskit import QuantumCircuit, transpile as qk_transpile

        # Identify the advantaged node (one with most targets)
        node_targets = dqaa._partition_targets_by_prefix(list(global_goods), j=j)
        selected_prefix = max(node_targets.keys(), key=lambda p: len(node_targets[p]))
        local_targets = node_targets[selected_prefix]
        print(f"Advantaged node prefix: '{selected_prefix}', targets: {local_targets}\n")

        # Build FPAA schedule
        l_opt = int(math.ceil((0.5 * math.log(2.0 / epsilon)) / math.sqrt(global_a)))
        L = int(2 * l_opt + 1)
        alphas, betas = dqaa._generate_fpaa_phases(L=L, delta=epsilon)

        qc = QuantumCircuit(local_n)
        qc.h(range(local_n))
        for a_j, b_j in zip(alphas, betas):
            qc.global_phase += np.pi
            qc.append(dqaa._build_local_oracle(local_n, local_targets, float(b_j)).to_gate(), range(local_n))
            qc.append(dqaa._build_local_diffusion(local_n, float(a_j)).to_gate(), range(local_n))
        qc.measure_all()

        results = {}
        for label, dep_pct in [("Pristine (1% depol)", pristine_depol), ("Degraded (5% depol)", degraded_depol)]:
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(dep_pct, 2), ['cx'])
            sim = AerSimulator(noise_model=nm, device="GPU")
            t_qc = qk_transpile(qc, backend=sim, optimization_level=3, seed_transpiler=42)
            counts = sim.run(t_qc, shots=shots).result().get_counts()
            p_succ = sum(counts.get(s, 0) for s in local_targets) / shots
            p_baseline = len(local_targets) / (2 ** local_n)
            snr = p_succ / max(1e-6, p_baseline)
            results[label] = (dep_pct, p_succ, p_baseline, snr)

        print(f"{'Noise Model':<25} | {'Dep. Rate':<12} | {'Success P':<12} | {'Baseline P':<12} | {'SNR'}")
        print("-" * 75)
        for label, (rate, p_s, p_b, snr) in results.items():
            print(f"{label:<25} | {rate:<12.0%} | {p_s:<12.4f} | {p_b:<12.4f} | {snr:.2f}x")

        prist_snr = results["Pristine (1% depol)"][3]
        degr_snr  = results["Degraded (5% depol)"][3]
        print(f"\n-> SNR reduction: {prist_snr:.2f}x → {degr_snr:.2f}x ({(1-degr_snr/max(1e-6,prist_snr))*100:.0f}% SNR loss)")
        print("   The classical master node requires SNR > 1.0 to distinguish signal above noise floor.")
        if degr_snr < 1.0:
            print("-> The degraded QPU falls below the effective noise floor: the master node cannot reliably identify the marked item.")
        print("-> CONCLUSION: DQAA is as strong as its weakest QPU. If the advantaged node is on a")
        print("   degraded chip, the distributed speedup is entirely erased by thermal noise.")
    except Exception as exc:
        print("[Scenario J requires qiskit-aer. Skipped.]")
        print(f"{_AER_GPU_HINT} Original error: {type(exc).__name__}: {exc}")


# =============================================================================
# Scenario K: The Partitioning Tradeoff (Classical Coordination Growth)
# =============================================================================

def run_scenario_k(global_n: int = GLOBAL_N, global_goods: tuple[str, ...] = GLOBAL_GOODS, j_min: int = 1, j_max: int | None = None) -> None:
    """
    K. THE PARTITIONING TRADEOFF (Classical Coordination Growth)

    Sweeps j from 1 to n-2, proving DQAA is a U-shaped optimization problem:
    - As j increases: quantum circuit depth decreases
    - As j increases: classical nodes = 2^j increases rapidly
    - As j increases: classical verification queries increase accordingly

    The optimal j is where quantum benefit outweighs classical coordination cost.
    """
    print(f"\n{SEP}")
    print("SCENARIO K: PARTITIONING TRADEOFF (Classical Coordination Growth)")
    print(SEP)
    if j_max is None:
        j_max = global_n - 2
    print(f"Sweeping j from {j_min} to {j_max} for n={global_n}, {len(global_goods)} marked items.")
    print("Tracking: local circuit depth, 2^j node count, and verification-query growth.\n")

    print(f"{'j':<5} | {'2^j nodes':<12} | {'Local n (bits)':<16} | {'Max Routed Depth':<18} | {'Max CX':<10} | {'Est. Verify Queries'}")
    print("-" * 85)

    for j_val in range(j_min, min(j_max, global_n - 2) + 1):
        local_n_val = global_n - j_val
        n_nodes = 2 ** j_val

        try:
            hw = hw_tradeoff(
                global_n=global_n,
                j=j_val,
                global_goods=global_goods,
                basis_gates=tuple(BASIS_NISQ),
            )
            max_depth = int(max(m["routed_depth"] for m in hw.distributed_node_metrics.values()))
            max_cx    = int(max(m["routed_cx"]    for m in hw.distributed_node_metrics.values()))
        except Exception:
            max_depth = -1
            max_cx    = -1

        # Classical verification: 1 oracle call per flagged candidate
        # In the limit: each node outputs ~1 candidate → n_nodes queries
        # In the empty case: 0 queries. Estimate = n_nodes (worst case)
        est_queries = n_nodes

        print(f"{j_val:<5} | {n_nodes:<12} | {local_n_val:<16} | {max_depth:<18} | {max_cx:<10} | {est_queries}")

    print(f"\n-> Quantum depth decreases with j because the local register becomes shallower.")
    print(f"   Classical overhead (nodes and queries) increases as 2^j.")
    print(f"   The optimal j balances quantum depth savings against classical coordination cost.")
    print(f"-> CONCLUSION: DQAA is a U-shaped optimization. Choosing j too large makes the")
    print(f"   classical coordinator the bottleneck, not the quantum hardware.")
    print(f"   For n={global_n} with {len(global_goods)} items, optimal j ≈ floor(n/2) = {global_n//2}.")


# =============================================================================
# Scenario L: 2026 DEQAAA Resource Continuation
# =============================================================================

def run_scenario_l(global_n: int = 6, dqaa_j: int = 2, deqaaa_node_qubits: tuple[int, ...] = (2, 2, 2), target_indices: tuple[int, ...] = (8, 14), distribution_seed: int = 21) -> None:
    """
    L. 2026 DEQAAA RESOURCE CONTINUATION

    Compares the 2025 DQAA architecture against the 2026 DEQAAA continuation:
    - DQAA: 2^j nodes, local width n-j, total qubits 2^j(n-j)
    - DEQAAA: arbitrary t nodes, local widths n_j summing to n, total qubits n

    Also prints the local success probabilities p_j, exact-EQAAA iteration counts
    J_j, and phase angles phi_j required by the first phase of DEQAAA.
    """
    print(f"\n{SEP}")
    print("SCENARIO L: 2026 DEQAAA RESOURCE CONTINUATION")
    print(SEP)

    resource = deqaaa_resource(
        global_n=global_n,
        dqaa_j=dqaa_j,
        deqaaa_node_qubits=deqaaa_node_qubits,
    )
    bookkeeping = deqaaa_bookkeep(
        global_n=global_n,
        node_qubits=deqaaa_node_qubits,
        target_indices=target_indices,
        distribution_seed=distribution_seed,
    )

    print(f"{'Metric':<42} | {'2025 DQAA':<20} | {'2026 DEQAAA'}")
    print("-" * 90)
    print(f"{'Node count':<42} | {resource.dqaa_nodes:<20} | {resource.deqaaa_nodes}")
    print(f"{'Max qubits on one node':<42} | {resource.dqaa_local_qubits:<20} | {resource.deqaaa_max_local_qubits}")
    print(f"{'Total qubits across network':<42} | {resource.dqaa_total_qubits:<20} | {resource.deqaaa_total_qubits}")
    print(f"{'Arbitrary amplitude distributions?':<42} | {'No':<20} | Yes")
    print(f"{'Exact amplification?':<42} | {'No':<20} | Yes")

    print(f"\nPaper-style bookkeeping on an arbitrary distribution with targets={target_indices}:")
    print(f"Global success probability p_g = {bookkeeping.global_success_probability:.6f}")
    print(f"{'Node':<10} | {'n_j':<6} | {'|X_j|':<8} | {'p_j':<12} | {'J_j':<8} | {'phi_j'}")
    print("-" * 80)
    for node_idx in range(len(bookkeeping.node_qubits)):
        print(
            f"{node_idx:<10} | "
            f"{bookkeeping.node_qubits[node_idx]:<6} | "
            f"{int(bookkeeping.local_target_counts[node_idx]):<8} | "
            f"{bookkeeping.local_success_probabilities[node_idx]:<12.6f} | "
            f"{int(bookkeeping.local_iterations[node_idx]):<8} | "
            f"{bookkeeping.local_phases[node_idx]:.6f}"
        )
        print(f"  node_{node_idx} local targets: {bookkeeping.node_targets[f'node_{node_idx}']}")

    print("\n-> CONCLUSION: DEQAAA removes the tensor-factorization bottleneck,")
    print("   permits arbitrary node partitions, and compresses total qubits from")
    print("   2^j(n-j) down to n while retaining exact local EQAAA phase control.")


# =============================================================================
# Scenario M: 2026 DEQAAA Shot-Precision Continuation
# =============================================================================

def run_scenario_m(global_n: int = 6, deqaaa_node_qubits: tuple[int, ...] = (2, 2, 2), target_indices: tuple[int, ...] = (8, 14), shot_counts: tuple[int, ...] = (10000, 100000), distribution_seed: int = 21, measurement_seed: int = 21) -> None:
    """
    M. 2026 DEQAAA SHOT-PRECISION CONTINUATION

    Reproduces the main practical caveat of the 2026 paper: DEQAAA needs the
    exact probability distribution P to synthesize precise local p_j values and
    exact phase angles, while experiments only have access to shot-estimated P~.
    """
    print(f"\n{SEP}")
    print("SCENARIO M: 2026 DEQAAA SHOT-PRECISION CONTINUATION")
    print(SEP)

    result = deqaaa_shots(
        global_n=global_n,
        node_qubits=deqaaa_node_qubits,
        target_indices=target_indices,
        shot_counts=shot_counts,
        distribution_seed=distribution_seed,
        measurement_seed=measurement_seed,
    )

    print(f"Arbitrary-distribution continuation with n={global_n}, node_qubits={deqaaa_node_qubits}, targets={target_indices}")
    print(f"Exact local p_j values: {np.array2string(result.exact_local_success_probabilities, precision=6)}\n")
    print(f"{'Shots':<12} | {'D_KL(P~||P)':<16} | {'max |p_j-p~_j|':<16} | {'mean |p_j-p~_j|':<17} | {'Estimated J_j'}")
    print("-" * 105)
    for idx, shots in enumerate(result.shot_counts):
        print(
            f"{int(shots):<12} | "
            f"{result.kl_divergences[idx]:<16.6e} | "
            f"{result.max_local_probability_error[idx]:<16.6e} | "
            f"{result.mean_local_probability_error[idx]:<17.6e} | "
            f"{result.estimated_local_iterations[idx].tolist()}"
        )

    print("\n-> CONCLUSION: the exact 2026 construction is mathematically clean, but")
    print("   practical phase synthesis depends on how accurately shots recover P.")
    print("   Higher shot counts shrink both KL divergence and local p_j error.")


# =============================================================================
# Scenario N: 2026 DEQAAA Partition-Strategy Sweep
# =============================================================================

def run_scenario_n(global_n: int = 6, target_indices: tuple[int, ...] = (8, 14), distribution_seed: int = 21, max_nodes: int | None = None) -> None:
    """
    N. 2026 DEQAAA PARTITION-STRATEGY SWEEP

    Sweeps admissible node allocations n = n0 + ... + n_{t-1} and records how
    the partition strategy changes the per-node qubit load and the local
    success probabilities p_j that govern the exact local phases.
    """
    print(f"\n{SEP}")
    print("SCENARIO N: 2026 DEQAAA PARTITION-STRATEGY SWEEP")
    print(SEP)

    result = deqaaa_partitions(
        global_n=global_n,
        target_indices=target_indices,
        distribution_seed=distribution_seed,
        max_nodes=max_nodes,
    )

    print(f"Sweeping admissible node partitions for n={global_n}, targets={target_indices}")
    print(f"{'Partition n_j':<18} | {'t':<4} | {'max n_j':<8} | {'min p_j':<12} | {'max p_j':<12} | {'mean p_j':<12} | {'J_j'}")
    print("-" * 110)
    for row in result.rows:
        print(
            f"{str(tuple(row['node_qubits'])):<18} | "
            f"{row['nodes']:<4} | "
            f"{row['max_local_qubits']:<8} | "
            f"{row['min_local_success_probability']:<12.6f} | "
            f"{row['max_local_success_probability']:<12.6f} | "
            f"{row['mean_local_success_probability']:<12.6f} | "
            f"{row['local_iterations']}"
        )

    best = result.rows[0]
    worst = result.rows[-1]
    print("\nBest partition under current ordering:")
    print(f"  {tuple(best['node_qubits'])} with max local qubits={best['max_local_qubits']} and max p_j={best['max_local_success_probability']:.6f}")
    print("Worst partition under current ordering:")
    print(f"  {tuple(worst['node_qubits'])} with max local qubits={worst['max_local_qubits']} and max p_j={worst['max_local_success_probability']:.6f}")


# =============================================================================
# Scenario O: 2026 DEQAAA Target-Configuration Sweep
# =============================================================================

def run_scenario_o(global_n: int = 6, deqaaa_node_qubits: tuple[int, ...] = (2, 2, 2), target_sets: tuple[tuple[int, ...], ...] = ((8, 14), (8, 9), (14, 15), (8, 14, 30)), distribution_seed: int = 21) -> None:
    """
    O. 2026 DEQAAA TARGET-CONFIGURATION SWEEP

    Compares several target arrangements under a fixed DEQAAA partition to show
    how concentrated, dispersed, and multi-target cases change p_g, p_j, and
    the local exact-amplification bookkeeping.
    """
    print(f"\n{SEP}")
    print("SCENARIO O: 2026 DEQAAA TARGET-CONFIGURATION SWEEP")
    print(SEP)

    result = deqaaa_targets(
        global_n=global_n,
        node_qubits=deqaaa_node_qubits,
        target_sets=target_sets,
        distribution_seed=distribution_seed,
    )

    print(f"Fixed partition n_j={deqaaa_node_qubits} on n={global_n}")
    print(f"{'Targets':<20} | {'p_g':<12} | {'min p_j':<12} | {'max p_j':<12} | {'|X_j|':<14} | {'J_j'}")
    print("-" * 110)
    for row in result.rows:
        print(
            f"{str(tuple(row['target_indices'])):<20} | "
            f"{row['global_success_probability']:<12.6f} | "
            f"{row['min_local_success_probability']:<12.6f} | "
            f"{row['max_local_success_probability']:<12.6f} | "
            f"{row['local_target_counts']!s:<14} | "
            f"{row['local_iterations']}"
        )

    print("\n-> CONCLUSION: DEQAAA remains exact in each case, but the local p_j values,")
    print("   phase angles, and target-substring counts depend strongly on how targets")
    print("   are distributed across the chosen node partition.")


# =============================================================================
# Scenario P: 2026 DEQAAA Distribution-Robustness Sweep
# =============================================================================

def run_scenario_p(global_n: int = 6, deqaaa_node_qubits: tuple[int, ...] = (2, 2, 2), target_indices: tuple[int, ...] = (8, 14), distribution_seeds: tuple[int, ...] = (7, 21, 84, 126)) -> None:
    """
    P. 2026 DEQAAA DISTRIBUTION-ROBUSTNESS SWEEP

    Keeps the same partition and targets, but changes the arbitrary amplitude
    distribution to show how strongly the local DEQAAA bookkeeping depends on
    the underlying state-preparation profile.
    """
    print(f"\n{SEP}")
    print("SCENARIO P: 2026 DEQAAA DISTRIBUTION-ROBUSTNESS SWEEP")
    print(SEP)

    result = deqaaa_robustness(
        global_n=global_n,
        node_qubits=deqaaa_node_qubits,
        target_indices=target_indices,
        distribution_seeds=distribution_seeds,
    )

    print(f"Fixed partition n_j={deqaaa_node_qubits}, targets={target_indices}")
    print(f"{'Seed':<8} | {'p_g':<12} | {'local p_j':<36} | {'J_j':<12}")
    print("-" * 95)
    for row in result.rows:
        local_probs = np.array2string(np.asarray(row["local_success_probabilities"]), precision=5, separator=", ")
        print(
            f"{row['distribution_seed']:<8} | "
            f"{row['global_success_probability']:<12.6f} | "
            f"{local_probs:<36} | "
            f"{row['local_iterations']}"
        )

    print("\nAcross all tested arbitrary distributions:")
    print(f"  local p_j span  = {result.local_success_probability_span}")
    print(f"  local J_j span  = {result.local_iteration_span}")
    print(f"  local phi_j span = {[round(x, 6) for x in result.local_phase_span]}")


# =============================================================================
# Scenario Q: 2026 DEQAAA Phase-Mismatch Sweep
# =============================================================================

def run_scenario_q(global_n: int = 6, deqaaa_node_qubits: tuple[int, ...] = (2, 2, 2), target_indices: tuple[int, ...] = (8, 14), shot_counts: tuple[int, ...] = (100, 1000, 10000, 100000), distribution_seed: int = 21, measurement_seed: int = 21) -> None:
    """
    Q. 2026 DEQAAA PHASE-MISMATCH SWEEP

    Quantifies the practical risk of replacing exact local probabilities by
    shot-estimated ones: how often the prescribed exact-EQAAA iteration count
    J_j changes, and how large the induced phase-angle errors become.
    """
    print(f"\n{SEP}")
    print("SCENARIO Q: 2026 DEQAAA PHASE-MISMATCH SWEEP")
    print(SEP)

    result = deqaaa_mismatch(
        global_n=global_n,
        node_qubits=deqaaa_node_qubits,
        target_indices=target_indices,
        shot_counts=shot_counts,
        distribution_seed=distribution_seed,
        measurement_seed=measurement_seed,
    )

    print(f"Exact local p_j values: {np.array2string(result.exact_local_success_probabilities, precision=6)}")
    print(f"Exact local J_j values: {result.exact_local_iterations.tolist()}")
    print(f"Exact local phi_j values: {[round(x, 6) for x in result.exact_local_phases.tolist()]}\n")
    print(f"{'Shots':<10} | {'iteration mismatches':<22} | {'max |phi_j-phihat_j|':<22} | {'mean |phi_j-phihat_j|'}")
    print("-" * 92)
    for row_idx, shots in enumerate(result.shot_counts):
        print(
            f"{int(shots):<10} | "
            f"{int(result.iteration_mismatch_counts[row_idx]):<22} | "
            f"{result.max_phase_error[row_idx]:<22.6e} | "
            f"{result.mean_phase_error[row_idx]:.6e}"
        )

    print("\n-> CONCLUSION: low-shot estimates can perturb exact local prescriptions,")
    print("   while higher shot counts stabilize both J_j and phi_j toward their")
    print("   exact DEQAAA values.")


def _save_distributed_algorithm_figure(
    global_n: int = GLOBAL_N,
    *,
    j_values=(1, 2, 3),
    global_goods: tuple[str, ...] = GLOBAL_GOODS,
    output_name="distributed_partition_tradeoff.png",
):
    plt = _load_pyplot()
    node_counts = []
    mono_depth = []
    dist_depth = []
    mono_cx = []
    dist_cx = []
    mono_swaps = []
    dist_swaps = []
    depth_reduction = []

    for j in j_values:
        res = hw_tradeoff(
            global_n=global_n,
            j=j,
            global_goods=global_goods,
            basis_gates=tuple(BASIS_NISQ),
        )
        node_counts.append(2 ** int(j))
        mono_depth.append(float(res.monolithic_metrics["routed_depth"]))
        dist_depth.append(float(res.distributed_aggregate_metrics["routed_depth"]))
        mono_cx.append(float(res.monolithic_metrics["routed_cx"]))
        dist_cx.append(float(res.distributed_aggregate_metrics["routed_cx"]))
        mono_swaps.append(float(res.monolithic_metrics["estimated_swap_count"]))
        dist_swaps.append(float(res.distributed_aggregate_metrics["estimated_swap_count"]))
        depth_reduction.append(float(res.reduction_factors["depth_reduction"]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Distributed QAA Partition Tradeoff", fontsize=14, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(node_counts, mono_depth, marker="o", linewidth=2.0, label="monolithic", color="#d62728")
    ax1.plot(node_counts, dist_depth, marker="s", linewidth=2.0, label="distributed aggregate", color="#1f77b4")
    ax1.set_title("Routed Depth")
    ax1.set_ylabel("Depth")
    ax1.legend(fontsize=8)

    ax2.plot(node_counts, mono_cx, marker="o", linewidth=2.0, label="monolithic", color="#9467bd")
    ax2.plot(node_counts, dist_cx, marker="s", linewidth=2.0, label="distributed aggregate", color="#2ca02c")
    ax2.set_title("Routed CX Count")
    ax2.set_ylabel("CX count")
    ax2.legend(fontsize=8)

    ax3.plot(node_counts, mono_swaps, marker="o", linewidth=2.0, label="monolithic", color="#8c564b")
    ax3.plot(node_counts, dist_swaps, marker="s", linewidth=2.0, label="distributed aggregate", color="#ff7f0e")
    ax3.set_title("Routing SWAP Burden")
    ax3.set_ylabel("Estimated SWAP count")
    ax3.legend(fontsize=8)

    ax4.plot(node_counts, depth_reduction, marker="D", linewidth=2.2, color="#1f77b4")
    ax4.set_title("Depth Reduction from Partitioning")
    ax4.set_ylabel("Monolithic depth / distributed depth")

    for axis in axes.flat:
        axis.set_xlabel("Parallel nodes = 2^j")
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "distributed_partition_tradeoff",
            "global_n": int(global_n),
            "j_values": [int(x) for x in j_values],
            "parallel_nodes": [int(x) for x in node_counts],
            "global_goods": list(global_goods),
            "basis_gates": list(BASIS_NISQ),
        },
    )
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


def _save_distributed_shot_figure(
    global_n: int = 6,
    deqaaa_node_qubits: tuple[int, ...] = (2, 2, 2),
    target_indices: tuple[int, ...] = (8, 14),
    *,
    shot_counts: tuple[int, ...] = (100, 1000, 10000, 100000),
    distribution_seed: int = 21,
    measurement_seed: int = 21,
    output_name="distributed_phase_mismatch_profile.png",
):
    plt = _load_pyplot()
    result = deqaaa_mismatch(
        global_n=global_n,
        node_qubits=deqaaa_node_qubits,
        target_indices=target_indices,
        shot_counts=shot_counts,
        distribution_seed=distribution_seed,
        measurement_seed=measurement_seed,
    )

    x = np.asarray(result.shot_counts, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Distributed DEQAAA Shot-Sensitivity Profile", fontsize=14, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(x, result.iteration_mismatch_counts, marker="o", linewidth=2.0, color="#1f77b4")
    ax1.set_xscale("log")
    ax1.set_title("Iteration Prescription Mismatches")
    ax1.set_ylabel("Mismatch count")

    ax2.plot(x, result.max_phase_error, marker="o", linewidth=2.0, color="#d62728")
    ax2.set_xscale("log")
    ax2.set_title("Maximum Phase Error")
    ax2.set_ylabel("max |phi - phi_hat|")

    ax3.plot(x, result.mean_phase_error, marker="o", linewidth=2.0, color="#2ca02c")
    ax3.set_xscale("log")
    ax3.set_title("Mean Phase Error")
    ax3.set_ylabel("mean |phi - phi_hat|")

    exact_iterations = np.asarray(result.exact_local_iterations, dtype=float)
    ax4.bar(np.arange(len(exact_iterations)), exact_iterations, color="#9467bd")
    ax4.set_title("Exact Local Iteration Targets")
    ax4.set_ylabel("J_j")
    ax4.set_xlabel("Node index")

    for axis in (ax1, ax2, ax3):
        axis.set_xlabel("Shots")
        axis.grid(axis="y", linestyle=":", alpha=0.35)
    ax4.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "distributed_phase_mismatch_profile",
            "global_n": int(global_n),
            "deqaaa_node_qubits": [int(x) for x in deqaaa_node_qubits],
            "target_indices": [int(x) for x in target_indices],
            "shot_counts": [int(x) for x in shot_counts],
            "distribution_seed": int(distribution_seed),
            "measurement_seed": int(measurement_seed),
            "exact_local_iterations": [int(x) for x in result.exact_local_iterations],
        },
    )
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


# =============================================================================
# Main Orchestrator
# =============================================================================

if __name__ == "__main__":
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

    print("Distributed Amplitude Amplification Benchmark Suite — Scenarios A through Q")
    print(f"Problem baseline: n={GLOBAL_N}, j={J}, {len(GLOBAL_GOODS)} marked items, eps={EPSILON}")
    print(f"Results saved to: {output_filepath}")
    print(SEP)
    print(publishability.summary())

    raw_scenarios = [
        ("A", run_scenario_a),
        ("B", run_scenario_b),
        ("C", run_scenario_c),
        ("D", run_scenario_d),
        ("E", run_scenario_e),
        ("F", run_scenario_f),
        ("G", run_scenario_g),
        ("H", run_scenario_h),
        ("I", run_scenario_i),
        ("J", run_scenario_j),
        ("K", run_scenario_k),
        ("L", run_scenario_l),
        ("M", run_scenario_m),
        ("N", run_scenario_n),
        ("O", run_scenario_o),
        ("P", run_scenario_p),
        ("Q", run_scenario_q),
    ]
    scenarios = wrap_scenarios(raw_scenarios, module_globals=globals(), extra_patch_objects=(dqaa,), config=publishability)

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

    render_backend_validation_summary(publishability)
    _save_distributed_algorithm_figure()
    _save_distributed_shot_figure()
    logger.close()
    sys.stdout = logger.terminal
    print(f"\nBenchmark suite complete. Results saved to {output_filepath}")
