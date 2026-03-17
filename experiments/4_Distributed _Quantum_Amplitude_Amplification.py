"""Distributed lucky-node verification for Theorem 3.

This module numerically validates the convexity guarantee used by distributed
quantum amplitude amplification:

    max_k p_k >= p

where:
- p is the global success probability in a 2^n search space,
- p_k is the local success probability in node k after partitioning by j
  prefix qubits into 2^j nodes.

The primary evidence artifact is a bar chart of all local p_k values with a
horizontal line at the global average p.

Compatibility note:
- Dataclass fields keep historical names ``global_a`` and ``local_ak`` for API
  stability; they represent global ``p`` and local ``p_k`` respectively.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sympy as sp
from one_click_utils import start_one_click_session

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Operator, Statevector, entropy, partial_trace
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore[assignment]
    Statevector = None  # type: ignore[assignment]
    DensityMatrix = None  # type: ignore[assignment]
    Operator = None  # type: ignore[assignment]
    partial_trace = None  # type: ignore[assignment]
    entropy = None  # type: ignore[assignment]
    transpile = None  # type: ignore[assignment]
    CouplingMap = None  # type: ignore[assignment]

try:
    from qiskit.circuit.library import PhaseOracleGate
except Exception:  # pragma: no cover
    PhaseOracleGate = None  # type: ignore[assignment]

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
except Exception:  # pragma: no cover
    AerSimulator = None  # type: ignore[assignment]
    NoiseModel = None  # type: ignore[assignment]

try:
    from qiskit.providers.fake_provider import GenericBackendV2
except Exception:  # pragma: no cover
    GenericBackendV2 = None  # type: ignore[assignment]


@dataclass
class LuckyNodeInstanceResult:
    """Results for one random global-search instance.

    ``global_a`` and ``local_ak`` are legacy field names carrying p and p_k.
    """

    n: int
    j: int
    num_good: int
    seed: Optional[int]
    global_a: float
    local_good_counts: np.ndarray
    local_ak: np.ndarray
    lucky_indices: list[int]
    max_ak: float
    max_gap: float
    theorem_holds: bool


@dataclass
class LuckyNodeMonteCarloResult:
    """Aggregate results over many random H_Good-subspace distributions.

    Gap statistics are computed for max_k(p_k - p), stored in legacy names.
    """

    n: int
    j: int
    num_good: int
    trials: int
    seed: Optional[int]
    max_gap_samples: np.ndarray
    lucky_count_samples: np.ndarray
    violation_count: int
    min_gap: float
    mean_gap: float
    std_gap: float
    max_gap: float


@dataclass
class DistributedFPAAResults:
    """Results for distributed FPAA execution (Algorithm 3 proof-of-concept).

    ``global_a`` is a legacy field name for global success probability p.
    """

    global_n: int
    j: int
    local_n: int
    global_goods: list[str]
    node_targets: dict[str, list[str]]
    global_a: float
    epsilon: float
    l_opt: int
    L: int
    alphas: np.ndarray
    betas: np.ndarray
    node_probabilities: dict[str, np.ndarray]
    node_good_success: dict[str, float]
    node_top_suffix: dict[str, str]
    node_top_probability: dict[str, float]
    lucky_nodes: list[str]
    unlucky_nodes: list[str]


@dataclass
class HardwareCompilationResults:
    """Compilation-resource comparison for monolithic vs distributed designs."""

    global_n: int
    j: int
    local_n: int
    global_goods: list[str]
    node_targets: dict[str, list[str]]
    basis_gates: list[str]
    topology: str
    monolithic_metrics: dict[str, float]
    distributed_node_metrics: dict[str, dict[str, float]]
    distributed_aggregate_metrics: dict[str, float]
    reduction_factors: dict[str, float]


@dataclass
class EntanglementObstructionResults:
    """Negative-proof results: local FPAA under separable vs entangled preparation."""

    global_n: int
    j: int
    local_n: int
    good_global: str
    good_suffix: str
    global_a_for_schedule: float
    epsilon: float
    L: int
    k_values: np.ndarray
    separable_good_probs: np.ndarray
    entangled_good_probs: np.ndarray
    separable_suffix_purity: float
    entangled_suffix_purity: float
    entangled_suffix_entropy_bits: float
    obstruction_ratio_peak: float


@dataclass
class NISQNoiseResults:
    """Density-matrix benchmark under hardware-calibrated noise."""

    backend_name: str
    backend_source: str
    global_n: int
    j: int
    local_n: int
    global_goods: list[str]
    node_targets: dict[str, list[str]]
    epsilon: float
    global_a_for_schedule: float
    L: int
    k_values: np.ndarray
    shots: int
    monolithic_ideal_success: np.ndarray
    monolithic_noisy_success: np.ndarray
    distributed_ideal_success: np.ndarray
    distributed_noisy_success: np.ndarray
    distributed_node_noisy_success: dict[str, np.ndarray]
    monolithic_random_baseline: float
    distributed_random_baseline: float


@dataclass
class OraclePartitionNodeMetrics:
    """Per-prefix compiler/synthesis metrics for one distributed sub-oracle."""

    prefix: str
    simplified_formula: str
    is_trivial: bool
    active_variable_count: int
    classical_compile_time_sec: float
    transpile_time_sec: float
    total_gates: int
    cx_gates: int
    depth: int
    estimated_swap_count: int


@dataclass
class CompilerResourceResults:
    """Module-6 results: automated AST partitioning + resource comparison."""

    global_n: int
    j: int
    local_n: int
    formula_format: str
    formula_text: str
    basis_gates: list[str]
    monolithic_compile_time_sec: float
    monolithic_transpile_time_sec: float
    monolithic_total_gates: int
    monolithic_cx_gates: int
    monolithic_depth: int
    monolithic_swap_count: int
    distributed_total_compile_time_sec: float
    distributed_total_transpile_time_sec: float
    distributed_node_metrics: dict[str, OraclePartitionNodeMetrics]
    distributed_avg_total_gates: float
    distributed_max_total_gates: int
    distributed_sum_total_gates: int
    gate_reduction_vs_avg: float
    gate_reduction_vs_max: float
    gate_reduction_vs_sum: float
    classical_time_ratio_dist_over_mono: float
    trivial_prefixes: list[str]


@dataclass
class NetworkStatisticsResults:
    """Module-7 end-to-end classical master-node extraction under finite shots."""

    global_n: int
    j: int
    local_n: int
    global_goods: list[str]
    node_targets: dict[str, list[str]]
    epsilon: float
    global_a_for_schedule: float
    L: int
    shots_per_node: int
    total_shots: int
    uniform_mean_per_state: float
    uniform_std_per_state: float
    sifting_sigma: float
    sifting_threshold: int
    node_counts: dict[str, dict[str, int]]
    flagged_candidates: list[str]
    verified_answers: list[str]
    false_positives: list[str]
    classical_queries_made: int


def _require_qiskit() -> None:
    if QuantumCircuit is None or Statevector is None:
        raise RuntimeError("Qiskit is required for distributed FPAA execution.")


def _require_qiskit_density_matrix() -> None:
    if QuantumCircuit is None or DensityMatrix is None or Operator is None or partial_trace is None:
        raise RuntimeError("Qiskit DensityMatrix tools are required for entanglement obstruction experiment.")


def _require_aer_noise() -> None:
    if AerSimulator is None or NoiseModel is None:
        raise RuntimeError("qiskit-aer with noise support is required for NISQ noise benchmark.")


def _require_aer_simulator() -> None:
    if AerSimulator is None:
        raise RuntimeError("qiskit-aer is required for network shot-noise benchmark.")


def _require_oracle_synthesis() -> None:
    if PhaseOracleGate is None:
        raise RuntimeError("PhaseOracleGate is required for automated oracle partitioning benchmark.")


def _require_qiskit_transpiler() -> None:
    if QuantumCircuit is None or transpile is None or CouplingMap is None:
        raise RuntimeError("Qiskit transpiler is required for hardware compilation trade-off experiment.")


def _validate_inputs(n: int, j: int, num_good: int, trials: Optional[int] = None) -> None:
    if n < 1:
        raise ValueError("n must be >= 1.")
    if j < 0 or j > n:
        raise ValueError("j must satisfy 0 <= j <= n.")
    if num_good < 1 or num_good > 2**n:
        raise ValueError("num_good must satisfy 1 <= num_good <= 2^n.")
    if trials is not None and trials < 1:
        raise ValueError("trials must be >= 1.")


def _compute_local_good_counts(good_states: np.ndarray, n: int, j: int) -> np.ndarray:
    num_nodes = 2**j
    suffix_bits = n - j
    counts = np.zeros(num_nodes, dtype=int)
    for state in good_states:
        node = int(state) >> suffix_bits
        counts[node] += 1
    return counts


def _generate_fpaa_phases(L: int, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytical FPAA schedule used identically by all distributed nodes."""
    if L < 1 or L % 2 == 0:
        raise ValueError("L must be a positive odd integer.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta/epsilon must be in (0, 1).")

    gamma_inv = float(np.cosh((1.0 / L) * np.arccosh(1.0 / delta)))
    gamma = 1.0 / gamma_inv
    sq_term = float(np.sqrt(max(0.0, 1.0 - gamma * gamma)))

    tol = 1e-12
    alpha = np.zeros(L, dtype=float)
    for idx in range(1, L + 1):
        theta = (2.0 * np.pi * idx) / L
        tan_val = 0.0 if np.isclose(np.sin(theta), 0.0, atol=tol) else float(np.tan(theta))
        denom = tan_val * sq_term
        alpha[idx - 1] = np.pi if np.isclose(denom, 0.0, atol=tol) else 2.0 * float(np.arctan2(1.0, denom))

    beta = -alpha[::-1]
    return alpha, beta


def _partition_targets_by_prefix(global_goods: list[str], j: int) -> dict[str, list[str]]:
    """Map global n-bit targets into node-local suffix targets keyed by j-bit prefix."""
    if not global_goods:
        raise ValueError("global_goods must not be empty.")
    n = len(global_goods[0])
    num_nodes = 2**j

    out: dict[str, list[str]] = {format(k, f"0{j}b"): [] for k in range(num_nodes)}
    for bitstring in global_goods:
        if len(bitstring) != n or any(ch not in "01" for ch in bitstring):
            raise ValueError("All global targets must be binary strings with identical length.")
        prefix, suffix = bitstring[:j], bitstring[j:]
        out[prefix].append(suffix)
    return out


def _build_local_oracle(local_n: int, targets: list[str], beta_phase: float) -> "QuantumCircuit":
    """S_t(beta) for one node-local oracle f_k."""
    _require_qiskit()
    qc = QuantumCircuit(local_n, name=f"S_fk({beta_phase:.3f})")
    if not targets:
        return qc

    controls = list(range(local_n - 1))
    target = local_n - 1
    for bitstring in targets:
        rev = bitstring[::-1]  # little-endian qubit mapping
        for q, bit in enumerate(rev):
            if bit == "0":
                qc.x(q)
        if local_n == 1:
            qc.p(beta_phase, 0)
        else:
            qc.mcp(beta_phase, controls, target)
        for q, bit in enumerate(rev):
            if bit == "0":
                qc.x(q)
    return qc


def _build_local_diffusion(local_n: int, alpha_phase: float) -> "QuantumCircuit":
    """S_s(alpha) diffusion about uniform superposition in node-local register."""
    _require_qiskit()
    qc = QuantumCircuit(local_n, name=f"S_s({alpha_phase:.3f})")
    qc.h(range(local_n))
    qc.x(range(local_n))
    if local_n == 1:
        qc.p(alpha_phase, 0)
    else:
        qc.mcp(alpha_phase, list(range(local_n - 1)), local_n - 1)
    qc.x(range(local_n))
    qc.h(range(local_n))
    return qc


def _build_phase_oracle(num_qubits: int, good_bitstrings: list[str], phase: float = np.pi) -> "QuantumCircuit":
    """Phase oracle that marks all listed basis states with e^{i*phase}."""
    _require_qiskit()
    qc = QuantumCircuit(num_qubits, name="oracle")
    if not good_bitstrings:
        return qc

    controls = list(range(num_qubits - 1))
    target = num_qubits - 1
    for bitstring in good_bitstrings:
        if len(bitstring) != num_qubits or any(ch not in "01" for ch in bitstring):
            raise ValueError("Invalid H_Good bitstring for oracle.")
        rev = bitstring[::-1]
        for q, bit in enumerate(rev):
            if bit == "0":
                qc.x(q)
        if num_qubits == 1:
            qc.p(phase, 0)
        else:
            qc.mcp(phase, controls, target)
        for q, bit in enumerate(rev):
            if bit == "0":
                qc.x(q)
    return qc


def _build_grover_diffusion(num_qubits: int, phase: float = np.pi) -> "QuantumCircuit":
    """Standard Grover diffusion operator with generalized phase."""
    _require_qiskit()
    qc = QuantumCircuit(num_qubits, name="diffusion")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.p(phase, 0)
    else:
        qc.mcp(phase, list(range(num_qubits - 1)), num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def _build_grover_step(num_qubits: int, good_bitstrings: list[str]) -> "QuantumCircuit":
    """One Grover iteration: oracle then diffusion from uniform superposition."""
    _require_qiskit()
    qc = QuantumCircuit(num_qubits, name=f"GroverStep_n{num_qubits}")
    qc.h(range(num_qubits))
    qc.append(_build_phase_oracle(num_qubits, good_bitstrings, phase=np.pi).to_gate(), range(num_qubits))
    qc.append(_build_grover_diffusion(num_qubits, phase=np.pi).to_gate(), range(num_qubits))
    return qc


def _extract_compilation_metrics(
    qc: "QuantumCircuit",
    coupling_map: "CouplingMap",
    basis_gates: list[str],
    seed_transpiler: int = 42,
) -> dict[str, float]:
    """Compile under connectivity constraints and return routing/depth metrics."""
    _require_qiskit_transpiler()

    logical = transpile(
        qc,
        basis_gates=basis_gates,
        optimization_level=3,
        seed_transpiler=seed_transpiler,
    )
    routed_native = transpile(
        qc,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        optimization_level=3,
        seed_transpiler=seed_transpiler,
    )
    # Keep SWAP as explicit instruction for routing-overhead accounting.
    routed_swap_view = transpile(
        qc,
        basis_gates=basis_gates + ["swap"],
        coupling_map=coupling_map,
        optimization_level=3,
        seed_transpiler=seed_transpiler,
    )

    ops_logical = logical.count_ops()
    ops_native = routed_native.count_ops()
    ops_swap_view = routed_swap_view.count_ops()

    logical_cx = int(ops_logical.get("cx", 0))
    routed_cx = int(ops_native.get("cx", 0))
    swap_count = int(ops_swap_view.get("swap", 0))
    single_q = int(sum(v for k, v in ops_native.items() if k != "cx"))

    return {
        "qubits": float(qc.num_qubits),
        "logical_depth_no_cmap": float(logical.depth()),
        "routed_depth": float(routed_native.depth()),
        "logical_cx_no_cmap": float(logical_cx),
        "routed_cx": float(routed_cx),
        "routed_total_gates": float(sum(ops_native.values())),
        "routed_single_qubit": float(single_q),
        "estimated_swap_count": float(swap_count),
        "estimated_routing_cx_overhead": float(max(0, routed_cx - logical_cx)),
    }


def experiment_lucky_node_verification(
    n: int = 8,
    j: int = 3,
    num_good: int = 12,
    seed: Optional[int] = 42,
) -> LuckyNodeInstanceResult:
    """Run one theorem-check instance and return all local/global statistics."""
    _validate_inputs(n, j, num_good)

    rng = np.random.default_rng(seed)
    n_global = 2**n
    n_local = 2 ** (n - j)

    good_states = rng.choice(n_global, size=num_good, replace=False)
    global_a = float(num_good / n_global)

    local_counts = _compute_local_good_counts(good_states, n=n, j=j)
    local_ak = local_counts / n_local

    max_ak = float(np.max(local_ak))
    max_gap = float(max_ak - global_a)
    lucky_indices = np.where(local_ak >= global_a)[0].astype(int).tolist()
    theorem_holds = bool(max_gap >= -1e-14)

    return LuckyNodeInstanceResult(
        n=n,
        j=j,
        num_good=num_good,
        seed=seed,
        global_a=global_a,
        local_good_counts=local_counts,
        local_ak=local_ak,
        lucky_indices=lucky_indices,
        max_ak=max_ak,
        max_gap=max_gap,
        theorem_holds=theorem_holds,
    )


def experiment_distributed_fpaa_execution(
    global_n: int = 6,
    j: int = 2,
    global_goods: tuple[str, ...] = ("110110", "111111", "011001"),
    epsilon: float = 0.3,
) -> DistributedFPAAResults:
    """Exact Hua-Qiu style distributed FPAA simulation with 4 independent nodes."""
    _require_qiskit()
    if global_n < 2:
        raise ValueError("global_n must be >= 2.")
    if j <= 0 or j >= global_n:
        raise ValueError("j must satisfy 1 <= j < global_n.")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0, 1).")

    targets = [s.strip() for s in global_goods if s.strip()]
    if not targets:
        raise ValueError("global_goods must contain at least one bitstring.")
    if any(len(t) != global_n for t in targets):
        raise ValueError("Every global target must have length global_n.")

    local_n = global_n - j
    node_targets = _partition_targets_by_prefix(targets, j=j)
    lucky_nodes = sorted([prefix for prefix, suffixes in node_targets.items() if len(suffixes) > 0])
    unlucky_nodes = sorted([prefix for prefix, suffixes in node_targets.items() if len(suffixes) == 0])

    global_a = float(len(targets) / (2**global_n))

    # The schedule length is derived from global p and epsilon (same for every node).
    l_opt = int(np.ceil((0.5 * np.log(2.0 / epsilon)) / np.sqrt(global_a)))
    L = int(2 * l_opt + 1)
    alphas, betas = _generate_fpaa_phases(L=L, delta=epsilon)

    node_probabilities: dict[str, np.ndarray] = {}
    node_good_success: dict[str, float] = {}
    node_top_suffix: dict[str, str] = {}
    node_top_probability: dict[str, float] = {}

    for prefix in sorted(node_targets.keys()):
        local_targets = node_targets[prefix]
        qc = QuantumCircuit(local_n, name=f"node_{prefix}")
        qc.h(range(local_n))

        for a_j, b_j in zip(alphas, betas):
            qc.global_phase += np.pi
            qc.append(_build_local_oracle(local_n, local_targets, float(b_j)).to_gate(), range(local_n))
            qc.append(_build_local_diffusion(local_n, float(a_j)).to_gate(), range(local_n))

        state = Statevector.from_instruction(qc)
        probs = np.abs(state.data) ** 2
        node_probabilities[prefix] = probs

        if local_targets:
            good_idxs = [int(sfx, 2) for sfx in local_targets]
            node_good_success[prefix] = float(np.sum(probs[good_idxs]))
        else:
            node_good_success[prefix] = 0.0

        top_idx = int(np.argmax(probs))
        node_top_suffix[prefix] = format(top_idx, f"0{local_n}b")
        node_top_probability[prefix] = float(probs[top_idx])

    return DistributedFPAAResults(
        global_n=global_n,
        j=j,
        local_n=local_n,
        global_goods=targets,
        node_targets=node_targets,
        global_a=global_a,
        epsilon=epsilon,
        l_opt=l_opt,
        L=L,
        alphas=alphas,
        betas=betas,
        node_probabilities=node_probabilities,
        node_good_success=node_good_success,
        node_top_suffix=node_top_suffix,
        node_top_probability=node_top_probability,
        lucky_nodes=lucky_nodes,
        unlucky_nodes=unlucky_nodes,
    )


def experiment_hardware_compilation_tradeoff(
    global_n: int = 6,
    j: int = 2,
    global_goods: tuple[str, ...] = ("110110", "111111", "011001"),
    basis_gates: tuple[str, ...] = ("cx", "id", "rz", "sx", "x"),
    seed_transpiler: int = 42,
) -> HardwareCompilationResults:
    """Benchmark NISQ compilation cost for monolithic vs distributed architecture."""
    _require_qiskit_transpiler()
    if global_n < 2:
        raise ValueError("global_n must be >= 2.")
    if j <= 0 or j >= global_n:
        raise ValueError("j must satisfy 1 <= j < global_n.")

    targets = [s.strip() for s in global_goods if s.strip()]
    if not targets:
        raise ValueError("global_goods must not be empty.")
    if any(len(t) != global_n for t in targets):
        raise ValueError("All global targets must be global_n-bit strings.")

    local_n = global_n - j
    node_targets = _partition_targets_by_prefix(targets, j=j)
    basis = list(basis_gates)

    cmap_global = CouplingMap.from_line(global_n)
    cmap_local = CouplingMap.from_line(local_n)

    # Monolithic: full n-qubit oracle with the exact H_Good set.
    qc_monolithic = _build_grover_step(global_n, targets)
    mono_metrics = _extract_compilation_metrics(
        qc_monolithic,
        coupling_map=cmap_global,
        basis_gates=basis,
        seed_transpiler=seed_transpiler,
    )

    # Distributed: compile each node-local oracle separately.
    node_metrics: dict[str, dict[str, float]] = {}
    for prefix in sorted(node_targets.keys()):
        qc_local = _build_grover_step(local_n, node_targets[prefix])
        node_metrics[prefix] = _extract_compilation_metrics(
            qc_local,
            coupling_map=cmap_local,
            basis_gates=basis,
            seed_transpiler=seed_transpiler,
        )

    # Aggregate distributed costs as "parallel critical-path node" (max per metric).
    keys = list(next(iter(node_metrics.values())).keys())
    aggregate: dict[str, float] = {}
    for key in keys:
        aggregate[key] = float(max(m[key] for m in node_metrics.values()))

    def _safe_factor(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("inf")

    factors = {
        "qubit_reduction": float(global_n / local_n),
        "depth_reduction": _safe_factor(mono_metrics["routed_depth"], aggregate["routed_depth"]),
        "cx_reduction": _safe_factor(mono_metrics["routed_cx"], aggregate["routed_cx"]),
        "swap_reduction": _safe_factor(mono_metrics["estimated_swap_count"], aggregate["estimated_swap_count"]),
        "routing_cx_overhead_reduction": _safe_factor(
            mono_metrics["estimated_routing_cx_overhead"],
            aggregate["estimated_routing_cx_overhead"],
        ),
    }

    return HardwareCompilationResults(
        global_n=global_n,
        j=j,
        local_n=local_n,
        global_goods=targets,
        node_targets=node_targets,
        basis_gates=basis,
        topology=f"line_{global_n} vs line_{local_n}",
        monolithic_metrics=mono_metrics,
        distributed_node_metrics=node_metrics,
        distributed_aggregate_metrics=aggregate,
        reduction_factors=factors,
    )


# Fixed ansatz parameters (seeded offline) yielding strong cross-register entanglement
# and a clear obstruction signal after tracing out prefix qubits.
_ENTANGLED_ANSATZ_PARAMS_N6 = np.array(
    [
        0.35948351,
        3.51275926,
        4.75030001,
        3.34186162,
        3.66308334,
        5.38382321,
        4.40029813,
        1.84351236,
        2.84175945,
        2.48329120,
        1.82946302,
        6.25486990,
        2.90781066,
        0.32142623,
        4.49088589,
        1.47390107,
        4.45853018,
        5.47827743,
    ],
    dtype=float,
)


def _build_separable_node_preparation(global_n: int, j: int, node_prefix: str) -> "QuantumCircuit":
    """Prepare A_sep = |node_prefix>_prefix ⊗ |+...+>_suffix."""
    _require_qiskit()
    local_n = global_n - j
    if len(node_prefix) != j or any(ch not in "01" for ch in node_prefix):
        raise ValueError("node_prefix must be a j-bit binary string.")

    qc = QuantumCircuit(global_n, name="A_separable")
    qc.h(range(local_n))
    # node_prefix is given most-significant first; map into qubits [global_n-1 ... local_n].
    for offset, bit in enumerate(node_prefix):
        if bit == "1":
            qubit = global_n - 1 - offset
            qc.x(qubit)
    return qc


def _build_entangled_cross_register_ansatz(global_n: int, j: int) -> "QuantumCircuit":
    """Hardware-efficient cross-register ansatz that entangles prefix and suffix."""
    _require_qiskit()
    if not (global_n == 6 and j == 2):
        raise ValueError("Entanglement obstruction ansatz is currently defined for global_n=6, j=2.")

    params = _ENTANGLED_ANSATZ_PARAMS_N6
    qc = QuantumCircuit(global_n, name="A_entangled")

    for q in range(global_n):
        qc.ry(float(params[q]), q)
    # Cross-register entanglers (prefix: qubits 4,5; suffix: qubits 0..3)
    qc.cx(4, 1)
    qc.cx(5, 2)
    qc.cx(0, 4)
    qc.cx(3, 5)
    qc.cz(4, 0)
    qc.cz(5, 3)
    for q in range(global_n):
        qc.rz(float(params[global_n + q]), q)
    qc.cx(4, 2)
    qc.cx(5, 1)
    for q in range(global_n):
        qc.ry(float(params[2 * global_n + q]), q)
    return qc


def _local_fpaa_density_curve(
    rho0: "DensityMatrix",
    good_suffix: str,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Apply local FPAA schedule to a suffix density matrix and return target curve."""
    _require_qiskit_density_matrix()
    local_n = int(np.log2(rho0.data.shape[0]))
    if len(good_suffix) != local_n or any(ch not in "01" for ch in good_suffix):
        raise ValueError("good_suffix must match local register width.")
    if len(alphas) != len(betas):
        raise ValueError("alphas and betas must have same length.")

    good_idx = int(good_suffix, 2)
    proj = np.zeros((2**local_n, 2**local_n), dtype=complex)
    proj[good_idx, good_idx] = 1.0

    rho = np.array(rho0.data, dtype=complex)
    out = np.zeros(len(alphas) + 1, dtype=float)
    out[0] = float(np.real(np.trace(proj @ rho)))

    for step, (alpha, beta) in enumerate(zip(alphas, betas), start=1):
        iterate = QuantumCircuit(local_n)
        iterate.global_phase += np.pi
        iterate.append(_build_local_oracle(local_n, [good_suffix], float(beta)).to_gate(), range(local_n))
        iterate.append(_build_local_diffusion(local_n, float(alpha)).to_gate(), range(local_n))
        u = Operator(iterate).data
        rho = u @ rho @ u.conj().T
        out[step] = float(np.real(np.trace(proj @ rho)))

    return out


def experiment_entanglement_obstruction(
    global_n: int = 6,
    j: int = 2,
    good_global: str = "110110",
    epsilon: float = 0.3,
    global_a_for_schedule: float = 3.0 / 64.0,
    L: Optional[int] = None,
) -> EntanglementObstructionResults:
    """Negative proof that DQAA local reflections fail for non-separable preparation."""
    _require_qiskit_density_matrix()
    if global_n < 2:
        raise ValueError("global_n must be >= 2.")
    if j <= 0 or j >= global_n:
        raise ValueError("j must satisfy 1 <= j < global_n.")
    if len(good_global) != global_n or any(ch not in "01" for ch in good_global):
        raise ValueError("good_global must be a binary string of length global_n.")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0, 1).")
    if not (0.0 < global_a_for_schedule <= 1.0):
        raise ValueError("global_a_for_schedule must be in (0, 1].")

    local_n = global_n - j
    node_prefix = good_global[:j]
    good_suffix = good_global[j:]

    if L is None:
        l_opt = int(np.ceil((0.5 * np.log(2.0 / epsilon)) / np.sqrt(global_a_for_schedule)))
        L = int(2 * l_opt + 1)
    if L < 1 or L % 2 == 0:
        raise ValueError("L must be a positive odd integer.")

    alphas, betas = _generate_fpaa_phases(L=L, delta=epsilon)

    # Valid DQAA precondition: separable state across prefix/suffix cut.
    A_sep = _build_separable_node_preparation(global_n=global_n, j=j, node_prefix=node_prefix)
    sv_sep = Statevector.from_instruction(A_sep)
    rho_sep_local = partial_trace(DensityMatrix(sv_sep), list(range(local_n, global_n)))

    # Invalid case: hardware-efficient ansatz with explicit cross-register entanglement.
    A_ent = _build_entangled_cross_register_ansatz(global_n=global_n, j=j)
    sv_ent = Statevector.from_instruction(A_ent)
    rho_ent_local = partial_trace(DensityMatrix(sv_ent), list(range(local_n, global_n)))

    sep_curve = _local_fpaa_density_curve(rho_sep_local, good_suffix=good_suffix, alphas=alphas, betas=betas)
    ent_curve = _local_fpaa_density_curve(rho_ent_local, good_suffix=good_suffix, alphas=alphas, betas=betas)

    sep_purity = float(np.real(np.trace(rho_sep_local.data @ rho_sep_local.data)))
    ent_purity = float(np.real(np.trace(rho_ent_local.data @ rho_ent_local.data)))
    ent_entropy = float(entropy(rho_ent_local, base=2)) if entropy is not None else float("nan")
    obstruction_ratio_peak = float(np.max(sep_curve) / max(1e-15, np.max(ent_curve)))

    return EntanglementObstructionResults(
        global_n=global_n,
        j=j,
        local_n=local_n,
        good_global=good_global,
        good_suffix=good_suffix,
        global_a_for_schedule=float(global_a_for_schedule),
        epsilon=epsilon,
        L=L,
        k_values=np.arange(L + 1, dtype=int),
        separable_good_probs=sep_curve,
        entangled_good_probs=ent_curve,
        separable_suffix_purity=sep_purity,
        entangled_suffix_purity=ent_purity,
        entangled_suffix_entropy_bits=ent_entropy,
        obstruction_ratio_peak=obstruction_ratio_peak,
    )


def _load_hardware_noise_backend(
    preferred_backend: Optional[str],
    min_qubits: int,
    seed: int = 42,
) -> tuple[Any, str]:
    """Load IBM fake backend if available; otherwise fallback to GenericBackendV2."""
    _require_qiskit_transpiler()

    candidate_names = []
    if preferred_backend:
        candidate_names.append(preferred_backend.strip())
    candidate_names.extend(["FakeGuadalupeV2", "FakeGuadalupe", "FakeManilaV2", "FakeManila"])

    providers = [
        "qiskit_ibm_runtime.fake_provider",
        "qiskit.providers.fake_provider",
    ]

    for module_name in providers:
        try:
            mod = __import__(module_name, fromlist=["*"])
        except Exception:
            continue
        for cls_name in candidate_names:
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            try:
                backend = cls()
            except Exception:
                continue
            n_qubits = int(getattr(backend, "num_qubits", 0) or 0)
            if n_qubits >= min_qubits:
                return backend, f"{module_name}.{cls_name}"

    if GenericBackendV2 is None:
        raise RuntimeError(
            "No IBM fake backend found and GenericBackendV2 is unavailable. "
            "Install qiskit-ibm-runtime for FakeGuadalupe/FakeManila or upgrade qiskit."
        )

    # Fallback still includes realistic per-qubit T1/T2, gate and readout error metadata.
    n_qubits = max(16, min_qubits)
    basis = ["cx", "id", "rz", "sx", "x"]
    backend = GenericBackendV2(
        n_qubits,
        basis_gates=basis,
        coupling_map=CouplingMap.from_line(n_qubits),
        seed=seed,
        noise_info=True,
    )
    return backend, "qiskit.providers.fake_provider.GenericBackendV2(fallback)"


def experiment_nisq_noise_resilience(
    global_n: int = 6,
    j: int = 2,
    global_goods: tuple[str, ...] = ("110110", "111111", "011001"),
    epsilon: float = 0.3,
    global_a_for_schedule: float = 3.0 / 64.0,
    L: Optional[int] = None,
    k_max: int = 7,
    shots: int = 4096,
    preferred_backend: Optional[str] = "FakeGuadalupeV2",
    seed: int = 42,
) -> NISQNoiseResults:
    """Density-matrix benchmark: monolithic collapse vs distributed survival under noise."""
    _require_qiskit_transpiler()
    _require_aer_noise()

    if global_n < 2:
        raise ValueError("global_n must be >= 2.")
    if j <= 0 or j >= global_n:
        raise ValueError("j must satisfy 1 <= j < global_n.")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0, 1).")
    if not (0.0 < global_a_for_schedule <= 1.0):
        raise ValueError("global_a_for_schedule must be in (0, 1].")
    if k_max < 0:
        raise ValueError("k_max must be >= 0.")
    if shots < 256:
        raise ValueError("shots must be >= 256 for stable statistics.")

    targets = [s.strip() for s in global_goods if s.strip()]
    if not targets:
        raise ValueError("global_goods must not be empty.")
    if any(len(s) != global_n or any(ch not in "01" for ch in s) for s in targets):
        raise ValueError("All global_goods must be binary strings of length global_n.")

    local_n = global_n - j
    node_targets = _partition_targets_by_prefix(targets, j=j)

    if L is None:
        l_opt = int(np.ceil((0.5 * np.log(2.0 / epsilon)) / np.sqrt(global_a_for_schedule)))
        L = int(2 * l_opt + 1)
    if L < 1 or L % 2 == 0:
        raise ValueError("L must be a positive odd integer.")

    alphas, betas = _generate_fpaa_phases(L=L, delta=epsilon)
    k_stop = int(min(k_max, L))
    k_values = np.arange(k_stop + 1, dtype=int)

    backend, backend_source = _load_hardware_noise_backend(
        preferred_backend=preferred_backend,
        min_qubits=global_n,
        seed=seed,
    )
    noise_model = NoiseModel.from_backend(backend)

    ideal_sim = AerSimulator(method="density_matrix", seed_simulator=seed)
    noisy_sim = AerSimulator(method="density_matrix", noise_model=noise_model, seed_simulator=seed)

    mono_ideal = np.zeros(len(k_values), dtype=float)
    mono_noisy = np.zeros(len(k_values), dtype=float)
    dist_ideal = np.zeros(len(k_values), dtype=float)
    dist_noisy = np.zeros(len(k_values), dtype=float)
    dist_node_noisy: dict[str, np.ndarray] = {
        prefix: np.zeros(len(k_values), dtype=float) for prefix in sorted(node_targets.keys())
    }

    for idx, k in enumerate(k_values):
        # Monolithic n=6 circuit
        qc_mono = QuantumCircuit(global_n)
        qc_mono.h(range(global_n))
        for step in range(k):
            qc_mono.global_phase += np.pi
            qc_mono.append(_build_phase_oracle(global_n, targets, float(betas[step])).to_gate(), range(global_n))
            qc_mono.append(_build_grover_diffusion(global_n, float(alphas[step])).to_gate(), range(global_n))
        qc_mono.measure_all()
        tc_mono = transpile(
            qc_mono,
            backend=backend,
            optimization_level=3,
            seed_transpiler=seed,
            initial_layout=list(range(global_n)),
        )
        c_mono_ideal = ideal_sim.run(tc_mono, shots=shots).result().get_counts()
        c_mono_noisy = noisy_sim.run(tc_mono, shots=shots).result().get_counts()
        mono_ideal[idx] = float(sum(c_mono_ideal.get(t, 0) for t in targets) / shots)
        mono_noisy[idx] = float(sum(c_mono_noisy.get(t, 0) for t in targets) / shots)

        # Distributed n=4 node circuits (same global schedule, independent execution)
        node_probs_ideal: list[float] = []
        node_probs_noisy: list[float] = []
        for prefix in sorted(node_targets.keys()):
            local_targets = node_targets[prefix]
            qc_local = QuantumCircuit(local_n)
            qc_local.h(range(local_n))
            for step in range(k):
                qc_local.global_phase += np.pi
                qc_local.append(_build_local_oracle(local_n, local_targets, float(betas[step])).to_gate(), range(local_n))
                qc_local.append(_build_local_diffusion(local_n, float(alphas[step])).to_gate(), range(local_n))
            qc_local.measure_all()
            tc_local = transpile(
                qc_local,
                backend=backend,
                optimization_level=3,
                seed_transpiler=seed,
                initial_layout=list(range(local_n)),
            )
            c_local_ideal = ideal_sim.run(tc_local, shots=shots).result().get_counts()
            c_local_noisy = noisy_sim.run(tc_local, shots=shots).result().get_counts()

            if local_targets:
                p_i = float(sum(c_local_ideal.get(sfx, 0) for sfx in local_targets) / shots)
                p_n = float(sum(c_local_noisy.get(sfx, 0) for sfx in local_targets) / shots)
            else:
                p_i = 0.0
                p_n = 0.0

            node_probs_ideal.append(p_i)
            node_probs_noisy.append(p_n)
            dist_node_noisy[prefix][idx] = p_n

        # Global distributed success: at least one node outputs an H_Good suffix.
        one_minus = 1.0
        for p in node_probs_ideal:
            one_minus *= 1.0 - p
        dist_ideal[idx] = 1.0 - one_minus

        one_minus = 1.0
        for p in node_probs_noisy:
            one_minus *= 1.0 - p
        dist_noisy[idx] = 1.0 - one_minus

    mono_baseline = float(len(targets) / (2**global_n))
    dist_baseline = 1.0
    for prefix in sorted(node_targets.keys()):
        m = len(node_targets[prefix])
        dist_baseline *= 1.0 - float(m / (2**local_n))
    dist_baseline = float(1.0 - dist_baseline)

    return NISQNoiseResults(
        backend_name=str(getattr(backend, "name", "unknown_backend")),
        backend_source=backend_source,
        global_n=global_n,
        j=j,
        local_n=local_n,
        global_goods=targets,
        node_targets=node_targets,
        epsilon=epsilon,
        global_a_for_schedule=float(global_a_for_schedule),
        L=L,
        k_values=k_values,
        shots=shots,
        monolithic_ideal_success=mono_ideal,
        monolithic_noisy_success=mono_noisy,
        distributed_ideal_success=dist_ideal,
        distributed_noisy_success=dist_noisy,
        distributed_node_noisy_success=dist_node_noisy,
        monolithic_random_baseline=mono_baseline,
        distributed_random_baseline=dist_baseline,
    )


def experiment_classical_network_statistics(
    global_n: int = 6,
    j: int = 2,
    global_goods: tuple[str, ...] = ("110110", "111111", "011001"),
    epsilon: float = 0.3,
    global_a_for_schedule: float = 3.0 / 64.0,
    L: Optional[int] = None,
    shots_per_node: int = 100,
    sifting_sigma: float = 4.0,
    seed: int = 42,
) -> NetworkStatisticsResults:
    """Finite-shot network simulation and master-node statistical sifting."""
    _require_qiskit()
    _require_aer_simulator()

    if global_n < 2:
        raise ValueError("global_n must be >= 2.")
    if j <= 0 or j >= global_n:
        raise ValueError("j must satisfy 1 <= j < global_n.")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0, 1).")
    if not (0.0 < global_a_for_schedule <= 1.0):
        raise ValueError("global_a_for_schedule must be in (0, 1].")
    if shots_per_node < 16:
        raise ValueError("shots_per_node must be >= 16.")
    if sifting_sigma <= 0.0:
        raise ValueError("sifting_sigma must be > 0.")

    targets = [s.strip() for s in global_goods if s.strip()]
    if not targets:
        raise ValueError("global_goods must not be empty.")
    if any(len(s) != global_n or any(ch not in "01" for ch in s) for s in targets):
        raise ValueError("All global_goods must be binary strings of length global_n.")

    local_n = global_n - j
    node_targets = _partition_targets_by_prefix(targets, j=j)

    if L is None:
        l_opt = int(np.ceil((0.5 * np.log(2.0 / epsilon)) / np.sqrt(global_a_for_schedule)))
        L = int(2 * l_opt + 1)
    if L < 1 or L % 2 == 0:
        raise ValueError("L must be a positive odd integer.")

    alphas, betas = _generate_fpaa_phases(L=L, delta=epsilon)

    sim = AerSimulator(seed_simulator=seed)
    node_counts: dict[str, dict[str, int]] = {}
    for prefix in sorted(node_targets.keys()):
        local_targets = node_targets[prefix]
        qc = QuantumCircuit(local_n)
        qc.h(range(local_n))
        for step in range(L):
            qc.global_phase += np.pi
            qc.append(_build_local_oracle(local_n, local_targets, float(betas[step])).to_gate(), range(local_n))
            qc.append(_build_local_diffusion(local_n, float(alphas[step])).to_gate(), range(local_n))
        qc.measure_all()
        tqc = transpile(qc, sim, optimization_level=3, seed_transpiler=seed)
        counts = sim.run(tqc, shots=shots_per_node).result().get_counts()
        node_counts[prefix] = {str(k): int(v) for k, v in counts.items()}

    num_local_states = 2**local_n
    uniform_mean = float(shots_per_node / num_local_states)
    p_uniform = 1.0 / num_local_states
    uniform_std = float(np.sqrt(shots_per_node * p_uniform * (1.0 - p_uniform)))
    threshold = int(np.ceil(uniform_mean + float(sifting_sigma) * uniform_std))

    good_set = set(targets)
    flagged: list[str] = []
    verified: list[str] = []
    false_pos: list[str] = []
    queries = 0

    for prefix in sorted(node_counts.keys()):
        counts = node_counts[prefix]
        for suffix, count in counts.items():
            if int(count) < threshold:
                continue
            candidate = f"{prefix}{suffix}"
            flagged.append(candidate)
            queries += 1
            if candidate in good_set:
                verified.append(candidate)
            else:
                false_pos.append(candidate)

    flagged = sorted(set(flagged))
    verified = sorted(set(verified))
    false_pos = sorted(set(false_pos))

    return NetworkStatisticsResults(
        global_n=global_n,
        j=j,
        local_n=local_n,
        global_goods=targets,
        node_targets=node_targets,
        epsilon=epsilon,
        global_a_for_schedule=float(global_a_for_schedule),
        L=L,
        shots_per_node=int(shots_per_node),
        total_shots=int(shots_per_node * (2**j)),
        uniform_mean_per_state=uniform_mean,
        uniform_std_per_state=uniform_std,
        sifting_sigma=float(sifting_sigma),
        sifting_threshold=threshold,
        node_counts=node_counts,
        flagged_candidates=flagged,
        verified_answers=verified,
        false_positives=false_pos,
        classical_queries_made=int(queries),
    )


class DQAA_Oracle_Synthesizer:
    """AST-level oracle partitioning compiler for distributed DQAA."""

    def __init__(
        self,
        global_n: int,
        j: int,
        formula_text: str,
        formula_format: str = "auto",
    ) -> None:
        _require_qiskit_transpiler()
        _require_oracle_synthesis()
        if global_n < 2:
            raise ValueError("global_n must be >= 2.")
        if j <= 0 or j >= global_n:
            raise ValueError("j must satisfy 1 <= j < global_n.")

        self.global_n = int(global_n)
        self.j = int(j)
        self.local_n = self.global_n - self.j
        self.formula_text = str(formula_text).strip()
        self.formula_format = formula_format.strip().lower()
        if self.formula_format not in {"auto", "boolean", "dimacs"}:
            raise ValueError("formula_format must be one of {'auto', 'boolean', 'dimacs'}.")

        self.global_var_names = [f"v{i}" for i in range(self.global_n)]
        self.prefix_var_names = [f"v{i}" for i in range(self.j)]
        self.suffix_var_names = [f"v{i}" for i in range(self.j, self.global_n)]

        self.formula_expr = self._parse_formula(self.formula_text, self.formula_format)
        self._validate_expression_symbols()

    @staticmethod
    def _parse_dimacs_to_sympy(dimacs_text: str) -> tuple[sp.Expr, int]:
        header_re = re.compile(r"^\s*p\s+cnf\s+(\d+)\s+(\d+)\s*$", flags=re.IGNORECASE)
        header_vars = None
        header_clauses = None
        data_tokens: list[int] = []

        for raw in dimacs_text.splitlines():
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            m = header_re.match(line)
            if m:
                header_vars = int(m.group(1))
                header_clauses = int(m.group(2))
                continue
            for tok in line.split():
                data_tokens.append(int(tok))

        if header_vars is None or header_clauses is None:
            raise ValueError("DIMACS header not found. Expected line like: 'p cnf <nvars> <nclauses>'.")

        clauses: list[list[int]] = []
        current: list[int] = []
        for lit in data_tokens:
            if lit == 0:
                clauses.append(current)
                current = []
            else:
                current.append(int(lit))
        if current:
            clauses.append(current)

        if len(clauses) != header_clauses:
            # Accept mild format drift but keep it explicit.
            header_clauses = len(clauses)

        clause_exprs: list[sp.Expr] = []
        for clause in clauses:
            if not clause:
                clause_exprs.append(sp.false)
                continue
            lits: list[sp.Expr] = []
            for lit in clause:
                idx = abs(int(lit)) - 1
                if idx < 0 or idx >= header_vars:
                    raise ValueError("DIMACS literal index out of range.")
                sym = sp.Symbol(f"v{idx}", boolean=True)
                lits.append(sym if lit > 0 else sp.Not(sym))
            clause_exprs.append(sp.Or(*lits))

        expr = sp.And(*clause_exprs) if clause_exprs else sp.true
        return expr, header_vars

    def _parse_formula(self, formula_text: str, formula_format: str) -> sp.Expr:
        if formula_format == "auto":
            is_dimacs = re.search(r"^\s*p\s+cnf\s+", formula_text, flags=re.IGNORECASE | re.MULTILINE) is not None
            formula_format = "dimacs" if is_dimacs else "boolean"

        if formula_format == "dimacs":
            expr, dimacs_vars = self._parse_dimacs_to_sympy(formula_text)
            if dimacs_vars > self.global_n:
                raise ValueError(
                    f"DIMACS declares {dimacs_vars} variables, but global_n={self.global_n}. "
                    "Increase global_n or provide a smaller formula."
                )
            return expr

        try:
            expr = sp.sympify(formula_text, evaluate=False)
        except Exception as exc:
            raise ValueError(f"Failed to parse boolean formula: {exc}") from exc
        if expr is True:
            return sp.true
        if expr is False:
            return sp.false
        return expr

    def _validate_expression_symbols(self) -> None:
        for sym in self.formula_expr.free_symbols:
            m = re.fullmatch(r"v(\d+)", sym.name)
            if m is None:
                raise ValueError(
                    f"Unsupported symbol '{sym}'. Use naming convention v0..v{self.global_n-1}."
                )
            idx = int(m.group(1))
            if idx < 0 or idx >= self.global_n:
                raise ValueError(
                    f"Symbol '{sym}' exceeds global register (global_n={self.global_n})."
                )

    @staticmethod
    def _sympy_expr_to_qiskit_string(expr: sp.Expr) -> str:
        return str(expr)

    @staticmethod
    def _safe_simplify(expr: sp.Expr) -> sp.Expr:
        try:
            return sp.simplify_logic(expr, force=True)
        except Exception:
            return sp.simplify(expr)

    def _embed_oracle_gate(
        self,
        expression: sp.Expr,
        full_var_order: list[str],
        total_qubits: int,
        name: str,
    ) -> tuple[QuantumCircuit, str, bool, int]:
        qc = QuantumCircuit(total_qubits, name=name)

        if expression == sp.false:
            return qc, "False", True, 0
        if expression == sp.true:
            # Constant true oracle is a global phase flip on every basis state.
            qc.global_phase += np.pi
            return qc, "True", True, 0

        expr_text = self._sympy_expr_to_qiskit_string(expression)
        oracle_gate = PhaseOracleGate(expr_text, var_order=full_var_order)
        raw_vars = list(oracle_gate.boolean_expression.args)
        if raw_vars:
            qubit_map = [full_var_order.index(vname) for vname in raw_vars]
            qc.append(oracle_gate, qubit_map)
        return qc, expr_text, False, len(raw_vars)

    @staticmethod
    def _transpile_metrics(
        qc: QuantumCircuit,
        basis_gates: list[str],
        coupling_map: CouplingMap,
        optimization_level: int,
        seed: int,
    ) -> tuple[QuantumCircuit, int, int, int, int]:
        routed = transpile(
            qc,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
        routed_swap = transpile(
            qc,
            basis_gates=basis_gates + ["swap"],
            coupling_map=coupling_map,
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
        ops = routed.count_ops()
        total = int(sum(ops.values()))
        cx = int(ops.get("cx", 0))
        depth = int(routed.depth())
        swap = int(routed_swap.count_ops().get("swap", 0))
        return routed, total, cx, depth, swap

    def compile_monolithic(
        self,
        basis_gates: list[str],
        coupling_map: CouplingMap,
        optimization_level: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        qc, formula_text, _, _ = self._embed_oracle_gate(
            self.formula_expr,
            full_var_order=self.global_var_names,
            total_qubits=self.global_n,
            name="oracle_global",
        )
        compile_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        routed, total, cx, depth, swap = self._transpile_metrics(
            qc,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=optimization_level,
            seed=seed,
        )
        transpile_sec = time.perf_counter() - t1

        return {
            "formula_text": formula_text,
            "classical_compile_time_sec": float(compile_sec),
            "transpile_time_sec": float(transpile_sec),
            "total_gates": int(total),
            "cx_gates": int(cx),
            "depth": int(depth),
            "estimated_swap_count": int(swap),
            "routed_circuit": routed,
        }

    def compile_prefix(
        self,
        prefix: str,
        basis_gates: list[str],
        coupling_map: CouplingMap,
        optimization_level: int = 3,
        seed: int = 42,
    ) -> OraclePartitionNodeMetrics:
        if len(prefix) != self.j or any(ch not in "01" for ch in prefix):
            raise ValueError("prefix must be a j-bit binary string.")

        t0 = time.perf_counter()
        symbol_lookup = {sym.name: sym for sym in self.formula_expr.free_symbols}
        subs = {
            symbol_lookup.get(f"v{i}", sp.Symbol(f"v{i}")): (prefix[i] == "1")
            for i in range(self.j)
        }
        simplified = self._safe_simplify(self.formula_expr.subs(subs))
        qc_local, formula_text, is_trivial, active_var_count = self._embed_oracle_gate(
            simplified,
            full_var_order=self.suffix_var_names,
            total_qubits=self.local_n,
            name=f"oracle_prefix_{prefix}",
        )
        compile_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        _, total, cx, depth, swap = self._transpile_metrics(
            qc_local,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=optimization_level,
            seed=seed,
        )
        transpile_sec = time.perf_counter() - t1

        return OraclePartitionNodeMetrics(
            prefix=prefix,
            simplified_formula=formula_text,
            is_trivial=is_trivial,
            active_variable_count=int(active_var_count),
            classical_compile_time_sec=float(compile_sec),
            transpile_time_sec=float(transpile_sec),
            total_gates=int(total),
            cx_gates=int(cx),
            depth=int(depth),
            estimated_swap_count=int(swap),
        )

    def compile_all_prefixes(
        self,
        basis_gates: list[str],
        coupling_map: CouplingMap,
        optimization_level: int = 3,
        seed: int = 42,
    ) -> dict[str, OraclePartitionNodeMetrics]:
        out: dict[str, OraclePartitionNodeMetrics] = {}
        for idx in range(2**self.j):
            prefix = format(idx, f"0{self.j}b")
            out[prefix] = self.compile_prefix(
                prefix=prefix,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                optimization_level=optimization_level,
                seed=seed,
            )
        return out


def experiment_automated_oracle_partitioning(
    global_n: int = 5,
    j: int = 2,
    formula_text: str = (
        "(v0 | v1 | ~v2) & (~v0 | v2 | v3) & (v1 | ~v3 | v4) & (~v1 | ~v4 | v2) & (v0 | v1)"
    ),
    formula_format: str = "auto",
    basis_gates: tuple[str, ...] = ("cx", "id", "rz", "sx", "x"),
    optimization_level: int = 3,
    seed: int = 42,
) -> CompilerResourceResults:
    """Automated compiler pass: AST inject prefix -> simplify -> synthesize sub-oracles."""
    _require_qiskit_transpiler()
    _require_oracle_synthesis()

    compiler = DQAA_Oracle_Synthesizer(
        global_n=global_n,
        j=j,
        formula_text=formula_text,
        formula_format=formula_format,
    )
    basis = list(basis_gates)
    cmap_global = CouplingMap.from_line(compiler.global_n)
    cmap_local = CouplingMap.from_line(compiler.local_n)

    mono = compiler.compile_monolithic(
        basis_gates=basis,
        coupling_map=cmap_global,
        optimization_level=optimization_level,
        seed=seed,
    )
    nodes = compiler.compile_all_prefixes(
        basis_gates=basis,
        coupling_map=cmap_local,
        optimization_level=optimization_level,
        seed=seed,
    )

    total_compile = float(sum(m.classical_compile_time_sec for m in nodes.values()))
    total_transpile = float(sum(m.transpile_time_sec for m in nodes.values()))
    gate_list = [m.total_gates for m in nodes.values()]
    avg_gates = float(np.mean(gate_list)) if gate_list else 0.0
    max_gates = int(max(gate_list)) if gate_list else 0
    sum_gates = int(sum(gate_list))

    def safe_ratio(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("inf")

    trivial_prefixes = sorted([p for p, m in nodes.items() if m.is_trivial])

    return CompilerResourceResults(
        global_n=compiler.global_n,
        j=compiler.j,
        local_n=compiler.local_n,
        formula_format=compiler.formula_format,
        formula_text=compiler.formula_text,
        basis_gates=basis,
        monolithic_compile_time_sec=float(mono["classical_compile_time_sec"]),
        monolithic_transpile_time_sec=float(mono["transpile_time_sec"]),
        monolithic_total_gates=int(mono["total_gates"]),
        monolithic_cx_gates=int(mono["cx_gates"]),
        monolithic_depth=int(mono["depth"]),
        monolithic_swap_count=int(mono["estimated_swap_count"]),
        distributed_total_compile_time_sec=total_compile,
        distributed_total_transpile_time_sec=total_transpile,
        distributed_node_metrics=nodes,
        distributed_avg_total_gates=avg_gates,
        distributed_max_total_gates=max_gates,
        distributed_sum_total_gates=sum_gates,
        gate_reduction_vs_avg=safe_ratio(float(mono["total_gates"]), avg_gates),
        gate_reduction_vs_max=safe_ratio(float(mono["total_gates"]), float(max_gates)),
        gate_reduction_vs_sum=safe_ratio(float(mono["total_gates"]), float(sum_gates)),
        classical_time_ratio_dist_over_mono=safe_ratio(
            total_compile + total_transpile,
            float(mono["classical_compile_time_sec"] + mono["transpile_time_sec"]),
        ),
        trivial_prefixes=trivial_prefixes,
    )


def run_lucky_node_monte_carlo(
    n: int = 8,
    j: int = 3,
    num_good: int = 12,
    trials: int = 2000,
    seed: Optional[int] = 1234,
) -> LuckyNodeMonteCarloResult:
    """Run many random instances to empirically verify universal validity."""
    _validate_inputs(n, j, num_good, trials=trials)

    rng = np.random.default_rng(seed)
    n_global = 2**n
    n_local = 2 ** (n - j)
    global_a = float(num_good / n_global)

    max_gap_samples = np.zeros(trials, dtype=float)
    lucky_count_samples = np.zeros(trials, dtype=int)

    for t in range(trials):
        good_states = rng.choice(n_global, size=num_good, replace=False)
        local_counts = _compute_local_good_counts(good_states, n=n, j=j)
        local_ak = local_counts / n_local
        max_gap_samples[t] = float(np.max(local_ak) - global_a)
        lucky_count_samples[t] = int(np.count_nonzero(local_ak >= global_a))

    violation_count = int(np.count_nonzero(max_gap_samples < -1e-14))

    return LuckyNodeMonteCarloResult(
        n=n,
        j=j,
        num_good=num_good,
        trials=trials,
        seed=seed,
        max_gap_samples=max_gap_samples,
        lucky_count_samples=lucky_count_samples,
        violation_count=violation_count,
        min_gap=float(np.min(max_gap_samples)),
        mean_gap=float(np.mean(max_gap_samples)),
        std_gap=float(np.std(max_gap_samples)),
        max_gap=float(np.max(max_gap_samples)),
    )


def plot_distributed_fpaa_histograms(
    result: DistributedFPAAResults,
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """Generate 4-panel output histogram for independent distributed nodes."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plot_distributed_fpaa_histograms.") from exc

    num_states = 2 ** result.local_n
    threshold = 1.0 - result.epsilon * result.epsilon
    uniform_baseline = 1.0 / num_states
    x = np.arange(num_states, dtype=int)
    labels = [format(i, f"0{result.local_n}b") for i in x]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle(
        "Distributed FPAA Execution (Hua-Qiu POC)\n"
        f"n={result.global_n}, j={result.j}, local_qubits={result.local_n}, "
        f"p={result.global_a:.5f}, epsilon={result.epsilon:.2f}, L={result.L}",
        fontsize=15,
    )

    for ax, prefix in zip(axes.flatten(), sorted(result.node_probabilities.keys())):
        probs = result.node_probabilities[prefix]
        is_lucky = prefix in result.lucky_nodes
        color = "tab:green" if is_lucky else "0.60"
        bars = ax.bar(x, probs, color=color, edgecolor="black", linewidth=0.7, alpha=0.88)

        # Highlight known H_Good suffixes on lucky nodes.
        for suffix in result.node_targets[prefix]:
            bars[int(suffix, 2)].set_color("tab:orange")

        ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.8)
        ax.axhline(uniform_baseline, color="black", linestyle=":", linewidth=1.2, alpha=0.8)

        node_id = int(prefix, 2)
        status = "Lucky" if is_lucky else "Unlucky"
        ax.set_title(
            f"Node {node_id} (prefix={prefix}) [{status}]\n"
            f"p_k (H_Good set)={result.node_good_success[prefix]:.4f}, "
            f"top={result.node_top_suffix[prefix]}:{result.node_top_probability[prefix]:.4f}",
            fontsize=11,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x[::2])
        ax.set_xticklabels([labels[idx] for idx in x[::2]], rotation=45)
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    for ax in axes[-1, :]:
        ax.set_xlabel(f"Local suffix outcome ({result.local_n}-bit)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Measurement probability p(outcome)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_entanglement_obstruction(
    result: EntanglementObstructionResults,
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """Plot positive vs negative trajectory under local FPAA after partition cut."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plot_entanglement_obstruction.") from exc

    threshold = 1.0 - result.epsilon * result.epsilon

    plt.figure(figsize=(10.5, 6.2))
    plt.plot(
        result.k_values,
        result.separable_good_probs,
        color="tab:blue",
        marker="o",
        linewidth=2.3,
        label="Separable preparation (valid tensor factorization)",
    )
    plt.plot(
        result.k_values,
        result.entangled_good_probs,
        color="tab:red",
        marker="x",
        linestyle="--",
        linewidth=2.3,
        label="Cross-register entangled preparation (obstruction)",
    )
    plt.axhline(
        threshold,
        color="0.35",
        linestyle=":",
        linewidth=1.6,
        label=f"Target threshold 1-epsilon^2={threshold:.2f}",
    )
    plt.title(
        "Negative Proof: Entanglement Obstruction in DQAA\n"
        f"H_Good={result.good_global}, L={result.L}, "
        f"peak ratio={result.obstruction_ratio_peak:.1f}x",
        fontsize=13,
    )
    plt.xlabel("Local FPAA prefix of schedule (k)")
    plt.ylabel(f"Local success probability p_k for H_Good suffix |{result.good_suffix}>")
    plt.ylim(0.0, 1.02)
    plt.xticks(result.k_values)
    plt.grid(alpha=0.28)
    plt.legend(loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_nisq_noise_resilience(
    result: NISQNoiseResults,
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """Plot noisy-vs-ideal success trajectories for monolithic and distributed runs."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plot_nisq_noise_resilience.") from exc

    plt.figure(figsize=(11, 6.3))
    k = result.k_values

    plt.plot(
        k,
        result.monolithic_ideal_success,
        color="tab:red",
        marker="o",
        linewidth=2.1,
        label=f"Monolithic ideal (n={result.global_n})",
    )
    plt.plot(
        k,
        result.monolithic_noisy_success,
        color="tab:red",
        marker="x",
        linestyle="--",
        linewidth=2.4,
        label=f"Monolithic noisy (n={result.global_n})",
    )
    plt.plot(
        k,
        result.distributed_ideal_success,
        color="tab:blue",
        marker="o",
        linewidth=2.1,
        label=f"Distributed ideal (2^{result.j} nodes, {result.local_n} qubits/node)",
    )
    plt.plot(
        k,
        result.distributed_noisy_success,
        color="tab:blue",
        marker="x",
        linestyle="--",
        linewidth=2.4,
        label=f"Distributed noisy (2^{result.j} nodes)",
    )

    plt.axhline(
        result.monolithic_random_baseline,
        color="tab:red",
        linestyle=":",
        linewidth=1.3,
        alpha=0.8,
        label=f"Monolithic random limit (|T|/2^n={result.monolithic_random_baseline:.4f})",
    )
    plt.axhline(
        result.distributed_random_baseline,
        color="tab:blue",
        linestyle=":",
        linewidth=1.3,
        alpha=0.8,
        label=f"Distributed random limit={result.distributed_random_baseline:.4f}",
    )

    plt.title(
        "NISQ Noise-Resilience Benchmark (Density Matrix)\n"
        f"backend={result.backend_name}, source={result.backend_source}, shots={result.shots}",
        fontsize=12.5,
    )
    plt.xlabel("FPAA schedule prefix length (k)")
    plt.ylabel("Observed success probability p_k")
    plt.ylim(0.0, 1.02)
    plt.xticks(k)
    plt.grid(alpha=0.28)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_classical_network_statistics(
    result: NetworkStatisticsResults,
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """Render per-node finite-shot histograms with statistical sifting threshold."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plot_classical_network_statistics.") from exc

    node_prefixes = sorted(result.node_counts.keys())
    num_states = 2**result.local_n
    suffixes = [format(i, f"0{result.local_n}b") for i in range(num_states)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle(
        "Classical Network Extraction Under Shot Noise\n"
        f"shots/node={result.shots_per_node}, threshold={result.sifting_threshold} "
        f"(mu={result.uniform_mean_per_state:.2f}, sigma={result.uniform_std_per_state:.2f})",
        fontsize=14,
    )

    flagged_set = set(result.flagged_candidates)
    verified_set = set(result.verified_answers)

    for ax, prefix in zip(axes.flatten(), node_prefixes):
        counts = result.node_counts[prefix]
        values = [int(counts.get(sfx, 0)) for sfx in suffixes]
        colors = []
        for sfx, c in zip(suffixes, values):
            candidate = f"{prefix}{sfx}"
            if candidate in verified_set:
                colors.append("tab:green")
            elif candidate in flagged_set:
                colors.append("tab:orange")
            else:
                colors.append("0.65")

        ax.bar(suffixes, values, color=colors, edgecolor="black", linewidth=0.7, alpha=0.88)
        ax.axhline(result.sifting_threshold, color="tab:red", linestyle="--", linewidth=1.7)
        ax.axhline(result.uniform_mean_per_state, color="black", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.set_title(
            f"Node {int(prefix, 2)} (prefix={prefix})\n"
            f"local_targets={result.node_targets[prefix]}",
            fontsize=11,
        )
        ax.set_ylim(0.0, float(result.shots_per_node))
        ax.set_xticks(range(0, num_states, 2))
        ax.set_xticklabels([suffixes[i] for i in range(0, num_states, 2)], rotation=45)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    for ax in axes[-1, :]:
        ax.set_xlabel("Measured local suffix")
    for ax in axes[:, 0]:
        ax.set_ylabel("Shot count")

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_lucky_node_barchart(
    result: LuckyNodeInstanceResult,
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """Plot local p_k bars and global average p line for Figure-X style evidence."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plot_lucky_node_barchart.") from exc

    num_nodes = len(result.local_ak)
    xs = np.arange(num_nodes, dtype=int)
    colors = ["tab:green" if ak >= result.global_a else "0.60" for ak in result.local_ak]

    plt.figure(figsize=(11, 5.8))
    plt.bar(xs, result.local_ak, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    plt.axhline(
        y=result.global_a,
        color="tab:red",
        linestyle="--",
        linewidth=2.2,
        label=f"Global average p={result.global_a:.4f}",
    )
    plt.xlabel(f"Distributed node index k (prefix partition, 2^{result.j} nodes)")
    plt.ylabel("Local success probability p_k")
    plt.title(
        f"Lucky Node Verification (n={result.n}, j={result.j}, M={result.num_good})\n"
        f"max(p_k)-p={result.max_gap:.4f}, lucky nodes={len(result.lucky_indices)}"
    )
    plt.xticks(xs)
    plt.ylim(0.0, max(1.0, float(np.max(result.local_ak) * 1.15)))
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    legend_elements = [
        plt.Line2D([0], [0], color="tab:red", linestyle="--", linewidth=2.2, label="Global p"),
        Patch(facecolor="tab:green", edgecolor="black", label="Lucky node (p_k >= p)"),
        Patch(facecolor="0.60", edgecolor="black", label="p_k < p"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_lucky_node_monte_carlo_evidence(
    mc: LuckyNodeMonteCarloResult,
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """Plot global evidence that max_k(p_k-p) never crosses below zero."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plot_lucky_node_monte_carlo_evidence.") from exc

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=False)

    ax1.hist(mc.max_gap_samples, bins=36, color="tab:blue", edgecolor="black", alpha=0.85)
    ax1.axvline(0.0, color="tab:red", linestyle="--", linewidth=2.0, label="Theorem boundary: 0")
    ax1.set_xlabel("max_k(p_k - p)")
    ax1.set_ylabel("Trial count")
    ax1.set_title(
        f"Monte Carlo Convexity Check (n={mc.n}, j={mc.j}, M={mc.num_good}, trials={mc.trials})\n"
        f"violations={mc.violation_count}, min_gap={mc.min_gap:.6f}"
    )
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right")

    bins = np.arange(
        int(np.min(mc.lucky_count_samples)),
        int(np.max(mc.lucky_count_samples)) + 2,
    ) - 0.5
    ax2.hist(mc.lucky_count_samples, bins=bins, color="tab:green", edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Number of lucky nodes in a trial")
    ax2.set_ylabel("Trial count")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def summarize_distributed_fpaa(result: DistributedFPAAResults) -> dict[str, Any]:
    """JSON-safe summary for the distributed FPAA module."""
    return {
        "global_n": result.global_n,
        "j": result.j,
        "local_n": result.local_n,
        "global_goods": result.global_goods,
        "node_targets": result.node_targets,
        "global_a": result.global_a,
        "epsilon": result.epsilon,
        "l_opt": result.l_opt,
        "L": result.L,
        "success_threshold_1_minus_epsilon_sq": float(1.0 - result.epsilon * result.epsilon),
        "lucky_nodes": result.lucky_nodes,
        "unlucky_nodes": result.unlucky_nodes,
        "node_good_success": result.node_good_success,
        "node_top_suffix": result.node_top_suffix,
        "node_top_probability": result.node_top_probability,
    }


def summarize_hardware_tradeoff(result: HardwareCompilationResults) -> dict[str, Any]:
    """JSON-safe summary + journal-table rows for hardware compilation benchmark."""
    table_rows: list[dict[str, Any]] = []
    table_rows.append(
        {
            "architecture": f"Monolithic (n={result.global_n})",
            **result.monolithic_metrics,
        }
    )
    for prefix in sorted(result.distributed_node_metrics.keys()):
        table_rows.append(
            {
                "architecture": f"DQAA node {int(prefix, 2)} (prefix={prefix}, n={result.local_n})",
                **result.distributed_node_metrics[prefix],
            }
        )
    table_rows.append(
        {
            "architecture": f"DQAA critical-path max node (n={result.local_n})",
            **result.distributed_aggregate_metrics,
        }
    )

    return {
        "global_n": result.global_n,
        "j": result.j,
        "local_n": result.local_n,
        "global_goods": result.global_goods,
        "node_targets": result.node_targets,
        "basis_gates": result.basis_gates,
        "topology": result.topology,
        "monolithic_metrics": result.monolithic_metrics,
        "distributed_node_metrics": result.distributed_node_metrics,
        "distributed_aggregate_metrics": result.distributed_aggregate_metrics,
        "reduction_factors": result.reduction_factors,
        "hardware_resource_table_rows": table_rows,
    }


def summarize_entanglement_obstruction(result: EntanglementObstructionResults) -> dict[str, Any]:
    """JSON-safe summary for the tensor-factorization limitation proof."""
    return {
        "global_n": result.global_n,
        "j": result.j,
        "local_n": result.local_n,
        "good_global": result.good_global,
        "good_suffix": result.good_suffix,
        "global_a_for_schedule": result.global_a_for_schedule,
        "epsilon": result.epsilon,
        "L": result.L,
        "threshold_1_minus_epsilon_sq": float(1.0 - result.epsilon * result.epsilon),
        "separable_suffix_purity": result.separable_suffix_purity,
        "entangled_suffix_purity": result.entangled_suffix_purity,
        "entangled_suffix_entropy_bits": result.entangled_suffix_entropy_bits,
        "separable_peak_probability": float(np.max(result.separable_good_probs)),
        "entangled_peak_probability": float(np.max(result.entangled_good_probs)),
        "obstruction_ratio_peak": result.obstruction_ratio_peak,
        "k_at_separable_peak": int(np.argmax(result.separable_good_probs)),
        "k_at_entangled_peak": int(np.argmax(result.entangled_good_probs)),
        "separable_curve": result.separable_good_probs.astype(float).tolist(),
        "entangled_curve": result.entangled_good_probs.astype(float).tolist(),
    }


def summarize_nisq_noise_resilience(result: NISQNoiseResults) -> dict[str, Any]:
    """JSON-safe summary for the density-matrix NISQ benchmark."""
    mono_noisy_peak = float(np.max(result.monolithic_noisy_success))
    dist_noisy_peak = float(np.max(result.distributed_noisy_success))
    mono_peak_k = int(np.argmax(result.monolithic_noisy_success))
    dist_peak_k = int(np.argmax(result.distributed_noisy_success))

    return {
        "backend_name": result.backend_name,
        "backend_source": result.backend_source,
        "global_n": result.global_n,
        "j": result.j,
        "local_n": result.local_n,
        "global_goods": result.global_goods,
        "node_targets": result.node_targets,
        "epsilon": result.epsilon,
        "global_a_for_schedule": result.global_a_for_schedule,
        "L": result.L,
        "shots": result.shots,
        "k_values": result.k_values.astype(int).tolist(),
        "monolithic_random_baseline": result.monolithic_random_baseline,
        "distributed_random_baseline": result.distributed_random_baseline,
        "monolithic_ideal_success": result.monolithic_ideal_success.astype(float).tolist(),
        "monolithic_noisy_success": result.monolithic_noisy_success.astype(float).tolist(),
        "distributed_ideal_success": result.distributed_ideal_success.astype(float).tolist(),
        "distributed_noisy_success": result.distributed_noisy_success.astype(float).tolist(),
        "distributed_node_noisy_success": {
            prefix: arr.astype(float).tolist()
            for prefix, arr in result.distributed_node_noisy_success.items()
        },
        "noisy_peak_monolithic": mono_noisy_peak,
        "noisy_peak_distributed": dist_noisy_peak,
        "noisy_peak_ratio_distributed_over_monolithic": float(
            dist_noisy_peak / max(1e-15, mono_noisy_peak)
        ),
        "k_at_noisy_peak_monolithic": mono_peak_k,
        "k_at_noisy_peak_distributed": dist_peak_k,
    }


def summarize_network_statistics(result: NetworkStatisticsResults) -> dict[str, Any]:
    """JSON-safe summary for end-to-end classical post-processing benchmark."""
    recovered_all = set(result.verified_answers) == set(result.global_goods)
    query_fraction = float(result.classical_queries_made / max(1, result.total_shots))

    return {
        "global_n": result.global_n,
        "j": result.j,
        "local_n": result.local_n,
        "global_goods": result.global_goods,
        "node_targets": result.node_targets,
        "epsilon": result.epsilon,
        "global_a_for_schedule": result.global_a_for_schedule,
        "L": result.L,
        "shots_per_node": result.shots_per_node,
        "total_shots": result.total_shots,
        "uniform_mean_per_state": result.uniform_mean_per_state,
        "uniform_std_per_state": result.uniform_std_per_state,
        "sifting_sigma": result.sifting_sigma,
        "sifting_threshold": result.sifting_threshold,
        "classical_queries_made": result.classical_queries_made,
        "classical_query_fraction_of_total_shots": query_fraction,
        "flagged_candidates": result.flagged_candidates,
        "verified_answers": result.verified_answers,
        "false_positives": result.false_positives,
        "recovered_all_targets": recovered_all,
        "node_counts": result.node_counts,
    }


def summarize_oracle_partitioning(result: CompilerResourceResults) -> dict[str, Any]:
    """JSON-safe summary for automated oracle partitioning compiler benchmark."""
    node_rows: list[dict[str, Any]] = []
    for prefix in sorted(result.distributed_node_metrics.keys()):
        m = result.distributed_node_metrics[prefix]
        node_rows.append(
            {
                "prefix": prefix,
                "simplified_formula": m.simplified_formula,
                "is_trivial": m.is_trivial,
                "active_variable_count": m.active_variable_count,
                "classical_compile_time_sec": m.classical_compile_time_sec,
                "transpile_time_sec": m.transpile_time_sec,
                "total_gates": m.total_gates,
                "cx_gates": m.cx_gates,
                "depth": m.depth,
                "estimated_swap_count": m.estimated_swap_count,
            }
        )

    compiler_resource_table_rows = [
        {
            "architecture": f"Monolithic (n={result.global_n})",
            "classical_time_sec": result.monolithic_compile_time_sec + result.monolithic_transpile_time_sec,
            "total_gates": result.monolithic_total_gates,
            "cx_gates": result.monolithic_cx_gates,
            "depth": result.monolithic_depth,
            "swap_count": result.monolithic_swap_count,
        },
        {
            "architecture": f"Distributed average node (n={result.local_n}, 2^{result.j} nodes)",
            "classical_time_sec": (
                result.distributed_total_compile_time_sec + result.distributed_total_transpile_time_sec
            )
            / (2**result.j),
            "total_gates": result.distributed_avg_total_gates,
            "cx_gates": float(np.mean([m.cx_gates for m in result.distributed_node_metrics.values()])),
            "depth": float(np.mean([m.depth for m in result.distributed_node_metrics.values()])),
            "swap_count": float(np.mean([m.estimated_swap_count for m in result.distributed_node_metrics.values()])),
        },
        {
            "architecture": f"Distributed max node (critical path, n={result.local_n})",
            "classical_time_sec": float(
                max(
                    m.classical_compile_time_sec + m.transpile_time_sec
                    for m in result.distributed_node_metrics.values()
                )
            ),
            "total_gates": result.distributed_max_total_gates,
            "cx_gates": int(max(m.cx_gates for m in result.distributed_node_metrics.values())),
            "depth": int(max(m.depth for m in result.distributed_node_metrics.values())),
            "swap_count": int(max(m.estimated_swap_count for m in result.distributed_node_metrics.values())),
        },
        {
            "architecture": f"Distributed sum over all nodes (2^{result.j} circuits)",
            "classical_time_sec": result.distributed_total_compile_time_sec + result.distributed_total_transpile_time_sec,
            "total_gates": result.distributed_sum_total_gates,
            "cx_gates": int(sum(m.cx_gates for m in result.distributed_node_metrics.values())),
            "depth": int(sum(m.depth for m in result.distributed_node_metrics.values())),
            "swap_count": int(sum(m.estimated_swap_count for m in result.distributed_node_metrics.values())),
        },
    ]

    return {
        "global_n": result.global_n,
        "j": result.j,
        "local_n": result.local_n,
        "formula_format": result.formula_format,
        "formula_text": result.formula_text,
        "basis_gates": result.basis_gates,
        "monolithic_compile_time_sec": result.monolithic_compile_time_sec,
        "monolithic_transpile_time_sec": result.monolithic_transpile_time_sec,
        "monolithic_total_gates": result.monolithic_total_gates,
        "monolithic_cx_gates": result.monolithic_cx_gates,
        "monolithic_depth": result.monolithic_depth,
        "monolithic_swap_count": result.monolithic_swap_count,
        "distributed_total_compile_time_sec": result.distributed_total_compile_time_sec,
        "distributed_total_transpile_time_sec": result.distributed_total_transpile_time_sec,
        "distributed_avg_total_gates": result.distributed_avg_total_gates,
        "distributed_max_total_gates": result.distributed_max_total_gates,
        "distributed_sum_total_gates": result.distributed_sum_total_gates,
        "gate_reduction_vs_avg": result.gate_reduction_vs_avg,
        "gate_reduction_vs_max": result.gate_reduction_vs_max,
        "gate_reduction_vs_sum": result.gate_reduction_vs_sum,
        "classical_time_ratio_dist_over_mono": result.classical_time_ratio_dist_over_mono,
        "trivial_prefixes": result.trivial_prefixes,
        "per_prefix_metrics": node_rows,
        "compiler_resource_table_rows": compiler_resource_table_rows,
    }


def save_oracle_partitioning_table_csv(result: CompilerResourceResults, path: str) -> None:
    """Save journal-ready compiler resource table CSV."""
    fields = ["architecture", "classical_time_sec", "total_gates", "cx_gates", "depth", "swap_count"]
    rows = summarize_oracle_partitioning(result)["compiler_resource_table_rows"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_hardware_tradeoff_table_csv(result: HardwareCompilationResults, path: str) -> None:
    """Persist hardware resource table to CSV for paper insertion."""
    import csv

    fields = [
        "architecture",
        "qubits",
        "logical_depth_no_cmap",
        "routed_depth",
        "logical_cx_no_cmap",
        "routed_cx",
        "routed_total_gates",
        "routed_single_qubit",
        "estimated_swap_count",
        "estimated_routing_cx_overhead",
    ]
    rows = summarize_hardware_tradeoff(result)["hardware_resource_table_rows"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarize_instance(result: LuckyNodeInstanceResult) -> dict[str, Any]:
    """Return JSON-safe summary for one instance."""
    return {
        "n": result.n,
        "j": result.j,
        "num_good": result.num_good,
        "seed": result.seed,
        "global_a": result.global_a,
        "local_good_counts": result.local_good_counts.astype(int).tolist(),
        "local_ak": result.local_ak.astype(float).tolist(),
        "lucky_indices": result.lucky_indices,
        "max_ak": result.max_ak,
        "max_gap": result.max_gap,
        "theorem_holds": result.theorem_holds,
    }


def summarize_monte_carlo(mc: LuckyNodeMonteCarloResult) -> dict[str, Any]:
    """Return JSON-safe summary for Monte Carlo validation."""
    return {
        "n": mc.n,
        "j": mc.j,
        "num_good": mc.num_good,
        "trials": mc.trials,
        "seed": mc.seed,
        "violation_count": mc.violation_count,
        "violation_rate": float(mc.violation_count / mc.trials),
        "min_gap": mc.min_gap,
        "mean_gap": mc.mean_gap,
        "std_gap": mc.std_gap,
        "max_gap": mc.max_gap,
        "lucky_node_count_min": int(np.min(mc.lucky_count_samples)),
        "lucky_node_count_mean": float(np.mean(mc.lucky_count_samples)),
        "lucky_node_count_max": int(np.max(mc.lucky_count_samples)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lucky Node numerical proof for distributed FPAA Theorem 3.")
    parser.add_argument("--n", type=int, default=8, help="Global search qubit count.")
    parser.add_argument("--j", type=int, default=3, help="Prefix qubits for partitioning.")
    parser.add_argument("--num-good", type=int, default=12, help="Number of H_Good states globally (M).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the single-instance run.")
    parser.add_argument("--trials", type=int, default=2000, help="Monte Carlo trial count.")
    parser.add_argument("--mc-seed", type=int, default=1234, help="Seed for Monte Carlo run.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="FPAA epsilon for distributed execution (threshold is 1-epsilon^2).",
    )
    parser.add_argument(
        "--global-goods",
        type=str,
        default="110110,111111,011001",
        help="Comma-separated n-bit H_Good strings for distributed FPAA.",
    )
    parser.add_argument("--out-prefix", type=str, default="lucky_node", help="Output artifact prefix.")
    parser.add_argument("--show-plots", action="store_true", help="Display matplotlib windows.")
    parser.add_argument(
        "--save-json",
        nargs="?",
        const="lucky_node_results.json",
        default=None,
        help="Optional path for writing summary JSON.",
    )
    parser.add_argument("--run-all", action="store_true", help="Run single-instance + Monte Carlo + plots.")
    parser.add_argument(
        "--run-distributed-fpaa",
        action="store_true",
        help="Run the exact n=6,j=2 Hua-Qiu distributed FPAA simulation and emit 4-panel histogram.",
    )
    parser.add_argument(
        "--run-hardware-tradeoff",
        action="store_true",
        help="Run monolithic-vs-distributed constrained transpilation benchmark and emit resource table.",
    )
    parser.add_argument(
        "--run-entanglement-obstruction",
        action="store_true",
        help="Run negative proof for tensor-factorization failure under cross-register entanglement.",
    )
    parser.add_argument(
        "--good-global",
        type=str,
        default="110110",
        help="Global H_Good bitstring used in entanglement obstruction module.",
    )
    parser.add_argument(
        "--schedule-global-a",
        type=float,
        default=3.0 / 64.0,
        help="Global success probability p used to derive local FPAA schedule in obstruction module.",
    )
    parser.add_argument(
        "--obstruction-L",
        type=int,
        default=None,
        help="Optional odd FPAA length for obstruction module (overrides formula if set).",
    )
    parser.add_argument(
        "--run-noise-benchmark",
        action="store_true",
        help="Run density-matrix NISQ noise benchmark (monolithic vs distributed).",
    )
    parser.add_argument(
        "--noise-backend",
        type=str,
        default="FakeGuadalupeV2",
        help="Preferred fake backend class name for NoiseModel.from_backend.",
    )
    parser.add_argument(
        "--noise-shots",
        type=int,
        default=4096,
        help="Shot count for noisy/ideal probability estimation.",
    )
    parser.add_argument(
        "--noise-k-max",
        type=int,
        default=7,
        help="Maximum schedule prefix length k used in the noise benchmark.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="Seed for backend fallback generation and simulators in noise benchmark.",
    )
    parser.add_argument(
        "--run-oracle-compiler",
        action="store_true",
        help="Run automated AST-level oracle partitioning compiler benchmark.",
    )
    parser.add_argument(
        "--compiler-formula",
        type=str,
        default="(v0 | v1 | ~v2) & (~v0 | v2 | v3) & (v1 | ~v3 | v4) & (~v1 | ~v4 | v2) & (v0 | v1)",
        help="Boolean formula (or DIMACS text if --compiler-format dimacs).",
    )
    parser.add_argument(
        "--compiler-format",
        type=str,
        default="auto",
        choices=["auto", "boolean", "dimacs"],
        help="Input format for compiler formula.",
    )
    parser.add_argument(
        "--compiler-opt-level",
        type=int,
        default=3,
        help="Transpiler optimization level for compiler benchmark.",
    )
    parser.add_argument(
        "--compiler-seed",
        type=int,
        default=42,
        help="Seed for transpiler in compiler benchmark.",
    )
    parser.add_argument(
        "--run-network-statistics",
        action="store_true",
        help="Run end-to-end finite-shot classical master-node extraction benchmark.",
    )
    parser.add_argument(
        "--network-shots",
        type=int,
        default=100,
        help="Shot count per distributed node for network statistics module.",
    )
    parser.add_argument(
        "--network-sigma",
        type=float,
        default=4.0,
        help="Sifting threshold in standard deviations above uniform shot-noise mean.",
    )
    parser.add_argument(
        "--network-seed",
        type=int,
        default=42,
        help="Simulator seed for network statistics module.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if (
        not args.run_all
        and not args.run_distributed_fpaa
        and not args.run_hardware_tradeoff
        and not args.run_entanglement_obstruction
        and not args.run_noise_benchmark
        and not args.run_oracle_compiler
        and not args.run_network_statistics
    ):
        result = experiment_lucky_node_verification(
            n=args.n,
            j=args.j,
            num_good=args.num_good,
            seed=args.seed,
        )
        print(json.dumps(summarize_instance(result), indent=2))
        return 0

    summary: dict[str, Any] = {}

    if args.run_all:
        result = experiment_lucky_node_verification(
            n=args.n,
            j=args.j,
            num_good=args.num_good,
            seed=args.seed,
        )
        mc = run_lucky_node_monte_carlo(
            n=args.n,
            j=args.j,
            num_good=args.num_good,
            trials=args.trials,
            seed=args.mc_seed,
        )

        bar_path = f"{args.out_prefix}_barchart.png"
        mc_path = f"{args.out_prefix}_monte_carlo.png"
        plot_lucky_node_barchart(result, save_path=bar_path, show_plot=bool(args.show_plots))
        plot_lucky_node_monte_carlo_evidence(mc, save_path=mc_path, show_plot=bool(args.show_plots))

        summary["lucky_node"] = {
            "single_instance": summarize_instance(result),
            "monte_carlo": summarize_monte_carlo(mc),
            "artifacts": {
                "barchart": bar_path,
                "monte_carlo_plot": mc_path,
            },
        }

    if args.run_distributed_fpaa:
        targets = tuple(t.strip() for t in args.global_goods.split(",") if t.strip())
        # Convenience default: if user did not override shared --n/--j, use the
        # canonical Hua-Qiu proof-of-concept parameters.
        dist_n = 6 if (args.n == 8 and args.j == 3) else args.n
        dist_j = 2 if (args.n == 8 and args.j == 3) else args.j
        dist = experiment_distributed_fpaa_execution(
            global_n=dist_n,
            j=dist_j,
            global_goods=targets,
            epsilon=args.epsilon,
        )
        dist_plot = f"{args.out_prefix}_distributed_fpaa.png"
        plot_distributed_fpaa_histograms(dist, save_path=dist_plot, show_plot=bool(args.show_plots))
        summary["distributed_fpaa"] = summarize_distributed_fpaa(dist)
        summary["distributed_fpaa_artifact"] = {"four_panel_histogram": dist_plot}

    if args.run_hardware_tradeoff:
        targets = tuple(t.strip() for t in args.global_goods.split(",") if t.strip())
        # Same convenience default as distributed module.
        tradeoff_n = 6 if (args.n == 8 and args.j == 3) else args.n
        tradeoff_j = 2 if (args.n == 8 and args.j == 3) else args.j
        hw = experiment_hardware_compilation_tradeoff(
            global_n=tradeoff_n,
            j=tradeoff_j,
            global_goods=targets,
        )
        table_csv = f"{args.out_prefix}_hardware_resource_table.csv"
        save_hardware_tradeoff_table_csv(hw, table_csv)
        summary["hardware_tradeoff"] = summarize_hardware_tradeoff(hw)
        summary["hardware_tradeoff_artifact"] = {"resource_table_csv": table_csv}

    if args.run_entanglement_obstruction:
        # Same convenience default as distributed module.
        obstruction_n = 6 if (args.n == 8 and args.j == 3) else args.n
        obstruction_j = 2 if (args.n == 8 and args.j == 3) else args.j
        obstruction = experiment_entanglement_obstruction(
            global_n=obstruction_n,
            j=obstruction_j,
            good_global=args.good_global.strip(),
            epsilon=args.epsilon,
            global_a_for_schedule=float(args.schedule_global_a),
            L=args.obstruction_L,
        )
        obstruction_plot = f"{args.out_prefix}_entanglement_obstruction.png"
        plot_entanglement_obstruction(obstruction, save_path=obstruction_plot, show_plot=bool(args.show_plots))
        summary["entanglement_obstruction"] = summarize_entanglement_obstruction(obstruction)
        summary["entanglement_obstruction_artifact"] = {"obstruction_plot": obstruction_plot}

    if args.run_noise_benchmark:
        # Same convenience default as distributed module.
        noise_n = 6 if (args.n == 8 and args.j == 3) else args.n
        noise_j = 2 if (args.n == 8 and args.j == 3) else args.j
        targets = tuple(t.strip() for t in args.global_goods.split(",") if t.strip())
        noise = experiment_nisq_noise_resilience(
            global_n=noise_n,
            j=noise_j,
            global_goods=targets,
            epsilon=args.epsilon,
            global_a_for_schedule=float(args.schedule_global_a),
            L=args.obstruction_L,
            k_max=int(args.noise_k_max),
            shots=int(args.noise_shots),
            preferred_backend=args.noise_backend.strip() if args.noise_backend else None,
            seed=int(args.noise_seed),
        )
        noise_plot = f"{args.out_prefix}_nisq_noise.png"
        plot_nisq_noise_resilience(noise, save_path=noise_plot, show_plot=bool(args.show_plots))
        summary["nisq_noise_benchmark"] = summarize_nisq_noise_resilience(noise)
        summary["nisq_noise_artifact"] = {"noise_plot": noise_plot}

    if args.run_oracle_compiler:
        # Module-6 default uses n=5, j=2 unless user overrides shared --n/--j.
        compiler_n = 5 if (args.n == 8 and args.j == 3) else args.n
        compiler_j = 2 if (args.n == 8 and args.j == 3) else args.j
        compiler_res = experiment_automated_oracle_partitioning(
            global_n=compiler_n,
            j=compiler_j,
            formula_text=args.compiler_formula,
            formula_format=args.compiler_format,
            optimization_level=int(args.compiler_opt_level),
            seed=int(args.compiler_seed),
        )
        compiler_csv = f"{args.out_prefix}_compiler_resource_table.csv"
        save_oracle_partitioning_table_csv(compiler_res, compiler_csv)
        summary["oracle_partitioning_compiler"] = summarize_oracle_partitioning(compiler_res)
        summary["oracle_partitioning_artifact"] = {"compiler_table_csv": compiler_csv}

    if args.run_network_statistics:
        # Same convenience default as distributed module.
        network_n = 6 if (args.n == 8 and args.j == 3) else args.n
        network_j = 2 if (args.n == 8 and args.j == 3) else args.j
        targets = tuple(t.strip() for t in args.global_goods.split(",") if t.strip())
        network = experiment_classical_network_statistics(
            global_n=network_n,
            j=network_j,
            global_goods=targets,
            epsilon=args.epsilon,
            global_a_for_schedule=float(args.schedule_global_a),
            L=args.obstruction_L,
            shots_per_node=int(args.network_shots),
            sifting_sigma=float(args.network_sigma),
            seed=int(args.network_seed),
        )
        network_plot = f"{args.out_prefix}_network_shot_noise.png"
        plot_classical_network_statistics(network, save_path=network_plot, show_plot=bool(args.show_plots))
        summary["network_statistics"] = summarize_network_statistics(network)
        summary["network_statistics_artifact"] = {"network_plot": network_plot}

    if args.save_json is not None:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="dqaa")
    if len(sys.argv) == 1:
        stem = Path(__file__).stem
        raise SystemExit(
            main(
                [
                    "--run-all",
                    "--run-distributed-fpaa",
                    "--run-hardware-tradeoff",
                    "--run-entanglement-obstruction",
                    "--run-noise-benchmark",
                    "--run-oracle-compiler",
                    "--run-network-statistics",
                    "--out-prefix",
                    stem,
                    "--save-json",
                    f"{stem}_summary.json",
                ]
            )
        )
    raise SystemExit(main())
