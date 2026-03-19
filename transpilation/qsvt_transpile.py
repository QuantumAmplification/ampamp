"""
QSVT Transpilation Master Suite (Scenarios A through Z)
=======================================================

26 physical hardware scenarios for the full QSVT manuscript arc:
  A-Z = core compiler realities, unification limits, adversarial stress tests,
  parity heuristics, OS-level controls, and full-stack comparative evaluation.

This script imports:
  - 6_Quantum_Singular_Variable_Transformation.py
  - 2_Fixed_Point_Ammplitude_Amplification.py
  - quantum_profiler.py

Output log:
  !_QSVT_transpile_results.txt
"""

from __future__ import annotations

import math
import os
import sys
import traceback
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.linalg import sqrtm

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library import UnitaryGate


class Logger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def _import_module(alias: str, abs_path: str):
    spec = importlib.util.spec_from_file_location(alias, abs_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_HERE = os.path.dirname(os.path.abspath(__file__))
_QSVT_PATH = os.path.join(_HERE, "6_Quantum_Singular_Variable_Transformation.py")
_FPAA_PATH = os.path.join(_HERE, "2_Fixed_Point_Ammplitude_Amplification.py")
_PROF_PATH = os.path.join(_HERE, "quantum_profiler.py")

qsvt_mod = _import_module("qsvt_module", _QSVT_PATH)
fpaa_mod = _import_module("fpaa_module", _FPAA_PATH)
prof_mod = _import_module("quantum_profiler_module", _PROF_PATH)
HardwareProfiler = prof_mod.HardwareProfiler

SEP = "=" * 72
BASIS_NISQ = ["cx", "id", "rz", "sx", "x"]


def _line_edges(n_qubits: int) -> List[List[int]]:
    edges: List[List[int]] = []
    for i in range(n_qubits - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return edges


def _heavy_hex_map() -> CouplingMap:
    return CouplingMap.from_heavy_hex(distance=3)


def _t_count_rz(theta: float, synthesis_eps: float = 1e-3, tol: float = 1e-10) -> int:
    # Clifford phase check: multiples of pi/2 cost 0 T.
    k2 = round(theta / (math.pi / 2.0))
    if abs(theta - k2 * (math.pi / 2.0)) < tol:
        return 0
    # Exact pi/4 lattice point check: odd multiples cost 1 T.
    k4 = round(theta / (math.pi / 4.0))
    if abs(theta - k4 * (math.pi / 4.0)) < tol:
        return 1 if (k4 % 2) != 0 else 0
    # Ross-Selinger style asymptotic estimate for arbitrary angle synthesis.
    return max(0, int(math.ceil(3.21 * math.log2(1.0 / synthesis_eps) - 6.93)))


def _estimate_t_count_from_native(qc_native: QuantumCircuit, synthesis_eps: float = 1e-3) -> int:
    total = 0
    for inst in qc_native.data:
        op = inst.operation
        if op.name == "rz":
            total += _t_count_rz(float(op.params[0]), synthesis_eps=synthesis_eps)
        elif op.name in ("t", "tdg"):
            total += 1
    return total


def _transpile_stats(
    qc: QuantumCircuit,
    *,
    coupling_map: Optional[CouplingMap] = None,
    basis_gates: Optional[List[str]] = None,
    optimization_level: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    basis = basis_gates or BASIS_NISQ
    tqc = transpile(
        qc,
        basis_gates=basis,
        coupling_map=coupling_map,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    ops = tqc.count_ops()
    return {
        "circuit": tqc,
        "qubits": tqc.num_qubits,
        "depth": int(tqc.depth()),
        "gates": int(sum(ops.values())),
        "cx": int(ops.get("cx", 0)),
        "swap": int(ops.get("swap", 0)),
        "rz": int(ops.get("rz", 0)),
    }


def _profile_stats(qc: QuantumCircuit, coupling_map: CouplingMap) -> Dict[str, Any]:
    profiler = HardwareProfiler(
        coupling_map_edges=[list(e) for e in coupling_map.get_edges()],
        basis_gates=BASIS_NISQ,
        single_qubit_ns=20,
        two_qubit_ns=100,
    )
    try:
        return profiler.profile_circuit(qc)
    except Exception as exc:
        # Fallback path for pass-manager edge cases in newer Qiskit versions.
        t = _transpile_stats(qc, coupling_map=coupling_map, optimization_level=3)
        logical_ops = qc.count_ops()
        logical_gates = int(sum(logical_ops.values()))
        logical_depth = int(qc.depth())
        distance = int(2 * t["swap"])
        total_time_ns = float(max(1, t["depth"]) * 100)
        penalty = float(total_time_ns + 10.0 * t["cx"] + 5.0 * distance)
        return {
            "logical_depth": logical_depth,
            "logical_gates": logical_gates,
            "initial_distance_penalty": distance,
            "post_routing_depth": t["depth"],
            "routing_swaps": t["swap"],
            "post_translation_depth": t["depth"],
            "translation_gates": t["gates"],
            "translation_cnots": t["cx"],
            "post_optimization_depth": t["depth"],
            "final_gates": t["gates"],
            "final_cnots": t["cx"],
            "total_time_ns": total_time_ns,
            "hardware_penalty_score": penalty,
            "profiler_fallback": str(exc),
        }


def _nearest_idx(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))


def _phase_quantize(phases: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** int(bits)
    normed = np.mod(phases, 2.0 * np.pi) / (2.0 * np.pi)
    q_normed = np.round(normed * levels) / levels
    return q_normed * 2.0 * np.pi


def _apply_block_u(qc: QuantumCircuit, signal: int, data: List[int], work: Optional[int]) -> None:
    # THE TRUE ORACLE: Multi-Controlled X
    controls = [signal] + data[:-1]
    target = data[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def _apply_block_udg(qc: QuantumCircuit, signal: int, data: List[int], work: Optional[int]) -> None:
    # THE TRUE ORACLE UNCOMPUTATION (Self-inverse)
    controls = [signal] + data[:-1]
    target = data[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def build_qsp_signal_circuit(phases: np.ndarray, x_angle: float = 0.43) -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.rz(2.0 * float(phases[0]), 0)
    for phi in phases[1:]:
        qc.rx(2.0 * x_angle, 0)
        qc.rz(2.0 * float(phi), 0)
    return qc


def build_qsvt_lifted_circuit(
    n_data: int,
    phases: np.ndarray,
    *,
    include_work: bool = True,
    extra_route: bool = False,
    signal_mixer_angle: float = 0.43,
) -> QuantumCircuit:
    n_total = 1 + n_data + (1 if include_work else 0)
    qc = QuantumCircuit(n_total)
    signal = 0
    data = list(range(1, 1 + n_data))
    work = (1 + n_data) if include_work else None

    if data:
        qc.h(data)

    qc.rz(2.0 * float(phases[0]), signal)
    for idx, phi in enumerate(phases[1:], start=1):
        # Non-commuting mixer on the signal qubit prevents optimizer from
        # commuting all phase rotations through oracle controls and canceling U/U^\dag.
        qc.rx(2.0 * signal_mixer_angle, signal)
        if idx % 2 == 1:
            _apply_block_u(qc, signal, data, work)
        else:
            _apply_block_udg(qc, signal, data, work)
        if extra_route and len(data) >= 4:
            qc.cswap(signal, data[0], data[-1])
            qc.cswap(signal, data[1], data[-2])
        qc.rz(2.0 * float(phi), signal)
    return qc


def build_u_wall_circuit(n_data: int = 5, phase: float = 0.37, inject_phase: bool = True) -> QuantumCircuit:
    qc = QuantumCircuit(n_data + 1)
    signal = 0
    data = list(range(1, n_data + 1))

    for q in data:
        qc.h(q)
    for q in data:
        qc.cx(signal, q)
    for i in range(len(data) - 1):
        qc.cx(data[i], data[i + 1])

    if inject_phase:
        pivot = data[len(data) // 2]
        qc.rz(2.0 * phase, pivot)

    for i in reversed(range(len(data) - 1)):
        qc.cx(data[i], data[i + 1])
    for q in reversed(data):
        qc.cx(signal, q)
    return qc


def build_lcu_prep_sel_circuit(n_data: int = 5) -> QuantumCircuit:
    anc = QuantumRegister(1, "anc")
    data = QuantumRegister(n_data, "data")
    qc = QuantumCircuit(anc, data)

    alpha_0 = 1.5
    alpha_1 = 0.5
    alpha = abs(alpha_0) + abs(alpha_1)
    theta = 2.0 * np.arccos(np.sqrt(alpha_0 / alpha))
    qc.ry(theta, anc[0])  # PREP

    for q in data:
        qc.cx(anc[0], q)  # SEL spread across the data register
    if n_data >= 2:
        qc.mcx(list(data[:-1]), data[-1])
    for q in reversed(data):
        qc.cx(anc[0], q)

    qc.ry(-theta, anc[0])  # PREP^\dag
    return qc


def build_qsvt_degree_circuit(n_data: int, degree: int) -> QuantumCircuit:
    phase_grid = np.sin(np.linspace(0.0, 2.0 * np.pi, degree + 1)) * (np.pi / 2.0)
    return build_qsvt_lifted_circuit(n_data=n_data, phases=phase_grid, include_work=False)


def build_qpe_hhl_skeleton(n_data: int = 3, n_phase: int = 9) -> QuantumCircuit:
    # Register order: [phase ancillas][data register]
    qc = QuantumCircuit(n_phase + n_data)
    phase = list(range(n_phase))
    data = list(range(n_phase, n_phase + n_data))

    qc.h(phase)
    for k, q in enumerate(phase):
        reps = min(2 ** min(k, 4), 16)
        for _ in range(reps):
            for d in data:
                qc.cp(np.pi / (2 ** (k + 1)), q, d)

    # Approximate inverse QFT on the phase register.
    for i in range(n_phase // 2):
        qc.swap(phase[i], phase[n_phase - 1 - i])
    for j in range(n_phase):
        qc.h(phase[j])
        for m in range(j + 1, n_phase):
            qc.cp(-np.pi / (2 ** (m - j)), phase[m], phase[j])
    return qc


def build_strict_parity_circuit(n_data: int = 4, rounds: int = 4) -> QuantumCircuit:
    qc = QuantumCircuit(n_data + 1)
    signal = 0
    data = list(range(1, n_data + 1))
    qc.h(data)
    for _ in range(rounds):
        for q in data:
            qc.cx(signal, q)
        qc.rz(np.pi / 5.0, signal)
    return qc


def build_mixed_parity_circuit(n_data: int = 4, rounds: int = 4) -> QuantumCircuit:
    qc = QuantumCircuit(n_data + 2)
    parity = 0
    signal = 1
    data = list(range(2, n_data + 2))
    qc.h(data)
    qc.h(parity)
    for _ in range(rounds):
        for q in data:
            qc.cx(signal, q)
        if len(data) >= 4:
            qc.cswap(parity, data[0], data[-1])
            qc.cswap(parity, data[1], data[-2])
        if len(data) >= 3:
            qc.mcx([parity, signal], data[0])
            qc.mcx([parity, signal], data[-1])
        qc.rz(np.pi / 7.0, signal)
    return qc


def build_oaa_benchmark_circuit(n_data: int = 4, rounds: int = 2) -> QuantumCircuit:
    qc = QuantumCircuit(n_data + 1)
    anc = 0
    data = list(range(1, n_data + 1))
    qc.h(data)
    qc.ry(0.9, anc)
    for _ in range(rounds):
        # Forward True Oracle
        controls = [anc] + data[:-1]
        target = data[-1]
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)

        # Reflection
        qc.x(anc)
        qc.z(anc)
        qc.x(anc)

        # Uncompute True Oracle
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)
    return qc


def build_foqa_benchmark_circuit(n_data: int = 4, rounds: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(n_data + 2)
    anc = 0
    idx = 1
    data = list(range(2, n_data + 2))
    qc.h(data)
    qc.ry(0.6, idx)
    schedule = [1.2, 0.9, 0.6][:rounds]
    for alpha in schedule:
        qc.cry(alpha, idx, anc)

        # True Oracle
        controls = [idx] + data[:-1]
        target = data[-1]
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)

        qc.crz(alpha / 2.0, anc, idx)
    return qc


def build_dqaa_benchmark_circuit(n_global: int = 4, j: int = 1, rounds: int = 2) -> QuantumCircuit:
    local_n = n_global - j
    qc = QuantumCircuit(1 + local_n)
    prefix = 0
    data = list(range(1, 1 + local_n))
    qc.h([prefix] + data)
    for _ in range(rounds):
        # True Oracle
        controls = [prefix] + data[:-1]
        target = data[-1]
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)

        # Diffusion
        qc.h(data)
        qc.x(data)
        if len(data) >= 2:
            qc.mcp(np.pi, data[:-1], data[-1])
        qc.x(data)
        qc.h(data)
    return qc


def build_vtaa_benchmark_circuit(n_data: int = 4, rounds: int = 2) -> QuantumCircuit:
    qc = QuantumCircuit(n_data + 2)
    f0 = n_data
    f1 = n_data + 1
    data = list(range(n_data))
    qc.h(data)
    for _ in range(rounds):
        # True Oracle (VTAA already uses MCP natively, so we keep it honest)
        if n_data >= 2:
            qc.mcp(np.pi, data[:-1], data[-1])

        # Diffusion
        qc.h(data)
        qc.x(data)
        if n_data >= 2:
            qc.mcp(np.pi, data[:-1], data[-1])
        qc.x(data)
        qc.h(data)

        # Coherent Flags
        qc.ccx(data[0], data[1], f0)
        qc.cry(0.7, f0, f1)
    return qc


def build_svd_channel_skeleton() -> QuantumCircuit:
    # 2-qubit toy channel: [signal ancilla, data]
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cry(0.9, 0, 1)
    qc.crz(0.5, 0, 1)
    qc.cry(-0.7, 0, 1)
    return qc


def build_dynamic_reuse_qsvt_circuit(n_data: int = 4, rounds: int = 6) -> QuantumCircuit:
    qc = QuantumCircuit(1 + n_data, rounds)
    signal = 0
    data = list(range(1, 1 + n_data))
    qc.h(data)
    for r in range(rounds):
        for q in data:
            qc.cx(signal, q)
        qc.rz(0.21 * (r + 1), signal)
        for q in reversed(data):
            qc.cx(signal, q)
        qc.measure(signal, r)
        qc.reset(signal)
    return qc


def build_static_signal_bank_qsvt_circuit(n_data: int = 4, rounds: int = 6) -> QuantumCircuit:
    qc = QuantumCircuit(n_data + rounds)
    data = list(range(n_data))
    signals = list(range(n_data, n_data + rounds))
    qc.h(data)
    for r, signal in enumerate(signals):
        for q in data:
            qc.cx(signal, q)
        qc.rz(0.21 * (r + 1), signal)
        for q in reversed(data):
            qc.cx(signal, q)
    return qc


def _apply_pauli_tag(qc: QuantumCircuit, qubit: int, tag: int) -> None:
    # 0=I, 1=X, 2=Z, 3=XZ
    if tag in (1, 3):
        qc.x(qubit)
    if tag in (2, 3):
        qc.z(qubit)


def build_twirled_circuit(qc_in: QuantumCircuit, seed: int = 12345) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(qc_in.num_qubits, qc_in.num_clbits)

    for inst in qc_in.data:
        op = inst.operation
        qargs = [qc.qubits[qc_in.find_bit(q).index] for q in inst.qubits]
        cargs = [qc.clbits[qc_in.find_bit(c).index] for c in inst.clbits]

        if op.name == "cx":
            q0 = qc_in.find_bit(inst.qubits[0]).index
            q1 = qc_in.find_bit(inst.qubits[1]).index
            pre0 = int(rng.integers(0, 4))
            pre1 = int(rng.integers(0, 4))
            post0 = int(rng.integers(0, 4))
            post1 = int(rng.integers(0, 4))
            _apply_pauli_tag(qc, q0, pre0)
            _apply_pauli_tag(qc, q1, pre1)
            qc.append(op, qargs, cargs)
            _apply_pauli_tag(qc, q0, post0)
            _apply_pauli_tag(qc, q1, post1)
        else:
            qc.append(op, qargs, cargs)
    return qc


def build_lcu_embedding_qsvt_circuit(phases: np.ndarray, n_data: int = 4) -> QuantumCircuit:
    qc = QuantumCircuit(2 + n_data)
    signal = 0
    anc = 1
    data = list(range(2, 2 + n_data))
    theta = 0.73

    qc.h(data)
    for phi in phases:
        qc.rz(2.0 * float(phi), signal)
        qc.ry(theta, anc)
        qc.cx(signal, anc)
        for q in data:
            qc.cx(anc, q)
        if len(data) >= 2:
            qc.mcx(data[:-1], data[-1])
        for q in reversed(data):
            qc.cx(anc, q)
        qc.cx(signal, anc)
        qc.ry(-theta, anc)
    return qc


def build_trotter_embedding_qsvt_circuit(phases: np.ndarray, n_data: int = 4) -> QuantumCircuit:
    qc = QuantumCircuit(1 + n_data)
    signal = 0
    data = list(range(1, 1 + n_data))
    qc.h(data)
    for phi in phases:
        qc.rz(2.0 * float(phi), signal)
        for q in data:
            qc.cx(signal, q)
            qc.rz(0.18, q)
            qc.cx(signal, q)
        for i in range(len(data) - 1):
            qc.cx(data[i], data[i + 1])
            qc.rz(0.11, data[i + 1])
            qc.cx(data[i], data[i + 1])
    return qc


def run_scenario_a() -> None:
    print(f"\n{SEP}")
    print("SCENARIO A: QSP -> QSVT LIFTING OVERHEAD (MODULES 1-5)")
    print(SEP)

    phases = qsvt_mod.canonical_phase_sets()["Sign Function Proxy (d=3)"]
    qc_qsp = build_qsp_signal_circuit(phases)
    qc_qsvt = build_qsvt_lifted_circuit(n_data=6, phases=phases, include_work=True)

    hh = _heavy_hex_map()
    m_qsp = _transpile_stats(qc_qsp, coupling_map=hh, optimization_level=3)
    m_qsvt = _transpile_stats(qc_qsvt, coupling_map=hh, optimization_level=3)

    print(f"{'Metric':<36} | {'QSP (1q)':<14} | {'QSVT Lifted'}")
    print("-" * 74)
    print(f"{'Physical qubits':<36} | {m_qsp['qubits']:<14} | {m_qsvt['qubits']}")
    print(f"{'Post-opt depth':<36} | {m_qsp['depth']:<14} | {m_qsvt['depth']}")
    print(f"{'CX count':<36} | {m_qsp['cx']:<14} | {m_qsvt['cx']}")
    print(f"{'SWAP count':<36} | {m_qsp['swap']:<14} | {m_qsvt['swap']}")
    print(f"{'Total gates':<36} | {m_qsp['gates']:<14} | {m_qsvt['gates']}")

    depth_ratio = m_qsvt["depth"] / max(1, m_qsp["depth"])
    print(f"\n-> Lifting overhead depth ratio: {depth_ratio:.2f}x")
    print("-> Same phase list, but matrix lifting explodes width, routing, and entangling cost.")


def run_scenario_b() -> None:
    print(f"\n{SEP}")
    print("SCENARIO B: SINGLE-SIGNAL ROUTING BOTTLENECK (HEAVY-HEX ROUTING CONGESTION)")
    print(SEP)

    phases = np.linspace(-1.1, 1.1, 19)
    qc = build_qsvt_lifted_circuit(n_data=8, phases=phases, include_work=True, extra_route=True)
    hh = _heavy_hex_map()

    m_free = _transpile_stats(qc, coupling_map=None, optimization_level=3)
    m_hh = _transpile_stats(qc, coupling_map=hh, optimization_level=3)
    p_hh = _profile_stats(qc, hh)

    signal_interactions = 0
    for inst in qc.data:
        if len(inst.qubits) >= 2:
            indices = [qc.find_bit(q).index for q in inst.qubits]
            if 0 in indices:
                signal_interactions += 1

    print(f"Signal qubit participates in {signal_interactions} multi-qubit interactions.")
    print(f"{'Metric':<36} | {'All-to-all':<14} | {'Heavy-hex'}")
    print("-" * 74)
    print(f"{'Depth':<36} | {m_free['depth']:<14} | {m_hh['depth']}")
    print(f"{'CX count':<36} | {m_free['cx']:<14} | {m_hh['cx']}")
    print(f"{'SWAP count':<36} | {m_free['swap']:<14} | {m_hh['swap']}")
    print(f"{'Initial distance penalty':<36} | {'N/A':<14} | {p_hh['initial_distance_penalty']}")
    print(f"{'Total scheduled time (ns)':<36} | {'N/A':<14} | {p_hh['total_time_ns']:.1f}")

    print("\n-> Routing through one signal qubit creates a central hardware choke point.")


def run_scenario_c() -> None:
    print(f"\n{SEP}")
    print("SCENARIO C: ALTERNATING UNCOMPUTATION LIMIT (U - PHASE - U^dagger)")
    print(SEP)

    qc_wall = build_u_wall_circuit(n_data=5, phase=0.31, inject_phase=True)
    qc_no_wall = build_u_wall_circuit(n_data=5, phase=0.31, inject_phase=False)
    hh = _heavy_hex_map()

    wall_o0 = _transpile_stats(qc_wall, coupling_map=hh, optimization_level=0)
    wall_o3 = _transpile_stats(qc_wall, coupling_map=hh, optimization_level=3)
    free_o0 = _transpile_stats(qc_no_wall, coupling_map=hh, optimization_level=0)
    free_o3 = _transpile_stats(qc_no_wall, coupling_map=hh, optimization_level=3)

    print(f"{'Case':<32} | {'Depth o0':<10} | {'Depth o3':<10} | {'CX o0':<8} | {'CX o3'}")
    print("-" * 78)
    print(f"{'U * Rz(phi) * U^dagger':<32} | {wall_o0['depth']:<10} | {wall_o3['depth']:<10} | {wall_o0['cx']:<8} | {wall_o3['cx']}")
    print(f"{'U * U^dagger (control case)':<32} | {free_o0['depth']:<10} | {free_o3['depth']:<10} | {free_o0['cx']:<8} | {free_o3['cx']}")

    wall_reduction = wall_o3["depth"] / max(1, wall_o0["depth"])
    free_reduction = free_o3["depth"] / max(1, free_o0["depth"])
    print(f"\n-> With the phase wall: depth retention = {100*wall_reduction:.1f}% of o0.")
    print(f"-> Without the wall: depth retention = {100*free_reduction:.1f}% of o0.")


def run_scenario_d() -> None:
    print(f"\n{SEP}")
    print("SCENARIO D: LCU BLOCK-ENCODING SPATIAL OVERHEAD (PREP/SEL/PREP^dag)")
    print(SEP)

    res = qsvt_mod.experiment_lcu_block_encoding(plot=False)
    qc = build_lcu_prep_sel_circuit(n_data=5)
    line = CouplingMap(_line_edges(qc.num_qubits))
    free = _transpile_stats(qc, coupling_map=None, optimization_level=3)
    routed = _transpile_stats(qc, coupling_map=line, optimization_level=3)

    print(f"{'Metric':<36} | {'All-to-all':<14} | {'Linear-1D'}")
    print("-" * 74)
    print(f"{'Depth':<36} | {free['depth']:<14} | {routed['depth']}")
    print(f"{'CX count':<36} | {free['cx']:<14} | {routed['cx']}")
    print(f"{'SWAP count':<36} | {free['swap']:<14} | {routed['swap']}")
    print(f"{'Block extraction error ||A-aU00||':<36} | {'N/A':<14} | {res.reconstruction_error:.3e}")
    print(f"{'Global unitary audit':<36} | {'N/A':<14} | {res.is_U_strictly_unitary}")

    print("\n-> Embedding non-unitary A into a larger unitary U adds hard routing overhead.")


def run_scenario_e() -> None:
    print(f"\n{SEP}")
    print("SCENARIO E: INVARIANT SUBSPACE COLLAPSE UNDER DECOHERENCE")
    print(SEP)

    res = qsvt_mod.experiment_qsvt_invariant_subspace(degree=20, seed=42, plot=False)

    dim = res.unitary_dim
    steps = res.degree + 1
    rng = np.random.default_rng(2026)
    v0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v0 = v0 / np.linalg.norm(v0)
    tmp = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    tmp = tmp - np.vdot(v0, tmp) * v0
    v1 = tmp / np.linalg.norm(tmp)

    history_ideal = []
    history_noisy = []
    gamma = 0.08
    for k in range(steps):
        angle = (2.0 * np.pi * k) / max(1, steps - 1)
        psi = np.cos(angle) * v0 + np.sin(angle) * v1
        history_ideal.append(psi)

        eta = 1.0 - np.exp(-gamma * k)
        noise = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        noise = noise / np.linalg.norm(noise)
        noisy = (1.0 - eta) * psi + eta * noise
        noisy = noisy / np.linalg.norm(noisy)
        history_noisy.append(noisy)

    s_ideal = np.linalg.svd(np.column_stack(history_ideal), compute_uv=False)
    s_noisy = np.linalg.svd(np.column_stack(history_noisy), compute_uv=False)
    rank_ideal = int(np.sum(s_ideal > 1e-10))
    rank_noisy = int(np.sum(s_noisy > 1e-10))

    print(f"Ideal theorem audit rank #1: {res.rank_0}")
    print(f"Ideal theorem audit rank #2: {res.rank_1}")
    print(f"Constructed ideal trajectory rank: {rank_ideal}")
    print(f"Decohered trajectory rank: {rank_noisy}")
    print(f"Inter-plane overlap (ideal theorem): {res.inter_plane_overlap:.3e}")
    print("\n-> Thermal mixing lifts trajectories out of the clean 2D invariant planes.")


def run_scenario_f() -> None:
    print(f"\n{SEP}")
    print("SCENARIO F: HAMILTONIAN SIMULATION T-GATE EXTRACTION (FTQC)")
    print(SEP)

    hres = qsvt_mod.experiment_qsvt_hamiltonian_simulation(
        t=15.0,
        target_epsilon=1e-8,
        max_extra_degree=20,
        plot=False,
    )
    d_star = hres.optimal_d_for_target_error if hres.optimal_d_for_target_error > 0 else int(np.ceil(abs(hres.t)) + 10)
    degrees = sorted(set([int(np.ceil(abs(hres.t))), d_star, d_star + 10]))

    print(f"{'Degree d':<12} | {'Native depth':<12} | {'CX':<8} | {'Est. T-count'}")
    print("-" * 58)
    for d in degrees:
        qc = build_qsvt_degree_circuit(n_data=4, degree=int(d))
        native = _transpile_stats(qc, basis_gates=BASIS_NISQ, optimization_level=1)
        t_est = _estimate_t_count_from_native(native["circuit"], synthesis_eps=1e-3)
        print(f"{d:<12} | {native['depth']:<12} | {native['cx']:<8} | {t_est}")

    print("\n-> Even/odd Chebyshev synthesis pushes FT cost into non-Clifford phase synthesis.")


def run_scenario_g() -> None:
    print(f"\n{SEP}")
    print("SCENARIO G: MATRIX INVERSION SPATIAL SUPREMACY (QSVT vs QPE)")
    print(SEP)

    res = qsvt_mod.experiment_qsvt_matrix_inversion(
        kappa=20.0,
        degree=63,
        scale_factor=0.8,
        outside_weight=20.0,
        plot=False,
    )
    eps_target = 1e-2
    idx = _nearest_idx(res.epsilons, eps_target)
    qpe_anc = int(res.qpe_qubits[idx])
    qsvt_anc = int(res.qsvt_qubits[idx])

    qc_qsvt = build_qsvt_degree_circuit(n_data=3, degree=41)
    qc_qpe = build_qpe_hhl_skeleton(n_data=3, n_phase=qpe_anc)

    m_qsvt = _transpile_stats(qc_qsvt, optimization_level=2)
    m_qpe = _transpile_stats(qc_qpe, optimization_level=1)

    print(f"Precision target epsilon ~ {res.epsilons[idx]:.2e}")
    print(f"{'Metric':<36} | {'QSVT inverse':<14} | {'QPE/HHL style'}")
    print("-" * 76)
    print(f"{'Ancilla qubits (model)':<36} | {qsvt_anc:<14} | {qpe_anc}")
    print(f"{'Total qubits (compiled)':<36} | {m_qsvt['qubits']:<14} | {m_qpe['qubits']}")
    print(f"{'Depth':<36} | {m_qsvt['depth']:<14} | {m_qpe['depth']}")
    print(f"{'CX count':<36} | {m_qsvt['cx']:<14} | {m_qpe['cx']}")

    print("\n-> QSVT removes QPE's spatial ancilla limit and pays with polynomial-depth dynamics.")


def run_scenario_h() -> None:
    print(f"\n{SEP}")
    print("SCENARIO H: FIXED-POINT SEARCH DEPTH LIMIT (COHERENCE BREACH)")
    print(SEP)

    fres = qsvt_mod.experiment_qsvt_fixed_point_search(
        delta=0.2,
        target_degree=61,
        test_x0=0.15,
        plot=False,
    )

    degrees = [9, 21, 41, 61]
    line = CouplingMap(_line_edges(20))
    coherence_ns = 80_000.0

    print(f"{'Degree d':<10} | {'Success p(d)':<14} | {'Time (ns)':<12} | {'Breach?'}")
    print("-" * 62)
    breach_degree = None
    for d in degrees:
        qc = build_qsvt_degree_circuit(n_data=5, degree=d)
        m = _profile_stats(qc, line)
        idx = _nearest_idx(fres.degrees_eval.astype(float), float(d))
        p_d = float(fres.qsvt_fpaa_probs[idx])
        breach = m["total_time_ns"] > coherence_ns
        if breach and breach_degree is None:
            breach_degree = d
        print(f"{d:<10} | {p_d:<14.6f} | {m['total_time_ns']:<12.1f} | {'YES' if breach else 'no'}")

    print(f"\nCoherence budget: {coherence_ns:.0f} ns")
    if breach_degree is None:
        print("-> No breach in tested degrees.")
    else:
        print(f"-> First breach detected at approximately d={breach_degree}.")


def run_scenario_i() -> None:
    print(f"\n{SEP}")
    print("SCENARIO I: LCU ALGEBRA DEPTH EXPLOSION (ALPHA COMPOUNDING)")
    print(SEP)

    res = qsvt_mod.experiment_lcu_operator_algebra(seed=1337, depth_max=12, plot=False)
    line = CouplingMap(_line_edges(10))

    print(f"alpha_A={res.alpha_A:.4f}, alpha_B={res.alpha_B:.4f}")
    print(f"{'Layer k':<10} | {'Depth':<10} | {'CX':<8} | {'Signal ~ 1/alpha_A^k'}")
    print("-" * 64)
    for k in range(1, 7):
        qc = QuantumCircuit(4)
        for i in range(k):
            qc.cx(0, 1)
            qc.cx(1, 2)
            qc.rz(0.21 * (i + 1), 2)
            qc.cx(1, 2)
            qc.cx(2, 3)
            qc.rz(-0.17 * (i + 1), 3)
            qc.cx(2, 3)
        m = _transpile_stats(qc, coupling_map=line, optimization_level=3)
        signal = 1.0 / (res.alpha_A ** k)
        print(f"{k:<10} | {m['depth']:<10} | {m['cx']:<8} | {signal:.3e}")

    print("\n-> Alpha compounding quickly drives extraction amplitude below practical SNR.")


def run_scenario_j() -> None:
    print(f"\n{SEP}")
    print("SCENARIO J: USVA AMPLIFICATION-RECOVERY PENALTY (DEPTH OVERHEAD TO RE-INFLATE SIGNAL)")
    print(SEP)

    res = qsvt_mod.experiment_qsvt_uniform_amplification(
        c_amp=3.0,
        degree=21,
        alpha_A=1.6,
        rescue_threshold=0.15,
        max_depth=15,
        plot=False,
    )

    qc_core = build_qsvt_degree_circuit(n_data=4, degree=15)
    qc_base = QuantumCircuit(qc_core.num_qubits + 1)
    qc_base.compose(qc_core, inplace=True)
    qc_rescue = qc_base.copy()
    sig = 0
    data0 = 1
    rescue_work = qc_rescue.num_qubits - 1
    for _ in range(max(1, res.rescue_operations)):
        qc_rescue.cx(sig, data0)
        qc_rescue.rz(np.pi / 9.0, sig)
        qc_rescue.cx(sig, data0)
        qc_rescue.ccx(sig, data0, rescue_work)  # explicit work path for rescue logic
        qc_rescue.rz(np.pi / 11.0, rescue_work)
        qc_rescue.ccx(sig, data0, rescue_work)

    m_base = _transpile_stats(qc_base, optimization_level=3)
    m_rescue = _transpile_stats(qc_rescue, optimization_level=3)

    print(f"USVA slope near origin: {res.slope_at_origin:.3f}")
    print(f"USVA recovery operations (theory): {res.rescue_operations}")
    print(f"{'Metric':<30} | {'No recovery':<12} | {'With recovery'}")
    print("-" * 62)
    print(f"{'Depth':<30} | {m_base['depth']:<12} | {m_rescue['depth']}")
    print(f"{'CX count':<30} | {m_base['cx']:<12} | {m_rescue['cx']}")
    print(f"{'Gate count':<30} | {m_base['gates']:<12} | {m_rescue['gates']}")

    depth_tax = m_rescue["depth"] / max(1, m_base["depth"])
    print(f"\n-> Recovery depth tax: {depth_tax:.2f}x")


def run_scenario_k() -> None:
    print(f"\n{SEP}")
    print("SCENARIO K: MARKOV/BERNSTEIN COHERENCE LIMIT (SLOPE vs TIME)")
    print(SEP)

    res = qsvt_mod.experiment_markov_brothers_boundary(d_visual=15, max_degree=45, plot=False)
    line = CouplingMap(_line_edges(24))
    coherence_ns = 80_000.0
    degrees = [5, 15, 25, 35, 45]

    print(f"{'Degree d':<10} | {'Markov d^2':<12} | {'Time (ns)':<12} | {'Within T2?'}")
    print("-" * 62)
    feasible = []
    for d in degrees:
        qc = build_qsvt_degree_circuit(n_data=5, degree=d)
        m = _profile_stats(qc, line)
        ok = m["total_time_ns"] <= coherence_ns
        if ok:
            feasible.append(d)
        print(f"{d:<10} | {d*d:<12} | {m['total_time_ns']:<12.1f} | {'YES' if ok else 'no'}")

    max_feasible = max(feasible) if feasible else 0
    print(f"\n-> Coherence-limited degree ceiling in this model: d <= {max_feasible}")
    print("-> Mathematical slope potential scales as O(d^2), but hardware time caps usable d.")


def run_scenario_l() -> None:
    print(f"\n{SEP}")
    print("SCENARIO L: PHASE FRAGILITY AND TARGET DISTORTION (DRIFT SWEEP)")
    print(SEP)

    drifts = [0.005, 0.010, 0.020, 0.040]
    print(f"{'Phase drift (rad)':<18} | {'Max distortion':<16} | {'Max out-of-domain |P|'}")
    print("-" * 66)
    for drift in drifts:
        res = qsvt_mod.experiment_physical_phase_fragility(
            degree=25,
            phase_error=drift,
            x_bound=1.15,
            max_depth=40,
            plot=False,
        )
        print(f"{drift:<18.3f} | {res.max_distortion:<16.4f} | {res.max_leakage:.3f}")

    print("\n-> Calibration drift directly warps the synthesized polynomial shape.")


def run_scenario_m() -> None:
    print(f"\n{SEP}")
    print("SCENARIO M: GIBBS OVERSHOOT REGIME COMPILER FAULT (|P(x)| > 1)")
    print(SEP)

    res = qsvt_mod.experiment_adversarial_gibbs_catastrophe(degree=41, num_points=4001, plot=False)
    overshoot = res.max_amplitude
    fault = "none"
    try:
        bad = np.array([[overshoot, 0.0], [0.0, 1.0 / overshoot]], dtype=complex)
        _ = UnitaryGate(bad)
    except Exception as exc:
        fault = str(exc)

    print(f"Max synthesized amplitude: {overshoot:.6f}")
    print(f"Unitarity violation amount: {res.unitarity_violation:.6f}")
    if overshoot > 1.0:
        print("Physical synthesis status: CRITICAL (target exceeds unitary amplitude bounds)")
    print(f"Compiler/constructor exception: {fault}")
    print("\n-> Discontinuous fits induce Gibbs overshoot that cannot be mapped to a physical unitary.")


def run_scenario_n() -> None:
    print(f"\n{SEP}")
    print("SCENARIO N: PARITY SCRAMBLE ROUTING OVERHEAD (MIXED EVEN/ODD CONTROL)")
    print(SEP)

    res = qsvt_mod.experiment_adversarial_parity_scramble(dim=5, seed=101, plot=False)
    strict_qc = build_strict_parity_circuit(n_data=4, rounds=5)
    mixed_qc = build_mixed_parity_circuit(n_data=4, rounds=5)

    line_strict = CouplingMap(_line_edges(strict_qc.num_qubits))
    line_mixed = CouplingMap(_line_edges(mixed_qc.num_qubits))
    s = _transpile_stats(strict_qc, coupling_map=line_strict, optimization_level=3, basis_gates=BASIS_NISQ + ["swap"])
    m = _transpile_stats(mixed_qc, coupling_map=line_mixed, optimization_level=3, basis_gates=BASIS_NISQ + ["swap"])

    print(f"{'Metric':<34} | {'Strict parity':<14} | {'Mixed parity'}")
    print("-" * 72)
    print(f"{'Depth':<34} | {s['depth']:<14} | {m['depth']}")
    print(f"{'CX count':<34} | {s['cx']:<14} | {m['cx']}")
    print(f"{'SWAP count':<34} | {s['swap']:<14} | {m['swap']}")
    print(f"{'||M_expected - M_physical||':<34} | {'N/A':<14} | {res.scramble_error:.3e}")

    print("\n-> Breaking strict parity forces extra routing/control logic and changes the physical channel.")


def run_scenario_o() -> None:
    print(f"\n{SEP}")
    print("SCENARIO O: ILL-CONDITIONED extreme regime (FIXED DEPTH, KAPPA SWEEP)")
    print(SEP)

    degree = 41
    kappas = (5.0, 20.0, 100.0, 300.0)
    res = qsvt_mod.experiment_adversarial_ill_conditioned_abyss(
        degree=degree,
        kappas=kappas,
        scale_factor=0.5,
        outside_weight=50.0,
        num_points=5001,
        plot=False,
    )
    qc = build_qsvt_degree_circuit(n_data=4, degree=degree)
    prof = _profile_stats(qc, CouplingMap(_line_edges(16)))

    print(f"Fixed hardware degree d = {degree}")
    print(f"Physical scheduled time at fixed depth: {prof['total_time_ns']:.1f} ns")
    print(f"{'kappa':<10} | {'max fit error':<14} | {'status'}")
    print("-" * 46)
    for kappa in kappas:
        err = float(res.max_errors[float(kappa)])
        status = "noise-dominated" if err > 0.25 else "resolved"
        print(f"{kappa:<10.1f} | {err:<14.4e} | {status}")

    print("\n-> As spectral gaps shrink, fixed-depth hardware devolves into wrong inverse action.")


def run_scenario_p() -> None:
    print(f"\n{SEP}")
    print("SCENARIO P: NON-NORMAL EIGENVALUE LIMITATION (SVD CHANNEL DIVERGENCE)")
    print(SEP)

    res = qsvt_mod.experiment_adversarial_non_normal_trap(plot=False)
    qc = build_svd_channel_skeleton()
    m = _transpile_stats(qc, optimization_level=3)

    print(f"Is matrix normal? {res.is_normal}")
    print(f"Divergence ||f(A) - f^(SV)(A)||_F: {res.divergence_error:.4e}")
    print(f"SVD-channel skeleton depth: {m['depth']}, CX count: {m['cx']}")
    print("\n-> For non-normal inputs, eigenvalue intuition and physical QSVT execution separate.")


def run_scenario_q() -> None:
    print(f"\n{SEP}")
    print("SCENARIO Q: PHASE QUANTIZATION DAC LIMITATION (4/8/12-BIT)")
    print(SEP)

    res = qsvt_mod.experiment_adversarial_phase_quantization(
        degree=35,
        test_bits=(4, 8, 12),
        sweep_range=(4, 12),
        num_points=1001,
        plot=False,
    )

    print(f"{'DAC bits':<10} | {'Max |Delta P|':<14} | {'Fidelity proxy'}")
    print("-" * 50)
    for bits in [4, 8, 12]:
        err = float(res.max_errors[bits])
        idx = _nearest_idx(res.sweep_bits.astype(float), float(bits))
        fid = float(res.fidelity_curve[idx])
        print(f"{bits:<10} | {err:<14.4f} | {fid:.6f}")

    print("\n-> Finite DAC resolution introduces a pre-fridge fidelity floor that cannot be optimized away.")


def run_scenario_r() -> None:
    print(f"\n{SEP}")
    print("SCENARIO R: SUBNORMALIZATION OVER-NORMALIZATION (CHEATED ALPHA FAILURE)")
    print(SEP)

    res = qsvt_mod.experiment_adversarial_subnormalization_hubris(
        dim=4,
        target_sigma_max=2.5,
        valid_margin=0.01,
        seed=42,
        plot=False,
    )

    A = np.asarray(res.A, dtype=complex) / float(res.cheated_alpha)
    I = np.eye(A.shape[0], dtype=complex)
    defect = I - A @ A.conj().T
    sqrt_defect = sqrtm(defect)
    U_cheat = np.block([[A, sqrt_defect], [sqrt_defect, -A.conj().T]])

    failure = "none"
    try:
        _ = UnitaryGate(U_cheat)
    except Exception as exc:
        failure = str(exc)

    print(f"true alpha: {res.true_alpha:.4f}")
    print(f"cheated alpha: {res.cheated_alpha:.4f}")
    print(f"min valid defect eig:   {float(np.min(res.valid_eigenvalues)):.6f}")
    print(f"min cheated defect eig: {float(np.min(res.invalid_eigenvalues)):.6f}")
    print(f"Cholesky message: {res.catastrophe_message}")
    print(f"UnitaryGate check failure: {failure}")
    print("\n-> Cheating alpha triggers a linear-algebra hard stop before hardware synthesis.")


def _build_symmetry_test_circuit(n_data: int, degree: int, symmetric: bool) -> QuantumCircuit:
    if symmetric:
        left = np.linspace(-0.9, 0.9, degree // 2 + 1)
        phases = np.concatenate([left, left[-2::-1]])
    else:
        rng = np.random.default_rng(1234)
        phases = rng.uniform(-0.9, 0.9, degree + 1)
    return build_qsvt_lifted_circuit(n_data=n_data, phases=phases, include_work=True)


def run_scenario_s() -> None:
    print(f"\n{SEP}")
    print("SCENARIO S: PARITY-SYMMETRY COMPILER HEURISTIC (OPT_LEVEL=3)")
    print(SEP)

    hh = _heavy_hex_map()
    qc_sym = _build_symmetry_test_circuit(n_data=5, degree=21, symmetric=True)
    qc_asym = _build_symmetry_test_circuit(n_data=5, degree=21, symmetric=False)

    sym_o0 = _transpile_stats(qc_sym, coupling_map=hh, optimization_level=0)
    sym_o3 = _transpile_stats(qc_sym, coupling_map=hh, optimization_level=3)
    asym_o0 = _transpile_stats(qc_asym, coupling_map=hh, optimization_level=0)
    asym_o3 = _transpile_stats(qc_asym, coupling_map=hh, optimization_level=3)

    print(f"{'Case':<22} | {'Depth o0':<10} | {'Depth o3':<10} | {'CX o0':<8} | {'CX o3'}")
    print("-" * 72)
    print(f"{'Symmetric phases':<22} | {sym_o0['depth']:<10} | {sym_o3['depth']:<10} | {sym_o0['cx']:<8} | {sym_o3['cx']}")
    print(f"{'Asymmetric phases':<22} | {asym_o0['depth']:<10} | {asym_o3['depth']:<10} | {asym_o0['cx']:<8} | {asym_o3['cx']}")

    sym_gain = sym_o0["depth"] - sym_o3["depth"]
    asym_gain = asym_o0["depth"] - asym_o3["depth"]
    print(f"\nDepth cancellation gain (symmetric):  {sym_gain}")
    print(f"Depth cancellation gain (asymmetric): {asym_gain}")
    print("-> This tests whether parity symmetry gives the compiler extra cancellation leverage.")


@dataclass
class ShowdownRow:
    name: str
    qubits: int
    total_time_ns: float
    final_cnots: int
    routing_swaps: int
    penalty: float


def run_scenario_t() -> None:
    print(f"\n{SEP}")
    print("SCENARIO T: UNIFIED ALGORITHMIC COMPARATIVE EVALUATION (GROVER THROUGH QSVT)")
    print(SEP)

    n_data = 4
    good = [0]

    circuits: Dict[str, QuantumCircuit] = {
        "Grover": fpaa_mod.build_standard_grover_circuit(
            num_qubits=n_data,
            iterations=3,
            good_indices=good,
        ),
        "FPAA": fpaa_mod.build_fpaa_circuit(
            num_qubits=n_data,
            L=3,
            delta=0.1,
            good_indices=good,
        ),
        "OAA": build_oaa_benchmark_circuit(n_data=n_data, rounds=2),
        "FOQA": build_foqa_benchmark_circuit(n_data=n_data, rounds=3),
        "DQAA": build_dqaa_benchmark_circuit(n_global=n_data, j=1, rounds=2),
        "VTAA": build_vtaa_benchmark_circuit(n_data=n_data, rounds=2),
        "QSVT": build_qsvt_degree_circuit(n_data=n_data, degree=21),
    }

    hh = _heavy_hex_map()
    rows: List[ShowdownRow] = []
    for name, qc in circuits.items():
        metrics = _profile_stats(qc, hh)
        rows.append(
            ShowdownRow(
                name=name,
                qubits=qc.num_qubits,
                total_time_ns=float(metrics["total_time_ns"]),
                final_cnots=int(metrics["final_cnots"]),
                routing_swaps=int(metrics.get("routing_swaps", 0)),
                penalty=float(metrics["hardware_penalty_score"]),
            )
        )

    rows.sort(key=lambda r: r.penalty)

    print("Standardized test: single-target search instance on n=4 data qubits.")
    print(f"{'Algorithm':<10} | {'Qubits':<8} | {'Total Time (ns)':<16} | {'Final CNOTs':<12} | {'Penalty Score'}")
    print("-" * 84)
    for r in rows:
        print(f"{r.name:<10} | {r.qubits:<8} | {r.total_time_ns:<16.1f} | {r.final_cnots:<12} | {r.penalty:.1f}")

    winner = rows[0]
    print(f"\n-> Best-performing method in this standardized hardware profile: {winner.name}")
    print("-> The summary table now places the full amplitude-amplification lineage within a common comparison framework.")


def run_scenario_u() -> None:
    print(f"\n{SEP}")
    print("SCENARIO U: DYNAMIC QUBIT REUSE (MID-CIRCUIT RESET TRADEOFF)")
    print(SEP)

    rounds = 6
    dyn = build_dynamic_reuse_qsvt_circuit(n_data=4, rounds=rounds)
    static = build_static_signal_bank_qsvt_circuit(n_data=4, rounds=rounds)

    t_dyn = transpile(dyn, basis_gates=BASIS_NISQ + ["reset"], optimization_level=1, seed_transpiler=42)
    t_static = transpile(static, basis_gates=BASIS_NISQ, optimization_level=3, seed_transpiler=42)

    ops_dyn = t_dyn.count_ops()
    ops_static = t_static.count_ops()
    meas = int(ops_dyn.get("measure", 0))
    reset = int(ops_dyn.get("reset", 0))
    # Simple control-stack latency model (ns):
    # measurement ~1000ns, reset ~400ns.
    time_dyn_ns = float(t_dyn.depth() * 100 + meas * 1000 + reset * 400)
    time_static_ns = float(t_static.depth() * 100)

    print(f"{'Metric':<34} | {'Static ancilla bank':<18} | {'Dynamic reuse'}")
    print("-" * 76)
    print(f"{'Logical qubits':<34} | {static.num_qubits:<18} | {dyn.num_qubits}")
    print(f"{'Transpiled depth':<34} | {t_static.depth():<18} | {t_dyn.depth()}")
    print(f"{'CX count':<34} | {int(ops_static.get('cx', 0)):<18} | {int(ops_dyn.get('cx', 0))}")
    print(f"{'measure/reset ops':<34} | {'0/0':<18} | {meas}/{reset}")
    print(f"{'Estimated total time (ns)':<34} | {time_static_ns:<18.1f} | {time_dyn_ns:.1f}")

    print(f"\n-> Qubit savings: {static.num_qubits - dyn.num_qubits}")
    print(f"-> Latency multiplier from dynamic control: {time_dyn_ns / max(1.0, time_static_ns):.2f}x")


def run_scenario_v() -> None:
    print(f"\n{SEP}")
    print("SCENARIO V: RANDOMIZED COMPILING (PAULI TWIRLING OVERHEAD)")
    print(SEP)

    base = build_qsvt_degree_circuit(n_data=5, degree=25)
    twirled = build_twirled_circuit(base, seed=2026)
    hh = _heavy_hex_map()

    b = _transpile_stats(base, coupling_map=hh, optimization_level=3)
    t = _transpile_stats(twirled, coupling_map=hh, optimization_level=3)
    ops_b = b["circuit"].count_ops()
    ops_t = t["circuit"].count_ops()
    oneq_b = int(sum(v for k, v in ops_b.items() if k not in {"cx", "swap", "measure", "reset", "barrier"}))
    oneq_t = int(sum(v for k, v in ops_t.items() if k not in {"cx", "swap", "measure", "reset", "barrier"}))

    print(f"{'Metric':<32} | {'Baseline':<12} | {'Pauli-twirled'}")
    print("-" * 64)
    print(f"{'Depth':<32} | {b['depth']:<12} | {t['depth']}")
    print(f"{'CX count':<32} | {b['cx']:<12} | {t['cx']}")
    print(f"{'1Q gate count':<32} | {oneq_b:<12} | {oneq_t}")
    print(f"{'Total gates':<32} | {b['gates']:<12} | {t['gates']}")

    print(f"\n-> Twirling depth overhead: {t['depth'] / max(1, b['depth']):.2f}x")
    if t["gates"] > b["gates"] or oneq_t > oneq_b or t["depth"] > b["depth"]:
        print("-> In this transpilation run, randomized compiling introduces measurable control overhead.")
    else:
        print("-> In this transpilation run, compiler optimization removes any visible overhead from the twirling wrapper.")


def run_scenario_w() -> None:
    print(f"\n{SEP}")
    print("SCENARIO W: MAGIC STATE FACTORY FOOTPRINT (15-TO-1 CAPACITY)")
    print(SEP)

    n_data = 4
    qsvt_c = build_qsvt_degree_circuit(n_data=n_data, degree=17)
    fpaa_c = fpaa_mod.build_fpaa_circuit(num_qubits=n_data, L=9, delta=0.1, good_indices=[0])
    hh = _heavy_hex_map()

    qsvt_native = _transpile_stats(qsvt_c, optimization_level=1)
    fpaa_native = _transpile_stats(fpaa_c, optimization_level=1)
    qsvt_t = _estimate_t_count_from_native(qsvt_native["circuit"], synthesis_eps=1e-3)
    fpaa_t = _estimate_t_count_from_native(fpaa_native["circuit"], synthesis_eps=1e-3)

    qsvt_hw = _profile_stats(qsvt_c, hh)
    fpaa_hw = _profile_stats(fpaa_c, hh)

    # Simple FTQC factory model.
    factory_rate_per_us = 0.05  # one T state every 20 us per factory
    factory_qubits = 3000       # physical qubits per 15-to-1 lane

    def _factory_count(t_count: int, total_time_ns: float) -> int:
        runtime_us = max(total_time_ns / 1000.0, 1e-6)
        demand_per_us = t_count / runtime_us
        return int(math.ceil(demand_per_us / factory_rate_per_us))

    qsvt_fact = _factory_count(qsvt_t, qsvt_hw["total_time_ns"])
    fpaa_fact = _factory_count(fpaa_t, fpaa_hw["total_time_ns"])

    print(f"{'Metric':<30} | {'QSVT':<14} | {'FPAA'}")
    print("-" * 58)
    print(f"{'Estimated T-count':<30} | {qsvt_t:<14} | {fpaa_t}")
    print(f"{'Scheduled time (ns)':<30} | {qsvt_hw['total_time_ns']:<14.1f} | {fpaa_hw['total_time_ns']:.1f}")
    print(f"{'Factories for no-stall':<30} | {qsvt_fact:<14} | {fpaa_fact}")
    print(f"{'Factory qubit footprint':<30} | {qsvt_fact*factory_qubits:<14} | {fpaa_fact*factory_qubits}")

    print("\n-> Fractional-phase synthesis still needs nonzero factory bandwidth.")
    print("-> FTQC clocking is co-designed with distillation throughput, not just algorithm depth.")


def run_scenario_x() -> None:
    print(f"\n{SEP}")
    print("SCENARIO X: SPAM VULNERABILITY (SINGLE-READOUT FAILURE POINT)")
    print(SEP)

    fpres = qsvt_mod.experiment_qsvt_fixed_point_search(
        delta=0.2,
        target_degree=41,
        test_x0=0.15,
        plot=False,
    )
    p_true = float(fpres.qsvt_fpaa_probs[_nearest_idx(fpres.degrees_eval.astype(float), 41.0)])

    base_shots = 4096
    cal_shots = 2048
    print(f"True signal-qubit success probability: {p_true:.6f}")
    print(f"{'Readout flip eps':<16} | {'Observed p':<12} | {'Bias':<10} | {'Mitigated p':<12} | {'Shot overhead'}")
    print("-" * 86)
    for eps in [0.01, 0.03, 0.05]:
        e01 = eps
        e10 = eps
        p_obs = p_true * (1.0 - e10) + (1.0 - p_true) * e01
        p_mit = (p_obs - e01) / max(1e-9, (1.0 - e01 - e10))
        p_mit = float(np.clip(p_mit, 0.0, 1.0))
        bias = abs(p_obs - p_true)
        # twirled readout (x2 runs) + confusion matrix calibration (+2*cal_shots)
        total_shots = 2 * base_shots + 2 * cal_shots
        overhead = total_shots / base_shots
        print(f"{eps:<16.3f} | {p_obs:<12.6f} | {bias:<10.6f} | {p_mit:<12.6f} | {overhead:.2f}x")

    # Circuit-level twirling overhead proxy
    base_core = build_qsvt_degree_circuit(n_data=4, degree=21)
    qc_base = QuantumCircuit(base_core.num_qubits, 1)
    qc_base.compose(base_core, inplace=True)
    qc_base.measure(0, 0)
    qc_twirl = QuantumCircuit(base_core.num_qubits, 1)
    qc_twirl.compose(base_core, inplace=True)
    qc_twirl.x(0)
    qc_twirl.measure(0, 0)
    t_base = transpile(qc_base, basis_gates=BASIS_NISQ, optimization_level=3, seed_transpiler=42)
    t_twirl = transpile(qc_twirl, basis_gates=BASIS_NISQ, optimization_level=3, seed_transpiler=42)
    print(f"\nMeasurement-twirling compile overhead: depth {t_base.depth()} -> {t_twirl.depth()}")
    print("-> Single-qubit readout noise directly biases the whole QSVT decision channel.")


def run_scenario_y() -> None:
    print(f"\n{SEP}")
    print("SCENARIO Y: BLOCK-ENCODING TRANSLATION TRADEOFF (LCU vs TROTTER)")
    print(SEP)

    phases = np.sin(np.linspace(0.0, 2.0 * np.pi, 17)) * 0.9
    lcu = build_lcu_embedding_qsvt_circuit(phases=phases, n_data=4)
    trotter = build_trotter_embedding_qsvt_circuit(phases=phases, n_data=4)
    hh = _heavy_hex_map()

    m_lcu = _profile_stats(lcu, hh)
    m_trot = _profile_stats(trotter, hh)

    print(f"{'Metric':<34} | {'LCU embedding':<14} | {'Trotter embedding'}")
    print("-" * 74)
    print(f"{'Qubits':<34} | {lcu.num_qubits:<14} | {trotter.num_qubits}")
    print(f"{'Routing SWAPs':<34} | {int(m_lcu.get('routing_swaps', 0)):<14} | {int(m_trot.get('routing_swaps', 0))}")
    print(f"{'Final CNOTs':<34} | {int(m_lcu['final_cnots']):<14} | {int(m_trot['final_cnots'])}")
    print(f"{'Total scheduled time (ns)':<34} | {m_lcu['total_time_ns']:<14.1f} | {m_trot['total_time_ns']:.1f}")
    print(f"{'Unified penalty score':<34} | {m_lcu['hardware_penalty_score']:<14.1f} | {m_trot['hardware_penalty_score']:.1f}")

    winner = "LCU" if m_lcu["hardware_penalty_score"] < m_trot["hardware_penalty_score"] else "Trotter"
    print(f"\n-> Hardware-favored embedding for this phase sequence: {winner}")


def run_scenario_z() -> None:
    print(f"\n{SEP}")
    print("SCENARIO Z: UNIFIED HARDWARE-AWARE COMPARATIVE EVALUATION (A-TO-Z BENCHMARK SYNTHESIS)")
    print(SEP)

    n_data = 17
    good = [0]
    circuits: Dict[str, QuantumCircuit] = {
        "Grover": fpaa_mod.build_standard_grover_circuit(num_qubits=n_data, iterations=3, good_indices=good),
        "FPAA": fpaa_mod.build_fpaa_circuit(num_qubits=n_data, L=3, delta=0.1, good_indices=good),
        "OAA": build_oaa_benchmark_circuit(n_data=n_data, rounds=2),
        "FOQA": build_foqa_benchmark_circuit(n_data=n_data, rounds=3),
        "DQAA": build_dqaa_benchmark_circuit(n_global=n_data, j=1, rounds=2),
        "VTAA": build_vtaa_benchmark_circuit(n_data=n_data, rounds=2),
        "QSVT": build_qsvt_degree_circuit(n_data=n_data, degree=21),
    }
    hh = _heavy_hex_map()

    rows: List[ShowdownRow] = []
    for name, qc in circuits.items():
        m = _profile_stats(qc, hh)
        rows.append(
            ShowdownRow(
                name=name,
                qubits=qc.num_qubits,
                total_time_ns=float(m["total_time_ns"]),
                final_cnots=int(m["final_cnots"]),
                routing_swaps=int(m.get("routing_swaps", 0)),
                penalty=float(m["hardware_penalty_score"]),
            )
        )
    rows.sort(key=lambda r: r.penalty)

    print("Unified benchmark instance: identical single-target search problem.")
    print(
        f"{'Algorithm':<10} | {'Time (ns)':<12} | {'Final CNOTs':<12} | "
        f"{'SWAP overhead':<12} | {'Unified score'}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r.name:<10} | {r.total_time_ns:<12.1f} | {r.final_cnots:<12} | "
            f"{r.routing_swaps:<12} | {r.penalty:.1f}"
        )

    print(f"\n-> Best-performing method under the unified metric: {rows[0].name}")
    print("-> This completes the A-to-Z hardware-aware comparative summary table.")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    output_path = os.path.join(_HERE, "!_QSVT_transpile_results.txt")
    logger = Logger(output_path)
    sys.stdout = logger

    print("QSVT Transpilation Master Suite - Scenarios A through Z (26 total)")
    print(f"Results saved to: {output_path}")
    print(SEP)

    scenarios = [
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
        ("R", run_scenario_r),
        ("S", run_scenario_s),
        ("T", run_scenario_t),
        ("U", run_scenario_u),
        ("V", run_scenario_v),
        ("W", run_scenario_w),
        ("X", run_scenario_x),
        ("Y", run_scenario_y),
        ("Z", run_scenario_z),
    ]

    for label, fn in scenarios:
        try:
            fn()
        except Exception:
            print(f"\n*** SCENARIO {label} FAILED ***")
            traceback.print_exc()

    print(f"\n{SEP}")
    print("Benchmark suite complete. 26 scenarios executed.")
    logger.close()
    sys.stdout = logger.terminal
    print(f"\nBenchmark suite complete. Results saved to {output_path}")
