from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXVChain
from qiskit.quantum_info import Statevector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ampamp.fixed_point import FixedPointEngine
from ampamp.grover import GroverEngine
from ampamp.transpilation import TranspilationProfileConfig, TranspilationProfiler
from _shared_gpu_library import Logger, run_interactive_scenario_repl

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
STANDARD_QUBITS = 12


BASIS = ["cx", "id", "rz", "sx", "x"]


def _l_from_L(L: int) -> int:
    L = int(L)
    if L < 1 or (L % 2 == 0):
        raise ValueError("L must be an odd integer >= 1")
    return (L - 1) // 2


def _build_fpaa(n_qubits: int = 5, L: int = 3, delta: float = 0.1, good_indices: Sequence[int] = (0,)) -> QuantumCircuit:
    return FixedPointEngine(int(L), float(delta)).build_fixed_point_circuit(int(n_qubits), list(good_indices))


def _build_grover_equiv(n_qubits: int = 5, L: int = 3, good_indices: Sequence[int] = (0,)) -> QuantumCircuit:
    grover = GroverEngine(int(n_qubits), list(good_indices))
    return grover.construct_circuit(iterations=_l_from_L(int(L)))


def _tx(qc: QuantumCircuit, *, coupling_map: List[List[int]] | None = None, basis: Sequence[str] = BASIS, level: int = 3) -> Dict[str, float]:
    t_qc = transpile(qc, coupling_map=coupling_map, basis_gates=list(basis), optimization_level=int(level))
    ops = t_qc.count_ops()
    return {
        "depth": float(t_qc.depth()),
        "size": float(t_qc.size()),
        "cx": float(ops.get("cx", 0)),
        "rz": float(ops.get("rz", 0)),
        "t": float(ops.get("t", 0) + ops.get("tdg", 0)),
    }


def _prof(qc: QuantumCircuit, *, coupling_map: List[List[int]] | None = None) -> Dict[str, float]:
    cfg = TranspilationProfileConfig(coupling_map_edges=coupling_map, basis_gates=tuple(BASIS))
    m = TranspilationProfiler(cfg).profile_circuit(qc)
    return {
        "score": float(m["hardware_penalty_score"]),
        "time_ns": float(m["total_time_ns"]),
        "final_cnots": float(m["final_cnots"]),
        "swaps": float(m["routing_swaps"]),
        "distance": float(m["initial_distance_penalty"]),
    }


def run_scenario_a_unrolling_baseline(n_qubits: int = 5, good_indices: List[int] = [0], L: int = 3, optimization_level: int = 3) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO A: CONTINUOUS-ANGLE UNROLLING BASELINE")
    print("=" * 70)
    l_fpaa = _l_from_L(L)
    fpaa = _build_fpaa(n_qubits, L, 0.1, good_indices)
    grover = _build_grover_equiv(n_qubits, L, good_indices)
    m_fpaa = _tx(fpaa, level=optimization_level)
    m_grov = _tx(grover, level=optimization_level)

    print(f"FPAA L={L} (l={l_fpaa}) depth={int(m_fpaa['depth'])} cx={int(m_fpaa['cx'])} rz={int(m_fpaa['rz'])}")
    print(f"Grover k={l_fpaa} depth={int(m_grov['depth'])} cx={int(m_grov['cx'])} rz={int(m_grov['rz'])}")
    if l_fpaa > 0:
        print(f"CX/iterate FPAA={m_fpaa['cx']/l_fpaa:.1f}, Grover={m_grov['cx']/l_fpaa:.1f}")
        print(f"RZ overhead factor={(m_fpaa['rz']/max(1.0,m_grov['rz'])):.2f}x")


def run_scenario_b_topological_routing(n_qubits: int = 5, good_indices: List[int] = [0], L: int = 3, optimization_level: int = 3) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO B: TOPOLOGICAL PARAMETER ROUTING")
    print("=" * 70)
    qc = _build_fpaa(n_qubits, L, 0.1, good_indices)
    linear = [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]
    heavy = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3]] if n_qubits >= 5 else linear

    m_all = _tx(qc, coupling_map=None, level=optimization_level)
    m_hex = _tx(qc, coupling_map=heavy, level=optimization_level)
    m_lin = _tx(qc, coupling_map=linear, level=optimization_level)

    print(f"All-to-all depth={int(m_all['depth'])}, cx={int(m_all['cx'])}")
    print(f"Heavy-hex depth={int(m_hex['depth'])}, cx={int(m_hex['cx'])}")
    print(f"Linear depth={int(m_lin['depth'])}, cx={int(m_lin['cx'])}")


def run_scenario_c_synthesis_annihilation_failure(n_qubits: int = 5, good_indices: List[int] = [0], L: int = 5) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO C: SYNTHESIS/ANNIHILATION LIMITS")
    print("=" * 70)
    qc = _build_fpaa(n_qubits, L, 0.1, good_indices)
    m0 = _tx(qc, level=0)
    m3 = _tx(qc, level=3)
    print(f"opt0 depth={int(m0['depth'])}, cx={int(m0['cx'])}, rz={int(m0['rz'])}")
    print(f"opt3 depth={int(m3['depth'])}, cx={int(m3['cx'])}, rz={int(m3['rz'])}")


def run_scenario_d_passband_tightening_breaking_point(n_qubits: int = 5, target_p: float = 0.1) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO D: PASSBAND TIGHTENING BREAKPOINT")
    print("=" * 70)
    print("L | depth | cx | rz")
    print("-" * 40)
    for L in [3, 5, 7, 9, 11]:
        qc = _build_fpaa(n_qubits, L, max(1e-4, target_p), [0])
        m = _tx(qc)
        print(f"{L:<2}| {int(m['depth']):<5} | {int(m['cx']):<3} | {int(m['rz'])}")


def run_scenario_e_high_density_rescue(n_qubits: int = 6, M: int = 48) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO E: HIGH-DENSITY REGIME")
    print("=" * 70)
    N = 2 ** int(n_qubits)
    M = int(M)
    if not (N // 2 < M < N):
        raise ValueError(f"Require N/2 < M < N. got N={N}, M={M}")
    good = list(range(M))
    fpaa = _build_fpaa(n_qubits, 5, 0.1, good)
    grov = _build_grover_equiv(n_qubits, 5, good)
    m_fpaa = _tx(fpaa)
    m_grov = _tx(grov)
    print(f"N={N}, M={M}, classical_success={M/N:.4f}")
    print(f"FPAA depth={int(m_fpaa['depth'])}, cx={int(m_fpaa['cx'])}")
    print(f"Grover depth={int(m_grov['depth'])}, cx={int(m_grov['cx'])}")


def run_scenario_f_fault_tolerant_t_gate_explosion(n_qubits: int = 4, L: int = 3, synthesis_eps: float = 1e-3) -> None:
    _ = synthesis_eps
    print("\n" + "=" * 70)
    print("SCENARIO F: FAULT-TOLERANT T-GATE EXPLOSION")
    print("=" * 70)
    ft_basis = ["h", "s", "sdg", "cx", "t", "tdg"]
    qc = _build_fpaa(n_qubits, L, 0.1, [0])
    m = _tx(qc, basis=ft_basis)
    print(f"FT depth={int(m['depth'])}, total_T={int(m['t'])}, cx={int(m['cx'])}")


def run_scenario_g_modular_nesting_tradeoff(n_qubits: int = 4, L1: int = 3, L2: int = 3) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO G: MODULAR NESTING TRADEOFF")
    print("=" * 70)
    mono = _build_fpaa(n_qubits, int(L1 * L2), 0.1, [0]) if (L1 * L2) % 2 == 1 else _build_fpaa(n_qubits, 9, 0.1, [0])
    nested = QuantumCircuit(n_qubits)
    nested.compose(_build_fpaa(n_qubits, L1, 0.1, [0]), inplace=True)
    nested.compose(_build_fpaa(n_qubits, L2, 0.1, [0]), inplace=True)
    m_mono = _tx(mono)
    m_nest = _tx(nested)
    print(f"Monolithic depth={int(m_mono['depth'])}, cx={int(m_mono['cx'])}, rz={int(m_mono['rz'])}")
    print(f"Nested depth={int(m_nest['depth'])}, cx={int(m_nest['cx'])}, rz={int(m_nest['rz'])}")


def run_scenario_h_coherent_calibration_trap(n_qubits: int = 4, L: int = 5) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO H: COHERENT CALIBRATION TRAP")
    print("=" * 70)
    base_engine = FixedPointEngine(L, 0.1)
    pert_engine = FixedPointEngine(L, 0.1)
    pert_engine.alphas = pert_engine.alphas * 1.03
    pert_engine.betas = pert_engine.betas * 1.03

    base = base_engine.build_fixed_point_circuit(n_qubits, [0])
    pert = pert_engine.build_fixed_point_circuit(n_qubits, [0])

    sv_base = Statevector.from_instruction(base)
    sv_pert = Statevector.from_instruction(pert)
    p0_base = abs(sv_base.data[0]) ** 2
    p0_pert = abs(sv_pert.data[0]) ** 2
    print(f"ideal target prob={p0_base:.6f}")
    print(f"+3% phase-scale target prob={p0_pert:.6f}")
    print(f"absolute drop={max(0.0, p0_base - p0_pert):.6f}")


def run_scenario_i_ancilla_assisted_mcp_decomposition(n_qubits: int = STANDARD_QUBITS) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO I: ANCILLA-ASSISTED MCP DECOMPOSITION")
    print("=" * 70)
    n_qubits = int(n_qubits)
    ctrl_count = n_qubits - 1
    anc = max(0, ctrl_count - 2)

    qc0 = QuantumCircuit(n_qubits)
    qc0.h(range(n_qubits))
    qc0.h(n_qubits - 1)
    qc0.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc0.h(n_qubits - 1)

    qc1 = QuantumCircuit(n_qubits + anc)
    qc1.h(range(n_qubits))
    vchain = MCXVChain(num_ctrl_qubits=ctrl_count, dirty_ancillas=False)
    qc1.h(n_qubits - 1)
    qc1.append(vchain, list(range(n_qubits - 1)) + [n_qubits - 1] + list(range(n_qubits, n_qubits + anc)))
    qc1.h(n_qubits - 1)

    m0 = _tx(qc0)
    m1 = _tx(qc1)
    print(f"No-ancilla depth={int(m0['depth'])}, cx={int(m0['cx'])}")
    print(f"With {anc} ancilla depth={int(m1['depth'])}, cx={int(m1['cx'])}")


def run_scenario_j_plateau_overhead_tax(p_target: float = 0.05, P_floor: float = 0.99) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO J: PLATEAU OVERHEAD TAX")
    print("=" * 70)
    p_target = float(p_target)
    P_floor = float(P_floor)
    if not (0 < p_target <= 1 and 0 < P_floor < 1):
        raise ValueError("Require 0<p_target<=1 and 0<P_floor<1")

    # Conservative odd-L proxy from fixed-point schedule scaling.
    L_est = int(math.ceil(math.log(2.0 / max(1e-12, 1.0 - P_floor)) / max(1e-9, math.sqrt(p_target))))
    if L_est % 2 == 0:
        L_est += 1
    query_complexity = max(0, L_est - 1)

    qc = _build_fpaa(6, L_est, 0.1, [0])
    m = _tx(qc)
    print(f"p_target={p_target:.4f}, P_floor={P_floor:.4f}")
    print(f"estimated odd L={L_est}, query_complexity~{query_complexity}")
    print(f"transpiled depth={int(m['depth'])}, cx={int(m['cx'])}, rz={int(m['rz'])}")


def run_scenario_k_unified_profiler_evaluation(n_qubits: int = STANDARD_QUBITS, p_density: float = 0.25) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO K: UNIFIED PROFILER EVALUATION")
    print("=" * 70)
    n_qubits = int(n_qubits)
    N = 2 ** n_qubits
    M = max(1, min(N - 1, int(round(float(p_density) * N))))
    good = list(range(M))

    fpaa = _build_fpaa(n_qubits, 5, 0.1, good)
    grov = _build_grover_equiv(n_qubits, 5, good)
    linear = [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]

    p_fpaa = _prof(fpaa, coupling_map=linear)
    p_grov = _prof(grov, coupling_map=linear)

    print(f"FPAA score={p_fpaa['score']:.2f}, time_ns={p_fpaa['time_ns']:.1f}, cnots={int(p_fpaa['final_cnots'])}, swaps={int(p_fpaa['swaps'])}")
    print(f"Grover score={p_grov['score']:.2f}, time_ns={p_grov['time_ns']:.1f}, cnots={int(p_grov['final_cnots'])}, swaps={int(p_grov['swaps'])}")


if __name__ == "__main__":
    output_filepath = os.path.join(_RESULT_DIR, "terminal_output.log")
    logger = Logger(output_filepath)
    sys.stdout = logger
    scenarios = [
        ("A", lambda: run_scenario_a_unrolling_baseline(5, [0], 3, 3)),
        ("B", lambda: run_scenario_b_topological_routing(5, [0], 3, 3)),
        ("C", lambda: run_scenario_c_synthesis_annihilation_failure(5, [0], 5)),
        ("D", lambda: run_scenario_d_passband_tightening_breaking_point(5, 0.1)),
        ("E", lambda: run_scenario_e_high_density_rescue(6, 48)),
        ("F", lambda: run_scenario_f_fault_tolerant_t_gate_explosion(4, 3, 1e-3)),
        ("G", lambda: run_scenario_g_modular_nesting_tradeoff(4, 3, 3)),
        ("H", lambda: run_scenario_h_coherent_calibration_trap(4, 5)),
        ("I", lambda: run_scenario_i_ancilla_assisted_mcp_decomposition(8)),
        ("J", lambda: run_scenario_j_plateau_overhead_tax(0.05, 0.99)),
        ("K", lambda: run_scenario_k_unified_profiler_evaluation(8, 0.25)),
    ]
    interactive = [
        ("A", run_scenario_a_unrolling_baseline),
        ("B", run_scenario_b_topological_routing),
        ("C", run_scenario_c_synthesis_annihilation_failure),
        ("D", run_scenario_d_passband_tightening_breaking_point),
        ("E", run_scenario_e_high_density_rescue),
        ("F", run_scenario_f_fault_tolerant_t_gate_explosion),
        ("G", run_scenario_g_modular_nesting_tradeoff),
        ("H", run_scenario_h_coherent_calibration_trap),
        ("I", run_scenario_i_ancilla_assisted_mcp_decomposition),
        ("J", run_scenario_j_plateau_overhead_tax),
        ("K", run_scenario_k_unified_profiler_evaluation),
    ]
    try:
        for _, fn in scenarios:
            fn()
        run_interactive_scenario_repl(interactive, sep="=" * 70)
    finally:
        logger.log.close()
        sys.stdout = logger.terminal
        print("FPAA library GPU suite complete.")
