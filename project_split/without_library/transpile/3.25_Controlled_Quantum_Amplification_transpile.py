"""
Controlled Quantum Amplification transpilation suite (Scenarios A through G)
============================================================================
Hardware-proxy comparison for the 2017 Controlled Quantum Amplification
framework, mirroring the structure of the other transpile-side benchmark
files in this repository.

The suite uses a small two-qubit search register together with one control
qubit. The search operator A = W G is compared against the controlled circuit
U that implements the CQAA wrapper.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import os
import sys
import traceback

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from aer_publishability import (
    parse_publishability_cli,
    prepare_backend_validation_artifacts,
    render_backend_validation_summary,
    run_cli_scenario,
    save_figure_with_metadata,
    wrap_scenarios,
)

from transpile_path_utils import ensure_directory_on_syspath, resolve_project_file


_HERE = os.fspath(ensure_directory_on_syspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)
_CQAA_PATH = os.fspath(
    resolve_project_file(__file__, "3.25_Controlled_Quantum_Amplification.py", preferred_dirs=("Theory Algorithms",))
)


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _import_cqaa():
    spec = importlib.util.spec_from_file_location("cqaa_module", _CQAA_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["cqaa_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


cqaa = _import_cqaa()


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
    print("Enter a label like A or G, or press Enter to finish.")
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
        print("Example: epsilon=0.08, repetitions=3")
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


SEP = "=" * 70
BASIS = ["cx", "id", "rz", "sx", "x"]


def _bit_reversal_permutation(num_qubits: int) -> list[int]:
    return [int(format(idx, f"0{num_qubits}b")[::-1], 2) for idx in range(2 ** num_qubits)]


def _to_qiskit_order_vector(vec: np.ndarray, num_qubits: int) -> np.ndarray:
    perm = _bit_reversal_permutation(num_qubits)
    return np.asarray(vec, dtype=complex)[perm]


def _to_qiskit_order_matrix(mat: np.ndarray, num_qubits: int) -> np.ndarray:
    perm = _bit_reversal_permutation(num_qubits)
    return np.asarray(mat, dtype=complex)[np.ix_(perm, perm)]


def _control_rotation(theta_tilde: float) -> QuantumCircuit:
    qc = QuantumCircuit(1, name="R_tilde")
    qc.ry(2.0 * theta_tilde, 0)
    return qc


def build_single_target_instance(
    epsilon: float = 0.10,
    rotation_angles: tuple[float, ...] = (0.58,),
    target_weights: tuple[float, ...] = (0.90, 0.25, 0.10),
):
    return cqaa.build_cqaa_instance(
        search_dim=4,
        epsilon=epsilon,
        rotation_angles=rotation_angles,
        target_weights=target_weights,
    )


def build_multiple_target_instance(
    epsilon_pi: float = 0.12,
    rotation_angles: tuple[float, ...] = (0.58,),
    target_weights: tuple[float, ...] = (0.90, 0.25, 0.10),
):
    return cqaa.build_cqaa_multiple_target_instance(
        search_dim=4,
        epsilon_pi=epsilon_pi,
        rotation_angles=rotation_angles,
        target_weights=target_weights,
    )


def _prepare_search_state(search_register: QuantumRegister, vector: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(search_register, name="PrepInit")
    qiskit_vec = _to_qiskit_order_vector(vector, len(search_register))
    qc.append(StatePreparation(qiskit_vec), search_register[:])
    return qc


def build_detection_operator_circuit(instance, repetitions: int = 1, include_prep: bool = True) -> QuantumCircuit:
    search = QuantumRegister(int(round(math.log2(instance.search_dim))), "search")
    qc = QuantumCircuit(search, name="A_power")
    if include_prep:
        qc.compose(_prepare_search_state(search, instance.initial_perp), inplace=True)
    a_qiskit = _to_qiskit_order_matrix(instance.detection_operator, len(search))
    gate_a = UnitaryGate(Operator(a_qiskit), label="A")
    for _ in range(repetitions):
        qc.append(gate_a, search[:])
    return qc


def build_controlled_cqaa_circuit(instance, repetitions: int = 1, include_prep: bool = True) -> QuantumCircuit:
    control = QuantumRegister(1, "ctrl")
    search = QuantumRegister(int(round(math.log2(instance.search_dim))), "search")
    qc = QuantumCircuit(control, search, name="CQAA")
    if include_prep:
        qc.compose(_prepare_search_state(search, instance.initial_perp), qubits=search[:], inplace=True)

    w_qiskit = _to_qiskit_order_matrix(instance.walk_operator, len(search))
    g_math = np.eye(instance.search_dim) - 2.0 * instance.target_projector
    g_qiskit = _to_qiskit_order_matrix(g_math, len(search))
    gate_w = UnitaryGate(Operator(w_qiskit), label="W")
    gate_g = UnitaryGate(Operator(g_qiskit), label="G")
    rot = _control_rotation(instance.theta_tilde)
    ctrl_g = gate_g.control(1, ctrl_state=0)
    ctrl_w = gate_w.control(1, ctrl_state=0)

    for _ in range(repetitions):
        # Implement |0_tilde><0_tilde| ⊗ G + |1_tilde><1_tilde| ⊗ I
        # by conjugating the computational-basis ctrl-0 block with R_tilde^dagger on the left
        # and R_tilde on the right.
        qc.compose(rot.inverse(), qubits=[control[0]], inplace=True)
        qc.append(ctrl_g, [control[0]] + search[:])
        qc.compose(rot, qubits=[control[0]], inplace=True)
        qc.append(ctrl_w, [control[0]] + search[:])
    return qc


def build_direct_matrix_cqaa_circuit(instance, repetitions: int = 1, include_prep: bool = True) -> QuantumCircuit:
    control = QuantumRegister(1, "ctrl")
    search = QuantumRegister(int(round(math.log2(instance.search_dim))), "search")
    qc = QuantumCircuit(control, search, name="CQAA_matrix")
    if include_prep:
        qc.compose(_prepare_search_state(search, instance.initial_perp), qubits=search[:], inplace=True)
    structured = build_controlled_cqaa_circuit(instance, repetitions=repetitions, include_prep=False)
    gate_u = UnitaryGate(Operator(structured), label="U_dense")
    qc.append(gate_u, [control[0]] + search[:])
    return qc


def _transpile_metrics(circuit: QuantumCircuit, topology: str = "line", optimization_level: int = 3) -> dict[str, float]:
    num_qubits = circuit.num_qubits
    if topology == "line":
        cmap = CouplingMap.from_line(num_qubits)
    elif topology == "ring":
        cmap = CouplingMap.from_ring(num_qubits)
    elif topology == "all":
        cmap = CouplingMap.from_full(num_qubits)
    else:
        raise ValueError("topology must be one of {'line', 'ring', 'all'}.")
    tqc = transpile(
        circuit,
        basis_gates=BASIS,
        coupling_map=cmap,
        optimization_level=optimization_level,
        seed_transpiler=7,
    )
    counts = tqc.count_ops()
    return {
        "depth": float(tqc.depth()),
        "size": float(tqc.size()),
        "cx": float(counts.get("cx", 0)),
        "sx": float(counts.get("sx", 0)),
        "rz": float(counts.get("rz", 0)),
        "num_qubits": float(tqc.num_qubits),
    }


def _single_target_success(instance, circuit: QuantumCircuit, controlled: bool) -> float:
    state = Statevector.from_instruction(circuit).data
    if controlled:
        target = _to_qiskit_order_vector(cqaa._controlled_target_state(instance), circuit.num_qubits)
        return float(abs(np.vdot(target, state)) ** 2)
    target = _to_qiskit_order_vector(instance.target_state, circuit.num_qubits)
    return float(abs(np.vdot(target, state)) ** 2)


def _marked_subspace_success(instance, circuit: QuantumCircuit) -> float:
    state = Statevector.from_instruction(circuit).data
    ket1_tilde = cqaa._tilde_basis(instance.theta_tilde)[1]
    projector = np.kron(cqaa._projector(ket1_tilde), instance.target_projector)
    projector = _to_qiskit_order_matrix(projector, circuit.num_qubits)
    return float(np.real(np.vdot(state, projector @ state)))


def run_scenario_a(epsilon: float = 0.10, max_iterations: int = 12) -> None:
    """
    A. Ideal success comparison for A^t and U^t.
    """
    print(f"\n{SEP}")
    print("SCENARIO A: IDEAL DETECTION-VERSUS-FINDING COMPARISON")
    print(SEP)
    instance = build_single_target_instance(epsilon=epsilon)
    print(f"Two-qubit search register, epsilon={epsilon:.4f}, theta_tilde={instance.theta_tilde:.6f}")
    print(f"{'t':<6} | {'A^t target prob':<18} | {'U^t controlled target prob'}")
    print("-" * 65)
    best_a = (-1, -1.0)
    best_u = (-1, -1.0)
    for t in range(max_iterations + 1):
        det_circ = build_detection_operator_circuit(instance, repetitions=t, include_prep=True)
        cqaa_circ = build_controlled_cqaa_circuit(instance, repetitions=t, include_prep=True)
        p_a = _single_target_success(instance, det_circ, controlled=False)
        p_u = _single_target_success(instance, cqaa_circ, controlled=True)
        if p_a > best_a[1]:
            best_a = (t, p_a)
        if p_u > best_u[1]:
            best_u = (t, p_u)
        print(f"{t:<6} | {p_a:<18.6f} | {p_u:.6f}")
    print(f"\nBest direct-detection overlap: t={best_a[0]}, p={best_a[1]:.6f}")
    print(f"Best CQAA overlap:            t={best_u[0]}, p={best_u[1]:.6f}")


def run_scenario_b(epsilon: float = 0.10, repetitions: int = 1, topology: str = "all") -> None:
    """
    B. Single-step compilation overhead of the CQAA wrapper.
    """
    print(f"\n{SEP}")
    print("SCENARIO B: SINGLE-STEP CQAA COMPILATION OVERHEAD")
    print(SEP)
    instance = build_single_target_instance(epsilon=epsilon)
    det = build_detection_operator_circuit(instance, repetitions=repetitions, include_prep=True)
    ctrl = build_controlled_cqaa_circuit(instance, repetitions=repetitions, include_prep=True)
    det_m = _transpile_metrics(det, topology=topology)
    ctrl_m = _transpile_metrics(ctrl, topology=topology)
    print(f"Topology={topology}, repetitions={repetitions}")
    print(f"{'Metric':<18} | {'Detect-only A^t':<16} | {'Controlled U^t'}")
    print("-" * 60)
    for key in ("depth", "size", "cx", "sx", "rz", "num_qubits"):
        print(f"{key:<18} | {det_m[key]:<16.0f} | {ctrl_m[key]:.0f}")


def run_scenario_c(epsilon: float = 0.10, repetitions: tuple[int, ...] = (1, 2, 4, 6), topology: str = "line") -> None:
    """
    C. Depth scaling of repeated A and U on a constrained topology.
    """
    print(f"\n{SEP}")
    print("SCENARIO C: REPEATED-APPLICATION DEPTH SCALING")
    print(SEP)
    instance = build_single_target_instance(epsilon=epsilon)
    print(f"Topology={topology}")
    print(f"{'t':<6} | {'depth(A^t)':<12} | {'depth(U^t)':<12} | {'cx(A^t)':<10} | {'cx(U^t)'}")
    print("-" * 65)
    for rep in repetitions:
        det_m = _transpile_metrics(build_detection_operator_circuit(instance, repetitions=rep, include_prep=True), topology=topology)
        ctrl_m = _transpile_metrics(build_controlled_cqaa_circuit(instance, repetitions=rep, include_prep=True), topology=topology)
        print(f"{rep:<6} | {det_m['depth']:<12.0f} | {ctrl_m['depth']:<12.0f} | {det_m['cx']:<10.0f} | {ctrl_m['cx']:.0f}")


def run_scenario_d(epsilon: float = 0.10, angle_factors: tuple[float, ...] = (0.70, 0.85, 1.00, 1.15, 1.30), topology: str = "line") -> None:
    """
    D. Angle sweep for the controlled wrapper on hardware metrics and ideal success.
    """
    print(f"\n{SEP}")
    print("SCENARIO D: CONTROL-ANGLE HARDWARE SWEEP")
    print(SEP)
    base = build_single_target_instance(epsilon=epsilon)
    print(f"{'factor':<8} | {'theta_tilde':<12} | {'depth(U)':<10} | {'cx(U)':<8} | {'success'}")
    print("-" * 62)
    for factor in angle_factors:
        theta_tilde = base.theta_tilde * float(factor)
        modified = build_single_target_instance(epsilon=epsilon)
        modified.theta_tilde = theta_tilde
        circ = build_controlled_cqaa_circuit(modified, repetitions=1, include_prep=True)
        metrics = _transpile_metrics(circ, topology=topology)
        success = _single_target_success(modified, circ, controlled=True)
        print(f"{factor:<8.2f} | {theta_tilde:<12.6f} | {metrics['depth']:<10.0f} | {metrics['cx']:<8.0f} | {success:.6f}")


def run_scenario_e(epsilon: float = 0.10, targets: tuple[tuple[float, ...], ...] = ((0.90, 0.25, 0.10), (0.70, 0.55, 0.10), (0.40, 0.70, 0.20))) -> None:
    """
    E. Target-direction sweep under a fixed walk operator.
    """
    print(f"\n{SEP}")
    print("SCENARIO E: TARGET-DIRECTION SWEEP")
    print(SEP)
    print(f"{'weights':<24} | {'A peak prob (t<=6)':<18} | {'U peak prob (t<=6)'}")
    print("-" * 70)
    for weights in targets:
        instance = build_single_target_instance(epsilon=epsilon, target_weights=weights)
        best_a = 0.0
        best_u = 0.0
        for rep in range(7):
            best_a = max(best_a, _single_target_success(instance, build_detection_operator_circuit(instance, repetitions=rep, include_prep=True), controlled=False))
            best_u = max(best_u, _single_target_success(instance, build_controlled_cqaa_circuit(instance, repetitions=rep, include_prep=True), controlled=True))
        print(f"{str(weights):<24} | {best_a:<18.6f} | {best_u:.6f}")


def run_scenario_f(epsilon_pi: float = 0.12, repetitions: tuple[int, ...] = (1, 2, 4, 6)) -> None:
    """
    F. Multiple-target marked-subspace continuation.
    """
    print(f"\n{SEP}")
    print("SCENARIO F: MULTIPLE-TARGET CONTINUATION")
    print(SEP)
    instance = build_multiple_target_instance(epsilon_pi=epsilon_pi)
    print(f"Marked-subspace dimension={instance.target_subspace_dimension}, epsilon_pi={epsilon_pi:.4f}")
    print(f"{'t':<6} | {'marked-subspace success':<24} | {'depth(U^t,line)'}")
    print("-" * 58)
    for rep in repetitions:
        circ = build_controlled_cqaa_circuit(instance, repetitions=rep, include_prep=True)
        success = _marked_subspace_success(instance, circ)
        depth = _transpile_metrics(circ, topology="line")["depth"]
        print(f"{rep:<6} | {success:<24.6f} | {depth:.0f}")


def run_scenario_g(epsilon: float = 0.10, repetitions: int = 2, topology: str = "line") -> None:
    """
    G. Structured CQAA circuit versus direct matrix implementation.
    """
    print(f"\n{SEP}")
    print("SCENARIO G: STRUCTURED VERSUS DIRECT CQAA IMPLEMENTATION")
    print(SEP)
    instance = build_single_target_instance(epsilon=epsilon)
    structured = build_controlled_cqaa_circuit(instance, repetitions=repetitions, include_prep=True)
    direct = build_direct_matrix_cqaa_circuit(instance, repetitions=repetitions, include_prep=True)
    structured_state = Statevector.from_instruction(structured).data
    direct_state = Statevector.from_instruction(direct).data
    state_gap = float(np.linalg.norm(structured_state - direct_state))
    success_gap = abs(_single_target_success(instance, structured, controlled=True) - _single_target_success(instance, direct, controlled=True))
    m_struct = _transpile_metrics(structured, topology=topology)
    m_direct = _transpile_metrics(direct, topology=topology)
    print(f"Statevector difference = {state_gap:.6e}")
    print(f"Success-probability gap = {success_gap:.6e}\n")
    print(f"{'Metric':<18} | {'Structured U^t':<16} | {'Direct matrix U^t'}")
    print("-" * 60)
    for key in ("depth", "size", "cx", "sx", "rz"):
        print(f"{key:<18} | {m_struct[key]:<16.0f} | {m_direct[key]:.0f}")


def _save_cqaa_algorithm_figure(
    epsilon: float = 0.10,
    *,
    repetitions=(0, 1, 2, 4, 6, 8),
    output_name="cqaa_detection_finding_profile.png",
):
    plt = _load_pyplot()
    instance = build_single_target_instance(epsilon=epsilon)

    t_vals = []
    a_success = []
    u_success = []
    a_depth = []
    u_depth = []
    dense_depth = []
    a_cx = []
    u_cx = []
    dense_cx = []

    for reps in repetitions:
        det = build_detection_operator_circuit(instance, repetitions=reps, include_prep=True)
        cqaa_structured = build_controlled_cqaa_circuit(instance, repetitions=reps, include_prep=True)
        cqaa_dense = build_direct_matrix_cqaa_circuit(instance, repetitions=reps, include_prep=True)

        det_metrics = _transpile_metrics(det, topology="line")
        structured_metrics = _transpile_metrics(cqaa_structured, topology="line")
        dense_metrics = _transpile_metrics(cqaa_dense, topology="line")

        t_vals.append(int(reps))
        a_success.append(_single_target_success(instance, det, controlled=False))
        u_success.append(_single_target_success(instance, cqaa_structured, controlled=True))
        a_depth.append(det_metrics["depth"])
        u_depth.append(structured_metrics["depth"])
        dense_depth.append(dense_metrics["depth"])
        a_cx.append(det_metrics["cx"])
        u_cx.append(structured_metrics["cx"])
        dense_cx.append(dense_metrics["cx"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("CQAA Detection-to-Finding Profile", fontsize=14, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(t_vals, a_success, marker="o", linewidth=2.0, label="A^t target overlap", color="#1f77b4")
    ax1.plot(t_vals, u_success, marker="s", linewidth=2.0, label="U^t controlled overlap", color="#d62728")
    ax1.set_title("Success Trajectory")
    ax1.set_ylabel("Success probability")
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(fontsize=8)

    ax2.plot(t_vals, a_depth, marker="o", linewidth=2.0, label="A^t", color="#2ca02c")
    ax2.plot(t_vals, u_depth, marker="s", linewidth=2.0, label="CQAA structured", color="#9467bd")
    ax2.plot(t_vals, dense_depth, marker="D", linewidth=2.0, label="CQAA dense", color="#ff7f0e")
    ax2.set_title("Routed Depth on Line Topology")
    ax2.set_ylabel("Depth")
    ax2.legend(fontsize=8)

    ax3.plot(t_vals, a_cx, marker="o", linewidth=2.0, label="A^t", color="#8c564b")
    ax3.plot(t_vals, u_cx, marker="s", linewidth=2.0, label="CQAA structured", color="#1f77b4")
    ax3.plot(t_vals, dense_cx, marker="D", linewidth=2.0, label="CQAA dense", color="#d62728")
    ax3.set_title("Entangling Cost on Line Topology")
    ax3.set_ylabel("CX count")
    ax3.legend(fontsize=8)

    structured_gap = [ud - ad for ud, ad in zip(u_depth, a_depth)]
    dense_gap = [dd - ud for dd, ud in zip(dense_depth, u_depth)]
    ax4.plot(t_vals, structured_gap, marker="o", linewidth=2.0, label="finding - detection depth", color="#2ca02c")
    ax4.plot(t_vals, dense_gap, marker="s", linewidth=2.0, label="dense - structured depth", color="#ff7f0e")
    ax4.set_title("Wrapper Overhead")
    ax4.set_ylabel("Additional depth")
    ax4.legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Repetitions t")
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "cqaa_detection_finding_profile",
            "epsilon": float(epsilon),
            "repetitions": [int(x) for x in repetitions],
            "topology": "line",
        },
    )
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


def _save_cqaa_angle_figure(
    epsilon: float = 0.10,
    *,
    angle_factors=(0.70, 0.85, 1.00, 1.15, 1.30),
    output_name="cqaa_angle_sweep_profile.png",
):
    plt = _load_pyplot()
    base = build_single_target_instance(epsilon=epsilon)
    factors = []
    theta_vals = []
    depth_vals = []
    cx_vals = []
    success_vals = []

    for factor in angle_factors:
        modified = build_single_target_instance(epsilon=epsilon)
        modified.theta_tilde = base.theta_tilde * float(factor)
        circ = build_controlled_cqaa_circuit(modified, repetitions=1, include_prep=True)
        metrics = _transpile_metrics(circ, topology="line")
        factors.append(float(factor))
        theta_vals.append(float(modified.theta_tilde))
        depth_vals.append(float(metrics["depth"]))
        cx_vals.append(float(metrics["cx"]))
        success_vals.append(_single_target_success(modified, circ, controlled=True))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("CQAA Control-Angle Sweep", fontsize=14, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(factors, theta_vals, marker="o", linewidth=2.2, color="#1f77b4")
    ax1.set_title("Effective Tilde Angle")
    ax1.set_ylabel("theta_tilde")

    ax2.plot(factors, success_vals, marker="o", linewidth=2.2, color="#2ca02c")
    ax2.set_title("Controlled Success")
    ax2.set_ylabel("Success probability")
    ax2.set_ylim(0.0, 1.05)

    ax3.plot(factors, depth_vals, marker="o", linewidth=2.2, color="#d62728")
    ax3.set_title("Routed Depth")
    ax3.set_ylabel("Depth")

    ax4.plot(factors, cx_vals, marker="o", linewidth=2.2, color="#9467bd")
    ax4.set_title("Entangling Cost")
    ax4.set_ylabel("CX count")

    for axis in axes.flat:
        axis.set_xlabel("Angle factor")
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "cqaa_angle_sweep_profile",
            "epsilon": float(epsilon),
            "angle_factors": [float(x) for x in angle_factors],
            "theta_tilde_values": [float(x) for x in theta_vals],
            "topology": "line",
        },
    )
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


if __name__ == "__main__":
    output_filepath = os.path.join(_RESULT_DIR, "terminal_output.log")
    os.chdir(_RESULT_DIR)

    logger = Logger(output_filepath)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger
    cli_argv, publishability = parse_publishability_cli(
        sys.argv[1:],
        default_max_qubits=20,
        default_shots=1024,
        default_log_dir=_RESULT_DIR,
    )
    prepare_backend_validation_artifacts(publishability)

    print("Controlled Quantum Amplification Benchmark Suite — Scenarios A through G")
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
    ]
    scenarios = wrap_scenarios(raw_scenarios, module_globals=globals(), extra_patch_objects=(cqaa,), config=publishability)

    cli_executed = run_cli_scenario(cli_argv, scenarios)
    if not cli_executed:
        for label, fn in scenarios:
            try:
                fn()
            except Exception:
                print(f"\n*** SCENARIO {label} FAILED ***")
                traceback.print_exc()

        run_interactive_scenario_repl(scenarios, sep=SEP)
    render_backend_validation_summary(publishability)
    _save_cqaa_algorithm_figure()
    _save_cqaa_angle_figure()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    logger.close()
    print(f"\nBenchmark suite complete. Results saved to {output_filepath}")

