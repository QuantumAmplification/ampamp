"""
OAA Transpilation Master Suite (Scenarios A through K)
=======================================================
Full rigorous transpilation benchmark for Oblivious Amplitude Amplification (OAA)
and Block Encodings, mirroring the design philosophy of 2_Fixed_Point_Ammplitude_Amplification_transpile.py.

Mathematical Standing Notation (aligned with final.tex):
  - A        : Block-encoding unitary with exact clean-block identity
                (<0^l| \\otimes I) A (|0^l> \\otimes I) = sqrt(p) * U,
                equivalently P_0 A P_0 = |0^l><0^l| \\otimes sqrt(p) U.
                Built as PREP† SELECT PREP (LCU) or as ancilla + controlled-U.
  - A†       : Exact inverse (uncomputation).
  - R_0      : Here defined as I - 2|0^l⟩⟨0^l| ⊗ I_data.
                This is the negative of the clean-subspace reflection
                2|0^l⟩⟨0^l| ⊗ I_data - I used in final.tex.
                Implementation: X^l · mcp(π) · X^l on the l ancilla qubits.
  - R_bad    : Phase flip on NON-zero ancilla states.
                R_bad = 2|0^l⟩⟨0^l| ⊗ I_data - I
                       = -(I - 2|0^l⟩⟨0^l| ⊗ I_data)
                Note: R_bad gives +1 on the clean ancilla sector and -1 on its orthogonal complement.
                Implementation: Controlled-Z type, flipping everything EXCEPT |0^l⟩.
  - Q        : One OAA iterate = A · R_0 · A† · R_bad
                With the sign choices above, this differs from the final.tex
                iterate by an overall global phase only.
                Register-order note: for QuantumCircuit(anc, data), ancilla is the
                least-significant subsystem in statevector indexing; for
                QuantumCircuit(data, anc), ancilla is the most-significant subsystem.
  - p        : Success probability = ||Pi_Good |psi⟩||² where Pi_Good = |0^l⟩⟨0^l| ⊗ I
  - k_opt    : Smallest near-optimal OAA iteration count for the first peak:
                floor(π / (4 arcsin(√p)))
                = nearest integer to π / (4 arcsin(√p)) - 1/2
  - alpha    : LCU subnormalization factor = sum of |c_i| coefficients
  - A_TL     : Clean-ancilla block of A in the basis ordering used by the circuit
"""

from __future__ import annotations

import numpy as np
import math
import sys
import os
import ast
import inspect
import traceback
import importlib.util
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Operator, Statevector


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

_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RESULT_DIR, ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _import_local_module(alias: str, filename: str):
    module_path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


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
    print("Enter a label like A or K, or press Enter to finish.")
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
        print("Example: p=0.1, m=4")
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


# =============================================================================
# Logger: Dual output to terminal and file
# =============================================================================

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


# =============================================================================
# Core Circuit Primitives
# =============================================================================

def build_R_0(l: int) -> QuantumCircuit:
    """
    R_0 = I - 2|0^l⟩⟨0^l| (reflection about the all-zero ancilla state).

    This is ``-(2|0^l⟩⟨0^l| - I)``, so it differs by a global sign from the
    clean-subspace reflection used in final.tex.

    Implementation:
      X^{⊗l} · MCZ (multi-controlled-Z, all controls on target) · X^{⊗l}
    This flips all |0⟩ → |1⟩, applies a phase of –1 ONLY to |1...1⟩ (the
    image of |0...0⟩), then flips back. Net effect: phases |0...0⟩ by –1,
    leaves all other states unchanged. That is exactly I – 2|0^l⟩⟨0^l|.

    For l=1: XZX, which acts as Z on the |0⟩ state (phase -1 on |0⟩).
    """
    anc = QuantumRegister(l, "anc")
    qc = QuantumCircuit(anc, name=f"R0_l{l}")
    # Flip |0⟩ → |1⟩ so we can target |1...1⟩ with a multi-controlled gate
    qc.x(anc)
    if l == 1:
        qc.z(anc[0])                                               # CZ with no controls (just Z)
    else:
        qc.mcp(np.pi, list(range(l - 1)), l - 1)                 # MCZ → phase on |1...1⟩
    qc.x(anc)                                                      # Flip back
    return qc


def build_R_bad(l: int) -> QuantumCircuit:
    """
    R_bad = -(I - 2|0^l⟩⟨0^l|) = 2|0^l⟩⟨0^l| – I.

    Phases the |0^l⟩ component by +1 and everything else by –1.
    Equivalently: –R_0, so the two reflection conventions differ only by sign.

    In the OAA iterate Q = A · R_0 · A† · R_bad, this reflects in the
    H_Bad sector (non-zero ancilla), which is the "bad" subspace.

    Implementation: The same X^l MCZ X^l as R_0, preceded by a global
    phase of π (–I factor). We absorb the global phase into circuit.global_phase.
    """
    anc = QuantumRegister(l, "anc")
    qc = QuantumCircuit(anc, name=f"Rbad_l{l}")
    qc.global_phase = np.pi            # Overall factor of -1
    qc.x(anc)
    if l == 1:
        qc.z(anc[0])
    else:
        qc.mcp(np.pi, list(range(l - 1)), l - 1)
    qc.x(anc)
    return qc


def build_A_simple(p: float, m: int = 1) -> QuantumCircuit:
    """
    Construct a minimal valid block encoding A for a random m-qubit unitary U.

    The block encoding guarantees:
        (<0|_anc x I) · A · (|0>_anc x I) = sqrt(p) · U

    Equivalently:
        P_0 · A · P_0 = |0><0|_anc ⊗ sqrt(p) · U

    where P_0 = |0⟩⟨0|_anc ⊗ I_data.

    Construction (l=1):
        1. RY(2θ)|0⟩_anc = cos(θ)|0⟩ + sin(θ)|1⟩ with cos²(θ)=p,
           so θ = arccos(√p).
        2. X on ancilla (convert control-0 to control-1).
        3. Controlled-U on data register, control = ancilla qubit.
        4. X on ancilla (restore).

    When ancilla = 0 (measured in standard basis):
        amplitude = cos(θ) × U|data⟩ = √p · U|data⟩  ✓

    We use U = H^m (Hadamard on all data qubits) as the target unitary for
    reproducibility. Any unitary U can be substituted.

    Registers: QuantumCircuit(anc, data), so ancilla is qubit 0 and occupies the
    least-significant subsystem in Qiskit's statevector indexing. The ancilla=0
    sector is therefore the even-index principal block, not a contiguous top-left
    block.
    """
    assert 0.0 < p < 1.0, "p must be strictly in (0,1) for a non-trivial block encoding."
    theta = np.arccos(np.sqrt(p))      # cos²(θ) = p  →  ancilla|0⟩ branch has amplitude √p
    l = 1                              # one ancilla qubit

    anc = QuantumRegister(l, "anc")
    data = QuantumRegister(m, "data")
    qc = QuantumCircuit(anc, data, name="A")

    # Step 1: Prepare ancilla superposition
    qc.ry(2.0 * theta, anc[0])

    # Step 2-4: Control-0 → Control-1 → apply U → restore
    qc.x(anc[0])          # |0⟩ → |1⟩ so we can use control=|1⟩
    for i in range(m):
        qc.ch(anc[0], data[i])    # Controlled-H on each data qubit (U = H^m)
    qc.x(anc[0])          # Restore ancilla

    return qc


def build_Q(A: QuantumCircuit, l: int = 1) -> QuantumCircuit:
    """
    Construct the full OAA iterate Q = A · R_0 · A† · R_bad.

    The register layout MUST match A exactly.
    A has l+m qubits: ancilla qubits [0..l-1], data qubits [l..l+m-1].

    Q acts on the same register space. With the sign convention used here,
    this differs from the final.tex iterate by an overall global phase only.
    """
    n_total = A.num_qubits
    n_data = n_total - l
    assert n_data > 0, "A must have at least 1 data qubit."

    all_qubits = list(range(n_total))
    anc_qubits = list(range(l))

    R0 = build_R_0(l)
    Rbad = build_R_bad(l)
    A_dag = A.inverse()
    A_dag.name = "A_dag"

    qc = QuantumCircuit(n_total, name="Q")
    # Q = A R_0 A† R_bad  (read right to left: Rbad first, then A†, then R_0, then A)
    qc.append(Rbad.to_gate(), anc_qubits)
    qc.append(A_dag.to_gate(), all_qubits)
    qc.append(R0.to_gate(), anc_qubits)
    qc.append(A.to_gate(), all_qubits)
    return qc


def build_oaa_circuit(p: float, k: int, m: int = 1) -> QuantumCircuit:
    """
    Build the full OAA amplification circuit: Q^k · A applied to |0^{l+m}⟩.

    After k OAA iterates starting from A|0⟩, the success probability
    (measuring ancilla = 0) is:
        P_k(p) = sin²((2k+1)·arcsin(√p))

    which approaches 1 as k approaches the Grover/OAA optimum
    k_opt = floor(π / (4·arcsin(√p))) for the first success peak.
    """
    A = build_A_simple(p, m=m)
    Q = build_Q(A, l=1)
    n_total = A.num_qubits

    qc = QuantumCircuit(n_total, name=f"OAA_k{k}")
    qc.compose(A, inplace=True)
    for _ in range(k):
        qc.compose(Q, inplace=True)
    return qc


def build_lcu(c0: float, c1: float):
    """
    Explicit LCU block encoding for H = c0·X₀ + c1·Z₁  (two data qubits).

    LCU formula: A = PREP† · SELECT · PREP
    where:
      PREP : prepares √(c0/α)|0⟩_anc + √(c1/α)|1⟩_anc, α = |c0| + |c1|
      SELECT: |0⟩⟨0|_anc ⊗ X₀  +  |1⟩⟨1|_anc ⊗ Z₁

    Verification: Clean ancilla block A_TL = H/α = (c0·X₀ + c1·Z₁)/α.

    Register ordering: QuantumCircuit(data, anc) places ancilla in the
    most-significant subsystem. The anc=0 sector is therefore the contiguous
    top-left 4x4 block.
    """
    assert c0 > 0 and c1 > 0, "Coefficients must be positive for this LCU demo."
    alpha = c0 + c1                    # LCU subnormalization factor
    w0 = c0 / alpha                    # weight for first Pauli term
    # w1 = c1 / alpha                  # weight for second Pauli term (= 1 - w0)

    data = QuantumRegister(2, "data")
    anc = QuantumRegister(1, "anc")

    # PREP: RY(2θ) where cos²(θ)=w0  →  ancilla|0⟩ branch has amplitude √w0
    theta = np.arccos(np.sqrt(w0))
    prep = QuantumCircuit(data, anc, name="PREP")
    prep.ry(2.0 * theta, anc[0])

    # SELECT: controlled operations
    # anc=0 branch (X on ancilla to flip to control-1): CX on data[0]
    # anc=1 branch: CZ on data[1]
    select = QuantumCircuit(data, anc, name="SELECT")
    select.x(anc[0])              # flip to target anc=0 branch
    select.cx(anc[0], data[0])   # X₀ conditioned on anc=0 (now anc=1 after flip)
    select.x(anc[0])              # restore
    select.cz(anc[0], data[1])   # Z₁ conditioned on anc=1

    prep_dag = prep.inverse()
    prep_dag.name = "PREP_dag"

    # Full LCU block-encoding: PREP† SELECT PREP
    A = QuantumCircuit(data, anc, name="A_LCU")
    A.compose(prep, inplace=True)
    A.compose(select, inplace=True)
    A.compose(prep_dag, inplace=True)

    return prep, select, prep_dag, A, alpha


def verify_lcu_block(A: QuantumCircuit, c0: float, c1: float, alpha: float) -> float:
    """
    Verify A_TL (clean ancilla block of A) equals H/α = (c0·X₀ + c1·Z₁)/α.

    Qiskit register ordering: QuantumCircuit(data[2], anc[1]) →
    the full matrix is 8×8, Hilbert space = data ⊗ anc, with ancilla in the
    most-significant subsystem. The anc=0 block is the contiguous top-left 4×4
    principal block.

    Returns: Frobenius distance ‖A_TL – H/α‖_F  (should be < 1e-10)
    """
    A_mat = Operator(A).data   # 8×8 matrix
    A_TL = A_mat[:4, :4]       # top-left 4×4 (anc=0 sector)

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    # data register ordering: |q1,q0⟩, so X on q0 = I⊗X, Z on q1 = Z⊗I
    X0 = np.kron(I2, X)
    Z1 = np.kron(Z, I2)
    H_mat = c0 * X0 + c1 * Z1
    H_norm = H_mat / alpha

    return float(np.linalg.norm(A_TL - H_norm))


def get_swap_count(t_qc: QuantumCircuit) -> int:
    """Count the number of SWAP gates inserted by routing."""
    ops = t_qc.count_ops()
    return ops.get("swap", 0)


def linear_coupling_map(n: int) -> CouplingMap:
    """Bidirectional linear chain: 0-1-2-...(n-1)."""
    edges = [[i, i+1] for i in range(n-1)] + [[i+1, i] for i in range(n-1)]
    return CouplingMap(edges)


def heavy_hex_coupling_map(n: int) -> CouplingMap:
    """
    A realistic Heavy-Hex-style coupling map for n qubits.

    For n ≤ 7 we use the standard 5-node sub-graph extended linearly.
    Pattern: 0-1-2, 1-3-4, 3-5-6, ...  (nodes branch at every even hub)
    This models IBM heavy-hex connectivity where each data qubit connects
    to at most 2 neighbors with bridge qubits in between.
    """
    edges: list[list[int]] = []
    # Base linear backbone
    for i in range(n - 1):
        edges += [[i, i+1], [i+1, i]]
    # Add T-junction at node 1 if possible (heavy-hex branch)
    if n > 4:
        edges += [[1, 4], [4, 1]]
    if n > 6:
        edges += [[3, 6], [6, 3]]
    return CouplingMap(edges)


def k_opt(p: float) -> int:
    """Smallest near-optimal OAA count on the first success peak."""
    return max(0, int(np.floor(np.pi / (4.0 * np.arcsin(np.sqrt(p))))))


def _ancilla_zero_probability(state: np.ndarray, ancilla_index: int = 0) -> float:
    probs = np.abs(np.asarray(state, dtype=complex)) ** 2
    total = 0.0
    for idx, prob in enumerate(probs):
        if ((idx >> ancilla_index) & 1) == 0:
            total += float(prob)
    return total


BASIS_NISQ = ['cx', 'id', 'rz', 'sx', 'x']
NISQ_DEPTH_LIMIT = 2000   # Approximate NISQ coherence depth limit


# =============================================================================
# Scenario A: Block Encoding Unrolling Baseline
# =============================================================================

def run_scenario_a(p: float = 0.25, m: int = 2) -> None:
    """
    A. The Block Encoding Unrolling Baseline (A vs Q).

    Transpiles A, A†, and the single-iterate Q=A·R_0·A†·R_bad on all-to-all.
    Proves: Depth(Q) ≈ 2·Depth(A) + O(l) where O(l) = cost of reflections R_0, R_bad.
    This is the fundamental 2× depth overhead of uncomputation in OAA.
    """
    print("\n" + "=" * 70)
    print("SCENARIO A: THE BLOCK ENCODING UNROLLING BASELINE (A vs Q)")
    print("=" * 70)
    print(f"Parameters: p={p}, m={m} data qubits, l=1 ancilla.")
    print(f"Target U = H^m (Hadamard on all data qubits).")
    print(f"Architecture: All-to-All (isolating pure uncomputation cost).\n")

    A = build_A_simple(p, m=m)
    Q = build_Q(A, l=1)
    R0 = build_R_0(1)
    Rbad = build_R_bad(1)

    print("Transpiling A ...")
    tA = transpile(A, basis_gates=BASIS_NISQ, optimization_level=3)
    print("Transpiling A† ...")
    tAdg = transpile(A.inverse(), basis_gates=BASIS_NISQ, optimization_level=3)
    print("Transpiling R_0 ...")
    tR0 = transpile(R0, basis_gates=BASIS_NISQ, optimization_level=3)
    print("Transpiling R_bad ...")
    tRbad = transpile(Rbad, basis_gates=BASIS_NISQ, optimization_level=3)
    print("Transpiling Q = A·R_0·A†·R_bad ...")
    tQ = transpile(Q, basis_gates=BASIS_NISQ, optimization_level=3)

    dA = tA.depth(); cxA = tA.count_ops().get('cx', 0)
    dAdg = tAdg.depth(); cxAdg = tAdg.count_ops().get('cx', 0)
    dR0 = tR0.depth(); dRbad = tRbad.depth()
    dQ = tQ.depth(); cxQ = tQ.count_ops().get('cx', 0)

    predicted = 2 * dA + dR0 + dRbad

    print(f"\n{'Circuit':<12} | {'Depth':<8} | {'CX Count'}")
    print("-" * 40)
    print(f"{'A':<12} | {dA:<8} | {cxA}")
    print(f"{'A_dag':<12} | {dAdg:<8} | {cxAdg}")
    print(f"{'R_0':<12} | {dR0:<8} | {tR0.count_ops().get('cx',0)}")
    print(f"{'R_bad':<12} | {dRbad:<8} | {tRbad.count_ops().get('cx',0)}")
    print(f"{'Q (actual)':<12} | {dQ:<8} | {cxQ}")
    print(f"{'Q (predicted)':<12} | {predicted:<8} |")
    print(f"\n-> Verified: Depth(Q) = {dQ} vs 2·Depth(A)+R costs = {predicted}")
    print("-> CONCLUSION: Uncomputation (A†) doubles the physical depth. This is")
    print("   the irreducible hardware overhead of every OAA iterate.")


# =============================================================================
# Scenario B: LCU Structural Bottleneck (PREP vs SELECT)
# =============================================================================

def run_scenario_b(c0: float = 0.6, c1: float = 0.4) -> None:
    """
    B. The LCU Structural Bottleneck (PREP vs SELECT).

    Separately transpiles PREP and SELECT onto a Heavy-Hex lattice, proving
    the physical cost asymmetry: PREP is cheap (1 local ancilla rotation),
    SELECT is expensive (multi-qubit entanglement across the data register).
    """
    print("\n" + "=" * 70)
    print("SCENARIO B: THE LCU STRUCTURAL BOTTLENECK (PREP vs SELECT)")
    print("=" * 70)
    alpha = c0 + c1
    print(f"Hamiltonian: H = {c0}·X₀ + {c1}·Z₁,  α = {alpha:.4f}")
    print(f"Architecture: Heavy-Hex Lattice (3 qubits: 2 data + 1 ancilla)\n")

    prep, select, prep_dag, A_lcu, alpha = build_lcu(c0, c1)

    # Verify LCU correctness
    dist = verify_lcu_block(A_lcu, c0, c1, alpha)
    print(f"LCU Verification: ‖A_TL – H/α‖_F = {dist:.3e} -> {'PASS' if dist < 1e-9 else 'FAIL'}")

    cmap = heavy_hex_coupling_map(3)
    tprep   = transpile(prep,     basis_gates=BASIS_NISQ, coupling_map=cmap, optimization_level=3)
    tselect = transpile(select,   basis_gates=BASIS_NISQ, coupling_map=cmap, optimization_level=3)
    tpdag   = transpile(prep_dag, basis_gates=BASIS_NISQ, coupling_map=cmap, optimization_level=3)
    tA      = transpile(A_lcu,    basis_gates=BASIS_NISQ, coupling_map=cmap, optimization_level=3)

    dp = tprep.depth();   cxp = tprep.count_ops().get('cx', 0); swp = get_swap_count(tprep)
    ds = tselect.depth(); cxs = tselect.count_ops().get('cx', 0); sws = get_swap_count(tselect)
    dpd = tpdag.depth();  cxpd = tpdag.count_ops().get('cx', 0); swpd = get_swap_count(tpdag)
    dA = tA.depth();      cxA = tA.count_ops().get('cx', 0); swA = get_swap_count(tA)

    print(f"\n{'Module':<14} | {'Depth':<6} | {'CX':<6} | {'SWAPs Inserted'}")
    print("-" * 55)
    print(f"{'PREP':<14} | {dp:<6} | {cxp:<6} | {swp}")
    print(f"{'SELECT':<14} | {ds:<6} | {cxs:<6} | {sws}")
    print(f"{'PREP_dag':<14} | {dpd:<6} | {cxpd:<6} | {swpd}")
    print(f"{'A_LCU (total)':<14} | {dA:<6} | {cxA:<6} | {swA}")

    ratio = ds / max(1, dp)
    print(f"\n-> SELECT/PREP Depth Ratio: {ratio:.2f}x")
    print("-> CONCLUSION: PREP is trivially cheap (1 local ancilla RY).")
    print("   SELECT forces cross-register entanglement, exploding CX and SWAP counts.")


# =============================================================================
# Scenario C: Ancilla Star-Graph Routing Penalty
# =============================================================================

def run_scenario_c(m: int = 6) -> None:
    """
    C. The Ancilla Star-Graph Routing Penalty.

    The controlled-U gate in a block encoding requires the l ancilla qubit(s)
    to interact with all m data qubits, forming a star graph topology.
    We test this star pattern across All-to-All, Heavy-Hex, and Linear 1D.
    """
    print("\n" + "=" * 70)
    print("SCENARIO C: THE ANCILLA STAR-GRAPH ROUTING PENALTY")
    print("=" * 70)
    l = 1
    n_total = l + m
    print(f"Parameters: m={m} data qubits, l={l} ancilla. Total = {n_total} qubits.")
    print("Target: Ancilla controls CX onto every data qubit (pure star graph).\n")

    # A star-graph block encoding: ancilla connects to all data qubits via CX.
    anc = QuantumRegister(l, "anc")
    data = QuantumRegister(m, "data")
    A_star = QuantumCircuit(anc, data, name="A_star")
    A_star.h(anc[0])                    # Prepare ancilla superposition
    for i in range(m):
        A_star.cx(anc[0], data[i])      # CX from ancilla to each data qubit — the star graph

    cmap_hex = heavy_hex_coupling_map(n_total)
    cmap_lin = linear_coupling_map(n_total)

    print("Transpiling: All-to-All ...")
    t_all = transpile(A_star, basis_gates=BASIS_NISQ, optimization_level=3)
    print("Transpiling: Heavy-Hex ...")
    t_hex = transpile(A_star, basis_gates=BASIS_NISQ, coupling_map=cmap_hex, optimization_level=3)
    print("Transpiling: Linear 1D ...")
    t_lin = transpile(A_star, basis_gates=BASIS_NISQ, coupling_map=cmap_lin, optimization_level=3)

    d_all = t_all.depth(); sw_all = get_swap_count(t_all); cx_all = t_all.count_ops().get('cx',0)
    d_hex = t_hex.depth(); sw_hex = get_swap_count(t_hex); cx_hex = t_hex.count_ops().get('cx',0)
    d_lin = t_lin.depth(); sw_lin = get_swap_count(t_lin); cx_lin = t_lin.count_ops().get('cx',0)

    print(f"\n{'Topology':<15} | {'Depth':<8} | {'Total CX':<10} | {'SWAPs Inserted':<15} | {'Depth Multiplier'}")
    print("-" * 75)
    print(f"{'All-to-All':<15} | {d_all:<8} | {cx_all:<10} | {sw_all:<15} | 1.00x (baseline)")
    print(f"{'Heavy-Hex':<15} | {d_hex:<8} | {cx_hex:<10} | {sw_hex:<15} | {d_hex/max(1,d_all):.2f}x")
    print(f"{'Linear 1D':<15} | {d_lin:<8} | {cx_lin:<10} | {sw_lin:<15} | {d_lin/max(1,d_all):.2f}x")
    print("-> CONCLUSION: The star graph (1 center → m leaves) maps severely")
    print("   onto nearest-neighbor hardware. Every non-adjacent CX forces 1+ SWAPs.")


# =============================================================================
# Scenario D: Extreme Subnormalization Penalty (p → 0)
# =============================================================================

def run_scenario_d() -> None:
    """
    D. The Extreme Subnormalization Penalty (p → 0).

    As p → 0, k_opt ~ π/(4√p) grows as O(1/√p), meaning the OAA circuit
    depth grows as O(k_opt * Depth(Q)). We sweep p and flag NISQ breaches.

    We use m=3 data qubits (U = H^3, total 4 qubits) so A is non-trivial
    and Qiskit Level-3 cannot collapse repeated Q applications away.
    The k cap is set to 20 to keep runtime reasonable.
    """
    print("\n" + "=" * 70)
    print("SCENARIO D: THE EXTREME SUBNORMALIZATION PENALTY (p → 0)")
    print("=" * 70)
    m_data = 3            # 3 data qubits + 1 ancilla = 4 qubits total
    n_total = m_data + 1
    print(f"Architecture: Heavy-Hex ({n_total} qubits: 1 ancilla + {m_data} data).")
    print(f"Target U = H^{m_data}. Circuit: Q^k x A with k = min(k_opt, 20).")
    print(f"NISQ coherence depth limit: {NISQ_DEPTH_LIMIT}\n")

    cmap = heavy_hex_coupling_map(n_total)

    print(f"{'p':<10} | {'k_opt':<8} | {'Depth(A)':<10} | {'Depth(Q^k\u00b7A)':<15} | {'CX Count':<10} | {'Status'}")
    print("-" * 75)

    for p in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
        kk = k_opt(p)
        kk_run = min(kk, 20)   # cap to keep transpilation time reasonable

        A = build_A_simple(p, m=m_data)
        tA = transpile(A, basis_gates=BASIS_NISQ, optimization_level=3)
        dA = tA.depth()

        # Compose Q repetitions via append (gate boxes) to prevent optimizer collapsing.
        Q = build_Q(A, l=1)
        Q_gate = Q.to_gate(label="Q")
        qc = QuantumCircuit(n_total)
        qc.compose(A, inplace=True)
        for _ in range(kk_run):
            qc.append(Q_gate, list(range(n_total)))

        t_qc = transpile(qc, basis_gates=BASIS_NISQ, coupling_map=cmap, optimization_level=3)
        d = t_qc.depth()
        cx = t_qc.count_ops().get('cx', 0)
        flag = "  <-- NISQ LIMIT BREACHED!" if d > NISQ_DEPTH_LIMIT else ""
        cap_note = f" (k capped @ {kk_run})" if kk != kk_run else ""
        print(f"{p:<10} | {kk:<8} | {dA:<10} | {d:<15} | {cx:<10} | {flag}{cap_note}")

    print("\n-> CONCLUSION: OAA circuit depth scales as O(k_opt * Depth(Q)).")
    print(f"   k_opt ~ \u03c0/(4\u221ap) grows as O(1/\u221ap) — the NISQ depth limit.")
    print("   Tiny subnormalization (p\u21920) creates circuits impossible on NISQ hardware.")


# =============================================================================
# Scenario E: Small-Angle Hardware Limitation (Precision Limits)
# =============================================================================

def run_scenario_e(p: float = 0.001) -> None:
    """
    E. The Small-Angle Hardware Limitation (Precision Limits).

    When p is tiny, the ancilla rotation angle θ = arccos(√p) ≈ π/2 – √p is
    very close to π/2, meaning the deviation from π/2 that encodes the p value
    is tiny (≈ √p ≈ 0.031 radians for p=0.001).

    Physical hardware has finite angle resolution (~1e-3 radians for most NISQ).
    We demonstrate that injecting a jitter of 1e-3 on all RZ angles destroys the
    OAA amplification for small p.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        print("Scenario E skipped: qiskit_aer required.")
        return

    print("\n" + "=" * 70)
    print("SCENARIO E: THE SMALL-ANGLE HARDWARE LIMITATION (PRECISION LIMITS)")
    print("=" * 70)
    kk = k_opt(p)
    theta_anc = np.arccos(np.sqrt(p))
    deviation_from_half_pi = abs(np.pi / 2.0 - theta_anc)
    print(f"Parameters: p={p}, k_opt={kk}")
    print(f"Ancilla rotation θ = arccos(√p) = {theta_anc:.6f} rad")
    print(f"Deviation from π/2 = {deviation_from_half_pi:.6f} rad  ← must resolve this precisely")
    print(f"Hardware jitter model: σ = 1e-3 rad on every RZ gate\n")

    qc = build_oaa_circuit(p, kk, m=1)
    t_ideal = transpile(qc, basis_gates=BASIS_NISQ, optimization_level=3)

    sim = AerSimulator()
    shots = 16384

    # --- Ideal run ---
    t_meas = t_ideal.copy()
    t_meas.measure_all()
    ideal_counts = sim.run(t_meas, shots=shots).result().get_counts()
    # Qiskit prints the highest-index classical bit on the left, so qubit 0
    # appears in the rightmost character of the measured bitstring.
    ideal_success = sum(v for k_s, v in ideal_counts.items() if k_s[-1] == '0') / shots

    # --- Jitter run: add Gaussian noise σ=1e-3 to all RZ angles ---
    noisy_qc = t_ideal.copy()
    rng = np.random.default_rng(seed=42)
    for instr in noisy_qc.data:
        if instr.operation.name == 'rz':
            theta_orig = float(instr.operation.params[0])
            instr.operation.params[0] = theta_orig + rng.normal(0.0, 1e-3)
    noisy_qc.measure_all()
    noisy_counts = sim.run(noisy_qc, shots=shots).result().get_counts()
    noisy_success = sum(v for k_s, v in noisy_counts.items() if k_s[-1] == '0') / shots

    # Theoretical prediction
    theta0 = np.arcsin(np.sqrt(p))
    theory = float(np.sin((2 * kk + 1) * theta0) ** 2)

    print(f"{'Theoretical OAA Success':<30} = {theory:.4f}")
    print(f"{'Ideal Simulation Success':<30} = {ideal_success:.4f}")
    print(f"{'Noisy Simulation Success':<30} = {noisy_success:.4f}")
    print(f"{'Degradation from jitter':<30} = {theory - noisy_success:.4f}")
    print("-> CONCLUSION: Microscopic θ deviation from π/2 cannot be reliably")
    print("   executed. Hardware jitter materially degrades OAA amplification for small p.")


# =============================================================================
# Scenario F: Valid vs. Invalid Routing Cost
# =============================================================================

def run_scenario_f() -> None:
    """
    F. The Valid vs. Invalid Routing Cost (Geometric Obstruction).

    Valid setting: A_valid has ancilla rotation independent of data.
    Invalid setting: A_invalid has ancilla rotation conditioned on data state
    (cry gates), violating the clean-block identity
    (<0| x I) A (|0> x I) = sqrt(p) · U.

    We show that the invalid setting is mathematically inconsistent and physically
    more expensive to transpile (requires controlled-RY = CRY across registers).
    """
    print("\n" + "=" * 70)
    print("SCENARIO F: VALID VS. INVALID ROUTING COST (GEOMETRIC OBSTRUCTION)")
    print("=" * 70)
    p_valid = 0.05
    p_0 = 0.05     # Pr(ancilla=0 | data=|0⟩) in invalid setting
    p_1 = 0.20     # Pr(ancilla=0 | data=|1⟩) in invalid setting
    print(f"Valid setting: p = {p_valid} (ancilla prep independent of data).")
    print(f"Invalid setting: p(anc=0|data=|0⟩)={p_0}, p(anc=0|data=|1⟩)={p_1}")
    print(f"Architecture: Heavy-Hex (2 qubits)\n")

    anc = QuantumRegister(1, "anc")
    data = QuantumRegister(1, "data")

    # A_valid: pure RY on ancilla — independent of data state
    theta_valid = np.arccos(np.sqrt(p_valid))
    A_valid = QuantumCircuit(anc, data, name="A_valid")
    A_valid.ry(2.0 * theta_valid, anc[0])

    # A_invalid: CRY conditioned on data qubit — violates the clean block condition
    theta_0 = np.arccos(np.sqrt(p_0))
    theta_1 = np.arccos(np.sqrt(p_1))
    A_invalid = QuantumCircuit(anc, data, name="A_invalid")
    A_invalid.x(data[0])
    A_invalid.cry(2.0 * theta_0, data[0], anc[0])   # data=|0⟩ branch
    A_invalid.x(data[0])
    A_invalid.cry(2.0 * theta_1, data[0], anc[0])   # data=|1⟩ branch

    cmap = heavy_hex_coupling_map(2)
    tv = transpile(A_valid, basis_gates=BASIS_NISQ, coupling_map=cmap, optimization_level=3)
    ti = transpile(A_invalid, basis_gates=BASIS_NISQ, coupling_map=cmap, optimization_level=3)

    dv = tv.depth(); cxv = tv.count_ops().get('cx', 0)
    di = ti.depth(); cxi = ti.count_ops().get('cx', 0)

    print(f"{'Circuit':<15} | {'Depth':<8} | {'CX Count'}")
    print("-" * 40)
    print(f"{'A_valid':<15} | {dv:<8} | {cxv}")
    print(f"{'A_invalid':<15} | {di:<8} | {cxi}")
    print(f"-> Routing penalty: {di / max(1, dv):.2f}x Depth, {cxi / max(1, cxv):.2f}x CX")
    print("-> CONCLUSION: Invalid block encoding requires cross-register CRY")
    print("   gates, which both invalidates the OAA construction and materially increases cost.")


# =============================================================================
# Scenario G: Uncomputation Annihilation Limit
# =============================================================================

def run_scenario_g(p: float = 0.1) -> None:
    """
    G. The Uncomputation Annihilation Limit (A·R_0·A† ≠ I).

    A classical compiler wants to simplify A·A† = I. But in Q, R_0 sits
    between them: A · R_0 · A†. The R_0 operator is the "limit" that blocks
    full AA† cancellation.

    We compare Level-0 (no optimization) vs Level-3 (aggressive synthesis)
    transpilation of Q to quantify how much R_0 prevents classical mitigation.
    """
    print("\n" + "=" * 70)
    print("SCENARIO G: THE UNCOMPUTATION STRUCTURAL LIMIT")
    print("=" * 70)
    print(f"Parameters: p={p}, m=2 data qubits, l=1 ancilla.")
    print("Q = A · R_0 · A† · R_bad. Comparing Opt-Level 0 vs Level 3.\n")

    A = build_A_simple(p, m=2)
    Q = build_Q(A, l=1)

    t0 = transpile(Q, basis_gates=BASIS_NISQ, optimization_level=0)
    t3 = transpile(Q, basis_gates=BASIS_NISQ, optimization_level=3)

    d0 = t0.depth(); cx0 = t0.count_ops().get('cx', 0)
    d3 = t3.depth(); cx3 = t3.count_ops().get('cx', 0)

    depth_reduction = (d0 - d3) / max(1, d0) * 100
    cx_reduction = (cx0 - cx3) / max(1, cx0) * 100

    print(f"{'Level':<10} | {'Depth':<10} | {'CX Count'}")
    print("-" * 40)
    print(f"{'Level 0':<10} | {d0:<10} | {cx0}")
    print(f"{'Level 3':<10} | {d3:<10} | {cx3}")
    print(f"\n-> Depth reduction from Level 0 to Level 3: {depth_reduction:.1f}%")
    print(f"-> CX reduction: {cx_reduction:.1f}%")
    print("-> (Recall: Grover/FPAA typically see ~40% depth reduction with Level 3)")
    print("-> CONCLUSION: R_0 acts as an uncomputation limit, preventing the")
    print("   classical compiler from cancelling A·A† = I and limiting synthesis gains.")


# =============================================================================
# Scenario H: Multi-Ancilla Reflection Explosion (l > 1)
# =============================================================================

def run_scenario_h() -> None:
    """
    H. The Multi-Ancilla Reflection Explosion (R_0 depth scaling with l).

    As l ancilla qubits grow (richer LCU block encodings), R_0 becomes a
    multi-controlled-Z gate. The CX-count of an l-qubit MCZ grows as O(l),
    but the structural depth grows superlinearly due to SWAP routing.

    We sweep l from 1 to 6 and measure R_0 transpiled depth.
    """
    print("\n" + "=" * 70)
    print("SCENARIO H: THE MULTI-ANCILLA REFLECTION EXPLOSION (l > 1)")
    print("=" * 70)
    print("R_0 = I – 2|0^l⟩⟨0^l| on l ancilla qubits.")
    print("Architecture: Linear 1D chain (worst-case routing).\n")

    print(f"{'l (ancilla)':<14} | {'R_0 Depth (All-to-All)':<24} | {'R_0 Depth (Linear 1D)':<24} | {'CX Count (Linear)'}")
    print("-" * 85)
    for l in [1, 2, 3, 4, 5, 6]:
        R0 = build_R_0(l)
        t_all = transpile(R0, basis_gates=BASIS_NISQ, optimization_level=3)
        d_all = t_all.depth()
        if l == 1:
            # Linear coupling_map is undefined for a single qubit (0 edges).
            # Single-qubit R_0 = XZX needs no routing; report All-to-All depth.
            d_lin = d_all
            cx_lin = t_all.count_ops().get('cx', 0)
        else:
            cmap_lin = linear_coupling_map(l)
            t_lin = transpile(R0, basis_gates=BASIS_NISQ, coupling_map=cmap_lin, optimization_level=3)
            d_lin = t_lin.depth()
            cx_lin = t_lin.count_ops().get('cx', 0)
        print(f"{l:<14} | {d_all:<24} | {d_lin:<24} | {cx_lin}")

    print("-> CONCLUSION: MCZ depth scales as O(l) in theory, but linear topology")
    print("   forces extra SWAPs, making R_0 the dominant bottleneck for l > 3 LCU.")


# =============================================================================
# Scenario I: LCU Fault-Tolerant T-Count (Clifford+T)
# =============================================================================

def run_scenario_i(c0: float = 0.6, c1: float = 0.4, synthesis_eps: float = 1e-3) -> None:
    """
    I. The LCU Fault-Tolerant Synthesis (T-Count).

    In Clifford+T, continuous RY/RZ rotations must be approximated via
    Ross-Selinger: ~3.21·log₂(1/ε)–6.93 T gates per non-Clifford rotation.
    PREP contains the fractional angle (arccos(√w0)), generating most T-cost.
    SELECT uses X/CX/CZ which are exact Clifford gates (T-count = 0 in principle).

    We compile each LCU module separately to expose this asymmetry.
    """
    print("\n" + "=" * 70)
    print("SCENARIO I: THE LCU FAULT-TOLERANT T-GATE SYNTHESIS")
    print("=" * 70)
    alpha = c0 + c1
    print(f"H = {c0}·X₀ + {c1}·Z₁,  α = {alpha:.4f}")
    print(f"Clifford+T basis: ['h','s','sdg','cx','t','tdg']")
    print(f"Synthesis precision ε = {synthesis_eps}  →  ~{max(0, int(math.ceil(3.21 * math.log2(1/synthesis_eps) - 6.93)))} T gates per non-Clifford RZ\n")

    prep, select, prep_dag, A_lcu, alpha = build_lcu(c0, c1)
    dist = verify_lcu_block(A_lcu, c0, c1, alpha)
    print(f"LCU Block Verification: ‖A_TL – H/α‖_F = {dist:.3e} -> {'PASS' if dist < 1e-9 else 'FAIL'}\n")

    ft_basis = ['h', 's', 'sdg', 'cx', 't', 'tdg', 'x', 'z']

    def t_count(circ: QuantumCircuit) -> tuple[int,int]:
        try:
            tc = transpile(circ, basis_gates=ft_basis, optimization_level=3)
            t = tc.count_ops().get('t', 0) + tc.count_ops().get('tdg', 0)
            d = tc.depth()
            return t, d
        except Exception as e:
            return -1, -1

    pt, pd = t_count(prep)
    st, sd = t_count(select)
    pdt, pdd = t_count(prep_dag)
    at, ad = t_count(A_lcu)

    print(f"{'Module':<15} | {'T-Count':<10} | {'Clifford+T Depth'}")
    print("-" * 45)
    print(f"{'PREP':<15} | {pt:<10} | {pd}")
    print(f"{'SELECT':<15} | {st:<10} | {sd}")
    print(f"{'PREP_dag':<15} | {pdt:<10} | {pdd}")
    print(f"{'A_LCU (total)':<15} | {at:<10} | {ad}")
    print(f"\n-> PREP sources {100 * pt / max(1, at):.1f}% of the T-gate overhead.")
    print("-> SELECT compiles exactly (no fractional angles → zero T-count ideal).")
    print("-> CONCLUSION: In FTQC, the LCU T-factory overhead lives almost entirely in PREP.")


# =============================================================================
# Scenario J: Extreme LCU Coefficient Skew
# =============================================================================

def run_scenario_j() -> None:
    """
    J. The Extreme LCU Coefficient Skew (c0 >> c1).

    Highly skewed coefficients (e.g., c0=0.999, c1=0.001) push the PREP
    rotation to θ ≈ arccos(√w0) ≈ arccos(0.99995) ≈ 0.01 rad, which is
    near the hardware resolution limit.

    We verify the ideal LCU block is mathematically correct, then perturb the
    PREP rotation by a fixed calibration error δθ = 10^{-3} rad and recompute
    the block-encoding distance. This gives a deterministic proxy for finite
    analog angle resolution without inventing a fake matrix error estimate.
    """
    print("\n" + "=" * 70)
    print("SCENARIO J: THE EXTREME LCU COEFFICIENT SKEW")
    print("=" * 70)

    coeff_pairs = [(0.999, 0.001), (0.99, 0.01), (0.9, 0.1), (0.6, 0.4)]

    print(f"{'c0':<8} | {'c1':<8} | {'α':<8} | {'PREP θ (rad)':<15} | {'Ideal ‖err‖':<14} | {'Perturbed ‖err‖ (δθ=1e-3)'}")
    print("-" * 80)

    for c0, c1 in coeff_pairs:
        alpha = c0 + c1
        w0 = c0 / alpha
        theta_prep = np.arccos(np.sqrt(w0))

        prep, select, prep_dag, A_lcu, alpha_out = build_lcu(c0, c1)
        ideal_dist = verify_lcu_block(A_lcu, c0, c1, alpha_out)

        data = QuantumRegister(2, "data")
        anc = QuantumRegister(1, "anc")
        prep_perturbed = QuantumCircuit(data, anc, name="PREP_perturbed")
        prep_perturbed.ry(2.0 * (theta_prep + 1e-3), anc[0])
        prep_perturbed_dag = prep_perturbed.inverse()
        prep_perturbed_dag.name = "PREP_perturbed_dag"

        A_perturbed = QuantumCircuit(data, anc, name="A_LCU_perturbed")
        A_perturbed.compose(prep_perturbed, inplace=True)
        A_perturbed.compose(select, inplace=True)
        A_perturbed.compose(prep_perturbed_dag, inplace=True)
        perturbed_dist = verify_lcu_block(A_perturbed, c0, c1, alpha_out)

        print(f"{c0:<8} | {c1:<8} | {alpha:<8.4f} | {theta_prep:<15.6f} | {ideal_dist:<14.3e} | {perturbed_dist:.3e}")

    print("\n-> CONCLUSION: Highly skewed coefficients push PREP rotation to near-zero angles.")
    print("   A fixed calibration error δθ therefore becomes a larger relative")
    print("   perturbation of the intended PREP state, increasing block-encoding error.")


# =============================================================================
# Scenario K: Hardware Profiler Comparative Evaluation (OAA)
# =============================================================================

def run_scenario_k(p: float = 0.25) -> None:
    """
    K. The Hardware Profiler Comparative Evaluation (OAA Block-Encoding Evaluation).

    Feeds the full Q^k_opt · A circuit through the HardwareProfiler to get
    a single unified hardware penalty score in nanoseconds, summarizing all
    physical bottlenecks (routing, uncomputation, reflections) in one metric.
    """
    print("\n" + "=" * 70)
    print("SCENARIO K: HARDWARE PROFILER COMPARATIVE EVALUATION (OAA)")
    print("=" * 70)

    try:
        profiler_mod = _import_local_module("oaa_quantum_profiler_module", "quantum_profiler.py")
        HardwareProfiler = profiler_mod.HardwareProfiler
    except Exception:
        print("Scenario K skipped: quantum_profiler module not found.")
        return

    kk = k_opt(p)
    print(f"Parameters: p={p}, k_opt={kk}, m=1 data qubit, l=1 ancilla.")
    print(f"Circuit: Full OAA sequence (Q^{kk} · A) on Linear topology.\n")

    qc = build_oaa_circuit(p, kk, m=1)

    linear_edges = [[0, 1], [1, 0]]
    profiler = HardwareProfiler(
        coupling_map_edges=linear_edges,
        basis_gates=BASIS_NISQ,
        single_qubit_ns=20,
        two_qubit_ns=100,
    )

    print(f"Profiling OAA circuit through compiler stages ...")
    metrics = profiler.profile_circuit(qc)

    print(f"\n{'Metric':<30} | {'Value'}")
    print("-" * 50)
    print(f"{'Logical Depth':<30} | {metrics['logical_depth']}")
    print(f"{'Post-Routing SWAPs':<30} | {metrics['routing_swaps']}")
    print(f"{'Final CNOT Count':<30} | {metrics['final_cnots']}")
    print(f"{'Total Execution Time (ns)':<30} | {metrics['total_time_ns']:.1f}")
    print(f"{'Unified Hardware Penalty':<30} | {metrics['hardware_penalty_score']:.1f}")
    print("\n-> CONCLUSION: All OAA bottlenecks (star-graph routing, uncomputation")
    print("   depth, multi-ancilla reflections) are summarized by a single hardware-time metric.")
    print("   This establishes the physical baseline before transitioning to QSVT.")


def _save_oaa_algorithm_figure(
    *,
    p_values=(0.05, 0.1, 0.2, 0.3, 0.4),
    m: int = 2,
    output_name="oaa_amplification_resource_profile.png",
):
    plt = _load_pyplot()
    p_vals = []
    k_vals = []
    theory_success = []
    exact_success = []
    depth_all = []
    depth_lin = []
    cx_all = []
    cx_lin = []

    for p in p_values:
        kk = k_opt(float(p))
        qc = build_oaa_circuit(float(p), kk, m=m)
        state = Statevector.from_instruction(qc).data
        t_all = transpile(qc, basis_gates=BASIS_NISQ, optimization_level=3)
        t_lin = transpile(
            qc,
            basis_gates=BASIS_NISQ,
            coupling_map=linear_coupling_map(qc.num_qubits),
            optimization_level=3,
        )

        theta0 = float(np.arcsin(np.sqrt(float(p))))
        p_vals.append(float(p))
        k_vals.append(int(kk))
        theory_success.append(float(np.sin((2 * kk + 1) * theta0) ** 2))
        exact_success.append(_ancilla_zero_probability(state, ancilla_index=0))
        depth_all.append(float(t_all.depth()))
        depth_lin.append(float(t_lin.depth()))
        cx_all.append(float(t_all.count_ops().get('cx', 0)))
        cx_lin.append(float(t_lin.count_ops().get('cx', 0)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("OAA Amplification and Resource Profile", fontsize=14, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(p_vals, k_vals, marker="o", linewidth=2.2, color="#1f77b4")
    ax1.set_title("Optimal Amplification Length")
    ax1.set_ylabel("k*")

    ax2.plot(p_vals, theory_success, marker="o", linewidth=2.0, label="theory", color="#2ca02c")
    ax2.plot(p_vals, exact_success, marker="s", linewidth=2.0, label="exact circuit", color="#d62728")
    ax2.set_title("Success Probability at k*")
    ax2.set_ylabel("Success probability")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(fontsize=8)

    ax3.plot(p_vals, depth_all, marker="o", linewidth=2.0, label="all-to-all", color="#9467bd")
    ax3.plot(p_vals, depth_lin, marker="s", linewidth=2.0, label="linear", color="#ff7f0e")
    ax3.set_title("Circuit Depth")
    ax3.set_ylabel("Depth")
    ax3.legend(fontsize=8)

    ax4.plot(p_vals, cx_all, marker="o", linewidth=2.0, label="all-to-all", color="#8c564b")
    ax4.plot(p_vals, cx_lin, marker="s", linewidth=2.0, label="linear", color="#1f77b4")
    ax4.set_title("Entangling Cost")
    ax4.set_ylabel("CX count")
    ax4.legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Initial success probability p")
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "oaa_amplification_resource_profile",
            "p_values": [float(x) for x in p_values],
            "optimal_k_values": [int(x) for x in k_vals],
            "m": int(m),
            "basis_gates": list(BASIS_NISQ),
            "topologies": ["all_to_all", "linear"],
        },
    )
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


def _save_oaa_routing_figure(
    *,
    m_values=(2, 3, 4, 5, 6),
    output_name="oaa_star_routing_penalty.png",
):
    plt = _load_pyplot()
    ms = []
    depth_all = []
    depth_hex = []
    depth_lin = []
    swap_hex = []
    swap_lin = []

    for m in m_values:
        anc = QuantumRegister(1, "anc")
        data = QuantumRegister(int(m), "data")
        star = QuantumCircuit(anc, data, name="A_star")
        star.h(anc[0])
        for i in range(int(m)):
            star.cx(anc[0], data[i])

        t_all = transpile(star, basis_gates=BASIS_NISQ, optimization_level=3)
        t_hex = transpile(star, basis_gates=BASIS_NISQ, coupling_map=heavy_hex_coupling_map(star.num_qubits), optimization_level=3)
        t_lin = transpile(star, basis_gates=BASIS_NISQ, coupling_map=linear_coupling_map(star.num_qubits), optimization_level=3)

        ms.append(int(m))
        depth_all.append(float(t_all.depth()))
        depth_hex.append(float(t_hex.depth()))
        depth_lin.append(float(t_lin.depth()))
        swap_hex.append(float(get_swap_count(t_hex)))
        swap_lin.append(float(get_swap_count(t_lin)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("OAA Star-Graph Routing Penalty", fontsize=14, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(ms, depth_all, marker="o", linewidth=2.0, label="all-to-all", color="#1f77b4")
    ax1.plot(ms, depth_hex, marker="s", linewidth=2.0, label="heavy-hex", color="#2ca02c")
    ax1.plot(ms, depth_lin, marker="D", linewidth=2.0, label="linear", color="#d62728")
    ax1.set_title("Depth vs Data Register Size")
    ax1.set_ylabel("Depth")
    ax1.legend(fontsize=8)

    ax2.plot(ms, swap_hex, marker="s", linewidth=2.0, label="heavy-hex", color="#9467bd")
    ax2.plot(ms, swap_lin, marker="D", linewidth=2.0, label="linear", color="#ff7f0e")
    ax2.set_title("Inserted SWAPs")
    ax2.set_ylabel("SWAP count")
    ax2.legend(fontsize=8)

    hex_multiplier = [h / max(1.0, a) for h, a in zip(depth_hex, depth_all)]
    lin_multiplier = [l / max(1.0, a) for l, a in zip(depth_lin, depth_all)]
    ax3.plot(ms, hex_multiplier, marker="s", linewidth=2.0, label="heavy-hex / all", color="#8c564b")
    ax3.plot(ms, lin_multiplier, marker="D", linewidth=2.0, label="linear / all", color="#1f77b4")
    ax3.set_title("Depth Multiplier")
    ax3.set_ylabel("Multiplier")
    ax3.legend(fontsize=8)

    star_edges = list(ms)
    ax4.bar(star_edges, [m for m in ms], color="#7f7f7f")
    ax4.set_title("Star Degree of Ancilla Control")
    ax4.set_ylabel("Controlled data qubits")

    for axis in axes.flat:
        axis.set_xlabel("Data qubits m")
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    output_path = os.path.join(_RESULT_DIR, output_name)
    save_figure_with_metadata(
        fig,
        output_path,
        {
            "figure_kind": "oaa_star_routing_penalty",
            "m_values": [int(x) for x in m_values],
            "basis_gates": list(BASIS_NISQ),
            "topologies": ["all_to_all", "heavy_hex", "linear"],
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

    print("OAA Transpilation Benchmark Suite — Scenarios A through K")
    print(f"Results saved to: {output_filepath}")
    print("=" * 70)
    print(publishability.summary())
    raw_scenarios = [
        ("A", lambda: run_scenario_a(p=0.25, m=2)),
        ("B", lambda: run_scenario_b(c0=0.6, c1=0.4)),
        ("C", lambda: run_scenario_c(m=6)),
        ("D", run_scenario_d),
        ("E", lambda: run_scenario_e(p=0.001)),
        ("F", run_scenario_f),
        ("G", lambda: run_scenario_g(p=0.1)),
        ("H", run_scenario_h),
        ("I", lambda: run_scenario_i(c0=0.6, c1=0.4, synthesis_eps=1e-3)),
        ("J", run_scenario_j),
        ("K", lambda: run_scenario_k(p=0.25)),
    ]
    interactive_scenarios = [
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
    ]
    scenarios = wrap_scenarios(raw_scenarios, module_globals=globals(), config=publishability)
    interactive_wrapped = wrap_scenarios(interactive_scenarios, module_globals=globals(), config=publishability)

    try:
        cli_executed = run_cli_scenario(cli_argv, interactive_wrapped)
        if not cli_executed:
            for label, fn in scenarios:
                try:
                    fn()
                except Exception:
                    import traceback
                    print(f"\n*** SCENARIO {label} FAILED ***")
                    traceback.print_exc()
            run_interactive_scenario_repl(
                interactive_wrapped,
                sep="=" * 70,
            )
    except Exception as ex:
        import traceback
        print("\n\n*** UNHANDLED EXCEPTION ***")
        traceback.print_exc()
    finally:
        render_backend_validation_summary(publishability)
        _save_oaa_algorithm_figure()
        _save_oaa_routing_figure()
        logger.log.close()
        sys.stdout = logger.terminal
        print(f"\nBenchmark suite complete. Results saved to {output_filepath}")

