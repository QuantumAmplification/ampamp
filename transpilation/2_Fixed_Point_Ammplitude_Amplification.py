"""FPAA architectural trade-off study (Section III).

This script keeps only the six requested modules:
1) Phase schedule synthesizer + Table-I analytical check
2) Generalized Grover iterate circuit (mcp-based)
3) Passband plateau analysis in the SU(2) subspace
4) Recursive nesting demonstration (circuit + numerical composition)
5) NISQ phase-noise sensitivity benchmark
6) Fault-tolerant Clifford+T compilation (T-count overhead)

Standing notation aligned with final.tex:
- H_Good / H_Bad: target and non-target subspaces
- Pi_Good / Pi_Bad: corresponding orthogonal projectors
- |All> = A|0>^{⊗n}: prepared input state before amplification
- p = ||Pi_Good|All>||^2 (for unstructured search, p=M/N)
- sin^2(theta0)=p and Grover step angle theta=2*theta0
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    from one_click_utils import start_one_click_session
except Exception:
    def start_one_click_session(script_file, *, figure_prefix=None, log_name="terminal_output.log"):
        import atexit
        import io
        import os
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

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Gate
    from qiskit.quantum_info import Statevector
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore[assignment]
    transpile = None  # type: ignore[assignment]
    Gate = object  # type: ignore[misc,assignment]
    Statevector = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Shared mathematical helpers
# -----------------------------------------------------------------------------

def chebyshev_t(order: float, x: float) -> float:
    """Chebyshev T_order(x), with safe extension for x>=1 used by FPAA schedules."""
    if x >= 1.0:
        return float(np.cosh(order * np.arccosh(x)))
    if abs(x) <= 1.0 and float(order).is_integer():
        return float(np.cos(order * np.arccos(x)))
    raise ValueError("Unsupported region for this helper.")


def fpaa_generalized_iterates(L: int) -> int:
    """Return l=(L-1)/2 for the odd FPAA parameter L=2l+1."""
    L_int = int(L)
    if L_int < 1 or L_int % 2 == 0:
        raise ValueError("L must be a positive odd integer.")
    return (L_int - 1) // 2


def fpaa_query_complexity(L: int) -> int:
    """FPAA query complexity is L-1=2l for odd L=2l+1."""
    return 2 * fpaa_generalized_iterates(L)


def _fpaa_gamma_inverse(L: int, delta: float) -> float:
    """gamma^{-1}=T_{1/L}(1/delta) for the Yoder-Low-Chuang schedule."""
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    return chebyshev_t(1.0 / L, 1.0 / delta)


def fpaa_inner_delta(L_outer: int, delta: float) -> float:
    """Retuned inner error bound for nesting, delta_1=T_{1/L_outer}(1/delta)^(-1)."""
    return 1.0 / _fpaa_gamma_inverse(L_outer, delta)


def fpaa_success_probability(L: int, delta: float, lam: float) -> float:
    """Closed-form FPAA success probability from Yoder-Low-Chuang Eq. (1)."""
    p = float(np.clip(lam, 0.0, 1.0))
    gamma_inv = _fpaa_gamma_inverse(L, delta)
    arg = gamma_inv * math.sqrt(max(0.0, 1.0 - p))
    return float(1.0 - delta * delta * chebyshev_t(float(L), arg) ** 2)


def passband_edge(L: int, delta: float) -> float:
    """w = 1 - T_{1/L}(1/delta)^(-2)."""
    gamma_inv = _fpaa_gamma_inverse(L, delta)
    return 1.0 - gamma_inv ** (-2)


def _require_qiskit() -> None:
    if QuantumCircuit is None:
        raise RuntimeError("Qiskit is required for circuit-level modules.")


def _validate_good_indices(num_qubits: int, good_indices: Sequence[int]) -> List[int]:
    """Validate computational-basis indices spanning H_Good."""
    dim = 2**num_qubits
    out: List[int] = []
    for idx in good_indices:
        i = int(idx)
        if i < 0 or i >= dim:
            raise ValueError(f"Marked index {i} out of range for {num_qubits} qubits.")
        out.append(i)
    if not out:
        raise ValueError("H_Good index set must not be empty.")
    return sorted(set(out))


def _good_state_to_indices(num_qubits: int, good_state: Optional[str]) -> List[int]:
    """Convert optional H_Good bitstring representative into basis indices."""
    if good_state is None:
        return [2**num_qubits - 1]
    if len(good_state) != num_qubits or any(ch not in "01" for ch in good_state):
        raise ValueError("H_Good state must be a binary bitstring of length num_qubits.")
    return [int(good_state, 2)]


# -----------------------------------------------------------------------------
# Module 1: Phase schedule synthesizer
# -----------------------------------------------------------------------------

def generate_fpaa_phases(L: int, delta: float, epsilon: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute analytical FPAA phases alpha_j and beta_j.

    Uses:
      gamma_inv = cosh((1/L) * arccosh(1/delta))
      alpha_j = 2 arccot(tan(2*pi*j/L) * sqrt(1-gamma^2))
      beta_{l-j+1} = -alpha_j

    The returned arrays therefore contain l=(L-1)/2 generalized Grover iterates,
    exactly as in Eq. (11) of Yoder-Low-Chuang.

    epsilon models systematic over-rotation: angle -> angle*(1+epsilon).
    """
    l = fpaa_generalized_iterates(L)

    gamma_inv = float(_fpaa_gamma_inverse(L, delta))
    gamma = 1.0 / gamma_inv
    sq_term = float(np.sqrt(max(0.0, 1.0 - gamma * gamma)))

    # Handle singular points robustly (tan near 0 or undefined).
    tol = 1e-12
    alpha = np.zeros(l, dtype=float)
    for j in range(1, l + 1):
        theta_j = (2.0 * np.pi * j) / L
        tan_val = 0.0 if np.isclose(np.sin(theta_j), 0.0, atol=tol) else float(np.tan(theta_j))
        denom = tan_val * sq_term
        alpha[j - 1] = np.pi if np.isclose(denom, 0.0, atol=tol) else 2.0 * float(np.arctan2(1.0, denom))

    beta = -alpha[::-1]

    if epsilon != 0.0:
        scale = 1.0 + float(epsilon)
        alpha = alpha * scale
        beta = beta * scale

    return alpha, beta


def test_table_i_exact_match(delta: float = 0.1, atol: float = 1e-8) -> None:
    """Rigorous L=3 analytical cross-check for the single FPAA phase pair."""
    L = 3
    alpha_dyn, beta_dyn = generate_fpaa_phases(L=L, delta=delta)

    gamma_inv = float(_fpaa_gamma_inverse(L, delta))
    gamma = 1.0 / gamma_inv
    sq_term = float(np.sqrt(max(0.0, 1.0 - gamma * gamma)))

    a1 = 2.0 * float(np.arctan2(1.0, np.tan(2.0 * np.pi / 3.0) * sq_term))
    alpha_ref = np.array([a1], dtype=float)
    beta_ref = np.array([-a1], dtype=float)

    if not np.allclose(alpha_dyn, alpha_ref, atol=atol):
        raise AssertionError(f"Alpha mismatch.\nExpected: {alpha_ref}\nGot: {alpha_dyn}")
    if not np.allclose(beta_dyn, beta_ref, atol=atol):
        raise AssertionError(f"Beta mismatch.\nExpected: {beta_ref}\nGot: {beta_dyn}")


def test_fpaa_closed_form_match(L: int = 3, delta: float = 0.1, atol: float = 1e-10) -> None:
    """Verify the synthesized phase schedule reproduces the exact Chebyshev map."""
    alphas, betas = generate_fpaa_phases(L=L, delta=delta)
    p_samples = np.linspace(max(passband_edge(L, delta), 0.05), 0.95, 11)
    for p in p_samples:
        simulated = simulate_2d_fpaa_sequence(float(p), alphas, betas)
        theoretical = fpaa_success_probability(L, delta, float(p))
        if not np.isclose(simulated, theoretical, atol=atol):
            raise AssertionError(
                f"Closed-form mismatch at p={p:.6f}: theory={theoretical:.12f}, sim={simulated:.12f}"
            )


# -----------------------------------------------------------------------------
# Module 2: Generalized Grover iterate (Qiskit)
# -----------------------------------------------------------------------------

def generalized_oracle(num_qubits: int, good_indices: Sequence[int], beta: float) -> "QuantumCircuit":
    """S_t(beta): apply target phase on basis states spanning H_Good."""
    _require_qiskit()
    good = _validate_good_indices(num_qubits, good_indices)
    qc = QuantumCircuit(num_qubits, name=f"S_t({beta:.3f})")

    controls = list(range(num_qubits - 1))
    target = num_qubits - 1

    for idx in good:
        bits = format(idx, f"0{num_qubits}b")[::-1]  # little-endian map
        for q, bit in enumerate(bits):
            if bit == "0":
                qc.x(q)
        if num_qubits == 1:
            qc.p(beta, 0)
        else:
            qc.mcp(beta, controls, target)
        for q, bit in enumerate(bits):
            if bit == "0":
                qc.x(q)
    return qc


def generalized_diffusion(num_qubits: int, alpha: float) -> "QuantumCircuit":
    """S_s(alpha): H^n X^n mcp(-alpha) X^n H^n."""
    _require_qiskit()
    qc = QuantumCircuit(num_qubits, name=f"S_s({alpha:.3f})")

    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.p(-alpha, 0)
    else:
        qc.mcp(-alpha, list(range(num_qubits - 1)), num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_fpaa_circuit_from_phases(
    num_qubits: int,
    good_indices: Sequence[int],
    alphas: np.ndarray,
    betas: np.ndarray,
    initialize_superposition: bool = True,
) -> "QuantumCircuit":
    """Build A followed by the l generalized Grover iterates of the FPAA schedule."""
    _require_qiskit()
    if len(alphas) != len(betas):
        raise ValueError("alphas and betas must have the same length.")
    good = _validate_good_indices(num_qubits, good_indices)

    qc = QuantumCircuit(num_qubits, name=f"FPAA_l{len(alphas)}")
    if initialize_superposition:
        qc.h(range(num_qubits))

    # Circuit order is left-to-right execution of G_1, ..., G_L.
    for a_j, b_j in zip(alphas, betas):
        qc.global_phase += np.pi  # explicit '-' in G definition
        qc.append(generalized_oracle(num_qubits, good, float(b_j)).to_gate(), range(num_qubits))
        qc.append(generalized_diffusion(num_qubits, float(a_j)).to_gate(), range(num_qubits))

    return qc


def build_fpaa_circuit(
    num_qubits: int,
    L: int,
    delta: float,
    epsilon: float = 0.0,
    good_state: Optional[str] = None,
    good_indices: Optional[Sequence[int]] = None,
) -> "QuantumCircuit":
    """Convenience builder from schedule parameters."""
    if good_indices is None:
        good_indices = _good_state_to_indices(num_qubits, good_state)
    alphas, betas = generate_fpaa_phases(L=L, delta=delta, epsilon=epsilon)
    return build_fpaa_circuit_from_phases(num_qubits, good_indices, alphas, betas, initialize_superposition=True)


def build_standard_grover_circuit(
    num_qubits: int,
    iterations: int,
    good_state: Optional[str] = None,
    good_indices: Optional[Sequence[int]] = None,
) -> "QuantumCircuit":
    """Grover as trivial FPAA schedule alpha=beta=pi."""
    if good_indices is None:
        good_indices = _good_state_to_indices(num_qubits, good_state)
    alphas = np.full(iterations, np.pi, dtype=float)
    betas = np.full(iterations, np.pi, dtype=float)
    return build_fpaa_circuit_from_phases(num_qubits, good_indices, alphas, betas, initialize_superposition=True)


def test_grover_fallback_rigor(
    num_qubits: int = 4,
    good_indices: Sequence[int] = (5,),
    L: int = 3,
    atol: float = 1e-8,
) -> None:
    """Verify generalized iterate collapses to standard Grover at alpha=beta=pi."""
    _require_qiskit()
    if Statevector is None:
        raise RuntimeError("qiskit.quantum_info.Statevector is required for fallback testing.")

    good = _validate_good_indices(num_qubits, good_indices)
    qc = build_standard_grover_circuit(num_qubits, L, good_indices=good)
    state = Statevector.from_instruction(qc)

    success_prob = float(sum(np.abs(state.data[idx]) ** 2 for idx in good))
    p_value = len(good) / (2**num_qubits)
    theta0 = np.arcsin(np.sqrt(p_value))
    theta = 2.0 * theta0
    theoretical_prob = float(np.sin((2 * L + 1) * theta / 2.0) ** 2)

    if not np.isclose(success_prob, theoretical_prob, atol=atol):
        raise AssertionError(f"Grover fallback failed: sim={success_prob}, theory={theoretical_prob}")


# -----------------------------------------------------------------------------
# Module 3: Passband plateau analysis (SU(2) simulator)
# -----------------------------------------------------------------------------

def simulate_2d_fpaa_sequence(lam: float, alphas: np.ndarray, betas: np.ndarray) -> float:
    """Simulate FPAA in invariant basis {|H_Bad>,|H_Good>} for input probability p."""
    if len(alphas) != len(betas):
        raise ValueError("alphas and betas must have same length.")

    p = float(np.clip(lam, 0.0, 1.0))
    if p <= 0.0:
        return 0.0

    # Initial vector and projectors in the 2D invariant subspace.
    s = np.array([math.sqrt(1.0 - p), math.sqrt(p)], dtype=complex)
    state = s.copy()
    I = np.eye(2, dtype=complex)
    Pi_t = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    Pi_s = np.outer(s, s.conj())

    for a_j, b_j in zip(alphas, betas):
        S_t = I - (1.0 - np.exp(1j * float(b_j))) * Pi_t
        S_s = I - (1.0 - np.exp(-1j * float(a_j))) * Pi_s
        state = -(S_s @ S_t) @ state

    return float(np.abs(state[1]) ** 2)


def _grover_closed_form_success(lam: float, iterations: int) -> float:
    p_value = float(np.clip(lam, 0.0, 1.0))
    if p_value <= 0.0:
        return 0.0
    theta = 2.0 * np.arcsin(np.sqrt(p_value))
    return float(np.sin((2 * iterations + 1) * theta / 2.0) ** 2)


def sweep_passband(
    L: int,
    delta: float,
    epsilon: float = 0.0,
    p_min: float = 1e-3,
    p_max: float = 1.0,
    num_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Sweep continuous p and compare FPAA vs standard Grover."""
    p_values = np.linspace(p_min, p_max, num_points)
    alphas, betas = generate_fpaa_phases(L=L, delta=delta, epsilon=epsilon)
    grover_iterations = fpaa_generalized_iterates(L)

    fpaa = np.array([simulate_2d_fpaa_sequence(p, alphas, betas) for p in p_values], dtype=float)
    grover = np.array([_grover_closed_form_success(p, grover_iterations) for p in p_values], dtype=float)

    w = passband_edge(L, delta)
    floor = 1.0 - delta * delta
    mask = p_values >= w
    min_passband = float(np.min(fpaa[mask])) if np.any(mask) else float("nan")
    max_violation = float(max(0.0, floor - min_passband)) if np.isfinite(min_passband) else float("nan")

    return {
        "p": p_values,
        "fpaa": fpaa,
        "grover": grover,
        "passband_edge": np.array([w]),
        "target_floor": np.array([floor]),
        "min_passband": np.array([min_passband]),
        "max_violation": np.array([max_violation]),
    }


def _plot_passband(curve: Dict[str, np.ndarray], output: str, title: str) -> None:
    import matplotlib.pyplot as plt

    p_values = curve["p"]
    edge = float(curve["passband_edge"][0])
    floor = float(curve["target_floor"][0])
    min_passband = float(curve["min_passband"][0])
    max_violation = float(curve["max_violation"][0])

    plt.figure(figsize=(10, 6))
    plt.plot(p_values, curve["grover"], "--", color="red", alpha=0.75, label="Standard Grover")
    plt.plot(p_values, curve["fpaa"], color="blue", linewidth=2.4, label="FPAA")
    plt.axvline(edge, color="green", linestyle=":", label=f"Passband edge w={edge:.4f}")
    plt.axhline(floor, color="black", linestyle=":", label=f"Target floor 1-delta^2={floor:.4f}")
    plt.fill_between(p_values, floor, 1.0, where=(p_values >= edge), color="green", alpha=0.08)
    plt.ylim(0.0, 1.02)
    plt.xlim(0.0, 1.0)
    plt.xlabel("Initial Success Probability (p)")
    plt.ylabel("Final Success Probability p_k")
    plt.title(f"{title}\nmin(FPAA|p>=w)={min_passband:.6f}, violation={max_violation:.2e}")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_passband_plateau(
    L: int,
    delta: float,
    output: str = "fpaa_passband.png",
    epsilon: float = 0.0,
    p_min: float = 1e-3,
    p_max: float = 1.0,
    num_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Generate Module-3 journal evidence plot."""
    curve = sweep_passband(
        L=L,
        delta=delta,
        epsilon=epsilon,
        p_min=p_min,
        p_max=p_max,
        num_points=num_points,
    )
    _plot_passband(curve, output, f"FPAA Passband (L={L}, delta={delta})")
    return curve


# -----------------------------------------------------------------------------
# Module 4: Recursive nesting demonstration
# -----------------------------------------------------------------------------

def _s0_reflection_gate(num_qubits: int, alpha: float) -> "Gate":
    """S_0(alpha): phase on |0...0>, wrapped as gate for nesting composition."""
    _require_qiskit()
    qc = QuantumCircuit(num_qubits, name=f"S0({alpha:.3f})")
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.p(-alpha, 0)
    else:
        qc.mcp(-alpha, list(range(num_qubits - 1)), num_qubits - 1)
    qc.x(range(num_qubits))
    return qc.to_gate()


def build_nested_fpaa_circuit(
    num_qubits: int,
    L1: int,
    L2: int,
    delta: float,
    good_state: Optional[str] = None,
    good_indices: Optional[Sequence[int]] = None,
) -> "QuantumCircuit":
    """Build nested circuit: outer source reflection about |psi_L1> = U_L1|0>."""
    _require_qiskit()
    if good_indices is None:
        good_indices = _good_state_to_indices(num_qubits, good_state)
    good = _validate_good_indices(num_qubits, good_indices)
    delta1 = fpaa_inner_delta(L2, delta)

    # Inner unitary U_L1 (as gate) and inverse.
    a1, b1 = generate_fpaa_phases(L1, delta1)
    u1 = build_fpaa_circuit_from_phases(num_qubits, good, a1, b1, initialize_superposition=True).to_gate(
        label=f"U_L{L1}"
    )
    u1_dag = u1.inverse()

    # Outer schedule (L2), with source reflection implemented via U1^dag S0 U1.
    a2, b2 = generate_fpaa_phases(L2, delta)
    qc = QuantumCircuit(num_qubits, name=f"Nested_{L1}x{L2}")
    qc.append(u1, range(num_qubits))
    for a_j, b_j in zip(a2, b2):
        qc.global_phase += np.pi
        qc.append(generalized_oracle(num_qubits, good, float(b_j)).to_gate(), range(num_qubits))
        qc.append(u1_dag, range(num_qubits))
        qc.append(_s0_reflection_gate(num_qubits, float(a_j)), range(num_qubits))
        qc.append(u1, range(num_qubits))
    return qc


def recursive_nesting_curves(
    L1: int = 3,
    L2: int = 3,
    delta: float = 0.1,
    p_min: float = 1e-4,
    p_max: float = 0.4,
    num_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Compare base L1, nested L1xL2, and native L1*L2 in SU(2) simulation."""
    delta1 = fpaa_inner_delta(L2, delta)
    a1, b1 = generate_fpaa_phases(L1, delta1)
    a2, b2 = generate_fpaa_phases(L2, delta)
    ac, bc = generate_fpaa_phases(L1 * L2, delta)
    p_values = np.linspace(p_min, p_max, num_points)

    base = np.array([simulate_2d_fpaa_sequence(x, a1, b1) for x in p_values], dtype=float)
    nested = np.array([simulate_2d_fpaa_sequence(p, a2, b2) for p in base], dtype=float)
    native = np.array([simulate_2d_fpaa_sequence(x, ac, bc) for x in p_values], dtype=float)

    diff = np.abs(nested - native)
    return {
        "p": p_values,
        "base_l1": base,
        "nested": nested,
        "native": native,
        "abs_diff": diff,
        "max_abs_diff": np.array([float(np.max(diff))]),
        "delta_l1": np.array([delta1]),
        "w_l1": np.array([passband_edge(L1, delta1)]),
        "w_comp": np.array([passband_edge(L1 * L2, delta)]),
    }


def plot_recursive_nesting_proof(
    L1: int = 3,
    L2: int = 3,
    delta: float = 0.1,
    output: str = "fpaa_nesting.png",
    p_min: float = 1e-4,
    p_max: float = 0.4,
    num_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Generate Module-4 overlay proof: nested (L1xL2) vs native (L1*L2)."""
    import matplotlib.pyplot as plt

    c = recursive_nesting_curves(L1, L2, delta, p_min, p_max, num_points)
    max_diff = float(c["max_abs_diff"][0])

    plt.figure(figsize=(10, 6))
    plt.plot(c["p"], c["base_l1"], color="gray", linestyle=":", linewidth=2.0, label=f"Base L={L1}")
    plt.plot(c["p"], c["native"], color="blue", linewidth=3.0, alpha=0.55, label=f"Native L={L1*L2}")
    plt.plot(c["p"], c["nested"], color="red", linestyle="--", linewidth=2.0, label=f"Nested {L1}x{L2}")
    plt.axvline(float(c["w_l1"][0]), color="gray", linestyle=":", alpha=0.6)
    plt.axvline(float(c["w_comp"][0]), color="blue", linestyle=":", alpha=0.6)
    plt.xlabel("Initial Success Probability (p)")
    plt.ylabel("Final Success Probability p_k")
    plt.title(f"Recursive Nesting: T_{L2}(T_{L1}(x)) = T_{L1*L2}(x)\nmax |nested-native|={max_diff:.2e}")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return c


def nesting_composition_error(
    L1: int = 3,
    L2: int = 3,
    samples: int = 1000,
    x_min: float = 1.0,
    x_max: float = 5.0,
) -> float:
    """Direct polynomial identity check: T_{L2}(T_{L1}(x)) - T_{L1*L2}(x)."""
    xs = np.linspace(x_min, x_max, samples)
    lhs = np.array([chebyshev_t(L2, chebyshev_t(L1, x)) for x in xs], dtype=float)
    rhs = np.array([chebyshev_t(L1 * L2, x) for x in xs], dtype=float)
    return float(np.max(np.abs(lhs - rhs)))


# -----------------------------------------------------------------------------
# Module 5: NISQ phase-noise sensitivity
# -----------------------------------------------------------------------------

def simulate_noisy_fpaa_sequence(lam: float, alphas: np.ndarray, betas: np.ndarray, epsilon: float) -> float:
    """Systematic over-rotation model for all phase gates."""
    return simulate_2d_fpaa_sequence(
        lam,
        np.asarray(alphas, dtype=float) * (1.0 + float(epsilon)),
        np.asarray(betas, dtype=float) * (1.0 + float(epsilon)),
    )


def noise_sensitivity_sweep(
    L: int,
    delta: float,
    epsilons: Iterable[float] = (0.0, 0.01, 0.05, 0.10),
    p_min: float = 1e-3,
    p_max: float = 1.0,
    num_points: int = 1000,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Run noisy passband sweeps and report degradation metrics per epsilon."""
    alphas, betas = generate_fpaa_phases(L=L, delta=delta)
    p_values = np.linspace(p_min, p_max, num_points)
    w = passband_edge(L, delta)
    floor = 1.0 - delta * delta
    mask = p_values >= w

    out: Dict[float, Dict[str, np.ndarray]] = {}
    for eps in epsilons:
        p_final = np.array([simulate_noisy_fpaa_sequence(x, alphas, betas, float(eps)) for x in p_values], dtype=float)
        min_passband = float(np.min(p_final[mask])) if np.any(mask) else float("nan")
        max_violation = float(max(0.0, floor - min_passband)) if np.isfinite(min_passband) else float("nan")
        out[float(eps)] = {
            "p": p_values,
            "fpaa": p_final,
            "passband_edge": np.array([w]),
            "target_floor": np.array([floor]),
            "min_passband": np.array([min_passband]),
            "max_violation": np.array([max_violation]),
        }
    return out


def _plot_noise(noise_curves: Dict[float, Dict[str, np.ndarray]], output: str, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    colors = {0.0: "blue", 0.01: "green", 0.05: "orange", 0.10: "red"}
    styles = {0.0: "-", 0.01: "--", 0.05: "--", 0.10: "--"}

    for eps, curve in sorted(noise_curves.items(), key=lambda kv: kv[0]):
        label = "Ideal (0% Error)" if eps == 0.0 else f"{100*eps:.0f}% Phase Error"
        plt.plot(
            curve["p"],
            curve["fpaa"],
            color=colors.get(float(eps), None),
            linestyle=styles.get(float(eps), "--"),
            linewidth=3.0 if eps == 0.0 else 2.0,
            label=label,
        )

    ref = noise_curves[sorted(noise_curves.keys())[0]]
    w = float(ref["passband_edge"][0])
    floor = float(ref["target_floor"][0])
    plt.axvline(w, color="black", linestyle=":", linewidth=1.2, label=f"Ideal passband edge w={w:.4f}")
    plt.axhline(floor, color="gray", linestyle=":", linewidth=1.2, label=f"Target floor 1-delta^2={floor:.4f}")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Initial Success Probability (p)")
    plt.ylabel("Final Success Probability p_k")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_nisq_robustness_benchmark(
    L: int = 5,
    delta: float = 0.1,
    output: str = "fpaa_noise.png",
    epsilons: Iterable[float] = (0.0, 0.01, 0.05, 0.10),
    p_min: float = 1e-3,
    p_max: float = 1.0,
    num_points: int = 1000,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Generate Module-5 degradation plot and return metrics."""
    curves = noise_sensitivity_sweep(L, delta, epsilons, p_min, p_max, num_points)
    _plot_noise(curves, output, f"NISQ Robustness: FPAA Plateau Degradation (L={L}, delta={delta})")
    return curves


# -----------------------------------------------------------------------------
# Module 6: Fault-tolerant compilation cost (Clifford+T)
# -----------------------------------------------------------------------------

@dataclass
class CompilationResult:
    algorithm: str
    iterations: int
    base_unitaries: str
    continuous_angles: str
    theoretical_depth: str
    t_gate_count: int
    overhead_multiplier: float
    passband_stability: str


def _is_clifford_phase(theta: float, tol: float = 1e-10) -> bool:
    """True when theta is a multiple of pi/2 (Clifford-only Z rotation)."""
    k = round(theta / (math.pi / 2.0))
    return abs(theta - k * (math.pi / 2.0)) < tol


def _t_count_for_pi_over_4_multiple(theta: float, tol: float = 1e-10) -> Optional[int]:
    """Exact T-count for RZ(k*pi/4): odd-k => 1 T, even-k => 0 T (up to Clifford/global)."""
    k = int(round(theta / (math.pi / 4.0)))
    if abs(theta - k * (math.pi / 4.0)) > tol:
        return None
    return 1 if (k % 2) != 0 else 0


def _estimate_t_per_rotation(synthesis_eps: float) -> int:
    """Rough Ross-Selinger style scaling for single-qubit rotation synthesis."""
    if synthesis_eps <= 0:
        raise ValueError("synthesis_eps must be > 0")
    return max(0, int(math.ceil(3.21 * math.log2(1.0 / synthesis_eps) - 6.93)))


def _estimate_t_count_from_native(qc: "QuantumCircuit", synthesis_eps: float) -> int:
    """Estimate T-count by charging non-Clifford RZ gates after native transpilation."""
    t_per = _estimate_t_per_rotation(synthesis_eps)
    total = 0
    for inst, qargs, cargs in qc.data:
        if inst.name == "rz":
            theta = float(inst.params[0])
            if _is_clifford_phase(theta):
                continue
            t_exact = _t_count_for_pi_over_4_multiple(theta)
            total += t_exact if t_exact is not None else t_per
        elif inst.name == "t":
            total += 1
    return int(total)


def benchmark_t_gate_blowup(
    num_qubits: int,
    L: int,
    delta: float,
    good_state: Optional[str] = None,
    good_indices: Optional[Sequence[int]] = None,
    synthesis_eps: float = 1e-3,
    optimization_level: int = 3,
) -> List[CompilationResult]:
    """Compile Grover vs FPAA into Clifford+T and compare query-oracle overhead proxies."""
    _require_qiskit()
    if transpile is None:
        raise RuntimeError("Qiskit transpiler is unavailable.")

    if good_indices is None:
        good_indices = _good_state_to_indices(num_qubits, good_state)
    good = _validate_good_indices(num_qubits, good_indices)

    grover_iterations = fpaa_generalized_iterates(L)

    # Match FPAA against the Grover baseline with the same number of generalized iterates.
    grover = build_standard_grover_circuit(num_qubits, grover_iterations, good_indices=good)
    fpaa = build_fpaa_circuit(num_qubits, L, delta, good_indices=good)

    def direct_t_count(qc: "QuantumCircuit") -> Optional[int]:
        try:
            tqc = transpile(qc, basis_gates=["cx", "h", "s", "t"], optimization_level=optimization_level)
            return int(tqc.count_ops().get("t", 0))
        except Exception:
            return None

    g_t = direct_t_count(grover)
    f_t = direct_t_count(fpaa)

    # If direct Clifford+T synthesis path is unavailable, estimate from native RZ counts.
    if g_t is None or f_t is None:
        g_native = transpile(grover, basis_gates=["rz", "sx", "x", "cx"], optimization_level=optimization_level)
        f_native = transpile(fpaa, basis_gates=["rz", "sx", "x", "cx"], optimization_level=optimization_level)
        if g_t is None:
            g_t = _estimate_t_count_from_native(g_native, synthesis_eps)
        if f_t is None:
            f_t = _estimate_t_count_from_native(f_native, synthesis_eps)

    g_t = int(g_t)
    f_t = int(f_t)
    mult = float(f_t / max(1, g_t))

    return [
        CompilationResult(
            algorithm="Standard Grover",
            iterations=grover_iterations,
            base_unitaries="D(pi)O(pi)",
            continuous_angles="No",
            theoretical_depth="O(n)",
            t_gate_count=g_t,
            overhead_multiplier=1.0,
            passband_stability="Unstable (Souffle)",
        ),
        CompilationResult(
            algorithm=f"FPAA (L={L})",
            iterations=grover_iterations,
            base_unitaries="G(alpha_j,beta_j)",
            continuous_angles="Yes",
            theoretical_depth="O(n)",
            t_gate_count=f_t,
            overhead_multiplier=mult,
            passband_stability=f"Theoretical >= 1 - delta^2 = {1.0 - delta*delta:.6f}",
        ),
    ]


def _save_resource_table(rows: List[CompilationResult], output_csv: str) -> None:
    """Persist Module-6 results table for manuscript insertion."""
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Algorithm",
                "Iterations",
                "Base Unitaries",
                "Continuous Angles?",
                "Theoretical Depth",
                "Fault-Tolerant T-Gate Count",
                "Overhead Multiplier",
                "Passband Stability",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.algorithm,
                    r.iterations,
                    r.base_unitaries,
                    r.continuous_angles,
                    r.theoretical_depth,
                    r.t_gate_count,
                    f"{r.overhead_multiplier:.3f}",
                    r.passband_stability,
                ]
            )



# -----------------------------------------------------------------------------
# CLI orchestration
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="FPAA Section III architectural trade-off study")
    parser.add_argument("--L", type=int, default=3, help="FPAA sequence length")
    parser.add_argument("--delta", type=float, default=0.1, help="Failure bound parameter")
    parser.add_argument("--qubits", type=int, default=4, help="Qubit count for circuit benchmarks")
    parser.add_argument("--good", type=str, default=None, help="H_Good basis-state bitstring, e.g. 1111")
    parser.add_argument("--out-prefix", type=str, default="fpaa", help="Output artifact prefix")
    parser.add_argument(
        "--synthesis-eps",
        type=float,
        default=1e-3,
        help="Per-rotation synthesis precision for FTQC T-count estimation fallback",
    )
    parser.add_argument("--run-all", action="store_true", help="Run all six modules and emit artifacts")
    args = parser.parse_args(argv)

    # Core rigor checks for Modules 1 and 2.
    test_table_i_exact_match(delta=0.1)
    print("Phase schedule test (L=3 analytical Table-I check): PASSED")
    test_fpaa_closed_form_match(L=args.L, delta=args.delta)
    print("Closed-form Chebyshev map test: PASSED")

    try:
        test_grover_fallback_rigor(num_qubits=4, good_indices=(5,), L=3)
        print("Generalized iterate test (Grover fallback at alpha=beta=pi): PASSED")
    except RuntimeError as exc:
        print(f"Generalized iterate test skipped: {exc}")

    if not args.run_all:
        a, b = generate_fpaa_phases(args.L, args.delta)
        print("alpha:", np.array2string(a, precision=10))
        print("beta :", np.array2string(b, precision=10))
        print(f"passband edge w = {passband_edge(args.L, args.delta):.10f}")
        return

    # Module 3: passband plateau proof.
    passband_png = f"{args.out_prefix}_passband.png"
    pass_curve = plot_passband_plateau(L=args.L, delta=args.delta, output=passband_png)
    print(f"Saved: {passband_png}")
    print(
        "Passband rigor:"
        f" w={float(pass_curve['passband_edge'][0]):.6f},"
        f" target={float(pass_curve['target_floor'][0]):.6f},"
        f" min(FPAA|p>=w)={float(pass_curve['min_passband'][0]):.6f},"
        f" violation={float(pass_curve['max_violation'][0]):.2e}"
    )

    # Module 5: NISQ robustness benchmark.
    noise_L = max(args.L, 5)
    noise_png = f"{args.out_prefix}_noise.png"
    noise = plot_nisq_robustness_benchmark(L=noise_L, delta=args.delta, output=noise_png)
    print(f"Saved: {noise_png}")
    print(f"NISQ robustness diagnostics (L={noise_L}):")
    for eps, c in sorted(noise.items(), key=lambda kv: kv[0]):
        print(
            f"  eps={100*eps:>4.0f}% | min(FPAA|p>=w)={float(c['min_passband'][0]):.6f}"
            f" | violation={float(c['max_violation'][0]):.2e}"
        )

    # Module 4: recursive nesting proof.
    nesting_png = f"{args.out_prefix}_nesting.png"
    nesting = plot_recursive_nesting_proof(L1=3, L2=3, delta=args.delta, output=nesting_png)
    print(f"Saved: {nesting_png}")
    print(f"Nesting overlap max |nested-native|: {float(nesting['max_abs_diff'][0]):.3e}")
    print(f"Chebyshev identity max error T3(T3(x))-T9(x): {nesting_composition_error(3, 3):.3e}")

    # Module 6: FTQC Clifford+T overhead table.
    try:
        rows = benchmark_t_gate_blowup(
            num_qubits=args.qubits,
            L=args.L,
            delta=args.delta,
            good_state=args.good,
            synthesis_eps=args.synthesis_eps,
        )
        table_csv = f"{args.out_prefix}_resource_overhead.csv"
        _save_resource_table(rows, table_csv)
        print(f"Saved: {table_csv}")
        for r in rows:
            print(
                f"{r.algorithm:16s} | iters={r.iterations:2d} | depth={r.theoretical_depth:4s} "
                f"| T-count={r.t_gate_count:8d} | x{r.overhead_multiplier:9.3f} | {r.passband_stability}"
            )
    except RuntimeError as exc:
        print(f"Module 6 skipped: {exc}")


if __name__ == "__main__":
    start_one_click_session(__file__, figure_prefix="fpaa")
    if len(sys.argv) == 1:
        script_stem = Path(__file__).stem
        main(["--run-all", "--out-prefix", script_stem])
    else:
        main()
