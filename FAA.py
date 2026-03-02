"""FPAA architectural trade-off study.

This module implements a full six-part workflow for fixed-point amplitude
amplification (FPAA):
1) phase schedule synthesis
2) generalized Grover iterate circuit construction
3) passband plateau analysis
4) recursive nesting demonstration
5) phase-noise sensitivity
6) fault-tolerant compilation overhead (T-count proxy)

The implementation supports:
- subspace statevector simulation (fast lambda sweep)
- Qiskit circuit construction for hardware-aware analyses
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Gate
    from qiskit.quantum_info import Statevector
except Exception:  # pragma: no cover - allows import without qiskit installed
    QuantumCircuit = None  # type: ignore[assignment]
    transpile = None  # type: ignore[assignment]
    Gate = object  # type: ignore[misc,assignment]
    Statevector = None  # type: ignore[assignment]


def chebyshev_t(order: float, x: float) -> float:
    """Chebyshev T_order(x), extended to fractional order for |x| >= 1."""
    x = float(x)
    order = float(order)
    if abs(x) <= 1.0 and float(order).is_integer():
        return float(np.cos(order * np.arccos(x)))
    if x >= 1.0:
        return float(np.cosh(order * np.arccosh(x)))
    if x <= -1.0 and float(order).is_integer():
        # Integer-order extension on (-inf, -1].
        n = int(round(order))
        return float(((-1) ** n) * np.cosh(n * np.arccosh(-x)))
    raise ValueError("Fractional order with |x| < 1 is not supported in this helper.")


def passband_edge(L: int, delta: float) -> float:
    """w = 1 - T_{1/L}(1/delta)^(-2)."""
    gamma_inv = chebyshev_t(1.0 / L, 1.0 / delta)
    return 1.0 - gamma_inv ** (-2)


def generate_fpaa_phases(L: int, delta: float, epsilon: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return FPAA phase arrays (alpha, beta) with stable trigonometric handling.

    This follows the analytical compiler-style synthesis:
    - gamma_inv = cosh((1/L) * arccosh(1/delta))
    - alpha_j = 2 * arccot(tan(2*pi*j/L) * sqrt(1-gamma^2)), j=1..L
    - beta_j = -alpha_{L-j+1}

    epsilon injects systematic over-rotation: angle -> angle * (1 + epsilon).
    """
    if L < 1:
        raise ValueError("L must be >= 1.")
    if L % 2 == 0:
        raise ValueError("L must be odd for standard FPAA phase scheduling.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")

    gamma_inv = float(np.cosh((1.0 / L) * np.arccosh(1.0 / delta)))
    gamma = 1.0 / gamma_inv
    sq_term = float(np.sqrt(max(0.0, 1.0 - gamma * gamma)))

    # Tight tolerance for handling tan singular/zero crossings robustly.
    tol = 1e-12
    alpha = np.zeros(L, dtype=float)
    for j in range(1, L + 1):
        theta_j = (2.0 * np.pi * j) / L
        sin_theta = float(np.sin(theta_j))
        tan_val = 0.0 if np.isclose(sin_theta, 0.0, atol=tol) else float(np.tan(theta_j))
        denominator = tan_val * sq_term
        alpha_j = np.pi if np.isclose(denominator, 0.0, atol=tol) else 2.0 * float(np.arctan2(1.0, denominator))
        alpha[j - 1] = alpha_j

    beta = np.zeros(L, dtype=float)
    for j in range(1, L + 1):
        beta[j - 1] = -alpha[L - j]

    if epsilon != 0.0:
        alpha = alpha * (1.0 + epsilon)
        beta = beta * (1.0 + epsilon)
    return alpha, beta


def test_table_i_exact_match(delta: float = 0.1, atol: float = 1e-8) -> None:
    """Verify dynamic phase synthesis against analytical L=3 Table-I-style equations."""
    L = 3
    alpha_dyn, beta_dyn = generate_fpaa_phases(L=L, delta=delta)

    gamma_inv = float(np.cosh((1.0 / L) * np.arccosh(1.0 / delta)))
    gamma = 1.0 / gamma_inv
    sq_term = float(np.sqrt(max(0.0, 1.0 - gamma * gamma)))

    a1_exact = 2.0 * float(np.arctan2(1.0, np.tan(2.0 * np.pi / 3.0) * sq_term))
    a2_exact = 2.0 * float(np.arctan2(1.0, np.tan(4.0 * np.pi / 3.0) * sq_term))
    a3_exact = float(np.pi)  # tan(2*pi)=0 -> arccot(0)=pi/2 -> alpha_3=pi

    alpha_exact = np.array([a1_exact, a2_exact, a3_exact], dtype=float)
    beta_exact = np.array([-a3_exact, -a2_exact, -a1_exact], dtype=float)

    if not np.allclose(alpha_dyn, alpha_exact, atol=atol):
        raise AssertionError(f"Alpha phases failed Table-I check.\nExpected: {alpha_exact}\nGot: {alpha_dyn}")
    if not np.allclose(beta_dyn, beta_exact, atol=atol):
        raise AssertionError(f"Beta phases failed Table-I check.\nExpected: {beta_exact}\nGot: {beta_dyn}")


def _require_qiskit() -> None:
    if QuantumCircuit is None:
        raise RuntimeError("Qiskit is required for circuit-level modules.")


def _validate_marked_indices(num_qubits: int, marked_indices: Sequence[int]) -> List[int]:
    dim = 2**num_qubits
    out: List[int] = []
    for idx in marked_indices:
        i = int(idx)
        if i < 0 or i >= dim:
            raise ValueError(f"Marked index {i} out of range for {num_qubits} qubits.")
        out.append(i)
    if not out:
        raise ValueError("marked_indices must not be empty.")
    return sorted(set(out))


def _marked_state_to_indices(num_qubits: int, marked_state: Optional[str]) -> List[int]:
    """Backwards-compatible conversion from bitstring target to marked index list."""
    if marked_state is None:
        return [2**num_qubits - 1]
    if len(marked_state) != num_qubits or any(ch not in "01" for ch in marked_state):
        raise ValueError("marked_state must be a binary bitstring of length num_qubits.")
    return [int(marked_state, 2)]


def generalized_oracle(num_qubits: int, marked_indices: Sequence[int], beta: float) -> "QuantumCircuit":
    """S_t(beta): apply continuous phase shift beta on each marked basis state."""
    _require_qiskit()
    marked = _validate_marked_indices(num_qubits, marked_indices)
    qc = QuantumCircuit(num_qubits, name=f"S_t({beta:.3f})")
    controls = list(range(num_qubits - 1))
    target = num_qubits - 1

    for idx in marked:
        # little-endian bit layout for qubit index alignment
        binary = format(idx, f"0{num_qubits}b")[::-1]
        for q, bit in enumerate(binary):
            if bit == "0":
                qc.x(q)
        if num_qubits == 1:
            qc.p(beta, 0)
        else:
            qc.mcp(beta, controls, target)
        for q, bit in enumerate(binary):
            if bit == "0":
                qc.x(q)
    return qc


def generalized_diffusion(num_qubits: int, alpha: float) -> "QuantumCircuit":
    """S_s(alpha): H^n X^n phase-on-|1...1> X^n H^n (equiv. reflection about |s>)."""
    _require_qiskit()
    qc = QuantumCircuit(num_qubits, name=f"S_s({alpha:.3f})")
    controls = list(range(num_qubits - 1))
    target = num_qubits - 1

    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.p(alpha, 0)
    else:
        qc.mcp(alpha, controls, target)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_fpaa_circuit_from_phases(
    num_qubits: int,
    marked_indices: Sequence[int],
    alphas: np.ndarray,
    betas: np.ndarray,
    initialize_superposition: bool = True,
) -> "QuantumCircuit":
    """Build U_L = G(alpha_L,beta_L)...G(alpha_1,beta_1), with G=-S_s(alpha)S_t(beta)."""
    _require_qiskit()
    if len(alphas) != len(betas):
        raise ValueError("Alphas and Betas arrays must have same length.")
    marked = _validate_marked_indices(num_qubits, marked_indices)
    qc = QuantumCircuit(num_qubits, name=f"FPAA_L{len(alphas)}")
    if initialize_superposition:
        qc.h(range(num_qubits))

    for j in range(len(alphas)):
        oracle_step = generalized_oracle(num_qubits, marked, float(betas[j]))
        diffusion_step = generalized_diffusion(num_qubits, float(alphas[j]))
        # Encode the explicit minus sign in G(alpha,beta).
        qc.global_phase += np.pi
        qc.append(oracle_step.to_gate(), range(num_qubits))
        qc.append(diffusion_step.to_gate(), range(num_qubits))
    return qc


def _phase_on_basis_state(
    qc: "QuantumCircuit",
    qubits: Sequence[int],
    angle: float,
    bitstring: Optional[str],
) -> None:
    """Apply a controlled phase to a computational basis bitstring."""
    n = len(qubits)
    if n == 0:
        return
    if bitstring is None:
        bitstring = "1" * n
    if len(bitstring) != n:
        raise ValueError("bitstring length must match number of qubits.")

    # Map target basis state -> all-ones control target.
    for i, bit in enumerate(bitstring):
        if bit == "0":
            qc.x(qubits[i])

    if n == 1:
        qc.p(angle, qubits[0])
    else:
        qc.mcp(angle, qubits[:-1], qubits[-1])

    for i, bit in enumerate(bitstring):
        if bit == "0":
            qc.x(qubits[i])


def apply_target_reflection(
    qc: "QuantumCircuit",
    data_qubits: Sequence[int],
    beta_j: float,
    marked_state: Optional[str] = None,
) -> None:
    """S_t(beta_j): phase on marked computational state."""
    _phase_on_basis_state(qc, data_qubits, beta_j, marked_state)


def apply_s0_reflection(qc: "QuantumCircuit", data_qubits: Sequence[int], alpha_j: float) -> None:
    """S_0(alpha_j): phase on |0...0>."""
    _phase_on_basis_state(qc, data_qubits, alpha_j, "0" * len(data_qubits))


def append_generalized_iterate(
    qc: "QuantumCircuit",
    data_qubits: Sequence[int],
    alpha_j: float,
    beta_j: float,
    A_gate: Optional["Gate"] = None,
    A_dagger_gate: Optional["Gate"] = None,
    marked_state: Optional[str] = None,
) -> None:
    """Append one generalized FPAA iterate:
    G(alpha_j, beta_j) = A * S_0(alpha_j) * A^† * S_t(beta_j)
    """
    apply_target_reflection(qc, data_qubits, beta_j, marked_state)
    if A_dagger_gate is not None:
        qc.append(A_dagger_gate, data_qubits)
    apply_s0_reflection(qc, data_qubits, alpha_j)
    if A_gate is not None:
        qc.append(A_gate, data_qubits)


def _hadamard_layer_gate(num_qubits: int) -> "Gate":
    _require_qiskit()
    had = QuantumCircuit(num_qubits, name="A")
    had.h(range(num_qubits))
    return had.to_gate()


def build_fpaa_circuit(
    num_qubits: int,
    L: int,
    delta: float,
    epsilon: float = 0.0,
    marked_state: Optional[str] = None,
    marked_indices: Optional[Sequence[int]] = None,
    A_gate: Optional["Gate"] = None,
) -> "QuantumCircuit":
    """Construct U_L = G(alpha_L,beta_L)...G(alpha_1,beta_1)."""
    _require_qiskit()
    alpha, beta = generate_fpaa_phases(L=L, delta=delta, epsilon=epsilon)
    if marked_indices is None:
        marked_indices = _marked_state_to_indices(num_qubits, marked_state)

    # Standard FPAA path: explicit generalized oracle/diffusion using mcp phase gates.
    if A_gate is None:
        return build_fpaa_circuit_from_phases(
            num_qubits=num_qubits,
            marked_indices=marked_indices,
            alphas=alpha,
            betas=beta,
            initialize_superposition=True,
        )

    # Generic A-gate path retained for recursive-nesting experiments.
    A_dag = A_gate.inverse()

    qc = QuantumCircuit(num_qubits, name=f"FPAA_L{L}")
    qc.append(A_gate, range(num_qubits))
    for j in range(L):
        append_generalized_iterate(
            qc,
            data_qubits=list(range(num_qubits)),
            alpha_j=float(alpha[j]),
            beta_j=float(beta[j]),
            A_gate=A_gate,
            A_dagger_gate=A_dag,
            marked_state=marked_state,
        )
    return qc


def build_standard_grover_circuit(
    num_qubits: int,
    iterations: int,
    marked_state: Optional[str] = None,
    marked_indices: Optional[Sequence[int]] = None,
) -> "QuantumCircuit":
    """Standard Grover with pi-phase reflections."""
    if marked_indices is None:
        marked_indices = _marked_state_to_indices(num_qubits, marked_state)
    alphas = np.full(iterations, math.pi, dtype=float)
    betas = np.full(iterations, math.pi, dtype=float)
    return build_fpaa_circuit_from_phases(
        num_qubits=num_qubits,
        marked_indices=marked_indices,
        alphas=alphas,
        betas=betas,
        initialize_superposition=True,
    )


def test_grover_fallback_rigor(
    num_qubits: int = 4,
    marked_indices: Sequence[int] = (5,),
    L: int = 3,
    atol: float = 1e-8,
) -> None:
    """Rigorous check: generalized iterates reduce to standard Grover when alpha=beta=pi."""
    _require_qiskit()
    if Statevector is None:
        raise RuntimeError("qiskit.quantum_info.Statevector is required for fallback testing.")

    marked = _validate_marked_indices(num_qubits, marked_indices)
    alphas = np.full(L, np.pi, dtype=float)
    betas = np.full(L, np.pi, dtype=float)
    qc = build_fpaa_circuit_from_phases(num_qubits, marked, alphas, betas, initialize_superposition=True)
    state = Statevector.from_instruction(qc)

    success_prob = float(sum(np.abs(state.data[idx]) ** 2 for idx in marked))
    lam = len(marked) / (2**num_qubits)
    theta = 2.0 * np.arcsin(np.sqrt(lam))
    theoretical_prob = float(np.sin((2 * L + 1) * theta / 2.0) ** 2)

    if not np.isclose(success_prob, theoretical_prob, atol=atol):
        raise AssertionError(
            f"Grover fallback failed. Sim={success_prob:.12f}, Theory={theoretical_prob:.12f}"
        )


def _reflection_about_state(state: np.ndarray, angle: float) -> np.ndarray:
    """I + (exp(i*angle)-1)|psi><psi|."""
    ket = state.reshape((-1, 1))
    return np.eye(len(state), dtype=complex) + (np.exp(1j * angle) - 1.0) * (ket @ ket.conj().T)


def simulate_2d_fpaa_sequence(lam: float, alphas: np.ndarray, betas: np.ndarray) -> float:
    """SU(2) subspace simulation in basis {|Bad>, |Good>} for continuous lambda."""
    if len(alphas) != len(betas):
        raise ValueError("alphas and betas must have the same length.")
    lam = float(lam)
    if lam <= 0.0:
        return 0.0
    lam = min(lam, 1.0)

    # |s> = sqrt(1-lam)|Bad> + sqrt(lam)|Good>
    s = np.array([math.sqrt(1.0 - lam), math.sqrt(lam)], dtype=complex)
    state = s.copy()
    I = np.eye(2, dtype=complex)
    Pi_t = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)  # |Good><Good|
    Pi_s = np.outer(s, s.conj())

    for a_j, b_j in zip(alphas, betas):
        S_t = I - (1.0 - np.exp(1j * float(b_j))) * Pi_t
        S_s = I - (1.0 - np.exp(1j * float(a_j))) * Pi_s
        G_j = -(S_s @ S_t)
        state = G_j @ state

    return float(np.abs(state[1]) ** 2)


def _two_level_success_grover(lambda_: float, iterations: int) -> float:
    lam = float(lambda_)
    if lam <= 0.0:
        return 0.0
    lam = min(lam, 1.0)
    theta = 2.0 * np.arcsin(np.sqrt(lam))
    return float(np.sin((2 * iterations + 1) * theta / 2.0) ** 2)


def sweep_passband(
    L: int,
    delta: float,
    epsilon: float = 0.0,
    lambda_min: float = 1e-3,
    lambda_max: float = 1.0,
    num_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Compute continuous-lambda passband curves via SU(2) subspace simulation."""
    lambdas = np.linspace(lambda_min, lambda_max, num_points)
    alpha, beta = generate_fpaa_phases(L=L, delta=delta, epsilon=epsilon)
    fpaa = np.array([simulate_2d_fpaa_sequence(lam, alpha, beta) for lam in lambdas])
    grover = np.array([_two_level_success_grover(lam, L) for lam in lambdas])
    edge = passband_edge(L, delta)
    target_floor = 1.0 - delta * delta
    mask = lambdas >= edge
    min_passband = float(np.min(fpaa[mask])) if np.any(mask) else float("nan")
    max_violation = float(max(0.0, target_floor - min_passband)) if np.isfinite(min_passband) else float("nan")
    return {
        "lambda": lambdas,
        "fpaa": fpaa,
        "grover": grover,
        "passband_edge": np.array([edge]),
        "target_floor": np.array([target_floor]),
        "min_passband": np.array([min_passband]),
        "max_violation": np.array([max_violation]),
    }


def noise_sensitivity_sweep(
    L: int,
    delta: float,
    epsilons: Iterable[float] = (0.0, 0.01, 0.05, 0.10),
    lambda_min: float = 1e-3,
    lambda_max: float = 1.0,
    num_points: int = 1000,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Module 5: passband degradation under systematic phase over-rotation."""
    alphas, betas = generate_fpaa_phases(L=L, delta=delta, epsilon=0.0)
    lambdas = np.linspace(lambda_min, lambda_max, num_points)
    edge = passband_edge(L, delta)
    target_floor = 1.0 - delta * delta
    passband_mask = lambdas >= edge

    out: Dict[float, Dict[str, np.ndarray]] = {}
    for eps in epsilons:
        noisy = np.array(
            [simulate_noisy_fpaa_sequence(lam, alphas, betas, float(eps)) for lam in lambdas],
            dtype=float,
        )
        min_passband = float(np.min(noisy[passband_mask])) if np.any(passband_mask) else float("nan")
        max_violation = (
            float(max(0.0, target_floor - min_passband)) if np.isfinite(min_passband) else float("nan")
        )
        out[float(eps)] = {
            "lambda": lambdas,
            "fpaa": noisy,
            "passband_edge": np.array([edge]),
            "target_floor": np.array([target_floor]),
            "min_passband": np.array([min_passband]),
            "max_violation": np.array([max_violation]),
        }
    return out


def simulate_noisy_fpaa_sequence(lam: float, alphas: np.ndarray, betas: np.ndarray, epsilon: float) -> float:
    """Systematic over-rotation model: phase -> phase * (1 + epsilon)."""
    noisy_alphas = np.asarray(alphas, dtype=float) * (1.0 + float(epsilon))
    noisy_betas = np.asarray(betas, dtype=float) * (1.0 + float(epsilon))
    return simulate_2d_fpaa_sequence(float(lam), noisy_alphas, noisy_betas)


def nesting_composition_error(
    L1: int = 3, L2: int = 3, samples: int = 1000, x_min: float = 1.0, x_max: float = 5.0
) -> float:
    """Module 4: numerical check of T_{L2}(T_{L1}(x)) = T_{L1*L2}(x)."""
    xs = np.linspace(x_min, x_max, samples)
    lhs = np.array([chebyshev_t(L2, chebyshev_t(L1, x)) for x in xs])
    rhs = np.array([chebyshev_t(L1 * L2, x) for x in xs])
    return float(np.max(np.abs(lhs - rhs)))


def build_nested_fpaa_circuit(
    num_qubits: int,
    L1: int,
    L2: int,
    delta: float,
    marked_state: Optional[str] = None,
    marked_indices: Optional[Sequence[int]] = None,
) -> "QuantumCircuit":
    """Construct nested FPAA circuit by reflecting about the L1-prepared state.

    Level 2 iterate uses:
    - S_t(beta2_j) as usual on marked states
    - U_L1^† S_0(alpha2_j) U_L1 for source reflection
    """
    _require_qiskit()
    if marked_indices is None:
        marked_indices = _marked_state_to_indices(num_qubits, marked_state)
    marked = _validate_marked_indices(num_qubits, marked_indices)

    # Level-1 schedule and unitary gate
    a1, b1 = generate_fpaa_phases(L1, delta)
    ql1 = build_fpaa_circuit_from_phases(
        num_qubits=num_qubits,
        marked_indices=marked,
        alphas=a1,
        betas=b1,
        initialize_superposition=True,
    )
    gate_l1 = ql1.to_gate(label=f"U_L{L1}")
    gate_l1_dag = gate_l1.inverse()
    gate_l1_dag.label = f"U_L{L1}_dagger"

    # Level-2 schedule
    a2, b2 = generate_fpaa_phases(L2, delta)
    qc = QuantumCircuit(num_qubits, name=f"Nested_{L1}x{L2}")
    qc.append(gate_l1, range(num_qubits))
    for alpha_j, beta_j in zip(a2, b2):
        # G(alpha,beta) = -S_s(alpha)S_t(beta)
        qc.global_phase += np.pi
        qc.append(generalized_oracle(num_qubits, marked, float(beta_j)).to_gate(), range(num_qubits))
        qc.append(gate_l1_dag, range(num_qubits))
        qc.append(apply_s0_reflection_as_gate(num_qubits, float(alpha_j)), range(num_qubits))
        qc.append(gate_l1, range(num_qubits))
    return qc


def apply_s0_reflection_as_gate(num_qubits: int, alpha: float) -> "Gate":
    """S_0(alpha): phase on |0...0>, as a reusable gate."""
    _require_qiskit()
    q = QuantumCircuit(num_qubits, name=f"S0({alpha:.3f})")
    q.x(range(num_qubits))
    if num_qubits == 1:
        q.p(alpha, 0)
    else:
        q.mcp(alpha, list(range(num_qubits - 1)), num_qubits - 1)
    q.x(range(num_qubits))
    return q.to_gate()


def recursive_nesting_curves(
    L1: int = 3,
    L2: int = 3,
    delta: float = 0.1,
    lambda_min: float = 1e-4,
    lambda_max: float = 0.4,
    num_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Numerical proof curves for nested (L1xL2) vs native (L1*L2)."""
    a1, b1 = generate_fpaa_phases(L1, delta)
    a_comp, b_comp = generate_fpaa_phases(L1 * L2, delta)
    lambdas = np.linspace(lambda_min, lambda_max, num_points)

    p_base = np.array([simulate_2d_fpaa_sequence(lam, a1, b1) for lam in lambdas])
    p_nested = np.array([simulate_2d_fpaa_sequence(p, a1, b1) for p in p_base])
    p_native = np.array([simulate_2d_fpaa_sequence(lam, a_comp, b_comp) for lam in lambdas])
    diff = np.abs(p_nested - p_native)

    return {
        "lambda": lambdas,
        "base_l1": p_base,
        "nested": p_nested,
        "native": p_native,
        "abs_diff": diff,
        "max_abs_diff": np.array([float(np.max(diff))]),
        "w_l1": np.array([passband_edge(L1, delta)]),
        "w_comp": np.array([passband_edge(L1 * L2, delta)]),
    }


def plot_recursive_nesting_proof(
    L1: int = 3,
    L2: int = 3,
    delta: float = 0.1,
    output: str = "fpaa_nesting.png",
    lambda_min: float = 1e-4,
    lambda_max: float = 0.4,
    num_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Module 4 journal plot: overlay nested and native composed sequences."""
    import matplotlib.pyplot as plt

    curves = recursive_nesting_curves(
        L1=L1,
        L2=L2,
        delta=delta,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        num_points=num_points,
    )
    lam = curves["lambda"]
    p_base = curves["base_l1"]
    p_nested = curves["nested"]
    p_native = curves["native"]
    w_l1 = float(curves["w_l1"][0])
    w_comp = float(curves["w_comp"][0])
    max_diff = float(curves["max_abs_diff"][0])

    plt.figure(figsize=(10, 6))
    plt.plot(lam, p_base, label=f"Base L={L1}", color="gray", linestyle=":", linewidth=2.0)
    plt.plot(lam, p_native, label=f"Native L={L1*L2}", color="blue", linewidth=3.2, alpha=0.55)
    plt.plot(lam, p_nested, label=f"Nested {L1}x{L2}", color="red", linestyle="--", linewidth=2.0)
    plt.axvline(w_l1, color="gray", alpha=0.6, linestyle=":")
    plt.axvline(w_comp, color="blue", alpha=0.6, linestyle=":")
    plt.title(f"Recursive Nesting Proof: T_{L2}(T_{L1}(x)) = T_{L1*L2}(x)\nmax |nested-native| = {max_diff:.2e}")
    plt.xlabel("Initial Success Probability (lambda)")
    plt.ylabel("Final Success Probability")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return curves


def _is_clifford_phase(theta: float, tol: float = 1e-10) -> bool:
    """True if phase is multiple of pi/2 (up to global sign)."""
    k = round(theta / (math.pi / 2.0))
    return abs(theta - k * (math.pi / 2.0)) < tol


def _estimate_t_per_rotation(eps: float) -> int:
    """Ross-Selinger style rough estimate for single-qubit Z rotation synthesis."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    return max(0, int(math.ceil(3.21 * math.log2(1.0 / eps) - 6.93)))


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


def _t_count_for_pi_over_4_multiple(theta: float, tol: float = 1e-10) -> Optional[int]:
    """Exact T-count for RZ(k*pi/4) up to global phase; None if not such a multiple."""
    k = int(round(theta / (math.pi / 4.0)))
    if abs(theta - k * (math.pi / 4.0)) > tol:
        return None
    # Odd multiples require one T (plus Clifford); even multiples are Clifford-only.
    return 1 if (k % 2) != 0 else 0


def _estimate_t_count_from_native(qc: "QuantumCircuit", synthesis_eps: float) -> int:
    """Estimate T-count from native decomposition by charging non-Clifford RZ rotations."""
    t_per = _estimate_t_per_rotation(synthesis_eps)
    t_count = 0
    for inst, qargs, cargs in qc.data:
        if inst.name == "rz":
            theta = float(inst.params[0])
            if _is_clifford_phase(theta):
                continue
            exact_quarter = _t_count_for_pi_over_4_multiple(theta)
            if exact_quarter is not None:
                t_count += exact_quarter
            else:
                t_count += t_per
        elif inst.name == "t":
            t_count += 1
    return int(t_count)


def benchmark_t_gate_blowup(
    num_qubits: int,
    L: int,
    delta: float,
    marked_state: Optional[str] = None,
    marked_indices: Optional[Sequence[int]] = None,
    synthesis_eps: float = 1e-3,
    optimization_level: int = 3,
) -> List[CompilationResult]:
    """Module 6 benchmark: equal-length Grover vs FPAA in Clifford+T cost."""
    _require_qiskit()
    if transpile is None:
        raise RuntimeError("Qiskit transpiler is unavailable.")
    if marked_indices is None:
        marked_indices = _marked_state_to_indices(num_qubits, marked_state)
    marked = _validate_marked_indices(num_qubits, marked_indices)

    grover = build_standard_grover_circuit(
        num_qubits=num_qubits,
        iterations=L,
        marked_state=marked_state,
        marked_indices=marked,
    )
    fpaa = build_fpaa_circuit(
        num_qubits=num_qubits,
        L=L,
        delta=delta,
        epsilon=0.0,
        marked_state=marked_state,
        marked_indices=marked,
    )

    def direct_t_count(qc: "QuantumCircuit") -> Optional[int]:
        try:
            tqc = transpile(
                qc,
                basis_gates=["cx", "h", "s", "t"],
                optimization_level=optimization_level,
            )
            return int(tqc.count_ops().get("t", 0))
        except Exception:
            return None

    grover_t = direct_t_count(grover)
    fpaa_t = direct_t_count(fpaa)

    # Fallback: transpile to native basis and estimate T from RZ synthesis precision.
    if grover_t is None or fpaa_t is None:
        grover_native = transpile(
            grover, basis_gates=["rz", "sx", "x", "cx"], optimization_level=optimization_level
        )
        fpaa_native = transpile(
            fpaa, basis_gates=["rz", "sx", "x", "cx"], optimization_level=optimization_level
        )

        if grover_t is None:
            grover_t = _estimate_t_count_from_native(grover_native, synthesis_eps)
        if fpaa_t is None:
            fpaa_t = _estimate_t_count_from_native(fpaa_native, synthesis_eps)

    grover_t = int(grover_t)
    fpaa_t = int(fpaa_t)
    multiplier = float(fpaa_t / max(1, grover_t))

    return [
        CompilationResult(
            algorithm="Standard Grover",
            iterations=L,
            base_unitaries="D(pi)O(pi)",
            continuous_angles="No",
            theoretical_depth="O(n)",
            t_gate_count=grover_t,
            overhead_multiplier=1.0,
            passband_stability="Unstable (Souffle)",
        ),
        CompilationResult(
            algorithm=f"FPAA (L={L})",
            iterations=L,
            base_unitaries="G(alpha_j,beta_j)",
            continuous_angles="Yes",
            theoretical_depth="O(n)",
            t_gate_count=fpaa_t,
            overhead_multiplier=multiplier,
            passband_stability=f"Theoretical >= 1 - delta^2 = {1.0 - delta * delta:.6f}",
        ),
    ]


def fault_tolerant_cost_table(
    num_qubits: int,
    L: int,
    delta: float,
    marked_state: Optional[str] = None,
    synthesis_eps: float = 1e-3,
) -> List[CompilationResult]:
    """Backwards-compatible wrapper for Module 6."""
    return benchmark_t_gate_blowup(
        num_qubits=num_qubits,
        L=L,
        delta=delta,
        marked_state=marked_state,
        synthesis_eps=synthesis_eps,
    )


def _plot_passband(curve: Dict[str, np.ndarray], output: str, title: str) -> None:
    import matplotlib.pyplot as plt

    lam = curve["lambda"]
    fpaa = curve["fpaa"]
    grover = curve["grover"]
    edge = float(curve["passband_edge"][0])
    target_floor = float(curve["target_floor"][0])
    min_passband = float(curve["min_passband"][0])
    max_violation = float(curve["max_violation"][0])

    plt.figure(figsize=(9, 5))
    plt.plot(lam, grover, label="Standard Grover", linewidth=1.6)
    plt.plot(lam, fpaa, label="FPAA", linewidth=1.8)
    plt.axvline(edge, linestyle="--", linewidth=1.2, label=f"Passband edge w={edge:.4f}")
    plt.axhline(target_floor, linestyle=":", linewidth=1.2, label=f"Target floor 1-delta^2={target_floor:.4f}")
    plt.fill_between(
        lam,
        target_floor,
        1.0,
        where=(lam >= edge),
        color="green",
        alpha=0.08,
        label="Guaranteed region (lambda >= w)",
    )
    plt.ylim(0.0, 1.02)
    plt.xlabel("Initial solution density λ")
    plt.ylabel("Success probability")
    plt.title(f"{title}\nmin(FPAA|lambda>=w)={min_passband:.6f}, max violation={max_violation:.2e}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_passband_plateau(
    L: int,
    delta: float,
    output: str = "fpaa_passband.png",
    epsilon: float = 0.0,
    lambda_min: float = 1e-3,
    lambda_max: float = 1.0,
    num_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Module 3: generate and save the primary FPAA passband evidence plot."""
    curve = sweep_passband(
        L=L,
        delta=delta,
        epsilon=epsilon,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        num_points=num_points,
    )
    _plot_passband(curve, output, f"FPAA vs Grover (L={L}, delta={delta})")
    return curve


def _plot_noise(noise_curves: Dict[float, Dict[str, np.ndarray]], output: str, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    colors = {0.0: "blue", 0.01: "green", 0.05: "orange", 0.10: "red"}
    styles = {0.0: "-", 0.01: "--", 0.05: "--", 0.10: "--"}
    for eps, curve in sorted(noise_curves.items(), key=lambda kv: kv[0]):
        lbl = "Ideal (0% Error)" if eps == 0.0 else f"{100*eps:.0f}% Phase Error"
        lw = 3.0 if eps == 0.0 else 2.0
        plt.plot(
            curve["lambda"],
            curve["fpaa"],
            linewidth=lw,
            linestyle=styles.get(float(eps), "--"),
            color=colors.get(float(eps), None),
            label=lbl,
        )
    first = noise_curves[sorted(noise_curves.keys())[0]]
    edge = float(first["passband_edge"][0])
    floor = float(first["target_floor"][0])
    plt.axvline(edge, color="black", linestyle=":", linewidth=1.2, label=f"Ideal Passband Edge w={edge:.4f}")
    plt.axhline(floor, color="gray", linestyle=":", linewidth=1.2, label=f"Target floor 1-delta^2={floor:.4f}")
    plt.ylim(0.0, 1.02)
    plt.xlim(0.0, 1.0)
    plt.xlabel("Initial solution density λ")
    plt.ylabel("Success probability")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_nisq_robustness_benchmark(
    L: int = 5,
    delta: float = 0.1,
    output: str = "fpaa_noise.png",
    epsilons: Iterable[float] = (0.0, 0.01, 0.05, 0.10),
    lambda_min: float = 1e-3,
    lambda_max: float = 1.0,
    num_points: int = 1000,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Generate NISQ phase-noise degradation plot and return numeric diagnostics."""
    curves = noise_sensitivity_sweep(
        L=L,
        delta=delta,
        epsilons=epsilons,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        num_points=num_points,
    )
    _plot_noise(curves, output, f"NISQ Robustness: FPAA Plateau Degradation (L={L}, delta={delta})")
    return curves


def _save_table(rows: List[CompilationResult], output: str) -> None:
    import csv

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
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
            writer.writerow(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="FPAA architectural trade-off experiments.")
    parser.add_argument("--L", type=int, default=3, help="FPAA sequence length.")
    parser.add_argument("--delta", type=float, default=0.1, help="Failure bound parameter.")
    parser.add_argument("--qubits", type=int, default=4, help="Qubit count for circuit-level modules.")
    parser.add_argument(
        "--synthesis-eps",
        type=float,
        default=1e-3,
        help="Target per-rotation synthesis precision used in FTQC T-count estimation.",
    )
    parser.add_argument("--marked", type=str, default=None, help="Marked state bitstring (e.g. 1111).")
    parser.add_argument("--out-prefix", type=str, default="fpaa", help="Output prefix for artifacts.")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run passband/noise/nesting checks and emit plots + table.",
    )
    args = parser.parse_args()

    # Module 1 check
    test_table_i_exact_match(delta=0.1)
    print("Phase schedule test (L=3 analytical Table-I check): PASSED")
    try:
        test_grover_fallback_rigor(num_qubits=4, marked_indices=(5,), L=3)
        print("Generalized iterate test (Grover fallback at alpha=beta=pi): PASSED")
    except RuntimeError as exc:
        print(f"Generalized iterate test skipped: {exc}")

    if not args.run_all:
        alpha, beta = generate_fpaa_phases(args.L, args.delta)
        print("alpha:", np.array2string(alpha, precision=10))
        print("beta :", np.array2string(beta, precision=10))
        print(f"passband edge w = {passband_edge(args.L, args.delta):.10f}")
        return

    # Module 3
    curve = plot_passband_plateau(L=args.L, delta=args.delta, output=f"{args.out_prefix}_passband.png")
    passband_png = f"{args.out_prefix}_passband.png"
    print(f"Saved: {passband_png}")
    print(
        "Passband rigor:"
        f" w={float(curve['passband_edge'][0]):.6f},"
        f" target={float(curve['target_floor'][0]):.6f},"
        f" min(FPAA|lambda>=w)={float(curve['min_passband'][0]):.6f},"
        f" violation={float(curve['max_violation'][0]):.2e}"
    )

    # Module 5
    noise_L = max(args.L, 5)
    noise_png = f"{args.out_prefix}_noise.png"
    noise = plot_nisq_robustness_benchmark(L=noise_L, delta=args.delta, output=noise_png)
    print(f"Saved: {noise_png}")
    print(f"NISQ robustness diagnostics (L={noise_L}):")
    for eps, curve_eps in sorted(noise.items(), key=lambda kv: kv[0]):
        print(
            f"  eps={100*eps:>4.0f}% | min(FPAA|lambda>=w)={float(curve_eps['min_passband'][0]):.6f}"
            f" | violation={float(curve_eps['max_violation'][0]):.2e}"
        )

    # Module 4 composition
    nesting_png = f"{args.out_prefix}_nesting.png"
    nesting = plot_recursive_nesting_proof(L1=3, L2=3, delta=args.delta, output=nesting_png)
    print(f"Saved: {nesting_png}")
    print(f"Nesting overlap max |nested-native|: {float(nesting['max_abs_diff'][0]):.3e}")
    comp_err = nesting_composition_error(L1=3, L2=3)
    print(f"Chebyshev identity max error T3(T3(x)) - T9(x): {comp_err:.3e}")

    # Module 6
    try:
        table = fault_tolerant_cost_table(
            num_qubits=args.qubits,
            L=args.L,
            delta=args.delta,
            marked_state=args.marked,
            synthesis_eps=args.synthesis_eps,
        )
        csv_out = f"{args.out_prefix}_resource_overhead.csv"
        _save_table(table, csv_out)
        print(f"Saved: {csv_out}")
        for row in table:
            print(
                f"{row.algorithm:16s} | iters={row.iterations:2d} | depth={row.theoretical_depth:4s} "
                f"| T-count={row.t_gate_count:6d} | x{row.overhead_multiplier:7.3f} | {row.passband_stability}"
            )
    except RuntimeError as exc:
        print(f"Module 6 skipped: {exc}")


if __name__ == "__main__":
    main()
