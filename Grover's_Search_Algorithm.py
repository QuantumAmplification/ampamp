import numpy as np


def grover_success_prob(lambda_value: float, k: int) -> float:
    """Standard Grover success probability in terms of lambda = M/N."""
    if lambda_value < 0 or lambda_value > 1:
        raise ValueError("lambda_value must be in [0, 1]")
    if lambda_value == 0:
        return 0.0

    theta = 2.0 * np.arcsin(np.sqrt(lambda_value))
    angle = (2 * k + 1) * theta / 2.0
    return float(np.sin(angle) ** 2)


def optimal_grover_iterations(lambda_value: float) -> int:
    """Near-optimal integer iteration count for standard Grover search."""
    if lambda_value <= 0:
        return 0

    theta = 2.0 * np.arcsin(np.sqrt(lambda_value))
    # k* ≈ floor(pi/(4*arcsin(sqrt(lambda))) - 1/2)
    k_star = int(np.floor(np.pi / (2.0 * theta) - 0.5))
    return max(0, k_star)


def run_mn_test_cases() -> None:
    """
    Evaluate requested test cases:
    1) M << N
    2) M = N/2
    3) M > N/2
    """
    print("\n" + "=" * 70)
    print("GROVER TEST CASES: M vs N")
    print("=" * 70)

    test_cases = [
        {"label": "Case 1 (M << N)", "N": 1024, "M": 3},
        {"label": "Case 2 (M = N/2)", "N": 1024, "M": 512},
        {"label": "Case 3 (M > N/2)", "N": 1024, "M": 768},
    ]

    for case in test_cases:
        label = case["label"]
        N = case["N"]
        M = case["M"]

        print(f"\n{label}:")
        print(f"  N = {N}, M = {M}")

        if N <= 0:
            print("  Invalid input: N must be positive.")
            continue
        if M < 0:
            print("  Invalid input: M cannot be negative.")
            continue
        if M > N:
            print("  Invalid input: M cannot be greater than N for a search space.")
            print("  Result: test case rejected as physically invalid.")
            continue

        lambda_value = M / N
        k_star = optimal_grover_iterations(lambda_value)
        success_prob = grover_success_prob(lambda_value, k_star)

        print(f"  λ = M/N = {lambda_value:.6f}")
        print(f"  k* (near-optimal Grover iterations) = {k_star}")
        print(f"  Grover success probability at k* = {success_prob:.6f}")


def run_optional_qiskit_demo() -> None:
    """
    Optional 3-qubit circuit demo for marked state '101'.

    This is not required for the M/N test cases. It can fail in some
    sandboxed environments due OpenMP/shared-memory limits in Aer.
    """
    try:
        import matplotlib.pyplot as plt
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
        from qiskit.visualization import plot_histogram
        from qiskit_aer import AerSimulator
    except Exception as exc:
        print("\nQiskit demo skipped (imports unavailable):", exc)
        return

    data = QuantumRegister(3, "data")
    anc = QuantumRegister(1, "anc")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(data, anc, creg)

    def apply_phase_oracle_mark_101(qcircuit: QuantumCircuit):
        qcircuit.x(data[1])
        qcircuit.mcp(np.pi, [data[0], data[1], data[2]], anc[0])
        qcircuit.x(data[1])

    def apply_diffusion_operator_3q(qcircuit: QuantumCircuit):
        qcircuit.h(data)
        qcircuit.x(data)
        qcircuit.h(data[2])
        qcircuit.ccx(data[0], data[1], data[2])
        qcircuit.h(data[2])
        qcircuit.x(data)
        qcircuit.h(data)

    qc.h(data)
    qc.x(anc[0])
    apply_phase_oracle_mark_101(qc)
    apply_diffusion_operator_3q(qc)
    qc.measure(data, creg)

    try:
        backend = AerSimulator()
        compiled = transpile(qc, backend, optimization_level=1)
        shots = 4096
        result = backend.run(compiled, shots=shots).result()
        counts = result.get_counts()

        marked_state = "101"
        marked_probability = counts.get(marked_state, 0) / shots
        print("\nQiskit circuit demo (target=101):")
        print("Counts:", counts)
        print(f"P({marked_state}) = {marked_probability:.4f}")

        plot_histogram(counts, title="Grover Search (3-bit, 1 iteration, target=101)")
        plt.tight_layout()
        plt.savefig("grover_circuit_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Histogram saved: grover_circuit_histogram.png")
    except Exception as exc:
        print("\nQiskit demo skipped (runtime failed):", exc)


def main() -> None:
    run_mn_test_cases()

    # Keep disabled by default for portability in restricted environments.
    run_qiskit_demo = False
    if run_qiskit_demo:
        run_optional_qiskit_demo()


if __name__ == "__main__":
    main()
