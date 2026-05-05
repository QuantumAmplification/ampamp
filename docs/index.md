# Amplitude Amplification

An open source quantum machine learning and simulation framework that accelerates the path from theoretical algorithms to practical, executable quantum circuit synthesis. 

Amplitude Amplification (`ampamp`) provides a robust, production-grade API for researchers and engineers building next-generation quantum solutions. Whether you are generating optimal polynomial sequences via Quantum Singular Value Transformation (QSVT), mapping out oblivious operator expansions, or deploying partitioned distributed quantum searches, `ampamp` ensures exact phase tracking and high-fidelity algorithmic representation.

---

## Ecosystem At A Glance

<div class="grid cards" markdown>

-   **Foundations**

    ---
    Standard Grover's Search algorithms. Contains precise analytical definitions of the Soufflé Problem, diffusion operators, and exact success probability tracking.
    
    [API Reference](api/grover.md)

-   **Oracle Construction**

    ---
    General phase and bit-flip oracle construction from marked indices, marked bitstrings, and Boolean formulae over variables such as `v0`, `v1`, and `v2`.

    [API Reference](api/oracles.md)

-   **Entanglement Count**

    ---
    Light or hard active-entangled-qubit counting for Qiskit circuits, with sampled checkpoints for constrained hardware and every-step tracing for stronger hardware.

    [API Reference](api/entanglement.md)

-   **Fixed-Point Search (FPAA)**

    ---
    Achieve monotonic algorithmic convergence. Leverages monotonic Chebyshev-derived phase schedules to safely amplify target states without precise knowledge of target counts.
    
    [API Reference](api/fixed_point.md)

-   **QSVT & SU(2) Calculus**

    ---
    A complete engine for Quantum Singular Value Transformations via $SU(2)$ homomorphisms. Extract SVD mappings natively on quantum operators with mathematically verified parity models.
    
    [API Reference](api/qsvt.md)

-   **Distributed AA**

    ---
    Scalable search space partitioning mechanisms engineered for classical-quantum distributed processing clusters and NISQ-era parallelization networks.
    
    [API Reference](api/distributed.md)

-   **Oblivious Amplification (OAA)**

    ---
    Robust block-encoding arrays and operator amplification capabilities for Oblivious algorithmic structures utilizing pristine Linear Combination of Unitaries (LCU) architectures.

    [API Reference](api/oblivious.md)

-   **Variable Time (VTAA)**

    ---
    Variable-Time Amplitude Amplification branch systems mapping algorithmic costs to multi-staged state circuits ensuring asymptotic $O(\sqrt{E[t^2]})$ expected runtime scaling.
    
    [API Reference](api/variable_time.md)

-   **FPOA**

    ---
    Fixed-Point Oblivious Amplitude Amplification defining discrete $SU(2)$ non-linear recurrence paths to mitigate over-rotation faults in Hamiltonian simulations.
    
    [API Reference](api/foqa.md)

-   **Algorithm Diagnostics**

    ---
    An extensive suite of independent hardware realism auditors. Verify block unitarity constraints, map phase damping trajectories, limit-test empirical bounding box ranks, and identify algorithmic fidelity collapse.
    
    [API Reference](api/diagnostics.md)

</div>

---

## Quick Start

Get started quickly by bringing `ampamp` into your Python environment locally.

```bash
pip install -e .
```

Instantiate precision-tuned engines capable of scaling to highly complex quantum states:

```python
from ampamp import FPAAAuditor, FixedPointEngine

# Establish a 15-iteration strictly monotonic Chebyshev schedule
engine = FixedPointEngine(L=15, delta=1e-3)

# Access the classical analytical hardware limits
auditor = FPAAAuditor(engine)
auditor.estimate_ftqc_cost(synthesis_epsilon=1e-4)

# Generate the synthesized circuit structure to inject directly into Qiskit primitives
circuit = engine.build_fixed_point_circuit(num_qubits=6, marked_indices=[2, 7])
```
