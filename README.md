# AmplitudeAmplification

This repository is under active development. It serves as a centralized collection of implementations, mathematical derivations, and research notes focused on quantum amplitude amplification and its modern evolution into the Quantum Singular Value Transformation (QSVT) framework.

Historically, quantum search was viewed through the lens of geometric rotations (Grover's). We are documenting the transition from those early "algorithmic zoology" days to a unified algebraic approach where quantum circuits are treated as polynomial transformations of singular values.

---

## Roadmap and Contents

### I. Foundations & Grover’s Algorithm

* **The Basics:** Introduction to the oracle model and uniform superposition.
* **Geometric Interpretation:** Understanding the Grover iteration as a rotation in a two-dimensional invariant subspace.
* **Query Complexity:** Establishing the $\mathcal{O}(\sqrt{N})$ bound.

### II. Fixed-Point Amplitude Amplification

Standard Grover's algorithm suffers from the "overshooting" problem—if you run the iterations too long, your success probability drops. We implement fixed-point methods to ensure a monotonic rise in success probability.

* **Generalized Grover Iterate:** Moving beyond simple $\pi$ phase shifts.
* **Chebyshev Mapping:** Using the fixed-point property to guarantee success above a specific signal threshold.
* **Phase Schedules:** Recursive nesting and recursive phase tables for $L=3$.

### III. Oblivious Amplitude Amplification (OAA)

Amplifying unitaries when the initial state is unknown or part of a larger system.

* **The Geometric Engine:** Reflections ($R_{bad}$, $R_{initial}$) and the Invariant Subspace Theorem.
* **Roadblocks:** Addressing the No-Cloning theorem and the reflection obstruction.
* **The Purified Setting:** Bypassing obstructions via ancilla padding and state purification.
* **The Grand Connection:** Establishing the equivalence between OAA and Block Encodings.

### IV. Distributed Quantum Amplitude Amplification

Research into scaling search algorithms for NISQ-era distributed architectures.

* **Operator Tensorization:** Breaking down the global search problem into local sub-functions.
* **The "Lucky Node" Guarantee:** Mathematical proof (Theorem 3) regarding node-level success in a distributed network.
* **Distributed Fixed-Point Iteration:** Parallel architecture execution and synchronized phase schedules.

### V. Variable-Time Amplitude Amplification (VTAA)

Optimizing search when the computational cost of checking a "marked" state varies.

* **Staged Amplification:** Paying an average stopping-time cost rather than the worst-case.
* **Applications:** Linear systems with adaptive precision.

### VI. Quantum Singular Value Transformation (QSVT)

The ultimate unification of quantum algorithms. We treat the entire quantum circuit as a way to apply a polynomial transformation to the singular values of a block-encoded matrix.

* **From Zoology to Algebra:** How QSVT unifies Grover, Phase Estimation, and Hamiltonian Simulation.
* **Quantum Signal Processing (QSP):** Polynomial synthesis in $SU(2)$ and the derivation of phase sequences.
* **Block-Encoding:** Treating data as a sub-block of a larger unitary matrix.
* **Universal Compiler:** Implementing matrix inversion (HHL), Markov chain searches, and fast QMA amplification.

### VII. Advanced Applications & Fundamental Limits

* **Principal Component Regression:** Quantum-accelerated machine learning via unified QSVT.
* **Optimal Hamiltonian Simulation:** Reaching the fundamental limits of simulation via Jacobi-Anger expansions.
* **Complexity Bounds:** Using the Markov Brothers’ Inequality to link polynomial degree to query complexity.

---

## Current Status

* **Code:** Preliminary Python/Qiskit implementations are available for Grover and Fixed-Point schedules.
* **Theory:** LaTeX notes for Sections I through IV are complete. We are currently drafting the "Unification" sections regarding QSVT.
* **Next Steps:** We are working on a classical phase synthesis tool to generate the phase sequences $(\phi_d)$ required for arbitrary polynomial transformations.
