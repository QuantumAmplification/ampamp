
# AmplitudeAmplification

This repository is under active development. It serves as a centralized collection of implementations, mathematical derivations, and research notes focused on quantum amplitude amplification and its modern evolution into the Quantum Singular Value Transformation (QSVT) framework.

Historically, quantum search was viewed through the lens of geometric rotations (Grover's). We are documenting the transition from those early "algorithmic zoology" days to a unified algebraic approach where quantum circuits are treated as polynomial transformations of singular values.

---

## Roadmap and Contents

### I. Foundations & Grover’s Algorithm (GSA)

This section implements the core logic of the standard Grover Search Algorithm. Our implementation focuses on the **Success Probability Dynamics** and the **Geometric Interpretation** of the state vector's rotation.

#### Mathematical Implementation
The Python core (`grover_success_prob`) models the algorithm as a rotation in a 2D invariant subspace. The probability of success after $k$ iterations is defined by:

$$P_{\text{success}} = \sin^2\left(\frac{(2k + 1)\theta}{2}\right)$$

where the angle of rotation $\theta$ is derived from the initial overlap (signal strength) $\lambda = M/N$:

$$\theta = 2 \arcsin(\sqrt{\lambda})$$


#### Code Insights:
* **Optimal Iteration Calculator:** The function `optimal_grover_iterations` prevents the "overshooting" problem by calculating the integer $k^*$ that brings the state closest to the target Hilbert space: 
  $$k^* = \left\lfloor \frac{\pi}{2\theta} - \frac{1}{2} \right\rfloor$$
* **Case Studies:** The implementation evaluates three distinct regimes:
    1. **$M \ll N$:** The classic "needle in a haystack" where $k \approx \frac{\pi}{4}\sqrt{N/M}$.
    2. **$M = N/2$:** A critical point where the algorithm reaches maximum success in a single step (or zero steps).
    3. **$M > N/2$:** Observations on high-signal environments where standard amplification may actually decrease success probability.
* **Qiskit 3-Qubit Demo:** A practical circuit implementation using a phase oracle targeting the $|101\rangle$ state and a standard diffusion operator (inversion about the mean).

---

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

### IV. Distributed Quantum Amplitude Amplification
Research into scaling search algorithms for NISQ-era distributed architectures.
* **Operator Tensorization:** Breaking down the global search problem into local sub-functions.
* **The "Lucky Node" Guarantee:** Mathematical proof (Theorem 3) regarding node-level success in a distributed network.

### V. Variable-Time Amplitude Amplification (VTAA)
Optimizing search when the computational cost of checking a "marked" state varies.
* **Staged Amplification:** Paying an average stopping-time cost rather than the worst-case.

### VI. Quantum Singular Value Transformation (QSVT)
The ultimate unification of quantum algorithms. 
* **From Zoology to Algebra:** How QSVT unifies Grover, Phase Estimation, and Hamiltonian Simulation.
* **Quantum Signal Processing (QSP):** Polynomial synthesis in $SU(2)$ and the derivation of phase sequences.
* **Block-Encoding:** Treating data as a sub-block of a larger unitary matrix.

---

## Current Status

* **Code:** Preliminary Python/Qiskit implementations are available for Grover (GSA) and Fixed-Point schedules.
* **Theory:** LaTeX notes for Sections I through IV are complete. 
* **Next Steps:** We are working on a classical phase synthesis tool to generate the phase sequences ($\phi_d$) required for arbitrary polynomial transformations.

---
