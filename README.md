
# AmplitudeAmplification

This repository is under active development. It serves as a centralized collection of implementations, mathematical derivations, and research notes focused on quantum amplitude amplification and its modern evolution into the Quantum Singular Value Transformation (QSVT) framework.

Historically, quantum search was viewed through the lens of geometric rotations (Grover's). We are documenting the transition from those early "algorithmic zoology" days to a unified algebraic approach where quantum circuits are treated as polynomial transformations of singular values.

---

## Roadmap and Contents

### I. Foundations & Grover’s Algorithm (GSA)

This section implements the `GroverGeometricLab`, a rigorous numerical environment designed to study 2D geometry, instability, and the physical limits of quantum search. Rather than just a functional search tool, this implementation serves as a diagnostic suite for the "Soufflé Problem" and invariant subspace dynamics.

#### Mathematical Implementation

The core engine models the algorithm as a precise rotation in a 2D invariant subspace spanned by the "good" (target) and "bad" (non-target) state vectors.

* **Success Probability:** Defined as $P_{\text{success}} = \sin^2\!\left(\frac{(2k+1)\,\theta}{2}\right)$.
* **Geometric Angle:** Derived from solution density $\lambda = M/N$ as $\theta = 2\,\arcsin\!\left(\sqrt{\lambda}\right)$.
* **Invariant Subspace Theorem:** The implementation includes a projection auditor that verifies $|a_k|^2 + |b_k|^2 = 1.0$ across all iterations, confirming the state never leaves the 2D plane within double-precision limits ($~10^{-15}$).

#### Diagnostic Modules:

* **The Soufflé Problem (Module 1 & 5):** * Tracks the "Soufflé Crash"—the point ($k \approx 2k^*$) where success probability collapses back to near-zero.
* Generates a **Sensitivity Heatmap** that maps success rates across a continuous grid of solution densities ($\lambda$) and iteration counts ($k$), highlighting the extreme precision required for GSA.


* **The No-Cloning Barrier (Module 3):** * Provides an empirical proof of the No-Cloning Theorem.
* Simulates a naive CNOT "copy" attempt and measures the resulting **purity collapse** ($Tr(\rho^2)$), demonstrating how system-environment entanglement destroys the quantum state's coherence.


* **Recursive Nesting & Scaling (Module 4):** * Analyzes the "sharpening effect" of nested Amplitude Amplification.
* Includes a **Gate Depth Auditor** that compares the exponential circuit depth of nested iterations against standard GSA for equivalent probability boosts.


* **Qiskit 3-Qubit Demo:** * A practical hardware-ready circuit targeting the $|101\rangle$ state.
* Implements a multi-controlled phase oracle and a standard diffusion operator (inversion about the mean).



#### Visual Evidence Artifacts:

The lab generates `grover_geometric_evidence.png`, which provides six panel-proofs:

1. **Success vs. Iteration:** Visualizing the theoretical curve against simulated probability points.
2. **Subspace Verification:** A flat line at $P=1.0$ confirming rotation purity.
3. **Unit Circle Path:** The trajectory of amplitudes $a_k$ and $b_k$ through the invariant plane.
4. **Self-Similar Scaling:** Comparison of Level-1 and Level-2 success curves.
5. **Purity Bar Chart:** Contrasting original state purity with the collapse following a naive copy.
6. **Sensitivity Heatmap:** Identifying "Soufflé Islands" of high success in the parameter space.
---

### II. Fixed-Point Amplitude Amplification (FPAA)

Standard Grover’s algorithm suffers from the "soufflé problem"—if iterations continue past the optimal point $k^*$, the success probability collapses. This section implements a Fixed-Point Amplitude Amplification (FPAA) suite designed to achieve monotonic convergence and robustness against over-rotation.

#### Mathematical Implementation

The FPAA engine replaces the standard $\pi$-pulse reflections with a **Generalized Grover Iterate** characterized by length-$L$ phase sequences $\{\phi, \varphi\}$.

* **Chebyshev Mapping:** The implementation uses a recursive phase schedule to map the initial signal $\lambda$ onto a high-degree Chebyshev polynomial plateau, ensuring success probability remains high for all $\lambda > \delta$.
* **Monotonicity Guarantee:** Unlike the oscillatory behavior of standard AA, the FPAA module verifies a "passband" where success probability stays near-unity regardless of additional iterations.

#### Architectural Trade-off Modules:

* **Phase Schedule Synthesizer (Module 1):** Generates optimized phase sequences and performs analytical checks against Table-I benchmarks.
* **Generalized Grover Circuit (Module 2):** Implements the $\phi$-controlled reflection operator using multi-controlled phase (MCP) gates.
* **Passband Plateau Analysis (Module 3):** Audits the SU(2) subspace dynamics to visualize the transition from oscillatory growth to a stable success plateau.
* **Recursive Nesting Proof (Module 4):** Provides a numerical and circuit-level demonstration of the identity $T_{L_1}(T_{L_2}(x)) = T_{L_1 \cdot L_2}(x)$, proving the self-similar scaling of nested AA.
* **NISQ Robustness Benchmark (Module 5):** Evaluates the algorithm's sensitivity to analog phase-noise, quantifying the "violation" of the passband under varying hardware error rates.
* **FTQC Resource Auditor (Module 6):** Calculates the Clifford+T compilation overhead, specifically tracking the **T-gate blowup** required for high-precision phase synthesis in fault-tolerant architectures.

#### Evidence Artifacts:

* **`fpaa_nesting.png`**: Visual proof of the Chebyshev composition identity and the sharpening effect of nested schedules.
* **`fpaa_resource_overhead.csv`**: A comparative data table mapping target precision ($\epsilon$) to the required T-count and circuit depth.
* **Passband Visualizations**: Detailed plots of the success probability $P(\lambda)$ showing the engineered plateau above the threshold $\delta$.
---

### III. Oblivious Amplitude Amplification (OAA)

This section implements the `ObliviousAmplificationLab`, a diagnostic suite for amplifying unitary operators when the initial state is unknown or part of a larger system. Unlike standard Grover search, OAA focuses on the "purified setting" to bypass fundamental obstructions like the No-Cloning theorem.

#### Mathematical Implementation

The OAA engine treats the problem through a **Block-Encoding** lens, where a target operator $H/\alpha$ is embedded into the top-left block of a larger unitary $U$.

* **The Geometric Engine:** Implements reflections $R_{bad}$ and $R_{initial}$ to perform rotations within the invariant subspace.
* **Subnormalization Rescue:** Tracks the success probability and fidelity as the signal $1/\alpha$ decays, implementing "rescue" iterations to restore the target block's amplitude.
* **Invariant Subspace Theorem:** Verification that the oblivious iteration correctly preserves the trajectory within the 2D plane despite the lack of state knowledge.

#### Diagnostic Modules:

* **Purified-Setting Audit:** * Demonstrates how to bypass the reflection obstruction using ancilla padding.
* Compares "valid" vs "violated" settings to show how improper state initialization leads to fidelity collapse.


* **Explicit LCU (Linear Combination of Unitaries) Audit:** * Constructs block-encodings using the LCU framework and audits the "distance to identity" for the resulting operators.
* Quantifies off-diagonal leakage and verifies the normalization factors ($\alpha$) required for strictly unitary dilation.


* **Fidelity Curve Analysis:** * Measures the Mean Squared Deviation (MSD) between the target operation and the physically realized circuit output.
* Quantifies the stability of the amplified block across multiple iteration depths $k$.



#### Evidence Artifacts:

* **`purified_audit`**: A comparative data set showing the success probability of the purified vs unpurified oblivious settings.
* **`lcu_audit`**: Detailed matrix-norm analysis (`||M_TL - target||_F`) evaluating the accuracy of the block-encoded LCU target.
* **Geometric Obstruction Data**: Numerical proofs showing the specific points where naive oblivious amplification fails without ancilla-based purification.

---
### IV. Distributed Quantum Amplitude Amplification

This section implements the `DistributedAmplificationLab`, focusing on the architectural challenges of scaling search algorithms across multi-node NISQ-era networks. The implementation provides the first empirical validation of the **Lucky-Node Guarantee** (Theorem 3) and the impact of oracle partitioning on local search densities.

#### Mathematical Implementation

The engine validates the convexity guarantee required for distributed search: $\max_k a_k \ge a$, where $a$ is the global marked-state fraction and $a_k$ is the local fraction within node $k$.

* **Operator Partitioning:** The implementation partitions a $2^n$ search space into $2^j$ nodes using prefix-qubit slicing.
* **Convexity Audit:** Numerically proves that at least one node in the distributed network is "lucky," possessing a solution density higher than the global average.
* **Classical Network Statistics:** Models the effects of shot noise and sifting ($\sigma$) on the ability to resolve local success probabilities across a noisy network.

#### Diagnostic Modules:

* **Lucky-Node Verification (Module 1):** Performs a full sweep of the partitioned search space to identify nodes where $a_k > a$, confirming the existence of high-density sub-functions.
* **AST-Level Oracle Partitioning:** Analyzes the logical structure of the global oracle to ensure partitioning preserves the "marked" state properties across node boundaries.
* **Network Shot-Noise Benchmark:** Simulates the variance in success-rate estimation as a function of `shots_per_node`, quantifying the risk of "false negatives" in distributed search.
* **Sifting & Signal Processing:** Evaluates the efficiency of sifting algorithms in filtering low-probability nodes to focus global amplification resources.

#### Evidence Artifacts:

* **`network_shot_noise.png`**: A visualization of local marked-state fractions ($a_k$) across all nodes, featuring a horizontal threshold at the global average ($a$) to highlight the "lucky nodes."
* **`compiler_table_csv`**: A detailed breakdown of the partitioning costs and local vs. global query complexities.
* **Convexity Heatmap**: Data showing how increasing the number of prefix qubits ($j$) affects the variance and maximum available local signal strength.
---

### V. Variable-Time Amplitude Amplification (VTAA)

This section implements the `VariableTimeAmplitudeAmplificationLab`, designed to optimize quantum search when the computational cost of checking "marked" states is non-uniform across the algorithmic branches. Rather than paying the worst-case halting time, VTAA leverages branch-level statistics to achieve an average-case speedup.

#### Mathematical Implementation

VTAA models a variable-time algorithm $A$ through a set of computational branches, each characterized by a halting time $t_i$, a branch weight $w_i$, and a conditional success probability $s_i$.

* **State Composition:** The state following algorithm $A$ is represented as $|\psi\rangle = \sum_i \sqrt{w_i} (\sqrt{s_i}|i,good\rangle + \sqrt{1-s_i}|i,bad\rangle)$, where the total success probability is $p = \sum_i w_i s_i$.
* **Asymptotic Efficiency:** The implementation provides parameterized estimates for the Ambainis VTAA theorem, which achieves optimal scaling by balancing the costs of different halting-time branches.
* **Grover Dynamics:** Standard AA dynamics are applied to the aggregate success probability $p$ using the SU(2) formula $p_k = \sin^2((2k+1) \arcsin(\sqrt{p}))$.

#### Diagnostic Modules:

* **Branch-Level Statistical Auditor:** Verifies exact identities for halting times, weights, and conditional probabilities to ensure a rigorous simulation of the variable-time environment.
* **Asymptotic Scaling Benchmark:** Explicitly compares the cost ratios of standard restart strategies, worst-case AA, and VTAA to demonstrate the theoretical speedup.
* **FTQC & NISQ Scaling:** Evaluates resource scaling (such as T-gate counts) and "Phase Leakage" for variable-time circuits in both fault-tolerant and open-system (noise-limited) regimes.
* **Subspace & Trajectory Audits:** Tracks the state evolution through the 2D invariant subspace and visualizes the "Phase Staircase" to identify precise peaks for stopping rules.

#### Evidence Artifacts:

* **`vtaa_subspace_audit.png`**: Visual verification that the variable-time iterations preserve the state trajectory within the 2D plane.
* **`vtaa_phase_staircase.png`**: A mapping of success probability evolution that identifies optimal amplification points for varied branch costs.
* **`vtaa_ftqc_scaling.png`**: Data visualization of computational resource growth in fault-tolerant architectures.
* **`vtaa_cost_ratio_report`**: A numerical summary reporting the standard-to-VTAA cost ratio (average vs. worst case).

### VI. Quantum Singular Value Transformation (QSVT)

This section implements the `QSVT Unification Laboratory`, the definitive "algebraic engine" of the repository. It treats quantum algorithms not as a collection of specialized circuits, but as a unified framework for transforming the singular values of an operator via polynomial synthesis.

#### Mathematical Implementation

The laboratory is built on a high-precision `SU2QSPEngine` that performs forward-model Quantum Signal Processing (QSP) to map phase sequences $\{\phi_d\}$ to functional transformations $P(x)$.

* **Polynomial Extraction:** Explicitly evaluates the generalized QSP sequence (Eq. 78) and extracts the resulting polynomial (Eq. 80).
* **Structural Audits:** Includes automated checkers for unitarity/boundedness ($|P(x)| \le 1$) and parity constraints (Eq. 82) to ensure mathematical validity before hardware execution.
* **Block-Encoding Synthesizer:** Provides tools to embed non-unitary operators $A$ into unitary dilations $U$ using subnormalization factors $\alpha$.

#### Diagnostic Modules:

* **Algorithmic Unification (Phase II):**
* **Hamiltonian Simulation:** Synthesizes even/odd polynomials via Jacobi-Anger expansion for optimal-time evolution.
* **Matrix Inversion (HHL 2.0):** Implements gapped-inverse transformations, demonstrating superior spatial scaling over standard QPE-based methods.


* **Operator Calculus (Phase III):**
* **LCU Algebra:** Audits the closure of block-encodings under addition and multiplication, tracking the growth of subnormalization ($\alpha$).
* **Uniform Singular Value Amplification (USVA):** Implements "signal rescue" dynamics to counter exponential amplitude decay in deep circuits.


* **Fundamental Limits & Realism (Phase IV):**
* **Markov/Bernstein Boundary:** Numerically saturates extremal derivative limits to prove query-complexity ceilings.
* **Hardware Fragility:** Quantifies the impact of analog phase-drift and out-of-domain leakage on polynomial fidelity.



#### Adversarial Edge Cases (Phase V):

The laboratory includes a suite of "stress tests" to identify where the QSVT framework breaks down:

* **Gibbs Catastrophe:** Visualizes the fatal unitarity violations caused by fitting discontinuous targets (like raw sign functions) without smoothing.
* **Non-Normal Trap:** Demonstrates the divergence between eigenvalue functions and the physical singular-value channel of QSVT for non-normal matrices.
* **Phase Quantization Limit:** Simulates the functional distortion caused by finite-bit DAC resolution in control electronics.

#### Evidence Artifacts:

The laboratory generates comprehensive visualizations including:

* **Convergence Audits:** Tracking approximation error $\|P_d - f\|_\infty$ across increasing degrees.
* **Resource Scaling Plots:** Comparing QSVT ancilla requirements against legacy algorithms like QPE.
* **Spectral Defect Maps:** Highlighting non-PSD matrices resulting from "cheating" the subnormalization alpha.

---

Current Status

    Code: The implementation has progressed from preliminary scripts to a suite of polished, high-fidelity diagnostic modules. Each module now includes rigorous debugging protocols, such as machine-precision invariant subspace audits and automated unitarity checkers.

    Theory: LaTeX documentation for Sections I through XII is effectively complete, with the empirical code now providing direct "proof-of-work" for the theoretical claims regarding Grover dynamics, FPAA plateaus, and QSVT unification.

    Refinement & Polishing: Recent updates have added hardware-realism layers, including NISQ phase-noise sensitivity benchmarks, FTQC Clifford+T resource auditing, and finite-bit DAC quantization simulations.

    Next Steps: With the integration of the SU2QSPEngine, the classical phase synthesis tool is now functional. Our remaining focus is on final structural de-duplication of the survey text and the generation of publication-quality visual artifacts for the "Adversarial QSVT" case studies.
---
