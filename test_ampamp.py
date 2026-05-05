import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

# Core Engines
from ampamp.grover import GroverEngine
from ampamp.fixed_point import FixedPointEngine
from ampamp.oblivious import ObliviousEngine
from ampamp.foqa import FOQAEngine
from ampamp.distributed import DQAAEngine, OracleSynthesizer
from ampamp.variable_time import VTAAEngine, VariableTimeBranch
from ampamp.qsvt import SU2QSPEngine, QSVTSynthesizer

# Auditors
from ampamp.diagnostics import (
    GroverAuditor,
    FPAAAuditor,
    ObliviousAuditor,
    FOQAAuditor,
    DistributedAuditor,
    VTAAAuditor,
    FundamentalLimitsAuditor,
    QSVTAuditor
)

def main():
    print("Testing the ampamp Quantum Library 🚀\n")

    # --- 1. Standard Grover Search ---
    print("--- 1. Standard Grover Search ---")
    n_qubits = 4
    N = 2**n_qubits
    marked_states = [3, 11]
    engine = GroverEngine(n_qubits, marked_states)
    k_opt = engine.k_optimal
    print(f"Optimal Grover iterations: {k_opt}\n")
    
    # --- 2. Fixed-Point Amplitude Amplification ---
    print("--- 2. Fixed-Point Amplitude Amplification ---")
    fp_engine = FixedPointEngine(L=5, delta=0.1)
    fp_auditor = FPAAAuditor(fp_engine)
    print(f"FPAA created with Chebyshev degree {fp_engine.L}")
    print(f"Palindromic zetas: {np.round(fp_engine.zetas, 3)}")
    print(f"Grover phase pairs alpha/beta: {np.round(fp_engine.alphas, 3)} / {np.round(fp_engine.betas, 3)}\n")

    # --- 3. Oblivious Amplitude Amplification ---
    print("--- 3. Oblivious Amplitude Amplification ---")
    ob_engine = ObliviousEngine(m_data_qubits=2, l_ancilla_qubits=1, p=0.6)
    ob_auditor = ObliviousAuditor(ob_engine)
    print(f"ObliviousEngine initialized (m={ob_engine.m}, p={ob_engine.p})\n")

    # --- 4. FOQA ---
    print("--- 4. Fixed-Point Oblivious Amplitude Amplification (FOQA) ---")
    foqa_engine = FOQAEngine(theta=0.5)
    foqa_auditor = FOQAAuditor(foqa_engine)
    results_foqa = foqa_auditor.audit_damping_regimes(iterations=10, mizel_c=1.4)
    print(f"FOQA Critical Schedule max success (10 steps): {np.max(results_foqa['critical']):.3f}\n")

    # --- 5. Distributed QAA ---
    print("--- 5. Distributed QAA ---")
    dqaa_engine = DQAAEngine(global_n=6, j_prefixes=2)
    dqaa_auditor = DistributedAuditor(dqaa_engine)
    partitions = dqaa_engine.partition_targets(["010101", "110011"])
    print(f"DQAA Partitions (prefixes=2): {partitions}\n")

    # --- 6. Variable Time AA ---
    print("--- 6. Variable Time Amplitude Amplification ---")
    branches = [
        VariableTimeBranch(1.0, 0.4, 0.8),
        VariableTimeBranch(2.0, 0.6, 0.9)
    ]
    vtaa_engine = VTAAEngine(branches)
    vtaa_auditor = VTAAAuditor(vtaa_engine)
    t_mean, t_rms, t_max = vtaa_engine.stopping_time_moments()
    bound = vtaa_engine.vtaa_asymptotic_bound()
    print(f"VTAA Moments: Mean={t_mean:.2f}, RMS={t_rms:.2f}, Max={t_max:.2f}")
    print(f"VTAA Asymptotic Bound: {bound:.2f}\n")

    # --- 7. QSVT ---
    print("--- 7. Quantum Singular Value Transformation (QSVT) ---")
    qsvt_engine = SU2QSPEngine()
    qsvt_auditor = QSVTAuditor(qsvt_engine)
    even, odd, alpha = QSVTSynthesizer.synthesize_jacobi_anger(degree=5, time=1.0)
    print(f"QSVT Jacobi-Anger LCU norm bound: {alpha:.3f}\n")

    # --- 8. Diagnostics Suite & Subspace Verification ---
    print("--- 8. Diagnostics & Subspace Verification ---")
    auditor = GroverAuditor(engine)
    results = auditor.verify_subspace_rotation(max_k=3)
    print("Tracking Grover Invariant Subspace Purity (|a|^2 + |b|^2 = 1.0):")
    for k, purity in enumerate(results['purity']):
        print(f"  Step {k}: Purity = {purity:.4f} | Success = {results['success'][k]:.4f}")

    print("\n--- 9. Fundamental Limits Auditor ---")
    fl_auditor = FundamentalLimitsAuditor()
    print("FundamentalLimitsAuditor initialized successfully.")

if __name__ == "__main__":
    main()
