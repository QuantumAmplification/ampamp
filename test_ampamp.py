import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

# Importing your newly pip-installed library!
from ampamp.foundations import GroverEngine
from ampamp.fixed_point import FixedPointEngine
from ampamp.diagnostics import GroverAuditor

def main():
    print("🚀 Testing the ampamp Quantum Library 🚀\n")

    # --- 1. Standard Grover Search ---
    print("--- 1. Standard Grover Search ---")
    n_qubits = 4
    N = 2**n_qubits
    marked_states = [3, 11] # Finding integers 3 and 11
    
    print(f"Search space: {N} items. Marked items: {marked_states}")
    engine = GroverEngine(n_qubits, marked_states)
    
    k_opt = engine.k_optimal
    print(f"Optimal iterations calculated by engine: {k_opt}")
    
    # Build and measure the circuit
    grover_circ = engine.construct_circuit(k_opt)
    grover_circ.measure_all()
    
    # Execute on a local simulator
    backend = AerSimulator()
    job = backend.run(transpile(grover_circ, backend), shots=1024)
    counts = job.result().get_counts()
    
    print("Measurement Results (Top 2 expected to be 0011 and 1011):")
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    for state, count in sorted_counts[:2]:
        print(f"  State |{state}> (Int: {int(state, 2)}): {count} shots")
    print()

    # --- 2. Fixed-Point Amplitude Amplification ---
    print("--- 2. Fixed-Point Amplitude Amplification ---")
    L = 5 # Sequence length (must be odd)
    delta = 0.1 # Monotonic error bound
    
    fp_engine = FixedPointEngine(L=L, delta=delta)
    print(f"Generated Chebyshev Phase Schedule (L={L}, delta={delta}):")
    print(f"  Alphas: {np.round(fp_engine.alphas, 3)}")
    print(f"  Betas:  {np.round(fp_engine.betas, 3)}")
    print()

    # --- 3. Diagnostics Suite ---
    print("--- 3. Diagnostics & Subspace Verification ---")
    auditor = GroverAuditor(engine)
    results = auditor.verify_subspace_rotation(max_k=3)
    
    print("Tracking Invariant Subspace Purity (|a|^2 + |b|^2 = 1.0):")
    for k, purity in enumerate(results['purity']):
        print(f"  Step {k}: Purity = {purity:.4f} | Success Prob = {results['success'][k]:.4f}")

if __name__ == "__main__":
    main()
    