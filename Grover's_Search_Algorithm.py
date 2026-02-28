import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram


from qiskit_aer import AerSimulator


# We solve an unstructured search problem over all 3-bit strings:
# {000, 001, 010, 011, 100, 101, 110, 111}
#
# Exactly one string x* satisfies a Boolean predicate f(x)=1.
# For all other strings f(x)=0.
#
# Here we hardcode x* = "101", so the oracle encodes:
# f(x)=1 if x == "101", else 0.
#
# Goal: find the marked string using one Grover iteration.
# One iteration is used here as a hardware-aware depth choice: it keeps
# the circuit shorter on noisy devices while still amplifying the target
# probability above the uniform baseline of 1/8.


def apply_phase_oracle_mark_101(qc: QuantumCircuit, data, ancilla):
    # Oracle construction for marked bitstring "101":
    # 1) Flip data[1] because the middle target bit is 0.
    # 2) Apply a multi-controlled Z phase flip on the all-ones condition.
    # 3) Undo the temporary X.

    qc.x(data[1])

    # Multi-controlled Z implemented as a multi-controlled phase (angle pi).
    # Ancilla is fixed in |1>, so this marks only data state |111> in the
    # transformed basis, which corresponds to original |101>.
    qc.mcp(np.pi, [data[0], data[1], data[2]], ancilla)

    qc.x(data[1])


def apply_diffusion_operator_3q(qc: QuantumCircuit, data):
    # Diffusion operator on the 3 data qubits.
    # This grows the marked amplitude and suppresses non-marked amplitudes.

    qc.h(data)
    qc.x(data)

    # Phase flip of |111> in this transformed basis.
    qc.h(data[2])
    qc.ccx(data[0], data[1], data[2])
    qc.h(data[2])

    qc.x(data)
    qc.h(data)


# 3 data qubits and 1 ancilla qubit.
data = QuantumRegister(3, "data")
anc = QuantumRegister(1, "anc")
creg = ClassicalRegister(3, "c")
qc = QuantumCircuit(data, anc, creg)

# Uniform superposition over all 8 candidates in the search space.
qc.h(data)

# Keep ancilla in |1> for oracle phase marking.
qc.x(anc[0])

# Exactly one Grover iteration.
apply_phase_oracle_mark_101(qc, data, anc[0])
apply_diffusion_operator_3q(qc, data)

# Measure only the data register, which holds the search answer.
qc.measure(data, creg)


# Run on Aer simulator.
backend = AerSimulator()
compiled = transpile(qc, backend, optimization_level=1)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Report whether marked state is above uniform baseline (1/8).
uniform_probability = 1 / 8
marked_state = "101"
marked_probability = counts.get(marked_state, 0) / shots

print("Counts:", counts)
print(f"P({marked_state}) = {marked_probability:.4f}")
print(f"Uniform baseline = {uniform_probability:.4f}")
print("Marked state amplified:", marked_probability > uniform_probability)

# Display histogram.
plot_histogram(counts, title="Grover Search (3-bit, 1 iteration, target=101)")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# IBM Quantum hardware execution 
# -----------------------------------------------------------------------------
# from qiskit_ibm_runtime import QiskitRuntimeService
#
# service = QiskitRuntimeService(channel="ibm_quantum")
# backend_hw = service.least_busy(operational=True, simulator=False, min_num_qubits=4)
#
# qc_hw = transpile(qc, backend_hw, optimization_level=1)
# hw_job = backend_hw.run(qc_hw, shots=4096)
# hw_result = hw_job.result()
# hw_counts = hw_result.get_counts()
#
# print("Hardware backend:", backend_hw.name)
# print("Hardware counts:", hw_counts)
# plot_histogram(hw_counts, title=f"Grover result on {backend_hw.name}")
# plt.tight_layout()
# plt.show()
