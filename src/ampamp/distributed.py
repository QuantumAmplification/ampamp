"""Distributed Quantum Amplitude Amplification module.

Provides classes and utilities for partitioning global amplitude amplification
problems into local sub-problems suitable for distributed quantum computing.
"""

import numpy as np
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracleGate

class DQAAEngine:
    """Core engine for Distributed Quantum Amplitude Amplification.

    Handles prefix/suffix partitioning and local circuit generation.
    """
    def __init__(self, global_n: int, j_prefixes: int):
        """Initializes the DQAA Engine.

        Args:
            global_n (int): The total number of qubits in the global system.
            j_prefixes (int): The number of prefix bits used for partitioning.

        Raises:
            ValueError: If j_prefixes is not strictly between 0 and global_n.
        """
        if j_prefixes >= global_n or j_prefixes <= 0:
            raise ValueError("Prefix count j must be strictly between 0 and global_n.")
        self.global_n = global_n
        self.j = j_prefixes
        self.local_n = global_n - j_prefixes

    def partition_targets(self, global_targets: list[str]) -> dict[str, list[str]]:
        """Maps global n-bit targets into node-local suffix targets.

        Args:
            global_targets (list[str]): A list of global target bitstrings.

        Returns:
            dict[str, list[str]]: A dictionary mapping prefix bitstrings to lists 
                of suffix target bitstrings for each node.
        """
        num_nodes = 2 ** self.j
        partitions = {format(k, f"0{self.j}b"): [] for k in range(num_nodes)}
        
        for bitstring in global_targets:
            prefix, suffix = bitstring[:self.j], bitstring[self.j:]
            partitions[prefix].append(suffix)
        return partitions

    def build_node_circuit(self, alphas: np.ndarray, betas: np.ndarray, local_targets: list[str]) -> QuantumCircuit:
        """Constructs the local FPAA circuit for a specific node.

        Args:
            alphas (np.ndarray): Array of phase shifts for the local diffusion operator.
            betas (np.ndarray): Array of phase shifts for the local oracle operator.
            local_targets (list[str]): List of suffix target bitstrings for this processing node.

        Returns:
            QuantumCircuit: The constructed local quantum circuit.
        """
        qc = QuantumCircuit(self.local_n)
        qc.h(range(self.local_n))
        
        for alpha, beta in zip(alphas, betas):
            qc.global_phase += np.pi
            # (Assuming you move your _build_local_oracle/diffusion helpers here)
            # qc.append(self._build_local_oracle(local_targets, beta), range(self.local_n))
            # qc.append(self._build_local_diffusion(alpha), range(self.local_n))
            
        return qc

class OracleSynthesizer:
    """AST-level oracle partitioning compiler.

    Uses SymPy to simplify global boolean formulas into node-local sub-oracles.
    """
    def __init__(self, global_n: int, j: int, formula_text: str):
        """Initializes the AST-level oracle partitioning compiler.

        Args:
            global_n (int): The total number of qubits in the global system.
            j (int): The number of prefix bits used for partitioning.
            formula_text (str): The global boolean formula to be synthesized.
        """
        self.global_n = global_n
        self.j = j
        self.local_n = global_n - j
        self.formula_expr = sp.sympify(formula_text, evaluate=False)
        self.suffix_vars = [f"v{i}" for i in range(j, global_n)]

    def compile_node_formula(self, prefix: str) -> QuantumCircuit:
        """Injects the node prefix into the AST, simplifies, and builds a PhaseOracle.

        Args:
            prefix (str): The binary prefix string representing the current node.

        Returns:
            QuantumCircuit: The synthesized local phase oracle circuit.
        """
        symbol_lookup = {sym.name: sym for sym in self.formula_expr.free_symbols}
        
        # Substitute the known prefix bits into the global formula
        subs = {
            symbol_lookup.get(f"v{i}", sp.Symbol(f"v{i}")): (prefix[i] == "1")
            for i in range(self.j)
        }
        
        simplified = sp.simplify_logic(self.formula_expr.subs(subs), force=True)
        
        # Convert back to a Qiskit Phase Oracle
        if simplified in (sp.true, sp.false):
            qc = QuantumCircuit(self.local_n)
            if simplified == sp.true:
                qc.global_phase += np.pi
            return qc
            
        oracle_gate = PhaseOracleGate(str(simplified), var_order=self.suffix_vars)
        qc = QuantumCircuit(self.local_n)
        qc.append(oracle_gate, range(self.local_n))
        return qc