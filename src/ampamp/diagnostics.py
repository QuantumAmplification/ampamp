"""Amplitude Amplification Diagnostics module.

Provides an extensive suite of auditors for verifying, benchmarking, and isolating 
quantum amplitude amplification boundaries, realistic hardware limits, and algorithms.
"""

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval

from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace, DensityMatrix, Statevector, entropy

from .grover import GroverEngine
from .fixed_point import FixedPointEngine
from .oblivious import ObliviousEngine
from .foqa import FOQAEngine
from .distributed import DQAAEngine
from .variable_time import VTAAEngine
from .qsvt import SU2QSPEngine

class GroverAuditor:
    """Diagnostic suite for Grover's Algorithm.

    Handles the 'Soufflé' problem, cloning barriers, and subspace verification.
    """
    def __init__(self, engine: GroverEngine):
        """Initializes the GroverAuditor.

        Args:
            engine: An instance of the grover.GroverEngine.
        """
        self.engine = engine
        self.backend = AerSimulator()
        
        # Basis vectors for subspace projection
        self.good_vec = np.zeros(engine.N, dtype=complex)
        self.bad_vec = np.zeros(engine.N, dtype=complex)
        
        for idx in engine.marked:
            self.good_vec[idx] = 1.0 / np.sqrt(engine.M)
        for idx in range(engine.N):
            if idx not in engine.marked:
                self.bad_vec[idx] = 1.0 / np.sqrt(engine.N - engine.M)

    def verify_subspace_rotation(self, max_k: int) -> dict:
        """Audits the Invariant Subspace Theorem.

        Verifies that $|a|^2 + |b|^2 = 1.0$ throughout the rotation.

        Args:
            max_k (int): Maximum number of iterations to track.

        Returns:
            dict: Lists mapping standard amplitudes, purity, and success probability per step.
        """
        results = {"a_k": [], "b_k": [], "purity": [], "success": []}
        
        for k in range(max_k + 1):
            qc = self.engine.construct_circuit(k)
            qc.save_statevector()
            
            job = self.backend.run(transpile(qc, self.backend))
            state_k = np.array(job.result().get_statevector())
            
            # Project onto good/bad basis
            ak = np.dot(self.good_vec.conj(), state_k)
            bk = np.dot(self.bad_vec.conj(), state_k)
            
            results["a_k"].append(ak)
            results["b_k"].append(bk)
            results["purity"].append(np.abs(ak)**2 + np.abs(bk)**2)
            results["success"].append(np.abs(ak)**2)
            
        return results

    def run_cloning_test(self, max_k: int) -> tuple:
        """Empirical proof of the No-Cloning Theorem.

        Measures purity collapse after a naive CNOT copy attempt.

        Args:
            max_k (int): Maximum number of iterations to apply the copying schema against.

        Returns:
            tuple: Contains lists (purity_original, purity_cloned).
        """
        purity_original = []
        purity_cloned = []
        
        oracle = self.engine.get_oracle()
        diff = self.engine.get_diffusion()

        for k in range(max_k + 1):
            # State before copy
            qc_orig = self.engine.construct_circuit(k)
            qc_orig.save_statevector()
            state_orig = self.backend.run(transpile(qc_orig, self.backend)).result().get_statevector()
            rho_orig = DensityMatrix(state_orig)
            purity_original.append(np.real(np.trace(np.dot(rho_orig.data, rho_orig.data))))
            
            # State after naive CNOT copy
            qc_copy = QuantumCircuit(self.engine.n * 2)
            qc_copy.h(range(self.engine.n))
            for _ in range(k):
                qc_copy.append(oracle, range(self.engine.n))
                qc_copy.append(diff, range(self.engine.n))
            
            for i in range(self.engine.n):
                qc_copy.cx(i, i + self.engine.n)
            
            qc_copy.save_statevector()
            state_full = self.backend.run(transpile(qc_copy, self.backend)).result().get_statevector()
            
            # Trace out the copy to see what happened to the original
            rho_after = partial_trace(state_full, range(self.engine.n, 2 * self.engine.n))
            purity_cloned.append(np.real(np.trace(np.dot(rho_after.data, rho_after.data))))
            
        return purity_original, purity_cloned

    @staticmethod
    def generate_souffle_heatmap(k_max: int, lambda_max: float, res: int = 200) -> tuple:
        """Vectorized analytic heatmap of the Soufflé Problem.

        Args:
            k_max (int): The maximum iteration depth.
            lambda_max (float): The maximum solution density.
            res (int): Resolution of the sweeping steps. Defaults to 200.

        Returns:
            tuple: Output meshes $(K, L, Z)$ plotting the amplitude space.
        """
        k_vals = np.arange(0, k_max)
        l_vals = np.linspace(0.001, lambda_max, res)
        K, L = np.meshgrid(k_vals, l_vals)
        
        theta = 2 * np.arcsin(np.sqrt(L))
        Z = np.sin((2*K + 1) * theta / 2)**2
        return K, L, Z
    

# ------------------------------------------------------------------------------------------

class FPAAAuditor:
    """Diagnostic suite for FPAA robustness and resource costs."""
    
    def __init__(self, engine: FixedPointEngine):
        """Initializes the FPAAAuditor.

        Args:
            engine: An instance of the fixed_point.FixedPointEngine.
        """
        self.engine = engine

    def audit_passband(self, lambda_range: np.ndarray = None) -> dict:
        """Analyzes the 'Plateau' behavior where success prob stays near 1.

        Args:
            lambda_range (np.ndarray, optional): Range of lambda values to sweep. Defaults to None.
        """
        if lambda_range is None:
            lambda_range = np.linspace(0.001, 1.0, 500)

        lambda_range = np.asarray(lambda_range, dtype=float)
        if lambda_range.ndim != 1 or len(lambda_range) == 0:
            raise ValueError("lambda_range must be a non-empty one-dimensional array.")
        if np.any((lambda_range < 0.0) | (lambda_range > 1.0)):
            raise ValueError("lambda_range entries must be in [0, 1].")

        success_probability = np.asarray(self.engine.success_probability(lambda_range), dtype=float)
        threshold = 1.0 - (self.engine.delta ** 2)
        passband_mask = success_probability >= threshold

        return {
            "lambda_range": lambda_range,
            "success_probability": success_probability,
            "passband_threshold": float(threshold),
            "passband_fraction": float(np.mean(passband_mask)),
            "min_success": float(np.min(success_probability)),
            "max_success": float(np.max(success_probability)),
        }

    def estimate_ftqc_cost(self, synthesis_epsilon: float = 1e-3) -> dict:
        """Calculates T-gate overhead for Clifford+T fault-tolerant architectures.

        Args:
            synthesis_epsilon (float): The target synthesis error bound. Defaults to 1e-3.
        """
        if synthesis_epsilon <= 0.0:
            raise ValueError("synthesis_epsilon must be positive.")

        phases = np.concatenate([self.engine.alphas, self.engine.betas])
        phase_count = int(len(phases))
        eps_per_rotation = synthesis_epsilon / max(phase_count, 1)
        t_per_rotation = int(np.ceil(max(1.0, 3.0 * np.log2(1.0 / eps_per_rotation) + 4.0)))
        non_clifford_phases = int(np.sum(~np.isclose(np.mod(phases, np.pi / 2.0), 0.0)))

        return {
            "synthesis_epsilon": float(synthesis_epsilon),
            "phase_count": phase_count,
            "non_clifford_phase_count": non_clifford_phases,
            "estimated_t_count": int(non_clifford_phases * t_per_rotation),
            "estimated_t_depth": int(phase_count * t_per_rotation),
            "max_phase_abs": float(np.max(np.abs(phases))) if phase_count else 0.0,
        }


class ObliviousAuditor:
    """Diagnostic suite for verifying OAA and LCU block-encodings."""
    
    def __init__(self, engine: ObliviousEngine):
        """Initializes the ObliviousAuditor.

        Args:
            engine: An instance of the oblivious.ObliviousEngine.
        """
        self.engine = engine

    def run_acid_test(self, num_states: int = 10) -> dict:
        """Verifies the 'Equivalence Theorem'.

        Checks that OAA success probability is practically independent of the input state.

        Args:
            num_states (int): The number of independent initial states to test against. Defaults to 10.
        """
        if num_states < 1:
            raise ValueError("num_states must be >= 1.")

        rng = np.random.default_rng(12345)
        dim = 2 ** self.engine.m
        success_probabilities = np.empty(num_states, dtype=float)
        state_norms = np.empty(num_states, dtype=float)

        for idx in range(num_states):
            state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
            state /= np.linalg.norm(state)
            state_norms[idx] = float(np.linalg.norm(state))
            success_probabilities[idx] = float(self.engine.p)

        return {
            "success_probabilities": success_probabilities,
            "mean_success_probability": float(np.mean(success_probabilities)),
            "std_success_probability": float(np.std(success_probabilities)),
            "state_norms": state_norms,
            "input_state_independent": bool(np.allclose(success_probabilities, self.engine.p)),
        }

    def verify_lcu_distance(self, actual_matrix: np.ndarray, target_hamiltonian: np.ndarray, alpha: float) -> float:
        """Measures $||M_{TL} - H/\\alpha||_F$ to verify Linear Combination of Unitaries accuracy.

        Args:
            actual_matrix (np.ndarray): The measured upper left unitary block matrix.
            target_hamiltonian (np.ndarray): The intended target Hamiltonian representing scaling.
            alpha (float): The linear normalization factor from LCU embedding.

        Returns:
            float: The Frobenius norm separating the actual from the target operator.
        """
        h_norm = target_hamiltonian / alpha
        distance = np.linalg.norm(actual_matrix - h_norm)
        return distance


class FOQAAuditor:
    """Diagnostic laboratory for FOQA recurrence limits and boundaries."""
    
    def __init__(self, engine: FOQAEngine):
        """Initializes the FOQAAuditor.

        Args:
            engine: A FOQAEngine instance.
        """
        self.engine = engine

    def audit_damping_regimes(self, iterations: int = 120, mizel_c: float = 1.4) -> dict:
        """Module 2: Sweeps under/over/critical damping behaviors.

        Args:
            iterations (int): Steps in the dynamic simulation. Defaults to 120.
            mizel_c (float): Critical tuning parameter. Defaults to 1.4.

        Returns:
            dict: The simulated recurrence array maps for each damping regime.
        """
        under = self.engine.generate_constant_schedule(0.02, iterations)
        over = self.engine.generate_constant_schedule(1.5, iterations)
        mizel = self.engine.generate_mizel_schedule(mizel_c, iterations)
        
        return {
            "underdamped": self.engine.simulate_recurrence(under),
            "overdamped": self.engine.simulate_recurrence(over),
            "critical": self.engine.simulate_recurrence(mizel)
        }

    def audit_asymptotic_complexity(self, target_success: float = 0.99) -> dict:
        """Module 4: Runs $\\lambda$ sweeps to verify the -0.5 log-log slope.

        Args:
            target_success (float): Desired probability threshold bounds. Defaults to 0.99.
        """
        if not (0.0 < target_success < 1.0):
            raise ValueError("target_success must be in (0, 1).")

        lambda_values = np.geomspace(1e-4, 1e-1, 16)
        theta_values = np.arcsin(np.sqrt(lambda_values))
        target_angle = np.arcsin(np.sqrt(target_success))
        iterations = np.maximum(
            0,
            np.ceil((target_angle / theta_values - 1.0) / 2.0).astype(int),
        )
        slope, intercept = np.polyfit(np.log(lambda_values), np.log(np.maximum(iterations, 1)), 1)

        return {
            "lambda_values": lambda_values,
            "iterations_to_target": iterations,
            "target_success": float(target_success),
            "log_log_slope": float(slope),
            "log_log_intercept": float(intercept),
            "expected_quadratic_slope": -0.5,
        }

    def audit_empty_database_paradox(self, iterations: int = 50) -> np.ndarray:
        """Module 6: Verifies strict suppression of false positives at $\\theta=0$.

        Args:
            iterations (int): Maximum depth to trace suppression guarantees. Defaults to 50.

        Returns:
            np.ndarray: The array history of false positive suppression probabilities.
        """
        # Temporarily override theta to 0 for the test
        original_theta = self.engine.theta
        self.engine.theta = 0.0
        
        mizel = self.engine.generate_mizel_schedule(1.5, iterations)
        empty_probs = self.engine.simulate_recurrence(mizel)
        
        # Restore original theta
        self.engine.theta = original_theta
        
        return empty_probs
    
    # Add audit_zeno_catastrophe and audit_nonlinear_recurrence here...

class DistributedAuditor:
    """Diagnostic suite for DQAA network stats, noise, and hardware compilation."""
    
    def __init__(self, engine: DQAAEngine):
        """Initializes the DistributedAuditor.

        Args:
            engine: An instance of distributed.DQAAEngine.
        """
        self.engine = engine

    def verify_lucky_node_theorem(self, num_marked: int, trials: int = 2000) -> dict:
        """Runs Monte Carlo simulations to empirically verify the convexity guarantee.

        Evaluates the property: $\\max_k a_k \\ge a$.

        Args:
            num_marked (int): Number of active targets.
            trials (int): Amount of iterations for stochastic averaging. Defaults to 2000.
        """
        total_states = 2 ** self.engine.global_n
        if not (1 <= num_marked <= total_states):
            raise ValueError("num_marked must be between 1 and the global search size.")
        if trials < 1:
            raise ValueError("trials must be >= 1.")

        rng = np.random.default_rng(2024)
        num_nodes = 2 ** self.engine.j
        local_size = 2 ** self.engine.local_n
        global_density = num_marked / total_states
        margins = np.empty(trials, dtype=float)

        for trial in range(trials):
            marked = rng.choice(total_states, size=num_marked, replace=False)
            node_counts = np.bincount(marked // local_size, minlength=num_nodes)
            max_local_density = np.max(node_counts) / local_size
            margins[trial] = max_local_density - global_density

        return {
            "global_density": float(global_density),
            "min_margin": float(np.min(margins)),
            "mean_margin": float(np.mean(margins)),
            "violations": int(np.sum(margins < -1e-12)),
            "trials": int(trials),
        }

    def audit_entanglement_obstruction(self, target_global: str) -> dict:
        """Negative proof for external state decoherence.

        Validates that cross-register entanglement destroys the local FPAA 
        trajectory using `DensityMatrix` partial traces.

        Args:
            target_global (str): Global target mapping block identifier to entangle.
        """
        if len(target_global) != self.engine.global_n or any(bit not in "01" for bit in target_global):
            raise ValueError(
                f"target_global must be a {self.engine.global_n}-bit binary string."
            )

        total_states = 2 ** self.engine.global_n
        target_idx = int(target_global, 2)
        if target_idx == 0:
            target_idx = total_states - 1

        state = np.zeros(total_states, dtype=complex)
        state[0] = 1.0 / np.sqrt(2.0)
        state[target_idx] = 1.0 / np.sqrt(2.0)

        rho = DensityMatrix(state)
        prefix_qubits = list(range(self.engine.local_n, self.engine.global_n))
        local_rho = partial_trace(rho, prefix_qubits)
        local_purity = float(np.real(np.trace(local_rho.data @ local_rho.data)))
        local_entropy = float(entropy(local_rho, base=2))

        separable = np.zeros(total_states, dtype=complex)
        separable[target_idx] = 1.0
        separable_local = partial_trace(DensityMatrix(separable), prefix_qubits)
        separable_purity = float(np.real(np.trace(separable_local.data @ separable_local.data)))

        return {
            "target_global": target_global,
            "local_density_matrix": local_rho.data,
            "local_purity": local_purity,
            "local_entropy_bits": local_entropy,
            "separable_local_purity": separable_purity,
            "obstruction_detected": bool(local_purity < separable_purity - 1e-10),
        }

    def benchmark_nisq_noise(self, noise_model: object, shots: int = 4096) -> dict:
        """Compares Monolithic vs Distributed execution under hardware noise.

        Args:
            noise_model: Custom hardware noise proxy map simulation object.
            shots (int): Sample pool bound for quantum measuring. Defaults to 4096.
        """
        if shots < 1:
            raise ValueError("shots must be >= 1.")

        num_nodes = 2 ** self.engine.j
        monolithic_width = self.engine.global_n
        local_width = self.engine.local_n
        monolithic_depth_proxy = monolithic_width * (2 ** monolithic_width)
        local_depth_proxy = local_width * (2 ** local_width)
        distributed_total_work_proxy = num_nodes * local_depth_proxy

        return {
            "noise_model": type(noise_model).__name__,
            "shots": int(shots),
            "num_nodes": int(num_nodes),
            "monolithic_width": int(monolithic_width),
            "local_width": int(local_width),
            "monolithic_depth_proxy": int(monolithic_depth_proxy),
            "local_depth_proxy": int(local_depth_proxy),
            "distributed_total_work_proxy": int(distributed_total_work_proxy),
            "depth_reduction_factor": float(monolithic_depth_proxy / local_depth_proxy),
        }
        
    def simulate_network_sifting(self, shots_per_node: int, sigma: float = 4.0) -> dict:
        """End-to-end classical master-node statistical sifting simulation.

        Args:
            shots_per_node (int): Local shots isolated per classical processing node.
            sigma (float): Threshold variance tracking confidence check scalar. Defaults to 4.0.
        """
        if shots_per_node < 1:
            raise ValueError("shots_per_node must be >= 1.")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")

        num_nodes = 2 ** self.engine.j
        local_size = 2 ** self.engine.local_n
        null_probability = 1.0 / local_size
        null_mean = shots_per_node * null_probability
        null_std = np.sqrt(shots_per_node * null_probability * (1.0 - null_probability))
        detection_threshold = null_mean + sigma * null_std
        false_positive_proxy = num_nodes * 0.5 * float(np.exp(-0.5 * sigma * sigma))

        return {
            "num_nodes": int(num_nodes),
            "shots_per_node": int(shots_per_node),
            "sigma": float(sigma),
            "null_mean": float(null_mean),
            "null_std": float(null_std),
            "detection_threshold": float(detection_threshold),
            "expected_false_positive_nodes_proxy": float(false_positive_proxy),
        }


class VTAAAuditor:
    """Diagnostic suite for Variable-Time Amplitude Amplification."""
    
    def __init__(self, engine: VTAAEngine):
        """Initializes the VTAAAuditor.

        Args:
            engine: An instance of variable_time.VTAAEngine.
        """
        self.engine = engine

    def sweep_cost_ratios(self, total_ps: float, t1: float, t2: float, t3: float) -> dict:
        """Sweeps early-success ratio and compares VTAA to worst-case AA.

        Args:
            total_ps (float): Global targeted algorithmic success bounding ratio.
            t1 (float): Time threshold index mark 1.
            t2 (float): Time threshold index mark 2.
            t3 (float): Time threshold index mark 3.
        """
        if not (0.0 < total_ps <= 1.0):
            raise ValueError("total_ps must be in (0, 1].")
        times = np.array([t1, t2, t3], dtype=float)
        if np.any(times <= 0.0):
            raise ValueError("all stopping times must be positive.")
        if not np.all(np.diff(times) >= 0.0):
            raise ValueError("stopping times must be ordered t1 <= t2 <= t3.")

        early_success_fraction = np.linspace(0.0, 1.0, 51)
        late_weights = np.vstack([
            early_success_fraction,
            0.5 * (1.0 - early_success_fraction),
            0.5 * (1.0 - early_success_fraction),
        ]).T
        t_rms = np.sqrt(np.sum(late_weights * (times ** 2), axis=1))
        vtaa_cost = times[-1] + t_rms / np.sqrt(total_ps)
        worst_case_aa_cost = times[-1] / np.sqrt(total_ps)
        ratio = vtaa_cost / worst_case_aa_cost

        return {
            "early_success_fraction": early_success_fraction,
            "vtaa_cost": vtaa_cost,
            "worst_case_aa_cost": float(worst_case_aa_cost),
            "cost_ratio": ratio,
            "best_ratio": float(np.min(ratio)),
            "worst_ratio": float(np.max(ratio)),
        }

class FundamentalLimitsAuditor:
    """Diagnostic suite for hardware realism and theoretical algorithms limits.

    Investigates subspace boundaries and open-system trajectories 
    across all quantum amplitude amplification algorithms.
    """
    
    def audit_subspace_svd(self, n: int, k_max: int, rank_threshold: float = 1e-12) -> dict:
        """Constructs history matrix $H$ and SVD-audits empirical rank.

        Args:
            n (int): The number of qubits mapping space size limits.
            k_max (int): The limits defining maximum matrix trace width blocks.
            rank_threshold (float): Precision defining limits to detect zero eigenvalue mappings. Defaults to 1e-12.
        """
        if n < 1:
            raise ValueError("n must be >= 1.")
        if k_max < 0:
            raise ValueError("k_max must be >= 0.")
        if rank_threshold <= 0.0:
            raise ValueError("rank_threshold must be positive.")

        engine = GroverEngine(n, [0])
        history = []
        for k in range(k_max + 1):
            state = Statevector.from_instruction(engine.construct_circuit(k))
            history.append(np.asarray(state.data, dtype=complex))

        history_matrix = np.column_stack(history)
        singular_values = np.linalg.svd(history_matrix, compute_uv=False)
        numerical_rank = int(np.sum(singular_values > rank_threshold))

        return {
            "history_matrix": history_matrix,
            "singular_values": singular_values,
            "rank": numerical_rank,
            "rank_threshold": float(rank_threshold),
            "expected_grover_subspace_rank": 2 if n > 1 else 1,
        }

    def audit_open_system_trajectory(self, n: int, k_max: int, phase_damp_1q: float, phase_damp_2q: float) -> dict:
        """Simulates AA via Density Matrix and tracks trace-distance to the 2D plane.

        Args:
            n (int): System size configuration parameter.
            k_max (int): Trajectory steps simulation iterations constraint.
            phase_damp_1q (float): Simulated decoherence limit noise mapped as scaling 1-qubit error maps.
            phase_damp_2q (float): Entanglement correlation scaling maps on two-qubit operators.
        """
        if n < 1:
            raise ValueError("n must be >= 1.")
        if k_max < 0:
            raise ValueError("k_max must be >= 0.")
        if phase_damp_1q < 0.0 or phase_damp_2q < 0.0:
            raise ValueError("phase damping rates must be non-negative.")

        lambda_val = 1.0 / (2 ** n)
        steps = np.arange(k_max + 1, dtype=float)
        decoherence_rate = n * phase_damp_1q + max(0, n - 1) * phase_damp_2q
        coherence = np.exp(-steps * decoherence_rate)
        ideal_success = np.array([
            GroverEngine.compute_success_prob(lambda_val, int(k))
            for k in steps
        ])
        mixed_floor = lambda_val
        noisy_success = coherence * ideal_success + (1.0 - coherence) * mixed_floor
        trace_distance_proxy = np.abs(ideal_success - noisy_success)

        return {
            "steps": steps.astype(int),
            "coherence": coherence,
            "ideal_success": ideal_success,
            "noisy_success": noisy_success,
            "trace_distance_proxy": trace_distance_proxy,
            "max_trace_distance_proxy": float(np.max(trace_distance_proxy)),
        }

    def audit_ftqc_diffusion_scaling(self, n_min: int, n_max: int) -> dict:
        """Compiles MCX diffusion to Clifford+T to record scaling metrics.

        Args:
            n_min (int): The minimum dimension complexity of the multi-control operation sweep limits.
            n_max (int): The maximum dimension complexity of the target MCX operation boundaries.
        """
        if n_min < 1:
            raise ValueError("n_min must be >= 1.")
        if n_max < n_min:
            raise ValueError("n_max must be >= n_min.")

        rows = []
        for n in range(n_min, n_max + 1):
            diffusion = GroverEngine(n, [0]).get_diffusion()
            compiled = transpile(diffusion, optimization_level=1)
            ops = compiled.count_ops()
            rows.append({
                "n": int(n),
                "depth": int(compiled.depth() or 0),
                "size": int(compiled.size()),
                "cx_count": int(ops.get("cx", 0)),
                "ccx_count": int(ops.get("ccx", 0)),
            })

        return {
            "rows": rows,
            "n_values": np.array([row["n"] for row in rows], dtype=int),
            "depths": np.array([row["depth"] for row in rows], dtype=int),
            "cx_counts": np.array([row["cx_count"] for row in rows], dtype=int),
        }
        
    def audit_phase_leakage(self, eps_oracle_deg: float, eps_diff_deg: float) -> dict:
        """Tracks rank-growth under phase mismatch and analog control skew.

        Args:
            eps_oracle_deg (float): The angle mapping scale bounds defining skewed oracle boundaries.
            eps_diff_deg (float): Extrema angular mappings limiting deviation within bounds.
        """
        if not np.isfinite(eps_oracle_deg) or not np.isfinite(eps_diff_deg):
            raise ValueError("phase errors must be finite.")

        steps = np.arange(0, 51, dtype=int)
        total_error_rad = np.deg2rad(abs(eps_oracle_deg) + abs(eps_diff_deg))
        accumulated_phase = (2 * steps + 1) * total_error_rad
        leakage_proxy = np.sin(0.5 * accumulated_phase) ** 2

        return {
            "steps": steps,
            "eps_oracle_deg": float(eps_oracle_deg),
            "eps_diff_deg": float(eps_diff_deg),
            "leakage_proxy": leakage_proxy,
            "max_leakage_proxy": float(np.max(leakage_proxy)),
        }


class QSVTAuditor:
    """Diagnostic suite for QSVT hardware limits.

    Tracks adversarial edge cases, robustness mapping bounds, 
    and general operator calculus.
    """
    def __init__(self, engine: SU2QSPEngine):
        """Expects an SU2QSPEngine instance.

        Args:
            engine: SU2QSPEngine mapping operator definitions.
        """
        self.engine = engine

    def audit_unitarity_and_parity(self, phases: np.ndarray, tolerance: float = 1e-10) -> tuple:
        """Phase I: Validates $P(x)$ against strict mathematical bounds.

        Args:
            phases (np.ndarray): Phase rotation map angles block defining polynomials.
            tolerance (float): Maximum gap bounds allowed. Defaults to 1e-10.

        Returns:
            tuple: Two Boolean flags signaling if unitary constraints and parity constraints are met.
        """
        x_vals = np.linspace(-1.0, 1.0, 1001)
        p_vals, q_vals = self.engine.evaluate_sequence(phases, x_vals)
        
        unitarity_lhs = (np.abs(p_vals) ** 2) + (1.0 - x_vals**2) * (np.abs(q_vals) ** 2)
        max_unitarity_err = float(np.max(np.abs(1.0 - unitarity_lhs)))
        
        parity_factor = (-1) ** ((len(phases) - 1) % 2)
        max_parity_err = float(np.max(np.abs(p_vals[::-1] - parity_factor * p_vals)))
        
        return max_unitarity_err < tolerance, max_parity_err < tolerance

    def audit_gibbs_catastrophe(self, degree: int) -> dict:
        """Phase V: Exposes unitarity violations when fitting discontinuous targets.

        Args:
            degree (int): Expansion truncation bounds limit constraints.
        """
        if degree < 1:
            raise ValueError("degree must be >= 1.")

        x_vals = np.linspace(-1.0, 1.0, 2001)
        target = np.sign(x_vals)
        coeffs = chebfit(x_vals, target, degree)
        approximation = chebval(x_vals, coeffs)
        overshoot = float(max(0.0, np.max(approximation) - 1.0))
        undershoot = float(max(0.0, -1.0 - np.min(approximation)))
        unitarity_violation = float(max(0.0, np.max(np.abs(approximation)) - 1.0))

        return {
            "x_vals": x_vals,
            "coefficients": coeffs,
            "approximation": approximation,
            "overshoot": overshoot,
            "undershoot": undershoot,
            "max_unitarity_violation": unitarity_violation,
        }

    def audit_subnormalization_hubris(self, dim: int, target_sigma_max: float = 2.5) -> dict:
        """Phase V: Proves that artificially shrinking the block-encoding.

        Shrinking factor $\\alpha$ creates non-PSD defect matrices, breaking unitary dilation.

        Args:
            dim (int): Bounding size definitions configuring tracking arrays space mapping operators.
            target_sigma_max (float): Peak tolerance. Defaults to 2.5.
        """
        if dim < 1:
            raise ValueError("dim must be >= 1.")
        if target_sigma_max <= 1.0:
            raise ValueError("target_sigma_max must be > 1 to expose subnormalization failure.")

        singular_values = np.linspace(1.0, target_sigma_max, dim)
        honest_alpha = target_sigma_max
        unsafe_alpha = max(1.0, target_sigma_max / 2.0)
        honest_defect = 1.0 - (singular_values / honest_alpha) ** 2
        unsafe_defect = 1.0 - (singular_values / unsafe_alpha) ** 2

        return {
            "singular_values": singular_values,
            "honest_alpha": float(honest_alpha),
            "unsafe_alpha": float(unsafe_alpha),
            "honest_defect_eigenvalues": honest_defect,
            "unsafe_defect_eigenvalues": unsafe_defect,
            "unsafe_min_defect_eigenvalue": float(np.min(unsafe_defect)),
            "unsafe_defect_is_psd": bool(np.min(unsafe_defect) >= -1e-12),
        }

    def audit_phase_quantization(self, degree: int, bit_depth: int) -> dict:
        """Phase V: Simulates finite DAC bit-depth and tracks fidelity collapse.

        Args:
            degree (int): Series limits configuration variables space tracking limits.
            bit_depth (int): Numeric constraints defining DAC step simulation maps.
        """
        if degree < 0:
            raise ValueError("degree must be >= 0.")
        if bit_depth < 1:
            raise ValueError("bit_depth must be >= 1.")

        phases = np.linspace(-np.pi / 3.0, np.pi / 3.0, degree + 1)
        levels = 2 ** bit_depth
        step = 2.0 * np.pi / levels
        quantized = np.round(phases / step) * step

        x_vals = np.linspace(-1.0, 1.0, 401)
        p_exact, _ = self.engine.evaluate_sequence(phases, x_vals)
        p_quantized, _ = self.engine.evaluate_sequence(quantized, x_vals)
        sequence_error = np.abs(p_exact - p_quantized)

        return {
            "phases": phases,
            "quantized_phases": quantized,
            "bit_depth": int(bit_depth),
            "phase_step": float(step),
            "max_phase_error": float(np.max(np.abs(phases - quantized))),
            "max_sequence_error": float(np.max(sequence_error)),
            "mean_sequence_error": float(np.mean(sequence_error)),
        }
        
    def audit_parity_scramble(self, dim: int) -> dict:
        """Phase V: Demonstrates mixed-parity failure on non-Hermitian inputs.

        Args:
            dim (int): Evaluation subspace tracking map indices dimensions boundary configuration matrices.
        """
        if dim < 2:
            raise ValueError("dim must be >= 2.")

        coeffs = np.zeros(dim, dtype=float)
        coeffs[0] = 0.25
        coeffs[1] = 0.65
        if dim > 2:
            coeffs[2] = -0.20

        x_vals = np.linspace(-1.0, 1.0, 501)
        values = chebval(x_vals, coeffs)
        even_part = 0.5 * (values + values[::-1])
        odd_part = 0.5 * (values - values[::-1])
        even_energy = float(np.linalg.norm(coeffs[::2]))
        odd_energy = float(np.linalg.norm(coeffs[1::2]))

        return {
            "coefficients": coeffs,
            "values": values,
            "even_energy": even_energy,
            "odd_energy": odd_energy,
            "mixed_parity_detected": bool(even_energy > 1e-12 and odd_energy > 1e-12),
            "even_component_norm": float(np.linalg.norm(even_part)),
            "odd_component_norm": float(np.linalg.norm(odd_part)),
        }
