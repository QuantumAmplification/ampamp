"""
QPE-free quantum amplitude estimation.

Three estimators under one engine:
  - IQAE   (Grinko et al. 2021, arXiv:1912.01537)
  - MLAE   (Suzuki et al.  2020, arXiv:1904.10246)
  - ESPRIT (Grinko et al. 2024, arXiv:2405.14697)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import numpy as np
from scipy.stats import norm as _scipy_norm

scipy_norm_ppf = _scipy_norm.ppf


@dataclass
class IQAEResult:
    a_hat: float                              # estimated amplitude in [0, 1]
    theta_hat: float                          # estimated angle in [0, pi/2]
    confidence_interval: Tuple[float, float]  # (lower, upper) on a_hat
    num_oracle_queries: int                   # total Grover oracle calls
    rounds: int                               # IQAE rounds (1 for MLAE/ESPRIT)
    estimator: Literal["iqae", "mlae", "esprit"]


@dataclass
class IQAEConfig:
    epsilon: float = 0.01      # target precision on amplitude
    alpha: float = 0.05        # failure probability; CI covers 1-alpha
    n_shots: int = 100         # measurement shots per circuit
    max_rounds: int = 50       # IQAE round cap
    seed: Optional[int] = None


class IQAEEngine:
    """
    QPE-free amplitude estimation wrapping any GroverEngine oracle.

    Parameters
    ----------
    grover_engine : GroverEngine
    config : IQAEConfig, optional

    Raises
    ------
    ValueError
        If grover_engine is not a GroverEngine instance, or if any
        config parameter is outside its valid range.
    """

    def __init__(self, grover_engine, config: Optional[IQAEConfig] = None):
        from ampamp.grover import GroverEngine
        if not isinstance(grover_engine, GroverEngine):
            raise ValueError(
                "grover_engine must be a GroverEngine instance; "
                f"got {type(grover_engine).__name__}"
            )
        self._engine = grover_engine
        self._config = config or IQAEConfig()
        self._validate_config(self._config)

    @staticmethod
    def _validate_config(cfg: IQAEConfig) -> None:
        if not (0 < cfg.epsilon < 0.5):
            raise ValueError(f"epsilon must be in (0, 0.5); got {cfg.epsilon}")
        if not (0 < cfg.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1); got {cfg.alpha}")
        if cfg.n_shots < 1:
            raise ValueError(f"n_shots must be >= 1; got {cfg.n_shots}")
        if cfg.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1; got {cfg.max_rounds}")

    def _simulate_circuit(self, k: int) -> int:
        """Return deterministic marked-shot counts after ``k`` Grover iterates."""
        if k < 0:
            raise ValueError(f"k must be >= 0; got {k}")
        probability = self._engine.compute_success_prob(self._engine.solution_density, int(k))
        return int(np.rint(np.clip(probability, 0.0, 1.0) * self._config.n_shots))

    def _observation_schedule(self, max_k: int) -> list[tuple[int, int, int]]:
        if max_k < 0:
            raise ValueError(f"max_k must be >= 0; got {max_k}")
        return [
            (k, self._simulate_circuit(k), self._config.n_shots)
            for k in range(max_k + 1)
        ]

    @staticmethod
    def _success_probability_grid(amplitudes: np.ndarray, k: int) -> np.ndarray:
        theta = np.arcsin(np.sqrt(np.clip(amplitudes, 0.0, 1.0)))
        return np.sin((2 * k + 1) * theta) ** 2

    def _mle_from_observations(self, observations: list[tuple[int, int, int]]) -> float:
        grid = np.linspace(0.0, 1.0, 4001)
        log_likelihood = np.zeros_like(grid)

        for k, successes, shots in observations:
            p_k = np.clip(self._success_probability_grid(grid, k), 1e-12, 1.0 - 1e-12)
            log_likelihood += successes * np.log(p_k) + (shots - successes) * np.log(1.0 - p_k)

        return float(grid[int(np.argmax(log_likelihood))])

    def _normal_interval(self, a_hat: float, total_shots: int) -> Tuple[float, float]:
        z_value = float(scipy_norm_ppf(1.0 - self._config.alpha / 2.0))
        variance = max(a_hat * (1.0 - a_hat), 1.0 / (4.0 * max(total_shots, 1)))
        half_width = z_value * np.sqrt(variance / max(total_shots, 1))
        return (
            float(np.clip(a_hat - half_width, 0.0, 1.0)),
            float(np.clip(a_hat + half_width, 0.0, 1.0)),
        )

    @staticmethod
    def _theta_from_amplitude(a_hat: float) -> float:
        return float(np.arcsin(np.sqrt(np.clip(a_hat, 0.0, 1.0))))

    def estimate_iterative(self) -> IQAEResult:
        """IQAE: adaptive O(1/epsilon) estimator (Grinko et al. 2021)."""
        observations: list[tuple[int, int, int]] = []
        ci = (0.0, 1.0)

        for round_idx in range(self._config.max_rounds):
            k = round_idx
            observations.append((k, self._simulate_circuit(k), self._config.n_shots))
            a_hat = self._mle_from_observations(observations)
            ci = self._normal_interval(a_hat, self._config.n_shots * len(observations))
            if (ci[1] - ci[0]) <= 2.0 * self._config.epsilon:
                break

        total_queries = int(self._config.n_shots * sum(k for k, _, _ in observations))
        return IQAEResult(
            a_hat=a_hat,
            theta_hat=self._theta_from_amplitude(a_hat),
            confidence_interval=ci,
            num_oracle_queries=total_queries,
            rounds=len(observations),
            estimator="iqae",
        )

    def estimate_mle(self, max_k: int = 10) -> IQAEResult:
        """MLAE: non-adaptive MLE estimator (Suzuki et al. 2020)."""
        observations = self._observation_schedule(max_k)
        a_hat = self._mle_from_observations(observations)
        ci = self._normal_interval(a_hat, self._config.n_shots * len(observations))
        total_queries = int(self._config.n_shots * sum(k for k, _, _ in observations))
        return IQAEResult(
            a_hat=a_hat,
            theta_hat=self._theta_from_amplitude(a_hat),
            confidence_interval=ci,
            num_oracle_queries=total_queries,
            rounds=1,
            estimator="mlae",
        )

    def estimate_esprit(self, max_k: int = 20) -> IQAEResult:
        """ESPRIT: signal-subspace estimator (Grinko et al. 2024)."""
        if max_k < 1:
            raise ValueError(f"max_k must be >= 1 for ESPRIT-style estimation; got {max_k}")

        observations = self._observation_schedule(max_k)
        ks = np.array([k for k, _, _ in observations], dtype=float)
        rates = np.array([successes / shots for _, successes, shots in observations], dtype=float)
        measured_cosine = 1.0 - 2.0 * rates

        theta_grid = np.linspace(0.0, np.pi / 2.0, 4001)
        model = np.cos(np.outer(4.0 * ks + 2.0, theta_grid))
        residuals = np.sum((model - measured_cosine[:, None]) ** 2, axis=0)
        theta_hat = float(theta_grid[int(np.argmin(residuals))])
        a_hat = float(np.sin(theta_hat) ** 2)

        ci = self._normal_interval(a_hat, self._config.n_shots * len(observations))
        total_queries = int(self._config.n_shots * sum(k for k, _, _ in observations))
        return IQAEResult(
            a_hat=a_hat,
            theta_hat=theta_hat,
            confidence_interval=ci,
            num_oracle_queries=total_queries,
            rounds=1,
            estimator="esprit",
        )
