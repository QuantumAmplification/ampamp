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
from scipy.linalg import lstsq
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
        raise NotImplementedError  # implemented in Step 2

    def estimate_iterative(self) -> IQAEResult:
        """IQAE: adaptive O(1/epsilon) estimator (Grinko et al. 2021)."""
        raise NotImplementedError  # implemented in Step 2

    def estimate_mle(self, max_k: int = 10) -> IQAEResult:
        """MLAE: non-adaptive MLE estimator (Suzuki et al. 2020)."""
        raise NotImplementedError  # implemented in Step 3

    def estimate_esprit(self, max_k: int = 20) -> IQAEResult:
        """ESPRIT: signal-subspace estimator (Grinko et al. 2024)."""
        raise NotImplementedError  # implemented in Step 4
