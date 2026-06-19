"""MILP (Mixed-Integer Linear Programming) expert.

Uses mean-variance optimisation (Markowitz) to determine portfolio weights.
A wrapper around ``scipy.optimize.minimize``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.experts.base import BaseExpert


class MILPExpert(BaseExpert):
    """Mean-variance optimisation expert.

    Given a window of historical returns, it solves the optimisation:

        maximise   w^T μ - λ * w^T Σ w
        subject to sum(w) == 1,  w >= 0

    where μ is the vector of mean returns, Σ is the covariance matrix,
    and λ is the risk-aversion coefficient.
    """

    def __init__(
        self,
        name: str = "MILP",
        window: int = 60,
        risk_aversion: float = 1.0,
        n_assets: int = 5,
    ) -> None:
        self.name = name
        self.window = window
        self.risk_aversion = risk_aversion
        self.n_assets = n_assets
        self._returns_buffer: list[np.ndarray] = []

    def get_weights(self, state: np.ndarray) -> np.ndarray:
        # state is unused in this simple version – we rely on the stored history
        if len(self._returns_buffer) < 2:
            return np.ones(self.n_assets + 1, dtype=np.float32) / (self.n_assets + 1)

        returns = np.asarray(self._returns_buffer[-self.window :])
        mu = returns.mean(axis=0)
        sigma = np.cov(returns, rowvar=False)
        n = sigma.shape[0]

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ mu)
            port_std = float(np.sqrt(w @ sigma @ w + 1e-10))
            return -port_ret / port_std

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n
        result = minimize(
            neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 200},
        )

        w = result.x if result.success else (np.ones(n) / n)
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-10)
        # prepend cash weight (0 % for now – could be learned)
        full_w = np.empty(self.n_assets + 1, dtype=np.float32)
        full_w[0] = 0.0
        full_w[1:] = w
        return full_w

    def update_history(self, daily_returns: np.ndarray) -> None:
        """Append a vector of per-asset returns for the current step."""
        self._returns_buffer.append(np.asarray(daily_returns, dtype=np.float64))
        if len(self._returns_buffer) > self.window * 2:
            self._returns_buffer.pop(0)
