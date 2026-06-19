"""Registry for managing and evaluating portfolio-management experts."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.experts.base import BaseExpert

logger = logging.getLogger(__name__)


class ExpertRegistry:
    """Holds expert instances and provides query / evaluation utilities."""

    def __init__(self) -> None:
        self._experts: dict[str, BaseExpert] = {}

    def register(self, expert: BaseExpert, name: str | None = None) -> None:
        key = name or expert.get_name()
        if key in self._experts:
            logger.warning("Overwriting existing expert: %s", key)
        self._experts[key] = expert

    def get_expert(self, name: str) -> BaseExpert:
        if name not in self._experts:
            msg = f"Expert '{name}' not found. Available: {list(self._experts)}"
            raise KeyError(msg)
        return self._experts[name]

    def list_experts(self) -> list[str]:
        return list(self._experts)

    def remove(self, name: str) -> None:
        self._experts.pop(name, None)

    def __len__(self) -> int:
        return len(self._experts)

    def __contains__(self, name: str) -> bool:
        return name in self._experts

    def __repr__(self) -> str:
        return f"ExpertRegistry({list(self._experts)})"

    @staticmethod
    def evaluate_on_dataset(
        experts: list[BaseExpert],
        states: list[np.ndarray],
        returns: list[np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Compute performance metrics for each expert over a dataset.

        Parameters
        ----------
        experts : list[BaseExpert]
            Experts to evaluate.
        states : list[np.ndarray]
            Sequence of observation states.
        returns : list[np.ndarray]
            Corresponding per-asset returns at each step.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict: expert_name -> {cumulative_return, sharpe,
            max_drawdown, volatility}.
        """
        results: dict[str, dict[str, float]] = {}
        n_steps = len(states)

        for expert in experts:
            weights_list: list[np.ndarray] = []
            for s in states:
                w = expert.get_weights(s)
                weights_list.append(w)

            port_returns: list[float] = []
            for t in range(n_steps):
                w = weights_list[t]
                r = returns[t]
                if len(w) > len(r):
                    w = w[: len(r) + 1]
                cash_w = w[0]
                asset_w = w[1 : 1 + len(r)]
                port_ret = cash_w * 0.0 + float(np.dot(asset_w, r))
                port_returns.append(port_ret)

            port_returns = np.array(port_returns, dtype=np.float64)
            cum_return = float(np.prod(1 + port_returns)) - 1.0
            vol = float(np.std(port_returns)) + 1e-10
            sharpe = float(np.mean(port_returns) / vol) * np.sqrt(252)

            # max drawdown
            cum = np.cumprod(1 + port_returns)
            running_max = np.maximum.accumulate(cum)
            dd = (cum - running_max) / (running_max + 1e-10)
            max_dd = float(np.min(dd))

            results[expert.get_name()] = {
                "cumulative_return": cum_return,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "volatility": vol,
            }

        return results
