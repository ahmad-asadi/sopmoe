"""TAC (Trend-Aware Controller) expert – simplified placeholder.

The paper [8] describes TAC as a trend-aware controller that dynamically
selects among a pool of experts based on recent performance and market
regime features.  This placeholder implements a selection mechanism based
on rolling Sharpe ratios.
"""

from __future__ import annotations

import numpy as np

from src.experts.base import BaseExpert
from src.experts.finrl_expert import FinRLExpert


class TACExpert(BaseExpert):
    """Trend-Aware Controller placeholder.

    Selects the sub-expert with the highest rolling Sharpe ratio over the
    last ``window`` steps.  If multiple are tied, uses an equal-weight blend.
    """

    def __init__(
        self,
        experts: list[FinRLExpert],
        name: str = "TAC",
        window: int = 21,
        n_assets: int = 5,
    ) -> None:
        self.name = name
        self.experts = experts
        self.window = window
        self.n_assets = n_assets
        self._return_histories: list[list[float]] = [[] for _ in experts]

    def get_weights(self, state: np.ndarray) -> np.ndarray:
        if not self.experts:
            w = np.ones(self.n_assets + 1, dtype=np.float32)
            return w / w.sum()

        # compute rolling Sharpe for each expert
        sharpes = []
        for i, _ in enumerate(self.experts):
            hist = self._return_histories[i][-self.window :]
            if len(hist) < 5:
                sharpes.append(0.0)
            else:
                mu = float(np.mean(hist))
                sigma = float(np.std(hist)) + 1e-10
                sharpes.append(mu / sigma)

        sharpes = np.array(sharpes, dtype=np.float64)

        # pick the best expert (or blend if tied)
        max_sharpe = sharpes.max()
        if max_sharpe <= 0:
            w = np.ones(self.n_assets + 1, dtype=np.float32)
            return w / w.sum()

        best_mask = sharpes >= max_sharpe - 1e-6
        if best_mask.sum() == 1:
            idx = int(np.argmax(sharpes))
            return self.experts[idx].get_weights(state)
        else:
            sub_weights = [self.experts[i].get_weights(state) for i in np.where(best_mask)[0]]
            combined = np.mean(sub_weights, axis=0)
            combined = np.clip(combined, 0, 1)
            return (combined / (combined.sum() + 1e-10)).astype(np.float32)

    def update_performance(self, expert_idx: int, daily_return: float) -> None:
        if 0 <= expert_idx < len(self._return_histories):
            self._return_histories[expert_idx].append(daily_return)
