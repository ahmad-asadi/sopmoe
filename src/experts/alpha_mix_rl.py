"""AlphaMixRL expert – simplified placeholder.

The paper describes AlphaMixRL as a hierarchical RL with ensemble routing.
This placeholder implements a simple performance-weighted ensemble of DRL
experts.
"""

from __future__ import annotations

import numpy as np

from src.experts.base import BaseExpert
from src.experts.finrl_expert import FinRLExpert


class AlphaMixRLExpert(BaseExpert):
    """Simplified AlphaMixRL: performance-weighted ensemble of sub-experts.

    Maintains a rolling window of returns for each sub-expert and computes
    an exponentially weighted average to combine their outputs.
    """

    def __init__(
        self,
        experts: list[FinRLExpert],
        name: str = "AlphaMixRL",
        decay: float = 0.9,
        n_assets: int = 5,
    ) -> None:
        self.name = name
        self.experts = experts
        self.decay = decay
        self.n_assets = n_assets
        self._rolling_returns: list[list[float]] = [[] for _ in experts]

    def get_weights(self, state: np.ndarray) -> np.ndarray:
        if not self.experts:
            w = np.ones(self.n_assets + 1, dtype=np.float32)
            return w / w.sum()

        sub_weights = []
        scores = []
        for i, expert in enumerate(self.experts):
            w = expert.get_weights(state)
            sub_weights.append(w)
            recent = self._rolling_returns[i][-20:] if self._rolling_returns[i] else [0.0]
            score = sum(r * (self.decay ** (len(recent) - j))
                        for j, r in enumerate(recent)) / (len(recent) + 1e-10)
            scores.append(max(score, 1e-6))

        scores = np.array(scores, dtype=np.float64)
        scores /= scores.sum() + 1e-10

        combined = sum(s * w for s, w in zip(scores, sub_weights))
        combined = np.clip(combined, 0, 1)
        return (combined / (combined.sum() + 1e-10)).astype(np.float32)

    def update_performance(self, expert_idx: int, daily_return: float) -> None:
        if 0 <= expert_idx < len(self._rolling_returns):
            self._rolling_returns[expert_idx].append(daily_return)
