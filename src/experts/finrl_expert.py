"""Wrapper around Stable-Baselines3 / FinRL DRL agents.

Each trained model is loaded and exposed through the ``BaseExpert`` interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from src.experts.base import BaseExpert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent name → SB3 class mapping
# ---------------------------------------------------------------------------
DRL_AGENTS: dict[str, type[BaseAlgorithm]] = {
    "A2C": A2C,
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC,
}

try:
    from sb3_contrib import TQC, RecurrentPPO, MaskablePPO

    DRL_AGENTS["TQC"] = TQC
    DRL_AGENTS["RecurrentPPO"] = RecurrentPPO
    # MaskablePPO requires a discrete action space – keep it listed but
    # let client code decide whether to use it based on the action space.
    DRL_AGENTS["MaskablePPO"] = MaskablePPO
except ImportError:
    logger.warning("sb3_contrib not available – TQC / RecurrentPPO / MaskablePPO skipped")


def get_drl_agent_cls(name: str) -> type[BaseAlgorithm]:
    name_lower = name.lower()
    for k, v in DRL_AGENTS.items():
        if k.lower() == name_lower:
            return v
    msg = f"Unknown DRL agent: {name}. Available: {list(DRL_AGENTS)}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Expert wrapper
# ---------------------------------------------------------------------------
class FinRLExpert(BaseExpert):
    """Loads a saved SB3 model and uses it to produce portfolio weights.

    The raw action from the model is passed through a softmax to ensure
    non-negative weights that sum to 1.
    """

    def __init__(
        self,
        name: str,
        model_path: str | Path,
        n_assets: int,
        device: str = "auto",
    ) -> None:
        self.name = name
        self.n_assets = n_assets
        self._model: BaseAlgorithm = get_drl_agent_cls(name).load(
            str(model_path), device=device
        )
        logger.info("Loaded %s model from %s", name, model_path)

    def get_weights(self, state: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(state, deterministic=True)
        weights = self._softmax(action, n_assets=self.n_assets)
        return weights

    @staticmethod
    def _softmax(action: np.ndarray, n_assets: int) -> np.ndarray:
        """Stable softmax that returns a probability vector of length n_assets."""
        a = np.asarray(action, dtype=np.float64).flatten()
        a = np.clip(a, -100, 100)
        if a.ndim == 0:
            a = a[None]
        exp = np.exp(a - a.max())
        w = exp / (exp.sum() + 1e-10)
        if len(w) == n_assets + 1:
            return w.astype(np.float32)
        out = np.ones(n_assets + 1, dtype=np.float32) / (n_assets + 1)
        out[: len(w)] = w[: len(w)]
        return out


# ---------------------------------------------------------------------------
# Helper to create a dummy model (for testing / placeholders)
# ---------------------------------------------------------------------------
class DummyDRLExpert(BaseExpert):
    """Placeholder DRL expert that returns uniform weights."""

    def __init__(self, name: str = "dummy", n_assets: int = 5) -> None:
        self.name = name
        self.n_assets = n_assets

    def get_weights(self, state: np.ndarray) -> np.ndarray:
        w = np.ones(self.n_assets + 1, dtype=np.float32)
        return w / w.sum()
