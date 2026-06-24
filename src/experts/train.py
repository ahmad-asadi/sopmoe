"""Training script for DRL-based portfolio management experts.

Creates the ``PortfolioTradingEnv``, trains each agent from stable-baselines3
(and sb3-contrib), and saves the model checkpoints to ``checkpoints/``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import A2C, DDPG, PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from src.experts.base import PortfolioTradingEnv
from src.experts.finrl_expert import DRL_AGENTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared hyper-parameters (kept equal across agents for fairness)
# ---------------------------------------------------------------------------
SHARED_PARAMS: dict = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "seed": 42,
    "policy_kwargs": {"net_arch": [256, 256]},
}

TOTAL_TIMESTEPS = 50_000  # increase for real training

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def create_env_from_df(
    df: pd.DataFrame,
    stock_dim: int,
    tech_indicator_list: list[str] | None = None,
) -> PortfolioTradingEnv:
    """Build a ``PortfolioTradingEnv`` from a pre-processed DataFrame.

    The DataFrame must be a MultiIndex (date, ticker) with at least a
    ``close`` column.
    """
    if tech_indicator_list is None:
        tech_indicator_list = []

    # ensure required columns
    required = ["close"]
    for col in required:
        if col not in df.columns:
            msg = f"DataFrame missing required column: {col}"
            raise ValueError(msg)

    # keep only numeric columns for tech indicators
    auto_tech = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in ("close",)]
    if tech_indicator_list:
        auto_tech = [t for t in tech_indicator_list if t in df.columns]

    env = PortfolioTradingEnv(
        df=df,
        stock_dim=stock_dim,
        tech_indicator_list=auto_tech,
        transaction_cost_pct=0.001,
        reward_scaling=1.0,
        initial_amount=1_000_000.0,
    )
    return env


# Agent-specific policy types
_AGENT_POLICIES: dict[str, str] = {
    "A2C": "MlpPolicy",
    "DDPG": "MlpPolicy",
    "PPO": "MlpPolicy",
    "SAC": "MlpPolicy",
    "TQC": "MlpPolicy",
    "RecurrentPPO": "MlpLstmPolicy",
    "MaskablePPO": "MlpPolicy",
}


def train_agent(
    env: PortfolioTradingEnv,
    agent_cls: type,
    total_timesteps: int,
    model_save_path: str | Path,
    agent_name: str = "agent",
) -> None:
    """Train a single DRL agent and save the model."""
    logger.info("Training %s for %d timesteps ...", agent_name, total_timesteps)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(5_000, total_timesteps // 10),
        save_path=str(Path(model_save_path).parent),
        name_prefix=agent_name,
    )

    # Only pass kwargs that the agent class accepts
    import copy
    import inspect
    sig = inspect.signature(agent_cls.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: copy.deepcopy(v) for k, v in SHARED_PARAMS.items() if k in valid_keys}

    policy_name = _AGENT_POLICIES.get(agent_name, "MlpPolicy")

    model = agent_cls(
        policy_name,
        env,
        **kwargs,
        verbose=0,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
    )
    model.save(str(model_save_path))
    logger.info("Saved %s model to %s", agent_name, model_save_path)


def train_all_drl_agents(
    df: pd.DataFrame,
    stock_dim: int,
    tech_indicator_list: list[str] | None = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
) -> dict[str, str]:
    """Train every available DRL agent and return a name → path mapping."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}
    for name, agent_cls in DRL_AGENTS.items():
        # Skip agents that are incompatible with continuous action spaces
        if name == "MaskablePPO":
            logger.warning("Skipping %s – requires a discrete action space", name)
            continue

        # Create a fresh env for each agent so the internal state is clean
        env = create_env_from_df(df, stock_dim, tech_indicator_list)
        path = checkpoint_dir / f"{name}"
        train_agent(env, agent_cls, total_timesteps, path, agent_name=name)
        saved_paths[name] = str(path)

    return saved_paths


