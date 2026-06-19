"""Expert pool – portfolio management experts with a common interface.

Provides:
- ``BaseExpert``        – abstract base class
- ``PortfolioTradingEnv` – Gym environment for portfolio optimisation
- ``FinRLExpert``       – wrapper around trained SB3 / FinRL agents
- ``MILPExpert``        – mean-variance optimisation
- ``AlphaMixRLExpert``   – performance-weighted ensemble (placeholder)
- ``TACExpert``          – trend-aware controller (placeholder)
- ``ExpertRegistry``    – registry with evaluation utilities
- ``train_all_drl_agents`` – training entry point
"""

from src.experts.base import BaseExpert, PortfolioTradingEnv
from src.experts.finrl_expert import DRL_AGENTS, DummyDRLExpert, FinRLExpert
from src.experts.milp_expert import MILPExpert
from src.experts.alpha_mix_rl import AlphaMixRLExpert
from src.experts.tac_expert import TACExpert
from src.experts.registry import ExpertRegistry
from src.experts.train import train_all_drl_agents

__all__ = [
    "BaseExpert",
    "PortfolioTradingEnv",
    "DRL_AGENTS",
    "DummyDRLExpert",
    "FinRLExpert",
    "MILPExpert",
    "AlphaMixRLExpert",
    "TACExpert",
    "ExpertRegistry",
    "train_all_drl_agents",
]
