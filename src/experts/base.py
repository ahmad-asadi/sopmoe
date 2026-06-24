"""Base classes and the portfolio trading environment for the expert pool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class BaseExpert(ABC):
    """Abstract base class for all portfolio management experts.

    Every expert must implement ``get_weights`` which, given an observation
    state, returns a probability vector over assets (including cash).
    """

    name: str = "base"

    @abstractmethod
    def get_weights(self, state: np.ndarray) -> np.ndarray:
        """Return portfolio weight vector given the current state.

        Parameters
        ----------
        state : np.ndarray
            Observation from the environment.

        Returns
        -------
        np.ndarray
            Weight vector of shape ``(n_assets,)`` that sums to 1.
        """

    def get_name(self) -> str:
        return self.name


class PortfolioTradingEnv(gym.Env):
    """A portfolio-management environment that uses continuous weight actions.

    The action is a vector of portfolio weights (including cash) that must
    sum to 1.  The reward is the portfolio log-return after transaction costs.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        tech_indicator_list: list[str],
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1.0,
        initial_amount: float = 1_000_000.0,
        lookback_window: int = 10,
    ):
        super().__init__()
        self.df = df
        self.stock_dim = stock_dim
        self.tech_indicator_list = tech_indicator_list
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_amount = initial_amount
        self.lookback_window = lookback_window

        unique_dates = df.index.get_level_values(0).unique()
        self.dates = unique_dates.sort_values()
        self.max_step = len(self.dates) - 1
        self.day = 0
        self.terminal = False

        # state: cash(1) + prices(stock_dim) + prev_weights(stock_dim + 1 for cash)
        #   + tech_indicator per stock (n_tech * stock_dim)
        state_dim = (
            1
            + stock_dim
            + (stock_dim + 1)
            + len(tech_indicator_list) * stock_dim
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        # Actions are raw scores (logits); softmax is applied in step()
        # Bounds must be finite for stable-baselines3 compatibility.
        bound = 10.0
        self.action_space = spaces.Box(
            low=-bound, high=bound, shape=(stock_dim + 1,), dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.day = 0
        self.terminal = False
        self.cash = float(self.initial_amount)
        self.current_weights = np.zeros(self.stock_dim + 1, dtype=np.float32)
        self.current_weights[0] = 1.0  # 100 % cash
        self.portfolio_value = self.initial_amount
        self.returns_history: list[float] = []
        return self._get_obs(), {}

    def step(self, actions: np.ndarray):
        assert not self.terminal, "Episode is already terminal"
        actions = np.asarray(actions, dtype=np.float64)
        # softmax to get valid portfolio weights
        exp_a = np.exp(actions - actions.max())
        actions = (exp_a / (exp_a.sum() + 1e-10)).astype(np.float32)

        prices = self._get_prices()
        prev_weights = self.current_weights.copy()

        # --- compute turnover and transaction costs ---
        turnover = np.abs(actions - prev_weights).sum()
        tc = self.transaction_cost_pct * turnover

        # --- price-relative change ---
        next_day = min(self.day + 1, self.max_step)
        next_prices = self._get_prices(next_day)
        price_rel = next_prices / (prices + 1e-10)

        # portfolio return (excluding cash, which has price_rel = 1)
        cash_weight = actions[0]
        asset_weights = actions[1:]
        asset_return = float(np.dot(asset_weights, price_rel))
        portfolio_return = max(cash_weight * 1.0 + asset_return - tc, 0.001)
        self.portfolio_value *= portfolio_return

        self.current_weights = actions.copy()
        self.cash = self.portfolio_value * self.current_weights[0]
        self.day = next_day
        self.terminal = self.day >= self.max_step
        log_ret = float(np.log(portfolio_return))
        self.returns_history.append(log_ret)

        reward = log_ret * self.reward_scaling

        return self._get_obs(), reward, self.terminal, False, {}

    def render(self, mode: str = "human"):
        print(f"Step {self.day}: value={self.portfolio_value:.2f}, "
              f"weights={self.current_weights}")

    def _get_prices(self, day: int | None = None) -> np.ndarray:
        d = self.dates[day if day is not None else self.day]
        row = self.df.loc[d]
        prices = row["close"]
        if isinstance(prices, (int, float, np.floating)):
            prices = np.array([prices], dtype=np.float32)
        else:
            prices = prices.values.astype(np.float32)
        return prices

    def _get_obs(self) -> np.ndarray:
        d = self.dates[self.day]
        row = self.df.loc[d]
        close = row["close"]
        if isinstance(close, (int, float, np.floating)):
            prices = np.array([close], dtype=np.float32)
            tech_values = np.concatenate(
                [np.array([row.get(t, 0.0)], dtype=np.float32) for t in self.tech_indicator_list]
            )
        else:
            prices = close.values.astype(np.float32)
            # Correctly handle tech indicator values by ensuring they are arrays and not NaN
            tech_vals_list = []
            for t in self.tech_indicator_list:
                val = row.get(t)
                if val is None:
                    tech_vals_list.append(np.zeros(len(prices), dtype=np.float32))
                elif hasattr(val, "values"):
                    tech_vals_list.append(val.values.astype(np.float32))
                elif isinstance(val, (int, float, np.floating)):
                    tech_vals_list.append(np.full(len(prices), float(val), dtype=np.float32))
                else:
                    tech_vals_list.append(np.zeros(len(prices), dtype=np.float32))
            
            tech_values = np.concatenate(tech_vals_list)
        
        # Final NaN cleanup to prevent model collapse
        cash_arr = np.array([self.cash / self.initial_amount], dtype=np.float32)
        obs = np.concatenate([cash_arr, prices, self.current_weights, tech_values])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


    def _get_info(self):
        return {"portfolio_value": self.portfolio_value, "step": self.day}
