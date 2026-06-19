# -*- coding: utf-8 -*-
"""Backtesting engine for portfolio strategy evaluation.

Provides:
- ``Backtester`` - runs full-period simulations for individual experts and
  switching strategies (including ablations).
- Utility functions for equity-curve tracking and turnover computation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from src.data import DataLoader, FeatureEngineer, StateBuilder
from src.experts.registry import ExpertRegistry
from src.inference.engine import InferenceEngine
from src.inference.utils import project_weights
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Backtester:
    """Simulates trading over the test period for expert(s) or switching strategies.

    Parameters
    ----------
    inference_engine :
        ``InferenceEngine`` instance for retrieval-based switching.
        Can be ``None`` when only running fixed experts.
    expert_registry :
        Registry with all candidate expert instances.
    data_loader :
        ``DataLoader`` instance used to load price / feature data.
    config :
        Dict-like config with keys:
        - ``initial_capital`` (float, default 100000.0)
        - ``transaction_cost_pct`` (float, default 0.001)
        - ``rebalance_freq`` (str, default ``"monthly"``)
        - ``lookback`` (int, default 60)
        - ``train_end`` / ``val_end`` (str, date splits)
        - ``test_start`` (str, override test start)
    """

    def __init__(
        self,
        inference_engine: InferenceEngine | None,
        expert_registry: ExpertRegistry,
        data_loader: DataLoader,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.inference_engine = inference_engine
        self.expert_registry = expert_registry
        self.data_loader = data_loader
        self.config = config or {}

        self.initial_capital = float(self.config.get("initial_capital", 100_000.0))
        self.transaction_cost_pct = float(self.config.get("transaction_cost_pct", 0.001))
        self.rebalance_freq = self.config.get("rebalance_freq", "monthly")
        self.L = int(self.config.get("lookback", 60))

        self._prepare_data()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Load data, compute features, build states, filter test period."""
        df = self.data_loader.load_data()

        fe = FeatureEngineer()
        df_tech, df_mkt = fe.compute_features(df)

        self.tech_feature_names = fe.tech_feature_names
        self.market_feature_names = fe.market_feature_names
        self.symbols = sorted(df.index.get_level_values("symbol").unique())
        self.n_assets = len(self.symbols)

        # Build all states (needs lookback from full history)
        state_builder = StateBuilder(
            window_length=self.L,
            tech_feature_names=self.tech_feature_names,
            market_feature_names=self.market_feature_names,
        )
        all_states = state_builder.build_states(df_tech, df_mkt)

        # Determine test start date
        test_start = self.config.get("test_start", "2022-01-01")
        if isinstance(test_start, str):
            test_start = pd.Timestamp(test_start)
        else:
            test_start = pd.Timestamp(test_start)

        self.test_states = [
            (s, t, ts) for s, t, ts in all_states if ts >= test_start
        ]

        # Compute daily asset returns from close prices
        close_df = df["close"].unstack("symbol")
        self.close_prices = close_df
        self.asset_returns = close_df.pct_change().fillna(0.0)

        # Store full dataframes for expert-state building
        self.df = df
        self.df_tech = df_tech
        self.df_mkt = df_mkt

        if len(self.test_states) == 0:
            logger.warning("No test states after filtering (start=%s)", test_start)
        else:
            first_ts = self.test_states[0][2]
            last_ts = self.test_states[-1][2]
            logger.info(
                "Backtester ready: %d test steps  [%s ... %s],  %d assets",
                len(self.test_states),
                first_ts.date(),
                last_ts.date(),
                self.n_assets,
            )

    # ------------------------------------------------------------------
    # Fixed-expert simulation
    # ------------------------------------------------------------------

    def run_expert(
        self,
        expert_name: str,
        return_equity_curve: bool = False,
    ) -> dict[str, Any]:
        """Run a fixed-expert strategy over the test period.

        Parameters
        ----------
        expert_name :
            Name of the registered expert to use.
        return_equity_curve :
            If ``True``, include the full equity curve in the result.

        Returns
        -------
        dict with keys:
            ``daily_returns`` (np.ndarray)
            ``equity_curve`` (np.ndarray, if requested)
            ``n_trades`` (int) - number of rebalances
        """
        expert = self.expert_registry.get_expert(expert_name)

        daily_returns: list[float] = []
        equity = [self.initial_capital]
        prev_weights: np.ndarray | None = None
        n_trades = 0

        for i, (S_tech, S_mkt, ts) in enumerate(self.test_states):
            expert_state = self._build_expert_state(ts, prev_weights)
            raw_weights = expert.get_weights(expert_state)
            projected = project_weights(raw_weights)

            if prev_weights is not None and not np.allclose(projected, prev_weights, atol=1e-6):
                n_trades += 1

            port_ret = self._portfolio_return(ts, projected, prev_weights)

            daily_returns.append(port_ret)
            equity.append(equity[-1] * (1.0 + port_ret))
            prev_weights = projected.copy()

            # Update internal state for experts that track history
            day_asset_returns = self.asset_returns.loc[ts].values
            if hasattr(expert, "update_history"):
                expert.update_history(day_asset_returns)

        result: dict[str, Any] = {
            "daily_returns": np.array(daily_returns, dtype=np.float64),
            "n_trades": n_trades,
        }
        if return_equity_curve:
            result["equity_curve"] = np.array(equity[1:], dtype=np.float64)
        return result

    # ------------------------------------------------------------------
    # Switching strategies (with ablation controls)
    # ------------------------------------------------------------------

    def run_switching(
        self,
        strategy_name: str = "proposed",
        use_retrieval: bool = True,
        use_llm: bool = True,
        use_uncertainty: bool = True,
        return_equity_curve: bool = False,
    ) -> dict[str, Any]:
        """Run a switching strategy over the test period.

        Parameters
        ----------
        strategy_name :
            Human-readable label for the strategy (used in results).
        use_retrieval :
            If ``False``, use a recent-gating heuristic instead of the
            full retrieval pipeline.
        use_llm :
            If ``False``, skip the LLM uncertainty call (use heuristic).
        use_uncertainty :
            If ``False``, set the uncertainty penalty :math:`\\lambda = 0`.
        return_equity_curve :
            If ``True``, include the full equity curve in the result.

        Returns
        -------
        dict with keys ``daily_returns``, ``equity_curve``, ``n_trades``,
        and ``selected_experts`` (list of chosen expert names).
        """
        daily_returns: list[float] = []
        equity = [self.initial_capital]
        prev_weights: np.ndarray | None = None
        n_trades = 0
        selected_experts: list[str] = []

        # Store original engine config so we can restore it
        orig_lambda = None
        orig_llm_client = None
        if self.inference_engine is not None:
            orig_lambda = self.inference_engine.lambda_
            orig_llm_client = getattr(self.inference_engine, "llm_client", None)

        try:
            self._configure_engine(use_llm, use_uncertainty)

            for i, (S_tech, S_mkt, ts) in enumerate(self.test_states):
                day_asset_returns = self.asset_returns.loc[ts].values

                if use_retrieval and self.inference_engine is not None:
                    selected, projected, meta = self._run_inference_step(
                        S_tech, S_mkt, ts, prev_weights
                    )
                else:
                    selected, projected = self._recent_gating(
                        ts, prev_weights, day_asset_returns
                    )

                selected_experts.append(selected)

                if prev_weights is not None and not np.allclose(projected, prev_weights, atol=1e-6):
                    n_trades += 1

                port_ret = self._portfolio_return(ts, projected, prev_weights)
                daily_returns.append(port_ret)
                equity.append(equity[-1] * (1.0 + port_ret))
                prev_weights = projected.copy()

                # Update history for all experts that track it
                for ename in self.expert_registry.list_experts():
                    exp = self.expert_registry.get_expert(ename)
                    if hasattr(exp, "update_history"):
                        exp.update_history(day_asset_returns)

        finally:
            self._restore_engine(orig_lambda, orig_llm_client)

        result: dict[str, Any] = {
            "daily_returns": np.array(daily_returns, dtype=np.float64),
            "n_trades": n_trades,
            "selected_experts": selected_experts,
        }
        if return_equity_curve:
            result["equity_curve"] = np.array(equity[1:], dtype=np.float64)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure_engine(self, use_llm: bool, use_uncertainty: bool) -> None:
        """Temporarily adjust the inference engine for ablation settings."""
        if self.inference_engine is None:
            return

        if not use_llm:
            self.inference_engine.llm_client = None

        if not use_uncertainty:
            self.inference_engine.lambda_ = 0.0

    def _restore_engine(
        self,
        orig_lambda: float | None,
        orig_llm_client: Callable[[str], str] | None,
    ) -> None:
        """Restore original inference engine configuration."""
        if self.inference_engine is None:
            return
        if orig_lambda is not None:
            self.inference_engine.lambda_ = orig_lambda
        self.inference_engine.llm_client = orig_llm_client

    def _run_inference_step(
        self,
        S_tech: torch.Tensor,
        S_mkt: torch.Tensor,
        ts: pd.Timestamp,
        prev_weights: np.ndarray | None,
    ) -> tuple[str, np.ndarray, dict[str, Any]]:
        """Run a single inference step and return (name, weights, metadata)."""
        if S_tech.dim() == 2:
            S_tech = S_tech.unsqueeze(0)
        if S_mkt.dim() == 1:
            S_mkt = S_mkt.unsqueeze(0)

        expert_state = self._build_expert_state(ts, prev_weights)

        return self.inference_engine.select_expert(
            tech_state=S_tech,
            mkt_state=S_mkt,
            timestamp=ts.isoformat() if ts is not None else None,
            expert_state=expert_state,
        )

    def _recent_gating(
        self,
        ts: pd.Timestamp,
        prev_weights: np.ndarray | None,
        day_asset_returns: np.ndarray,
    ) -> tuple[str, np.ndarray]:
        """No-retrieval fallback: select the expert with best recent Sharpe.

        Tracks a rolling window of per-expert portfolio returns and picks
        the one with the highest Sharpe over the window.  Blend is used
        when no history exists yet.
        """
        window = min(60, len(self.test_states))

        if not hasattr(self, "_recent_gating_buffer"):
            self._recent_gating_buffer: dict[str, list[float]] = {
                name: [] for name in self.expert_registry.list_experts()
            }

        # Evaluate all experts on today's state and record portfolio returns
        expert_returns: dict[str, float] = {}
        expert_weights: dict[str, np.ndarray] = {}

        for name in self.expert_registry.list_experts():
            exp = self.expert_registry.get_expert(name)
            expert_state = self._build_expert_state(ts, prev_weights)
            raw = exp.get_weights(expert_state)
            proj = project_weights(raw)

            port_ret = self._portfolio_return(ts, proj, prev_weights)
            expert_returns[name] = port_ret
            expert_weights[name] = proj

            self._recent_gating_buffer[name].append(port_ret)
            if len(self._recent_gating_buffer[name]) > window:
                self._recent_gating_buffer[name].pop(0)

        # If insufficient history, blend equally
        n_experts = len(self.expert_registry)
        if n_experts == 0:
            return "none", np.ones(self.n_assets + 1, dtype=np.float32) / (self.n_assets + 1)

        if len(self._recent_gating_buffer) < 10:
            avg_w = np.zeros(self.n_assets + 1, dtype=np.float64)
            for w in expert_weights.values():
                avg_w += w.astype(np.float64)
            avg_w /= n_experts
            return "blend", avg_w.astype(np.float32)

        # Select expert with highest recent Sharpe
        best_name = max(
            self._recent_gating_buffer,
            key=lambda n: self._rolling_sharpe(self._recent_gating_buffer[n]),
        )
        return best_name, expert_weights[best_name]

    @staticmethod
    def _rolling_sharpe(returns: list[float]) -> float:
        arr = np.array(returns, dtype=np.float64)
        if len(arr) < 5 or np.std(arr) < 1e-12:
            return float(np.mean(arr))
        return float(np.mean(arr) / np.std(arr)) * np.sqrt(252)

    def _portfolio_return(
        self,
        ts: pd.Timestamp,
        weights: np.ndarray,
        prev_weights: np.ndarray | None,
    ) -> float:
        """Compute the daily portfolio return given weights and asset returns.

        Includes transaction costs if configured.
        """
        day_returns = self.asset_returns.loc[ts].values

        cash_w = weights[0]
        asset_w = weights[1 : 1 + len(day_returns)]
        port_ret = cash_w * 0.0 + float(np.dot(asset_w, day_returns))

        if self.transaction_cost_pct > 0.0 and prev_weights is not None:
            turnover = float(np.abs(weights - prev_weights).sum())
            port_ret -= self.transaction_cost_pct * turnover

        return port_ret

    def _build_expert_state(
        self,
        ts: pd.Timestamp,
        prev_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Build the 1-D observation vector expected by `BaseExpert.get_weights`.

        The vector follows the ``PortfolioTradingEnv`` convention:
        ``[cash_ratio, prices..., prev_weights..., tech_vals...]``
        """
        try:
            day_data = self.df.xs(ts, level="date")
        except KeyError:
            dim = 1 + self.n_assets + (self.n_assets + 1) + len(self.tech_feature_names) * self.n_assets
            return np.zeros(dim, dtype=np.float32)

        prices = day_data["close"].values.astype(np.float32)

        if prev_weights is None:
            prev_weights_arr = np.zeros(self.n_assets + 1, dtype=np.float32)
            prev_weights_arr[0] = 1.0
        else:
            prev_weights_arr = prev_weights

        # Build flattened tech vector
        try:
            day_tech = self.df_tech.xs(ts, level="date")
            tech_vals: list[float] = []
            for feat in self.tech_feature_names:
                vals = day_tech[feat].values
                if isinstance(vals, (int, float, np.floating)):
                    tech_vals.append(float(vals))
                else:
                    tech_vals.extend(float(v) for v in vals)
        except (KeyError, TypeError):
            tech_vals = [0.0] * (len(self.tech_feature_names) * self.n_assets)

        cash_ratio = np.array([prev_weights_arr[0]], dtype=np.float32)
        state = np.concatenate(
            [cash_ratio, prices, prev_weights_arr, np.array(tech_vals, dtype=np.float32)]
        )
        return state.astype(np.float32)
