"""Offline indexing engine implementing Algorithm 1.

For each historical timestep:
  1. Build S_tech, S_mkt from data.
  2. Get embedding h_τ from the VAE.
  3. For each expert e: execute policy, evaluate performance over H days,
     generate SoP via LLM, store (h_τ, R_τ^e, SoP_τ^e) in the vector DB.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data.state_builder import StateBuilder
from src.embedding.vae import DualStreamVAE
from src.experts.registry import ExpertRegistry
from src.indexing.database import VectorDatabase
from src.indexing.prompts import build_sop_prompt, parse_sop_response
from src.utils.logging import get_logger

logger = get_logger(__name__)



class IndexingEngine:
    """Offline indexing pipeline (Algorithm 1).

    Parameters
    ----------
    vae :
        Trained ``DualStreamVAE`` in evaluation mode.
    expert_registry :
        Registry containing all experts to index.
    state_builder :
        ``StateBuilder`` instance for constructing S_tech / S_mkt.
    db :
        ``VectorDatabase`` instance for storing results.
    config :
        Dict-like object with at least:
        - ``H`` : evaluation horizon in days
        - ``L`` : lookback window (should match VAE)
        - ``mock_llm`` : bool, whether to skip real LLM calls
        - ``device`` : torch device string
        - ``llm`` : dict with model_name, api_base, api_key, etc.
    """

    def __init__(
        self,
        vae: DualStreamVAE,
        expert_registry: ExpertRegistry,
        state_builder: StateBuilder,
        db: VectorDatabase,
        config: dict[str, Any],
    ):
        self.vae = vae
        self.expert_registry = expert_registry
        self.state_builder = state_builder
        self.db = db
        self.H = int(config.get("H", 20))
        self.L = int(config.get("L", 60))
        self.device = str(config.get("device", "cpu"))
        self.llm_config = config.get("llm", {})
        self._sop_cache: dict[tuple[str, str], str] = {}

        self.vae = self.vae.to(self.device)
        self.vae.eval()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_indexing(
        self,
        df: pd.DataFrame,
        df_tech: pd.DataFrame,
        df_mkt: pd.DataFrame,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> int:
        """Run the offline indexing pipeline over the date range.
        
        Parameters
        ----------
        df :
            Raw OHLCV+features DataFrame with MultiIndex (date, symbol).
        df_tech :
            Technical feature DataFrame from ``FeatureEngineer``.
        df_mkt :
            Market feature DataFrame from ``FeatureEngineer``.
        start_date, end_date :
            Optional date range (default: use all available dates after L).

        Returns
        -------
        Number of records indexed.
        """
        logger.info("Starting offline indexing process...")
        timestamps = self._get_timestamps(df_tech, start_date, end_date)
        if not timestamps:
            logger.warning("No timestamps to index.")
            return 0

        logger.info(
            "Indexing %d timestamps, %d experts, H=%d",
            len(timestamps),
            len(self.expert_registry),
            self.H,
        )

        expert_names = self.expert_registry.list_experts()
        total = len(timestamps) * len(expert_names)
        pbar = tqdm(total=total, desc="Indexing", unit="rec")

        for ts in timestamps:
            logger.info("Processing timestamp: %s", ts.isoformat())
            tech_tensor, mkt_tensor = self.state_builder.build_state(
                ts, df_tech, df_mkt
            )
            embedding_np = self._compute_embedding(tech_tensor, mkt_tensor)

            forward_returns = self._get_forward_returns(ts, df)

            market_ctx = self._build_market_context(ts, df_mkt)

            for expert_name in expert_names:
                sop_text, sop_parsed = self._generate_sop(
                    ts=ts,
                    expert_name=expert_name,
                    df=df,
                    forward_returns=forward_returns,
                    market_ctx=market_ctx,
                )
                
                logger.info(
                    "Generated SoP for [%s] at %s: %s", 
                    expert_name, ts.isoformat(), sop_text
                )

                perf = self._evaluate_performance(
                    self._get_forward_weights(expert_name, ts, df),
                    forward_returns,
                )

                self.db.add_record(
                    embedding=embedding_np,
                    timestamp=ts.isoformat(),
                    expert_id=expert_name,
                    sop_text=sop_text,
                    cumulative_return=perf["cumulative_return"],
                    sharpe=perf["sharpe"],
                    drawdown=perf["max_drawdown"],
                    uncertainty_score=sop_parsed.get("confidence_bound_epsilon", 0.1),
                    extra=sop_parsed,
                )

                pbar.update(1)

        pbar.close()
        logger.info("Indexing complete – %d records stored", len(self.db))
        return len(self.db)


    # ------------------------------------------------------------------
    # Timestamp helpers
    # ------------------------------------------------------------------

    def _get_timestamps(
        self,
        df_tech: pd.DataFrame,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
    ) -> list[pd.Timestamp]:
        all_dates = sorted(df_tech.index.get_level_values("date").unique())
        # Need at least L days of lookback + H days forward
        min_date_idx = self.L - 1
        max_date_idx = len(all_dates) - self.H
        if max_date_idx <= min_date_idx:
            logger.warning(
                "Not enough data: need >= %d dates, have %d",
                self.L + self.H,
                len(all_dates),
            )
            return []

        available = all_dates[min_date_idx:max_date_idx]

        if start_date is not None:
            start = pd.Timestamp(start_date)
            available = [d for d in available if d >= start]
        if end_date is not None:
            end = pd.Timestamp(end_date)
            available = [d for d in available if d <= end]

        return available

    def _get_forward_returns(
        self, timestamp: pd.Timestamp, df: pd.DataFrame
    ) -> np.ndarray:
        """Get per-asset daily returns for days [τ+1, τ+H].

        Returns array of shape ``(H, n_assets)``.
        """
        all_dates = sorted(df.index.get_level_values("date").unique())
        symbols = sorted(df.index.get_level_values("symbol").unique())
        n_assets = len(symbols)

        if timestamp not in all_dates:
            return np.zeros((self.H, n_assets), dtype=np.float32)

        idx = all_dates.index(timestamp)
        fwd_dates = all_dates[idx + 1 : idx + 1 + self.H]

        returns_list: list[np.ndarray] = []
        for d in fwd_dates:
            day_data = df.loc[d]
            closes = day_data["close"].values.astype(np.float32)
            returns_list.append(closes)

        if not returns_list:
            return np.zeros((self.H, n_assets), dtype=np.float32)

        returns_arr = np.stack(returns_list, axis=0)

        # Compute price-relative returns if we have consecutive days
        if len(returns_arr) > 1:
            daily_returns = returns_arr[1:] / (returns_arr[:-1] + 1e-10) - 1.0
            daily_returns = np.clip(daily_returns, -0.5, 0.5)
            # Pad to H
            if len(daily_returns) < self.H:
                pad = np.zeros((self.H - len(daily_returns), n_assets), dtype=np.float32)
                daily_returns = np.concatenate([daily_returns, pad], axis=0)
            return daily_returns[:self.H]
        else:
            return np.zeros((self.H, n_assets), dtype=np.float32)

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _compute_embedding(self, tech_tensor: torch.Tensor, mkt_tensor: torch.Tensor) -> np.ndarray:
        tech_flat = tech_tensor.view(self.L, -1).unsqueeze(0).to(self.device)
        mkt_input = mkt_tensor.unsqueeze(0).to(self.device)
        h = self.vae.get_embedding(tech_flat, mkt_input)
        return h.cpu().numpy().astype(np.float32).flatten()

    # ------------------------------------------------------------------
    # Expert evaluation
    # ------------------------------------------------------------------

    def _get_expert_weights(self, expert_name: str, timestamp: pd.Timestamp, df: pd.DataFrame) -> np.ndarray:
        expert = self.expert_registry.get_expert(expert_name)
        state = self._build_expert_state(timestamp, df)
        return expert.get_weights(state)

    def _get_forward_weights(
        self, expert_name: str, timestamp: pd.Timestamp, df: pd.DataFrame
    ) -> list[np.ndarray]:
        """Return expert weights for each of the H forward days."""
        expert = self.expert_registry.get_expert(expert_name)
        all_dates = sorted(df.index.get_level_values("date").unique())
        if timestamp not in all_dates:
            n_assets = len(sorted(df.index.get_level_values("symbol").unique()))
            w = np.ones(n_assets + 1, dtype=np.float32) / (n_assets + 1)
            return [w]

        idx = all_dates.index(timestamp)
        fwd_dates = all_dates[idx: idx + self.H]

        weights: list[np.ndarray] = []
        for d in fwd_dates:
            state = self._build_expert_state(d, df)
            w = expert.get_weights(state)
            weights.append(w)
        return weights

    def _evaluate_performance(
        self,
        weights_list: list[np.ndarray],
        forward_returns: np.ndarray,
    ) -> dict[str, float]:
        """Compute cumulative return, Sharpe, max drawdown over H days.

        Uses daily rebalancing: each day's weights are applied to that day's returns.
        """
        n_steps = min(len(weights_list), len(forward_returns))
        if n_steps == 0:
            return {
                "cumulative_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }

        port_returns: list[float] = []
        for t in range(n_steps):
            w = weights_list[t]
            r = forward_returns[t]
            cash_w = w[0]
            asset_w = w[1 : 1 + len(r)]
            port_ret = cash_w * 0.0 + float(np.dot(asset_w, r))
            port_returns.append(port_ret)

        port_returns = np.array(port_returns, dtype=np.float64)
        cum_return = float(np.prod(1 + port_returns)) - 1.0
        vol = float(np.std(port_returns)) + 1e-10
        sharpe = float(np.mean(port_returns) / vol) * np.sqrt(252)

        cum = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / (running_max + 1e-10)
        max_dd = float(np.min(dd))

        return {
            "cumulative_return": cum_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }

    # ------------------------------------------------------------------
    # SoP generation
    # ------------------------------------------------------------------

    def _generate_sop(
        self,
        ts: pd.Timestamp,
        expert_name: str,
        df: pd.DataFrame,
        forward_returns: np.ndarray,
        market_ctx: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        cache_key = (ts.isoformat(), expert_name)
        if cache_key in self._sop_cache:
            cached = self._sop_cache[cache_key]
            return cached, json.loads(cached) if self._is_json(cached) else (cached, {})

        weights = self._get_expert_weights(expert_name, ts, df)
        perf = self._evaluate_performance([weights], forward_returns[:1])

        prompt = build_sop_prompt(
            timestamp=ts.isoformat(),
            market_context=market_ctx,
            expert_name=expert_name,
            allocation=weights.tolist(),
            performance_metrics=perf,
            uncertainty_estimate=0.1,
        )

        response_text = self._call_llm(prompt)

        parsed = parse_sop_response(response_text)
        sop_text = parsed.get("sop_text", response_text)

        self._sop_cache[cache_key] = response_text
        return sop_text, parsed

    def _call_llm(self, prompt: str) -> str:
        try:
            import litellm
            response = litellm.completion(
                model=self.llm_config.get("model_name", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are a financial performance analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.llm_config.get("temperature", 0.7),
                max_tokens=self.llm_config.get("max_tokens", 1024),
                api_base=self.llm_config.get("api_base"),
                api_key=self.llm_config.get("api_key"),
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("LLM call failed: %s – using fallback SoP", exc)
            return json.dumps({"sop_text": "Error generating SoP", "error": str(exc)})

    # ------------------------------------------------------------------
    # Market context
    # ------------------------------------------------------------------

    def _build_market_context(
        self, timestamp: pd.Timestamp, df_mkt: pd.DataFrame
    ) -> dict[str, Any]:
        if timestamp not in df_mkt.index:
            return {"volatility": "N/A", "trend": "N/A"}

        row = df_mkt.loc[timestamp]
        vol = row.get("mkt_volatility", None)
        mkt_ret = row.get("mkt_return", None)

        context: dict[str, Any] = {}
        if vol is not None:
            context["volatility"] = f"{float(vol):.6f}"
        else:
            context["volatility"] = "N/A"

        if mkt_ret is not None:
            trend = "bullish" if mkt_ret > 0 else "bearish"
            context["trend"] = trend
        else:
            context["trend"] = "N/A"

        return context

    # ------------------------------------------------------------------
    # Expert state construction
    # ------------------------------------------------------------------

    def _build_expert_state(
        self, timestamp: pd.Timestamp, df: pd.DataFrame
    ) -> np.ndarray:
        """Construct a state vector compatible with ``BaseExpert.get_weights``.
        
        The state must match the observation space of the trained models (302 dimensions).
        Since we are in offline indexing, we pad the state to the required size.
        """
        try:
            day_data = df.loc[timestamp]
        except KeyError:
            return np.zeros(302, dtype=np.float32)

        symbols = sorted(df.index.get_level_values("symbol").unique())
        n_assets = len(symbols)
        prices = day_data["close"].values.astype(np.float32)
        uniform_w = np.ones(n_assets + 1, dtype=np.float32) / (n_assets + 1)
        
        # Basic features
        tech_vals = []
        for col in day_data.columns:
            if col not in ("open", "high", "low", "close", "volume", "symbol"):
                vals = day_data[col].values
                if isinstance(vals, (int, float, np.floating)):
                    tech_vals.append(float(vals))
                else:
                    tech_vals.extend(float(v) for v in vals)
        
        state = np.concatenate([
            np.array([1.0], dtype=np.float32),
            prices,
            uniform_w,
            np.array(tech_vals, dtype=np.float32),
        ])
        
        # Pad or truncate to 302
        if len(state) < 302:
            state = np.pad(state, (0, 302 - len(state)), 'constant')
        else:
            state = state[:302]
            
        return state.astype(np.float32)


    @staticmethod
    def _is_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
