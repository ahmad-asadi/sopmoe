import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


class StateBuilder:
    def __init__(
        self,
        window_length: int = 30,
        tech_feature_names: Optional[List[str]] = None,
        market_feature_names: Optional[List[str]] = None,
    ):
        self.L = window_length
        self.tech_feature_names = tech_feature_names or [
            "log_return", "rsi", "macd", "macd_signal", "macd_hist",
            "atr", "realized_vol", "volume_change",
        ]
        self.market_feature_names = market_feature_names or [
            "mkt_return", "mkt_volatility", "mkt_liquidity",
        ]

    def build_state(
        self, timestamp: pd.Timestamp, df_tech: pd.DataFrame, df_mkt: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start = timestamp - pd.Timedelta(days=self.L - 1)
        end = timestamp

        dates = pd.date_range(start, end, freq="D")
        symbols = sorted(df_tech.index.get_level_values("symbol").unique())
        A = len(symbols)
        F1 = len(self.tech_feature_names)
        F2 = len(self.market_feature_names)

        S_tech = np.full((self.L, A, F1), np.nan)
        S_mkt = np.full((self.L, F2), np.nan)

        for i, date in enumerate(dates):
            for j, sym in enumerate(symbols):
                try:
                    row = df_tech.loc[(date, sym)]
                    S_tech[i, j, :] = [row.get(f, np.nan) for f in self.tech_feature_names]
                except (KeyError, TypeError):
                    pass
            if date in df_mkt.index:
                S_mkt[i, :] = df_mkt.loc[date, self.market_feature_names].values

        S_tech, S_mkt = self._zscore_normalize(S_tech, S_mkt)

        return torch.from_numpy(S_tech).float(), torch.from_numpy(S_mkt).float()

    def build_states(
        self, df_tech: pd.DataFrame, df_mkt: pd.DataFrame
    ) -> List[Tuple[torch.Tensor, torch.Tensor, pd.Timestamp]]:
        dates = sorted(df_tech.index.get_level_values("date").unique())
        states = []
        for t in dates[self.L - 1 :]:
            S_tech, S_mkt = self.build_state(t, df_tech, df_mkt)
            states.append((S_tech, S_mkt, t))
        logger.info(f"Built {len(states)} states")
        return states

    def _zscore_normalize(
        self, S_tech: np.ndarray, S_mkt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        eps = 1e-8
        mean_t = np.nanmean(S_tech, axis=0, keepdims=True)
        std_t = np.nanstd(S_tech, axis=0, keepdims=True)
        S_tech = np.where(std_t > eps, (S_tech - mean_t) / std_t, 0.0)
        mean_m = np.nanmean(S_mkt, axis=0, keepdims=True)
        std_m = np.nanstd(S_mkt, axis=0, keepdims=True)
        S_mkt = np.where(std_m > eps, (S_mkt - mean_m) / std_m, 0.0)
        S_tech = np.nan_to_num(S_tech, nan=0.0)
        S_mkt = np.nan_to_num(S_mkt, nan=0.0)
        return S_tech, S_mkt
