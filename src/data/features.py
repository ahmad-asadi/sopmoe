import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.tech_feature_names: List[str] = [
            "log_return",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr",
            "realized_vol",
            "volume_change",
        ]
        self.market_feature_names: List[str] = [
            "mkt_return",
            "mkt_volatility",
            "mkt_liquidity",
        ]

    def compute_features(
        self, df: pd.DataFrame, cache_path: Optional[Path | str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if cache_path is not None:
            cache_path = Path(cache_path)
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached features from {cache_path}")
            df_tech = pd.read_parquet(cache_path)
            df_mkt = pd.read_parquet(cache_path.with_name("mkt_features.parquet"))
            return df_tech, df_mkt

        df = df.copy().sort_index()
        symbols = df.index.get_level_values("symbol").unique()

        tech_parts = []
        for sym in symbols:
            sdf = df.xs(sym, level="symbol").copy()
            sdf = self._add_technical(sdf)
            sdf["symbol"] = sym
            tech_parts.append(sdf.reset_index())

        df_tech = pd.concat(tech_parts, ignore_index=True)
        df_tech = df_tech.set_index(["date", "symbol"])
        df_tech = self._drop_leading_nan(df_tech)

        df_mkt = self._add_market_features(df_tech)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_tech.to_parquet(cache_path)
            df_mkt.to_parquet(cache_path.with_name("mkt_features.parquet"))
            logger.info(f"Cached features to {cache_path}")

        return df_tech, df_mkt

    def _add_technical(self, sdf: pd.DataFrame) -> pd.DataFrame:
        sdf = sdf.sort_index()

        sdf["log_return"] = np.log(sdf["close"] / sdf["close"].shift(1))

        sdf["rsi"] = self._rsi(sdf["close"], 14)

        macd_line, signal, hist = self._macd(sdf["close"], 12, 26, 9)
        sdf["macd"] = macd_line
        sdf["macd_signal"] = signal
        sdf["macd_hist"] = hist

        sdf["atr"] = self._atr(sdf["high"], sdf["low"], sdf["close"], 14)

        sdf["realized_vol"] = sdf["log_return"].rolling(20).std()

        sdf["volume_change"] = sdf["volume"].pct_change()

        return sdf

    def _add_market_features(self, df_tech: pd.DataFrame) -> pd.DataFrame:
        daily = df_tech[["log_return", "volume"]].groupby("date")
        df_mkt = pd.DataFrame({
            "mkt_return": daily["log_return"].mean(),
            "mkt_volatility": daily["log_return"].std(),
            "mkt_liquidity": daily["volume"].sum(),
        })
        return df_mkt

    def _drop_leading_nan(self, df_tech: pd.DataFrame) -> pd.DataFrame:
        symbols = df_tech.index.get_level_values("symbol").unique()
        first_valid = df_tech.index.get_level_values("date").min()
        for sym in symbols:
            sym_data = df_tech.xs(sym, level="symbol")
            fv = sym_data[self.tech_feature_names].first_valid_index()
            if fv and fv > first_valid:
                first_valid = fv
        df_tech = df_tech[df_tech.index.get_level_values("date") >= first_valid]
        return df_tech

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()
