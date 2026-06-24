import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataLoader:
    def __init__(
        self,
        coin_list: List[str],
        start_date: str,
        end_date: str,
        data_dir: str = "./data/raw",
        cache_dir: Optional[str] = None,
    ):
        self.coin_list = coin_list
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/processed")

    def load_data(self) -> pd.DataFrame:
        dfs = []
        for coin in self.coin_list:
            path = self.data_dir / f"{coin}.csv"
            if path.exists():
                df = pd.read_csv(path, parse_dates=["Date"])
                logger.info(f"Loaded {coin} from {path}")
            else:
                logger.error(f"Data file {path} not found. Please run scripts/download_data.py first.")
                continue
            
            df["symbol"] = coin
            dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError("No data files were found in the data directory.")
            
        df = pd.concat(dfs, ignore_index=True)
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
        df = self._handle_missing(df)
        df = df.set_index(["date", "symbol"])
        logger.info(f"Final data shape: {df.shape}")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        date_range = pd.date_range(self.start_date, self.end_date, freq="D")
        symbols = sorted(df["symbol"].unique())
        full_idx = pd.MultiIndex.from_product([date_range, symbols], names=["date", "symbol"])
        df = df.set_index(["date", "symbol"]).reindex(full_idx)
        ohlcv = ["open", "high", "low", "close", "volume"]
        for sym in symbols:
            idx = pd.IndexSlice[:, sym]
            df.loc[idx, ohlcv] = df.loc[idx, ohlcv].ffill()
        complete = df[ohlcv].notna().groupby("date").all()
        valid_dates = complete[complete.all(axis=1)].index
        df = df.loc[valid_dates].reset_index()
        logger.info(f"After handling missing — {len(valid_dates)} valid dates, {len(symbols)} symbols")
        return df

    @staticmethod
    def split_data(

        df: pd.DataFrame,
        train_end: str = "2020-12-31",
        val_end: str = "2021-12-31",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dates = df.index.get_level_values("date")
        train = df[dates < pd.Timestamp(train_end)]
        val = df[(dates >= pd.Timestamp(train_end)) & (dates < pd.Timestamp(val_end))]
        test = df[dates >= pd.Timestamp(val_end)]
        logger.info(
            f"Train: {len(train)} rows, Val: {len(val)} rows, Test: {len(test)} rows"
        )
        return train, val, test
