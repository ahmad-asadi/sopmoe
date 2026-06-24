import ccxt
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DataDownloader:
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _standardize_df(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Ensure DF has columns: Date, Open, High, Low, Close, Volume"""
        df = df.copy()
        # Reset index to get Date/Timestamp as a column
        df = df.reset_index()
        
        # Rename the index column to 'Date'
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        
        # Normalize column names to Title Case
        cols = {col.lower(): col.capitalize() for col in df.columns}
        # Special case for 'date' which should remain 'Date'
        cols['date'] = 'Date'
        df.rename(columns=cols, inplace=True)
        
        # Ensure required columns exist
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                df[col] = 0.0 # Volume can be 0 for Forex
        
        return df[required]

    def fetch_crypto(self, start_date: str, end_date: str):
        logger.info("Fetching Crypto data from Yahoo Finance...")
        symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
            'DOGE-USD', 'ADA-USD', 'TRX-USD', 'AVAX-USD', 'SHIB-USD',
            'DOT-USD', 'LINK-USD', 'BCH-USD', 'LTC-USD', 'NEAR-USD',
            'MATIC-USD', 'UNI-USD', 'ATOM-USD', 'ETC-USD', 'XLM-USD',
            'TON-USD', 'APT-USD', 'ARB-USD', 'VET-USD', 'FIL-USD',
            'ICP-USD', 'MKR-USD', 'RNDR-USD', 'EGLD-USD', 'HBAR-USD'
        ]
        try:
            data = yf.download(symbols, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=False)
            for sym in symbols:
                if len(symbols) > 1:
                    if sym in data.columns.levels[0]:
                        df = data[sym].dropna(how='all')
                    else:
                        continue
                else:
                    df = data.dropna(how='all')
                
                if not df.empty:
                    std_df = self._standardize_df(df, sym)
                    filename = sym.replace('-', '_') + ".csv"
                    std_df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"Saved {sym} to {filename}")
        except Exception as e:
            logger.error(f"Failed to fetch Crypto data: {e}")

    def fetch_nasdaq(self, start_date: str, end_date: str):
        logger.info("Fetching NASDAQ data from Yahoo Finance...")
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'JPM', 'V', 'XOM', 'UNH', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'CVX',
            'BAC', 'KO', 'PEP', 'COST', 'TMO', 'ABBV', 'ACN', 'MRK', 'CRM',
            'DIS', 'MCD', 'LIN'
        ]
        try:
            data = yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=False)
            for ticker in tickers:
                # handle both single and multi-ticker returns from yfinance
                if len(tickers) > 1:
                    if ticker in data.columns.levels[0]:
                        df = data[ticker].dropna(how='all')
                    else:
                        continue
                else:
                    df = data.dropna(how='all')
                
                if not df.empty:
                    std_df = self._standardize_df(df, ticker)
                    std_df.to_csv(self.data_dir / f"{ticker}.csv", index=False)
                    logger.info(f"Saved {ticker}.csv")
        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ data: {e}")

    def fetch_forex(self, start_date: str, end_date: str):
        logger.info("Fetching Forex data from Yahoo Finance...")
        pairs = [
            'EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X',
            'USDCHF=X', 'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X',
            'EURAUD=X', 'EURCHF=X', 'EURCAD=X', 'EURNZD=X', 'GBPCHF=X',
            'GBPAUD=X', 'GBPCAD=X', 'GBPNZD=X', 'AUDJPY=X', 'AUDCHF=X',
            'AUDCAD=X', 'AUDNZD=X', 'CADJPY=X', 'CHFJPY=X', 'NZDJPY=X',
            'NZDCHF=X', 'NZDCAD=X', 'EURTRY=X', 'USDTRY=X', 'EURSEK=X'
        ]
        try:
            data = yf.download(pairs, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=False)
            for pair in pairs:
                if len(pairs) > 1:
                    if pair in data.columns.levels[0]:
                        df = data[pair].dropna(how='all')
                    else:
                        continue
                else:
                    df = data.dropna(how='all')

                if not df.empty:
                    std_df = self._standardize_df(df, pair)
                    filename = pair.replace('=', '_') + ".csv"
                    std_df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"Saved {filename}")
        except Exception as e:
            logger.error(f"Failed to fetch Forex data: {e}")

    def download_all(self, start_date: str, end_date: str):
        self.fetch_crypto(start_date, end_date)
        self.fetch_nasdaq(start_date, end_date)
        self.fetch_forex(start_date, end_date)
        logger.info("All data downloads completed.")
