import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from src.utils.logging import get_logger

logger = get_logger(__name__)

def plot_single_symbol(df, symbol, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.title(f"Price History: {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_market_aggregate(dfs, market_name, output_path):
    plt.figure(figsize=(15, 10))
    for symbol, df in dfs.items():
        # Normalize price to start at 1.0 for better comparison
        normalized_close = df['Close'] / df['Close'].iloc[0]
        plt.plot(df['Date'], normalized_close, label=symbol, alpha=0.7)
    
    plt.title(f"Normalized Price History: {market_name}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base 1.0)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def write_latex_table(markets, output_path):
    """Generates a LaTeX table with dataset statistics."""
    with open(output_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llll}\n")
        f.write("\\hline\n")
        f.write("Market & Symbol & Range & Samples \\\\ \\hline\n")
        
        for market_name, symbols in markets.items():
            for symbol, df in symbols.items():
                start_date = df['Date'].min().strftime('%Y-%m-%d')
                end_date = df['Date'].max().strftime('%Y-%m-%d')
                count = len(df)
                f.write(f"{market_name} & {symbol} & {start_date} to {end_date} & {count} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Market Data Summary}\n")
        f.write("\\label{tab:market_data}\n")
        f.write("\\end{table}\n")

def main():
    raw_data_dir = Path("./data/raw")
    output_dir = Path("./plots/raw_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_dir.exists():
        logger.error("Raw data directory not found. Please run scripts/download_data.py first.")
        return

    markets = {
        "Crypto": {},
        "NASDAQ": {},
        "Forex": {}
    }

    files = list(raw_data_dir.glob("*.csv"))
    logger.info(f"Found {len(files)} data files.")

    for file in files:
        symbol = file.stem
        try:
            df = pd.read_csv(file, parse_dates=["Date"])
            if df.empty:
                continue
            
            # Assign to market
            if "_X" in symbol:
                market = "Forex"
            elif "_USD" in symbol:
                market = "Crypto"
            else:
                market = "NASDAQ"
            
            markets[market][symbol] = df
            
            # Plot individual
            plot_single_symbol(df, symbol, output_dir / f"{symbol}.png")
            
        except Exception as e:
            logger.error(f"Failed to process {file}: {e}")

    # Plot aggregates
    for market_name, data in markets.items():
        if data:
            logger.info(f"Plotting aggregate for {market_name}...")
            plot_market_aggregate(data, market_name, output_dir / f"{market_name}_aggregate.png")

    # Write LaTeX table
    logger.info("Generating LaTeX table...")
    write_latex_table(markets, output_dir / "market_summary.tex")

    logger.info(f"All plots and table saved to {output_dir}")

if __name__ == "__main__":
    main()
