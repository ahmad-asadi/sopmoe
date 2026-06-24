#!/usr/bin/env python3
"""Production training script for DRL experts.
Loads real market data, computes technical indicators, and trains agents.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from typing import List

from src.data import DataLoader, FeatureEngineer
from src.experts.train import train_all_drl_agents
from src.utils.helpers import load_config, set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train DRL experts on real data")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps per agent")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data for training (0.0 to 1.0)")
    parser.add_argument("--market", type=str, default=None, help="Market name for checkpointing")
    parser.add_argument("--coin-list", type=str, default=None, help="Comma-separated list of coins to override config")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)

    if args.coin_list:
        cfg.data.coin_list = args.coin_list.split(",")

    logger.info("=" * 60)
    logger.info(f"EXPERT TRAINING PIPELINE - Market: {args.market or 'Default'}")
    logger.info("=" * 60)

    # 1. Data Loading
    logger.info("Loading raw data...")
    loader = DataLoader(
        coin_list=cfg.data.coin_list,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        data_dir=str(cfg.data.raw_dir),
        cache_dir=str(cfg.data.processed_dir),
    )
    df = loader.load_data()
    logger.info("Data loaded: %d rows", len(df))

    # 2. Feature Engineering
    logger.info("Computing technical features...")
    fe = FeatureEngineer()
    # Use market-specific cache to avoid conflicts
    cache_name = f"features_{args.market}.parquet" if args.market else "features.parquet"
    df_tech, df_mkt = fe.compute_features(
        df, cache_path=Path(cfg.data.processed_dir) / cache_name
    )
    logger.info("Features computed: tech %s, mkt %s", df_tech.shape, df_mkt.shape)

    # Merge tech features back into the main df
    tech_cols = fe.tech_feature_names
    df = df.join(df_tech[tech_cols])
    
    # 3. Train/Test Split
    all_dates = sorted(df.index.get_level_values("date").unique())
    split_idx = int(len(all_dates) * args.train_split)
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]
    
    train_end_date = train_dates[-1]
    test_start_date = test_dates[0]
    
    logger.info("Data split: Train dates [%s to %s], Test dates [%s to %s]", 
                all_dates[0].date(), train_end_date.date(), 
                test_start_date.date(), all_dates[-1].date())

    # Filter training data
    df_train = df[df.index.get_level_values("date") <= train_end_date]
    
    # 4. Training
    n_assets = len(cfg.data.coin_list)
    checkpoint_dir = Path("checkpoints") / (args.market if args.market else "default")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training for %d experts...", len(cfg.experts.list))
    saved_paths = train_all_drl_agents(
        df=df_train,
        stock_dim=n_assets,
        tech_indicator_list=fe.tech_feature_names,
        total_timesteps=args.timesteps,
        checkpoint_dir=checkpoint_dir,
    )

    for name, path in saved_paths.items():
        logger.info("Expert %s saved to %s", name, path)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    logger.info("EXPERT TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Data Loading
    logger.info("Loading raw data...")
    loader = DataLoader(
        coin_list=cfg.data.coin_list,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        data_dir=str(cfg.data.raw_dir),
        cache_dir=str(cfg.data.processed_dir),
    )
    df = loader.load_data()
    logger.info("Data loaded: %d rows", len(df))

    # 2. Feature Engineering
    # We compute features on the FULL dataset first to avoid edge effects at the split
    logger.info("Computing technical features...")
    fe = FeatureEngineer()
    df_tech, df_mkt = fe.compute_features(
        df, cache_path=Path(cfg.data.processed_dir) / "features.parquet"
    )
    logger.info("Features computed: tech %s, mkt %s", df_tech.shape, df_mkt.shape)

    # Merge tech features back into the main df to ensure PortfolioTradingEnv gets them
    # df is (date, symbol), df_tech is (date, symbol)
    # We only want to join the technical columns, not the OHLCV ones that are already in df
    tech_cols = fe.tech_feature_names
    df = df.join(df_tech[tech_cols])
    
    # 3. Train/Test Split
    all_dates = sorted(df.index.get_level_values("date").unique())
    split_idx = int(len(all_dates) * args.train_split)
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]
    
    train_end_date = train_dates[-1]
    test_start_date = test_dates[0]
    
    logger.info("Data split: Train dates [%s to %s], Test dates [%s to %s]", 
                all_dates[0].date(), train_end_date.date(), 
                test_start_date.date(), all_dates[-1].date())

    # Filter training data
    df_train = df[df.index.get_level_values("date") <= train_end_date]
    
    # 4. Training
    n_assets = len(cfg.data.coin_list)
    checkpoint_dir = Path("checkpoints")
    
    logger.info("Starting training for %d experts...", len(cfg.experts.list))
    saved_paths = train_all_drl_agents(
        df=df_train,
        stock_dim=n_assets,
        tech_indicator_list=fe.tech_feature_names,
        total_timesteps=args.timesteps,
        checkpoint_dir=checkpoint_dir,
    )

    for name, path in saved_paths.items():
        logger.info("Expert %s saved to %s", name, path)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
