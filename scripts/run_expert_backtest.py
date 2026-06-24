#!/usr/bin/env python3
"""Expert evaluation pipeline.
Runs backtests for all trained experts and saves granular results.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import DataLoader, FeatureEngineer
from src.experts.registry import ExpertRegistry
from src.experts.finrl_expert import FinRLExpert
from src.backtest.engine import Backtester
from src.backtest.metrics import compute_metrics, metrics_dataframe
from src.backtest.plotting import plot_equity_curves, plot_drawdowns
from src.utils.helpers import load_config, set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Output configuration
EVAL_ROOT = Path("evaluations/expert_backtests")
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

def build_expert_registry(cfg, market=None) -> ExpertRegistry:
    registry = ExpertRegistry()
    n_assets = len(cfg.data.coin_list)
    checkpoint_dir = Path("checkpoints") / (market if market else "default")
    
    if not checkpoint_dir.exists():
        logger.warning("Checkpoint directory %s does not exist.", checkpoint_dir)
        return registry

    for name in cfg.experts.list:
        # Try various case and extension combinations
        possible_names = [name, name.upper(), f"{name.upper()}.zip", f"{name}.zip"]
        ckpt_path = None
        for pn in possible_names:
            p = checkpoint_dir / pn
            if p.exists():
                ckpt_path = p
                break
        
        if ckpt_path:
            logger.info("Loading expert %s from %s", name, ckpt_path)
            registry.register(FinRLExpert(name=name, model_path=ckpt_path, n_assets=n_assets))
        else:
            logger.warning("Checkpoint not found for %s at %s. Skipping.", name, checkpoint_dir)

    
    return registry

def main():
    parser = argparse.ArgumentParser(description="Expert Evaluation Pipeline")
    parser.add_argument("--test-start", default="2022-01-01", help="Start date for test period")
    parser.add_argument("--market", type=str, default=None, help="Market name for evaluation")
    parser.add_argument("--coin-list", type=str, default=None, help="Comma-separated list of coins to override config")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)

    if args.coin_list:
        cfg.data.coin_list = args.coin_list.split(",")

    # Set output directory based on market
    eval_root = Path("results") / (args.market if args.market else "default")
    eval_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"EXPERT EVALUATION PIPELINE - Market: {args.market or 'Default'}")
    logger.info("=" * 60)

    # 1. Data Preparation
    loader = DataLoader(
        coin_list=cfg.data.coin_list,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        data_dir=str(cfg.data.raw_dir),
        cache_dir=str(cfg.data.processed_dir),
    )
    df = loader.load_data()

    # 2. Expert Registry
    registry = build_expert_registry(cfg, market=args.market)
    if len(registry) == 0:
        logger.error("No experts loaded for market %s. Check checkpoints directory.", args.market)
        return

    # 3. Backtester Setup
    bt_config = {
        "initial_capital": float(cfg.backtest.initial_capital),
        "transaction_cost_pct": float(cfg.backtest.transaction_cost),
        "rebalance_freq": str(cfg.backtest.rebalance_freq),
        "lookback": int(cfg.window.L),
        "test_start": args.test_start,
    }
    
    backtester = Backtester(
        inference_engine=None,
        expert_registry=registry,
        data_loader=loader,
        config=bt_config,
    )

    # 4. Individual Expert Evaluation
    all_metrics = {}
    all_equity = {}

    for expert_name in registry.list_experts():
        logger.info("Evaluating expert: %s", expert_name)
        
        result = backtester.run_expert(expert_name, return_equity_curve=True)
        returns = result["daily_returns"]
        
        metrics = compute_metrics(returns)
        all_metrics[expert_name] = metrics
        all_equity[expert_name] = returns 
        
        expert_dir = eval_root / expert_name
        expert_dir.mkdir(parents=True, exist_ok=True)
        
        with open(expert_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        plot_equity_curves(
            {expert_name: returns}, 
            title=f"Equity Curve - {expert_name} ({args.market})", 
            save_path=expert_dir / "equity_curve.png",
            initial_capital=bt_config["initial_capital"]
        )
        
        plot_drawdowns(
            returns,
            strategy_name=expert_name,
            save_path=expert_dir / "drawdown.png"
        )

    # 5. Aggregated Evaluation
    logger.info("Generating aggregated results...")
    
    df_summary = metrics_dataframe(all_metrics)
    df_summary.to_csv(eval_root / "summary.csv")
    
    plot_equity_curves(
        all_equity,
        title=f"Aggregated Expert Performance - {args.market}",
        save_path=eval_root / "aggregate_equity_curves.png",
        initial_capital=bt_config["initial_capital"]
    )
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sharpes = [m["sharpe_ratio"] for m in all_metrics.values()]
    sns.boxplot(y=sharpes, color="skyblue")
    sns.stripplot(y=sharpes, color="blue", alpha=0.5)
    plt.title("Sharpe Ratio Distribution")
    plt.ylabel("Sharpe Ratio")
    
    plt.subplot(1, 2, 2)
    returns_vals = [m["cumulative_return"] for m in all_metrics.values()]
    sns.boxplot(y=returns_vals, color="salmon")
    sns.stripplot(y=returns_vals, color="red", alpha=0.5)
    plt.title("Cumulative Return Distribution")
    plt.ylabel("Total Return")
    
    plt.tight_layout()
    plt.savefig(eval_root / "metrics_distribution.png")
    plt.close()

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("Results saved to %s", eval_root)
    logger.info("=" * 60)

    logger.info("EXPERT EVALUATION PIPELINE")
    logger.info("=" * 60)

    # 1. Data Preparation
    loader = DataLoader(
        coin_list=cfg.data.coin_list,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        data_dir=str(cfg.data.raw_dir),
        cache_dir=str(cfg.data.processed_dir),
    )
    df = loader.load_data()

    # We need to compute features for the Backtester's internal _prepare_data
    # The Backtester does this internally, but we pass the loader.
    
    # 2. Expert Registry
    registry = build_expert_registry(cfg)
    if len(registry) == 0:
        logger.error("No experts loaded. Check checkpoints directory.")
        return

    # 3. Backtester Setup
    bt_config = {
        "initial_capital": float(cfg.backtest.initial_capital),
        "transaction_cost_pct": float(cfg.backtest.transaction_cost),
        "rebalance_freq": str(cfg.backtest.rebalance_freq),
        "lookback": int(cfg.window.L),
        "test_start": args.test_start,
    }
    
    # VAE/Inference Engine is ignored as per requirements
    backtester = Backtester(
        inference_engine=None,
        expert_registry=registry,
        data_loader=loader,
        config=bt_config,
    )

    # 4. Individual Expert Evaluation
    all_metrics = {}
    all_equity = {}

    for expert_name in registry.list_experts():
        logger.info("Evaluating expert: %s", expert_name)
        
        # Run backtest
        result = backtester.run_expert(expert_name, return_equity_curve=True)
        returns = result["daily_returns"]
        equity = result["equity_curve"]
        
        # Compute metrics
        metrics = compute_metrics(returns)
        all_metrics[expert_name] = metrics
        all_equity[expert_name] = returns # store returns for plotting
        
        # Save individual outputs
        expert_dir = EVAL_ROOT / expert_name
        expert_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics JSON
        with open(expert_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        # Equity Curve Plot
        plot_equity_curves(
            {expert_name: returns}, 
            title=f"Equity Curve - {expert_name}", 
            save_path=expert_dir / "equity_curve.png",
            initial_capital=bt_config["initial_capital"]
        )
        
        # Drawdown Plot
        plot_drawdowns(
            returns,
            strategy_name=expert_name,
            save_path=expert_dir / "drawdown.png"
        )

    # 5. Aggregated Evaluation
    logger.info("Generating aggregated results...")
    
    # Summary CSV
    df_summary = metrics_dataframe(all_metrics)
    df_summary.to_csv(EVAL_ROOT / "summary.csv")
    
    # Aggregate Equity Curve
    plot_equity_curves(
        all_equity,
        title="Aggregated Expert Performance",
        save_path=EVAL_ROOT / "aggregate_equity_curves.png",
        initial_capital=bt_config["initial_capital"]
    )
    
    # Metrics Distribution Plot
    plt.figure(figsize=(12, 6))
    
    # Plot Sharpe Ratios
    plt.subplot(1, 2, 1)
    sharpes = [m["sharpe_ratio"] for m in all_metrics.values()]
    sns.boxplot(y=sharpes, color="skyblue")
    sns.stripplot(y=sharpes, color="blue", alpha=0.5)
    plt.title("Sharpe Ratio Distribution")
    plt.ylabel("Sharpe Ratio")
    
    # Plot Cumulative Returns
    plt.subplot(1, 2, 2)
    returns_vals = [m["cumulative_return"] for m in all_metrics.values()]
    sns.boxplot(y=returns_vals, color="salmon")
    sns.stripplot(y=returns_vals, color="red", alpha=0.5)
    plt.title("Cumulative Return Distribution")
    plt.ylabel("Total Return")
    
    plt.tight_layout()
    plt.savefig(EVAL_ROOT / "metrics_distribution.png")
    plt.close()

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("Results saved to %s", EVAL_ROOT)
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
