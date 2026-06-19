#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full backtesting and evaluation pipeline.

Produces the four main result tables (Tables 3-6) from the paper:
    Table 3 - Fixed expert baselines
    Table 4 - Switching strategy comparison
    Table 5 - Uncertainty ablation (Return‑only vs Risk‑Aware)
    Table 6 - Component ablation (NR, R‑LLM, R+L‑NoU, Full)

Usage:
    python scripts/run_backtest.py [--mock-llm] [--vae-checkpoint PATH]
                                   [--index-dir DIR] [--no-plots]

Outputs:
    results/performance_summary.csv   - All metrics in one table
    results/table_3_fixed_experts.csv
    results/table_4_switching.csv
    results/table_5_uncertainty_ablation.csv
    results/table_6_component_ablation.csv
    plots/equity_curves_[...].png
    plots/drawdown_[...].png
    plots/metrics_comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import (
    Backtester,
    compute_metrics,
    metrics_dataframe,
    format_metrics_row,
    plot_equity_curves,
    plot_drawdowns,
    plot_metrics_comparison,
)
from src.data import DataLoader, FeatureEngineer, StateBuilder
from src.experts.finrl_expert import DummyDRLExpert
from src.experts.registry import ExpertRegistry
from src.utils.helpers import load_config, set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Expert registry
# ---------------------------------------------------------------------------

def build_expert_registry(cfg) -> ExpertRegistry:
    registry = ExpertRegistry()
    n_assets = len(cfg.data.coin_list)
    for name in cfg.experts.list:
        registry.register(DummyDRLExpert(name=name, n_assets=n_assets))
    # Also add the baselines that the paper compares against
    registry.register(DummyDRLExpert(name="uniform", n_assets=n_assets))
    logger.info("Registered %d experts: %s", len(registry), registry.list_experts())
    return registry


# ---------------------------------------------------------------------------
# VAE + Index loading helpers
# ---------------------------------------------------------------------------

def _load_vae(cfg, fe, L, device):
    n_assets = len(cfg.data.coin_list)
    f1 = len(fe.tech_feature_names)
    f2 = len(fe.market_feature_names)
    tech_dim = n_assets * f1
    mkt_dim = f2

    from src.embedding.utils import load_vae_model, DualStreamVAE

    ckpt_dir = Path(cfg.embedding.train.checkpoint_dir)
    ckpts = sorted(ckpt_dir.glob("vae_epoch_*.pt"))
    ckpt_path = str(ckpts[-1]) if ckpts else None

    if ckpt_path and Path(ckpt_path).exists():
        vae = load_vae_model(cfg, ckpt_path, tech_dim, mkt_dim, L=L)
        logger.info("VAE loaded from %s", ckpt_path)
    else:
        logger.warning("No VAE checkpoint - creating untrained model")
        vae_cfg = cfg.embedding.vae
        vae = DualStreamVAE(
            input_dim_tech=tech_dim,
            input_dim_mkt=mkt_dim,
            latent_dim_tech=vae_cfg.latent_dim_tech,
            latent_dim_mkt=vae_cfg.latent_dim_mkt,
            d_model=vae_cfg.d_model,
            nhead=vae_cfg.nhead,
            num_layers=vae_cfg.num_layers,
            dropout=vae_cfg.dropout,
            L=L,
        )
        vae = vae.to(device)
        vae.eval()
    return vae


def _load_index(cfg):
    from src.indexing.database import VectorDatabase

    latent_dim = cfg.embedding.vae.latent_dim_tech + cfg.embedding.vae.latent_dim_mkt
    index_dir = Path("data/index")
    if (index_dir / "faiss.index").exists():
        db = VectorDatabase(
            dim=latent_dim,
            index_path=str(index_dir / "faiss.index"),
            db_path=str(index_dir / "metadata.db"),
        )
        logger.info("Index loaded: %d records", len(db))
    else:
        logger.warning("No index found - using empty DB")
        db = VectorDatabase(dim=latent_dim)
    return db


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full backtesting evaluation")
    parser.add_argument("--mock-llm", action="store_true", help="Use heuristic uncertainty")
    parser.add_argument("--vae-checkpoint", default=None, help="Path to VAE checkpoint")
    parser.add_argument("--index-dir", default="data/index", help="Directory with index files")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config & seed
    # ------------------------------------------------------------------
    cfg = load_config()
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("BACKTESTING EVALUATION PIPELINE")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    loader = DataLoader(
        coin_list=cfg.data.coin_list,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        data_dir=str(cfg.data.raw_dir),
        cache_dir=str(cfg.data.processed_dir),
    )
    df = loader.load_data()
    logger.info("Data loaded: %d rows", len(df))

    fe = FeatureEngineer()
    df_tech, df_mkt = fe.compute_features(
        df, cache_path=Path(str(cfg.data.processed_dir)) / "features.parquet"
    )
    logger.info("Features computed: tech %s, mkt %s", df_tech.shape, df_mkt.shape)

    # ------------------------------------------------------------------
    # VAE + Index
    # ------------------------------------------------------------------
    L = cfg.window.L
    vae = _load_vae(cfg, fe, L, device)

    db = _load_index(cfg)

    # ------------------------------------------------------------------
    # Expert registry
    # ------------------------------------------------------------------
    registry = build_expert_registry(cfg)

    # ------------------------------------------------------------------
    # Inference engine
    # ------------------------------------------------------------------
    llm_client = None if args.mock_llm else None  # Replace with real LLM client

    from src.inference.engine import InferenceEngine

    engine = InferenceEngine(
        vae=vae,
        expert_registry=registry,
        db=db,
        llm_client=llm_client,
        config={
            "K": cfg.retrieval.K,
            "lambda_": cfg.uncertainty["lambda"],
            "device": device,
            "performance_metric": "cumulative_return",
        },
    )

    # ------------------------------------------------------------------
    # Backtester config
    # ------------------------------------------------------------------
    bt_config = {
        "initial_capital": float(cfg.backtest.initial_capital),
        "transaction_cost_pct": float(cfg.backtest.transaction_cost),
        "rebalance_freq": str(cfg.backtest.rebalance_freq),
        "lookback": int(cfg.window.L),
        "test_start": "2022-01-01",
    }

    backtester = Backtester(
        inference_engine=engine,
        expert_registry=registry,
        data_loader=loader,
        config=bt_config,
    )

    # ==================================================================
    # TABLE 3 - Fixed Expert Baselines
    # ==================================================================
    logger.info("\n" + "-" * 60)
    logger.info("TABLE 3 - Fixed Expert Baselines")
    logger.info("-" * 60)

    table3_results: dict[str, dict[str, float]] = {}
    all_equity: dict[str, np.ndarray] = {}

    for expert_name in registry.list_experts():
        logger.info("  Running expert: %s ...", expert_name)
        result = backtester.run_expert(expert_name, return_equity_curve=True)
        metrics = compute_metrics(result["daily_returns"])
        table3_results[expert_name] = metrics
        all_equity[expert_name] = result["equity_curve"]
        logger.info("    %s", format_metrics_row(metrics))

    df_table3 = metrics_dataframe(table3_results)
    df_table3.to_csv(RESULTS_DIR / "table_3_fixed_experts.csv")
    logger.info("Table 3 saved → results/table_3_fixed_experts.csv")
    _print_table(df_table3, title="Table 3 - Fixed Expert Baselines")

    # Add Best Fixed (the single best expert on test set by Sharpe)
    best_fixed_name = df_table3["sharpe_ratio"].idxmax()
    best_fixed_sharpe = df_table3.loc[best_fixed_name, "sharpe_ratio"]
    logger.info("  Best Fixed expert: %s (Sharpe=%.3f)", best_fixed_name, best_fixed_sharpe)

    # ==================================================================
    # TABLE 4 - Switching Strategy Comparison
    # ==================================================================
    logger.info("\n" + "-" * 60)
    logger.info("TABLE 4 - Switching Strategy Comparison")
    logger.info("-" * 60)

    switching_strategies = {
        "Best Fixed": ("best_fixed", False, False, False),
        "Recent Gating": ("recent_gating", False, False, False),
        "T-VAE": ("t_vae", True, False, False),
        "Proposed RAG": ("proposed_rag", True, True, True),
    }

    table4_results: dict[str, dict[str, float]] = {}
    switching_equity: dict[str, np.ndarray] = {}

    for label, (sname, use_ret, use_llm, use_unc) in switching_strategies.items():
        if label == "Best Fixed":
            eq = all_equity.get(best_fixed_name, np.array([]))
            ret = backtester.run_expert(best_fixed_name)["daily_returns"]
            metrics = compute_metrics(ret)
            table4_results[label] = metrics
            switching_equity[label] = eq
        else:
            logger.info("  Running switching: %s ...", label)
            result = backtester.run_switching(
                strategy_name=sname,
                use_retrieval=use_ret,
                use_llm=use_llm,
                use_uncertainty=use_unc,
                return_equity_curve=True,
            )
            metrics = compute_metrics(result["daily_returns"])
            table4_results[label] = metrics
            switching_equity[label] = result["equity_curve"]
        logger.info("    %s", format_metrics_row(metrics))

    df_table4 = metrics_dataframe(table4_results)
    df_table4.to_csv(RESULTS_DIR / "table_4_switching.csv")
    logger.info("Table 4 saved → results/table_4_switching.csv")
    _print_table(df_table4, title="Table 4 - Switching Strategy Comparison")

    # ==================================================================
    # TABLE 5 - Uncertainty Ablation
    # ==================================================================
    logger.info("\n" + "-" * 60)
    logger.info("TABLE 5 - Uncertainty Ablation")
    logger.info("-" * 60)

    uncertainty_variants = {
        "Return-Only (lambda=0)": ("return_only", True, True, False),
        "Risk-Aware (lambda={})".format(cfg.uncertainty["lambda"]): ("risk_aware", True, True, True),
    }

    table5_results: dict[str, dict[str, float]] = {}
    uncertainty_equity: dict[str, np.ndarray] = {}

    for label, (sname, use_ret, use_llm, use_unc) in uncertainty_variants.items():
        logger.info("  Running: %s ...", label)
        result = backtester.run_switching(
            strategy_name=sname,
            use_retrieval=use_ret,
            use_llm=use_llm,
            use_uncertainty=use_unc,
            return_equity_curve=True,
        )
        metrics = compute_metrics(result["daily_returns"])
        table5_results[label] = metrics
        uncertainty_equity[label] = result["equity_curve"]
        logger.info("    %s", format_metrics_row(metrics))

    df_table5 = metrics_dataframe(table5_results)
    df_table5.to_csv(RESULTS_DIR / "table_5_uncertainty_ablation.csv")
    logger.info("Table 5 saved → results/table_5_uncertainty_ablation.csv")
    _print_table(df_table5, title="Table 5 - Uncertainty Ablation")

    # ==================================================================
    # TABLE 6 - Component Ablation
    # ==================================================================
    logger.info("\n" + "-" * 60)
    logger.info("TABLE 6 - Component Ablation")
    logger.info("-" * 60)

    component_variants = {
        "NR (No Retrieval)": ("nr", False, False, False),
        "R-LLM (No LLM)": ("r_llm", True, False, True),
        "R+L-NoU (No Uncertainty)": ("r_l_nou", True, True, False),
        "Full (R+L+U)": ("full", True, True, True),
    }

    table6_results: dict[str, dict[str, float]] = {}
    ablation_equity: dict[str, np.ndarray] = {}

    for label, (sname, use_ret, use_llm, use_unc) in component_variants.items():
        logger.info("  Running: %s ...", label)
        result = backtester.run_switching(
            strategy_name=sname,
            use_retrieval=use_ret,
            use_llm=use_llm,
            use_uncertainty=use_unc,
            return_equity_curve=True,
        )
        metrics = compute_metrics(result["daily_returns"])
        table6_results[label] = metrics
        ablation_equity[label] = result["equity_curve"]
        logger.info("    %s", format_metrics_row(metrics))

    df_table6 = metrics_dataframe(table6_results)
    df_table6.to_csv(RESULTS_DIR / "table_6_component_ablation.csv")
    logger.info("Table 6 saved → results/table_6_component_ablation.csv")
    _print_table(df_table6, title="Table 6 - Component Ablation")

    # ==================================================================
    # Combined summary CSV
    # ==================================================================
    all_results = {}
    all_results.update(table3_results)
    all_results.update(table4_results)
    all_results.update(table5_results)
    all_results.update(table6_results)

    df_all = metrics_dataframe(all_results)
    df_all.to_csv(RESULTS_DIR / "performance_summary.csv")
    logger.info("Full summary saved → results/performance_summary.csv")
    _print_table(df_all, title="Combined Performance Summary")

    # ==================================================================
    # Plots
    # ==================================================================
    if not args.no_plots:
        logger.info("\n" + "-" * 60)
        logger.info("Generating plots ...")

        # Equity curves for switching strategies
        plot_equity_curves(
            switching_equity,
            title="Equity Curves - Switching Strategies",
            save_path=str(PLOTS_DIR / "equity_curves_switching.png"),
            initial_capital=float(cfg.backtest.initial_capital),
        )

        # Equity curves for component ablation
        plot_equity_curves(
            ablation_equity,
            title="Equity Curves - Component Ablation",
            save_path=str(PLOTS_DIR / "equity_curves_ablation.png"),
            initial_capital=float(cfg.backtest.initial_capital),
        )

        # Drawdown for Best Fixed and Proposed RAG
        for label in ["Best Fixed", "Proposed RAG"]:
            if label in switching_equity:
                eq = switching_equity[label]
                ret = np.diff(eq) / eq[:-1]
                plot_drawdowns(
                    ret,
                    strategy_name=label,
                    save_path=str(PLOTS_DIR / f"drawdown_{label.lower().replace(' ', '_')}.png"),
                )

        # Metrics comparison bar chart
        comparison_df = df_all.loc[
            [k for k in all_results if k in df_all.index]
        ]
        plot_metrics_comparison(
            comparison_df,
            title="Strategy Comparison - All Variants",
            save_path=str(PLOTS_DIR / "metrics_comparison.png"),
        )

        logger.info("Plots saved to %s", PLOTS_DIR)

    # ==================================================================
    # Summary
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("BACKTESTING COMPLETE")
    logger.info("=" * 60)
    logger.info("Results: %s", RESULTS_DIR)
    if not args.no_plots:
        logger.info("Plots:   %s", PLOTS_DIR)

    # Print key findings
    print("\n--- Key Comparisons ---")
    for table_name, df in [
        ("Table 4 (Switching)", df_table4),
        ("Table 6 (Ablation)", df_table6),
    ]:
        best_sharpe = df["sharpe_ratio"].idxmax()
        best_val = df.loc[best_sharpe, "sharpe_ratio"]
        print(f"  {table_name}: Best Sharpe = {best_sharpe} ({best_val:.3f})")

    # Validate: Full model should outperform R+L-NoU
    if "Full (R+L+U)" in df_table6.index and "R+L-NoU (No Uncertainty)" in df_table6.index:
        full_sharpe = df_table6.loc["Full (R+L+U)", "sharpe_ratio"]
        nou_sharpe = df_table6.loc["R+L-NoU (No Uncertainty)", "sharpe_ratio"]
        if full_sharpe >= nou_sharpe:
            logger.info("✓ Full model outperforms R+L-NoU (Sharpe: %.3f > %.3f)", full_sharpe, nou_sharpe)
        else:
            logger.warning("⚠ Full model does NOT outperform R+L-NoU (Sharpe: %.3f < %.3f)", full_sharpe, nou_sharpe)


def _print_table(df: pd.DataFrame, title: str = "") -> None:
    """Pretty-print a metrics DataFrame to stderr / log."""
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(f"\n{title}")
    print(df.to_string())
    print()


if __name__ == "__main__":
    main()
