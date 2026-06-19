#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimentation harness for sensitivity analysis.

Runs the full backtest over a grid of configurations (K x lambda x LLM model x
embedding dimension), saves individual results, and aggregates them into a
summary CSV with summary statistics.

Usage:
    # Run all experiment configs in src/config/experiment/
    python scripts/run_experiments.py

    # Run specific experiments only
    python scripts/run_experiments.py --experiments K5_lambda0.1 K10_lambda0.5

    # Run experiments with a specific LLM model
    python scripts/run_experiments.py --llm-model gpt-4

    # Dry run (validate configs without running backtest)
    python scripts/run_experiments.py --dry-run

Outputs:
    results/experiments/<exp_name>/metrics.csv   - Individual experiment results
    results/experiments/summary.csv              - Aggregated results all experiments
    results/experiments/summary_statistics.csv   - Mean/std across experiments
    results/experiments/sensitivity_heatmap.png  - Heatmap of Sharpe by K x lambda
    results/experiments/sensitivity_lines.png    - Line plots of metrics vs K
"""

from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import (
    Backtester,
    compute_metrics,
    metrics_dataframe,
    format_metrics_row,
)
from src.data import DataLoader, FeatureEngineer
from src.experts.finrl_expert import DummyDRLExpert
from src.experts.registry import ExpertRegistry
from src.utils.helpers import load_config, set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path("results/experiments")
PLOTS_DIR = Path("plots/experiments")

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

K_VALUES = [5, 10, 20]
LAMBDA_VALUES = [0.1, 0.5, 1.0]
LLM_MODELS = ["gpt-3.5-turbo", "gpt-4", "local-model"]

DEFAULT_EXPERIMENTS = [
    f"K{k}_lambda{l}" for k, l in itertools.product(K_VALUES, LAMBDA_VALUES)
]


def build_registry(cfg) -> ExpertRegistry:
    registry = ExpertRegistry()
    n_assets = len(cfg.data.coin_list)
    for name in cfg.experts.list:
        registry.register(DummyDRLExpert(name=name, n_assets=n_assets))
    registry.register(DummyDRLExpert(name="uniform", n_assets=n_assets))
    return registry


def load_vae_and_index(cfg, fe, L, device):
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
        logger.warning("No VAE checkpoint – creating untrained model")
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


def load_index(cfg):
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
        logger.warning("No index found – using empty DB")
        db = VectorDatabase(dim=latent_dim)
    return db


def load_data(cfg):
    loader = DataLoader(
        coin_list=cfg.data.coin_list,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        data_dir=str(cfg.data.raw_dir),
        cache_dir=str(cfg.data.processed_dir),
    )
    df = loader.load_data()
    fe = FeatureEngineer()
    df_tech, df_mkt = fe.compute_features(
        df, cache_path=Path(str(cfg.data.processed_dir)) / "features.parquet"
    )
    return loader, fe, df, df_tech, df_mkt


def run_single_experiment(
    cfg,
    exp_name: str,
    exp_dir: Path,
) -> dict[str, dict[str, float]]:
    """Run the full backtest for a single experiment configuration.

    Returns a dict mapping strategy names to metric dicts.
    """
    logger.info("=" * 60)
    logger.info("RUNNING EXPERIMENT: %s", exp_name)
    logger.info("  K=%s, lambda=%s, model=%s",
                cfg.retrieval.K, cfg.uncertainty["lambda"], cfg.llm.model_name)
    logger.info("=" * 60)

    exp_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    loader, fe, df, df_tech, df_mkt = load_data(cfg)

    L = cfg.window.L
    vae = load_vae_and_index(cfg, fe, L, device)
    db = load_index(cfg)
    registry = build_registry(cfg)

    llm_client = None

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

    all_results: dict[str, dict[str, float]] = {}

    # ---- Fixed experts ----
    logger.info("Running fixed experts ...")
    for expert_name in registry.list_experts():
        result = backtester.run_expert(expert_name)
        metrics = compute_metrics(result["daily_returns"])
        all_results[f"fixed_{expert_name}"] = metrics

    # ---- Switching strategies ----
    switching_strategies = {
        "Best Fixed": ("best_fixed", False, False, False),
        "Recent Gating": ("recent_gating", False, False, False),
        "T-VAE": ("t_vae", True, False, False),
        "Proposed RAG": ("proposed_rag", True, True, True),
    }
    logger.info("Running switching strategies ...")
    for label, (sname, use_ret, use_llm, use_unc) in switching_strategies.items():
        result = backtester.run_switching(
            strategy_name=sname,
            use_retrieval=use_ret,
            use_llm=use_llm,
            use_uncertainty=use_unc,
        )
        metrics = compute_metrics(result["daily_returns"])
        all_results[label] = metrics
        logger.info("  %s: %s", label, format_metrics_row(metrics))

    # ---- Uncertainty ablation ----
    uncertainty_variants = {
        "Return-Only (lambda=0)": ("return_only", True, True, False),
        f"Risk-Aware (lambda={cfg.uncertainty['lambda']})": (
            "risk_aware", True, True, True
        ),
    }
    logger.info("Running uncertainty ablation ...")
    for label, (sname, use_ret, use_llm, use_unc) in uncertainty_variants.items():
        result = backtester.run_switching(
            strategy_name=sname,
            use_retrieval=use_ret,
            use_llm=use_llm,
            use_uncertainty=use_unc,
        )
        metrics = compute_metrics(result["daily_returns"])
        all_results[label] = metrics
        logger.info("  %s: %s", label, format_metrics_row(metrics))

    # ---- Component ablation ----
    component_variants = {
        "NR (No Retrieval)": ("nr", False, False, False),
        "R-LLM (No LLM)": ("r_llm", True, False, True),
        "R+L-NoU (No Uncertainty)": ("r_l_nou", True, True, False),
        "Full (R+L+U)": ("full", True, True, True),
    }
    logger.info("Running component ablation ...")
    for label, (sname, use_ret, use_llm, use_unc) in component_variants.items():
        result = backtester.run_switching(
            strategy_name=sname,
            use_retrieval=use_ret,
            use_llm=use_llm,
            use_uncertainty=use_unc,
        )
        metrics = compute_metrics(result["daily_returns"])
        all_results[label] = metrics
        logger.info("  %s: %s", label, format_metrics_row(metrics))

    # ---- Save individual results ----
    df_exp = metrics_dataframe(all_results)
    df_exp.to_csv(exp_dir / "metrics.csv")
    logger.info("Results saved to %s", exp_dir / "metrics.csv")

    # Save config copy
    OmegaConf.save(cfg, exp_dir / "config.yaml")
    logger.info("Config saved to %s", exp_dir / "config.yaml")

    return all_results


def create_sensitivity_plots(summary_df: pd.DataFrame, save_dir: Path) -> None:
    """Generate heatmaps and line plots for sensitivity analysis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not available – skipping plots")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter to the Proposed RAG strategy for K × lambda sensitivity
    rag_df = summary_df[summary_df["strategy"] == "Proposed RAG"].copy()
    if rag_df.empty:
        logger.warning("No 'Proposed RAG' results for sensitivity plots")
        return

    # ---- Heatmap: Sharpe ratio as function of K and lambda ----
    if "K" in rag_df.columns and "lambda" in rag_df.columns:
        pivot = rag_df.pivot_table(
            index="lambda", columns="K", values="sharpe_ratio", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    ax=ax, cbar_kws={"label": "Sharpe Ratio"})
        ax.set_title("Sharpe Ratio Sensitivity: K × lambda (Proposed RAG)")
        ax.set_xlabel("K (number of neighbours)")
        ax.set_ylabel("lambda (uncertainty penalty)")
        fig.tight_layout()
        path = save_dir / "sensitivity_heatmap.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap saved to %s", path)

        # ---- Line plot: metrics vs K for each lambda ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for metric, ax, label in [
            ("sharpe_ratio", axes[0], "Sharpe Ratio"),
            ("annualised_return", axes[1], "Annualised Return"),
        ]:
            for lam in sorted(rag_df["lambda"].unique()):
                subset = rag_df[rag_df["lambda"] == lam].sort_values("K")
                ax.plot(subset["K"], subset[metric],
                        marker="o", label=f"lambda={lam}")
            ax.set_xlabel("K")
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs K")
            ax.legend()
        fig.tight_layout()
        path = save_dir / "sensitivity_lines.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Line plot saved to %s", path)

    # ---- Bar plot: LLM model comparison ----
    llm_df = summary_df[
        summary_df["strategy"] == "Proposed RAG"
    ].dropna(subset=["llm_model"])
    if not llm_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = llm_df.groupby("llm_model")["sharpe_ratio"].agg(["mean", "std"])
        grouped.plot(kind="bar", y="mean", yerr="std", ax=ax,
                     capsize=4, legend=False, color="steelblue")
        ax.set_title("Sharpe Ratio by LLM Model (Proposed RAG)")
        ax.set_xlabel("LLM Model")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        path = save_dir / "llm_sensitivity.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("LLM sensitivity plot saved to %s", path)


def aggregate_results(experiments: list[dict], results_dir: Path) -> pd.DataFrame:
    """Aggregate all experiment results into a single DataFrame.

    Reads results from both the freshly-completed experiments list and any
    existing experiment directories under ``results_dir``.
    """
    rows = []

    seen_dirs: set[Path] = set()
    for exp in experiments:
        seen_dirs.add(exp["dir"])
        exp_name = exp["name"]
        exp_dir = exp["dir"]
        csv_path = exp_dir / "metrics.csv"
        if not csv_path.exists():
            logger.warning("Missing metrics file: %s", csv_path)
            continue
        df = pd.read_csv(csv_path, index_col=0)
        for strategy in df.index:
            row = df.loc[strategy].to_dict()
            row["experiment"] = exp_name
            row["strategy"] = strategy
            row["K"] = exp.get("K", None)
            row["lambda"] = exp.get("lambda", None)
            row["llm_model"] = exp.get("llm_model", None)
            row["embed_dim"] = exp.get("embed_dim", None)
            rows.append(row)

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child in seen_dirs:
            continue
        csv_path = child / "metrics.csv"
        if not csv_path.exists():
            continue
        exp_name = child.name
        exp_meta = parse_experiment_name(exp_name)
        df = pd.read_csv(csv_path, index_col=0)
        for strategy in df.index:
            row = df.loc[strategy].to_dict()
            row["experiment"] = exp_name
            row["strategy"] = strategy
            row["K"] = exp_meta.get("K", None)
            row["lambda"] = exp_meta.get("lambda", None)
            row["llm_model"] = exp_meta.get("llm_model", None)
            row["embed_dim"] = exp_meta.get("embed_dim", None)
            rows.append(row)

    if not rows:
        logger.warning("No results to aggregate")
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    return summary


def compute_summary_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std across experiment runs for each strategy."""
    if summary.empty:
        return pd.DataFrame()

    metric_cols = [
        "cumulative_return", "annualised_return", "annualised_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown",
    ]
    available = [c for c in metric_cols if c in summary.columns]

    stats_list = []
    for strategy in summary["strategy"].unique():
        subset = summary[summary["strategy"] == strategy]
        stats = {"strategy": strategy}
        for col in available:
            vals = pd.to_numeric(subset[col], errors="coerce").dropna()
            if len(vals) > 0:
                stats[f"{col}_mean"] = vals.mean()
                stats[f"{col}_std"] = vals.std()
            else:
                stats[f"{col}_mean"] = None
                stats[f"{col}_std"] = None
        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def parse_experiment_name(name: str) -> dict:
    """Extract structured metadata from experiment name (e.g. K5_lambda0.1)."""
    meta: dict = {"name": name}
    parts = name.split("_")
    for part in parts:
        if part.startswith("K"):
            try:
                meta["K"] = int(part[1:])
            except ValueError:
                pass
        elif part.startswith("lambda"):
            try:
                meta["lambda"] = float(part[6:])
            except ValueError:
                pass
        elif part.startswith("llm_"):
            meta["llm_model"] = part[4:]
        elif part.startswith("embed"):
            try:
                meta["embed_dim"] = int(part.replace("embed", "").replace("dim", ""))
            except ValueError:
                pass
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimentation harness for sensitivity analysis"
    )
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="List of experiment config names (e.g. K5_lambda0.1). "
             "If omitted, runs all configs in src/config/experiment/.",
    )
    parser.add_argument(
        "--llm-model", default=None,
        help="Override LLM model for all experiments (e.g. gpt-4).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configs without running backtests.",
    )
    parser.add_argument(
        "--config-dir", default="src/config/experiment",
        help="Directory containing experiment override YAML files.",
    )
    parser.add_argument(
        "--base-config", default="src/config",
        help="Directory containing the base config.yaml.",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        logger.error("Config directory not found: %s", config_dir)
        sys.exit(1)

    # Determine which experiment configs to run
    if args.experiments is not None:
        exp_names = args.experiments
    else:
        exp_names = []
        for f in sorted(config_dir.iterdir()):
            if f.suffix in (".yaml", ".yml") and f.name != "default.yaml":
                exp_names.append(f.stem)

    if not exp_names:
        logger.warning("No experiment configs found in %s", config_dir)
        return

    logger.info("Found %d experiment config(s): %s", len(exp_names), exp_names)

    # Override LLM model if specified
    llm_override = {}
    if args.llm_model:
        llm_override = {"llm": {"model_name": args.llm_model}}

    completed_experiments: list[dict] = []
    failed_experiments: list[str] = []

    for exp_name in exp_names:
        exp_meta = parse_experiment_name(exp_name)
        exp_dir = RESULTS_DIR / exp_name
        exp_config_path = config_dir / f"{exp_name}.yaml"

        if not exp_config_path.exists():
            logger.warning("Config file not found: %s – skipping", exp_config_path)
            failed_experiments.append(exp_name)
            continue

        if args.dry_run:
            logger.info("[DRY RUN] Would run %s from %s", exp_name, exp_config_path)
            continue

        try:
            # Load base config
            cfg = load_config(config_path=str(args.base_config), config_name="config")

            # Load and merge experiment override
            override = OmegaConf.load(exp_config_path)
            if llm_override:
                override = OmegaConf.merge(override, OmegaConf.create(llm_override))
            cfg = OmegaConf.merge(cfg, override)

            # Run experiment
            _ = run_single_experiment(cfg, exp_name, exp_dir)

            completed_experiments.append({
                **exp_meta,
                "dir": exp_dir,
            })
            logger.info("✓ Experiment '%s' completed successfully", exp_name)

        except Exception as exc:
            logger.error("Experiment '%s' failed: %s", exp_name, exc)
            traceback.print_exc()
            failed_experiments.append(exp_name)

    # ---- Aggregation ----
    if not completed_experiments:
        logger.warning("No experiments completed successfully")
        return

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATING RESULTS")
    logger.info("=" * 60)

    summary = aggregate_results(completed_experiments, RESULTS_DIR)
    if summary.empty:
        logger.warning("No results to aggregate")
        return

    # Save full summary
    summary_csv = RESULTS_DIR / "summary.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    logger.info("Full summary saved to %s", summary_csv)

    # Compute and save summary statistics
    stats = compute_summary_statistics(summary)
    if not stats.empty:
        stats_csv = RESULTS_DIR / "summary_statistics.csv"
        stats.to_csv(stats_csv, index=False)
        logger.info("Summary statistics saved to %s", stats_csv)
        print("\n--- Summary Statistics ---")
        pd.set_option("display.max_columns", 12)
        pd.set_option("display.width", 160)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print(stats.to_string(index=False))

    # ---- Visualisation ----
    create_sensitivity_plots(summary, PLOTS_DIR)

    # ---- Summary ----
    n_ok = len(completed_experiments)
    n_fail = len(failed_experiments)
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENTATION COMPLETE")
    logger.info("  Successful: %d", n_ok)
    logger.info("  Failed:     %d", n_fail)
    logger.info("  Results:    %s", RESULTS_DIR)
    logger.info("  Plots:      %s", PLOTS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
