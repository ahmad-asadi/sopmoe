#!/usr/bin/env python3
"""Online inference pipeline entry point (Algorithm 2).

Usage:
    python scripts/run_inference.py

This script:
  1. Loads configuration, data, and the pre-trained VAE.
  2. Loads the FAISS + SQLite index.
  3. Creates an ``InferenceEngine`` with the expert registry.
  4. Runs a single test timestep and prints the selected expert + weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import DataLoader, FeatureEngineer, StateBuilder
from src.embedding.utils import load_vae_model
from src.experts.finrl_expert import DummyDRLExpert
from src.experts.registry import ExpertRegistry
from src.indexing.database import VectorDatabase
from src.inference.engine import InferenceEngine
from src.utils.helpers import load_config, set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_expert_registry(cfg) -> ExpertRegistry:
    registry = ExpertRegistry()
    n_assets = len(cfg.data.coin_list)
    for name in cfg.experts.list:
        registry.register(DummyDRLExpert(name=name, n_assets=n_assets))
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Online inference pipeline")
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Path to VAE checkpoint (default: latest in checkpoints/vae/)",
    )
    parser.add_argument(
        "--index-dir",
        default="data/index",
        help="Directory containing faiss.index and metadata.db",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use heuristic uncertainty instead of LLM",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Specific timestamp to test (default: latest available)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = load_config()
    set_seed(cfg.seed)
    logger.info("Configuration loaded")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir = cfg.data.raw_dir
    processed_dir = cfg.data.processed_dir

    loader = DataLoader(
        coin_list=cfg.data.coin_list,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        data_dir=str(data_dir),
        cache_dir=str(processed_dir),
    )
    df = loader.load_data()
    logger.info("Loaded data: %d rows", len(df))

    fe = FeatureEngineer()
    df_tech, df_mkt = fe.compute_features(
        df, cache_path=Path(processed_dir) / "features.parquet"
    )
    logger.info("Tech: %s, Market: %s", df_tech.shape, df_mkt.shape)

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------
    n_assets = len(cfg.data.coin_list)
    f1 = len(fe.tech_feature_names)
    f2 = len(fe.market_feature_names)
    L = cfg.window.L
    tech_dim = n_assets * f1
    mkt_dim = f2

    if args.vae_checkpoint:
        ckpt_path = args.vae_checkpoint
    else:
        ckpt_dir = Path(cfg.embedding.train.checkpoint_dir)
        ckpts = sorted(ckpt_dir.glob("vae_epoch_*.pt"))
        ckpt_path = str(ckpts[-1]) if ckpts else None

    if ckpt_path and Path(ckpt_path).exists():
        vae = load_vae_model(cfg, ckpt_path, tech_dim, mkt_dim, L=L)
        logger.info("VAE loaded from %s", ckpt_path)
    else:
        logger.warning("No VAE checkpoint – creating untrained model")
        from src.embedding.vae import DualStreamVAE
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = vae.to(device)
        vae.eval()
        logger.warning("Using untrained VAE")

    # ------------------------------------------------------------------
    # Expert registry
    # ------------------------------------------------------------------
    registry = build_expert_registry(cfg)

    # ------------------------------------------------------------------
    # Load index
    # ------------------------------------------------------------------
    index_dir = Path(args.index_dir)
    latent_dim = cfg.embedding.vae.latent_dim_tech + cfg.embedding.vae.latent_dim_mkt

    if (index_dir / "faiss.index").exists():
        db = VectorDatabase(
            dim=latent_dim,
            index_path=str(index_dir / "faiss.index"),
            db_path=str(index_dir / "metadata.db"),
        )
        logger.info("Index loaded: %d records", len(db))
    else:
        logger.warning("No index found at %s – using empty DB", index_dir)
        db = VectorDatabase(dim=latent_dim)

    # ------------------------------------------------------------------
    # Inference engine
    # ------------------------------------------------------------------
    llm_client = None if args.mock_llm else None  # Replace with real client if available

    engine = InferenceEngine(
        vae=vae,
        expert_registry=registry,
        db=db,
        llm_client=llm_client,
        config={
            "K": cfg.retrieval.K,
            "lambda_": cfg.uncertainty.lambda_,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    )

    # ------------------------------------------------------------------
    # Run one inference step
    # ------------------------------------------------------------------
    state_builder = StateBuilder(window_length=L)
    dates = sorted(df_tech.index.get_level_values("date").unique())
    test_date = pd.Timestamp(args.timestamp) if args.timestamp else dates[-1]

    logger.info("Running inference for %s", test_date)
    tech_tensor, mkt_tensor = state_builder.build_state(test_date, df_tech, df_mkt)

    # Build expert state from raw data
    expert_state = _build_expert_state(test_date, df, n_assets)

    selected, weights, meta = engine.select_expert(
        tech_state=tech_tensor,
        mkt_state=mkt_tensor,
        timestamp=test_date.isoformat(),
        expert_state=expert_state,
    )

    print("\n" + "=" * 60)
    print(f"Timestamp:     {test_date}")
    print(f"Selected expert: {selected}")
    print(f"Utility:        {meta['utility']:.4f}")
    print(f"All utilities:  {meta['utilities']}")
    print(f"Weighted returns: {meta['weighted_returns']}")
    print(f"Uncertainties:  {meta['uncertainties']}")
    print(f"Portfolio weights (sum={weights.sum():.4f}):")
    print(f"  {np.round(weights, 4)}")
    print("=" * 60)

    db.close()


def _build_expert_state(
    timestamp: pd.Timestamp, df: pd.DataFrame, n_assets: int
) -> np.ndarray:
    """Build a minimal state vector for ``BaseExpert.get_weights``."""
    try:
        day_data = df.loc[timestamp]
    except KeyError:
        return np.zeros(1 + n_assets + (n_assets + 1) + 2 * n_assets, dtype=np.float32)

    prices = day_data["close"].values.astype(np.float32)
    uniform_w = np.ones(n_assets + 1, dtype=np.float32) / (n_assets + 1)

    tech_cols = [c for c in day_data.columns
                 if c not in ("open", "high", "low", "close", "volume")]
    tech_vals: list[float] = []
    for ti in tech_cols[:2]:
        vals = day_data[ti].values
        if isinstance(vals, (int, float, np.floating)):
            tech_vals.append(float(vals))
        else:
            tech_vals.extend(float(v) for v in vals)

    return np.concatenate([
        np.array([1.0], dtype=np.float32),
        prices,
        uniform_w,
        np.array(tech_vals, dtype=np.float32),
    ]).astype(np.float32)


if __name__ == "__main__":
    import pandas as pd  # noqa: F811
    main()
