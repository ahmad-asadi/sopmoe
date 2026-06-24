#!/usr/bin/env python3
"""Offline indexing pipeline entry point (Algorithm 1).

Usage:
    python scripts/run_indexing.py  [--mock-llm]  [--start-date DATE] [--end-date DATE]

This script:
  1. Loads configuration and data
  2. Builds technical / market features
  3. Loads the pre-trained VAE model
  4. Creates or loads expert registry
  5. Runs the indexing engine (builds SoPs, stores embeddings + metadata)
  6. Saves the FAISS index and SQLite database to ``data/index/``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import DataLoader, FeatureEngineer, StateBuilder
from src.embedding.utils import load_vae_model
from src.experts.registry import ExpertRegistry
from src.experts.finrl_expert import FinRLExpert
from src.indexing.database import VectorDatabase
from src.indexing.engine import IndexingEngine
from src.utils.helpers import load_config, set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_expert_registry(cfg) -> ExpertRegistry:
    """Populate the registry with trained or dummy experts."""
    registry = ExpertRegistry()
    n_assets = len(cfg.data.coin_list)

    # Use FinRLExpert and load models from checkpoints directory
    checkpoint_dir = Path("checkpoints")
    for name in cfg.experts.list:
        model_path = checkpoint_dir / name
        registry.register(FinRLExpert(name=name, model_path=model_path, n_assets=n_assets))

    logger.info("Registered %d experts", len(registry))
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline indexing pipeline")
    parser.add_argument("--mock-llm", action="store_true", help="Skip real LLM calls")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Override start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Override end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Path to VAE checkpoint (default: latest in checkpoints/vae/)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = load_config()
    set_seed(cfg.seed)

    logger.info("Configuration loaded")
    logger.info("mock_llm = %s", args.mock_llm)

    # ------------------------------------------------------------------
    # Data loading & feature engineering
    # ------------------------------------------------------------------
    data_dir = cfg.data.raw_dir
    processed_dir = cfg.data.processed_dir

    loader = DataLoader(
        coin_list=cfg.data.coin_list,
        start_date=args.start_date or cfg.data.start_date,
        end_date=args.end_date or cfg.data.end_date,
        data_dir=str(data_dir),
        cache_dir=str(processed_dir),
    )
    df = loader.load_data()
    logger.info("Loaded data: %d rows", len(df))

    fe = FeatureEngineer()
    df_tech, df_mkt = fe.compute_features(df, cache_path=Path(processed_dir) / "features.parquet")
    logger.info("Tech features: %s, Market features: %s", df_tech.shape, df_mkt.shape)

    # ------------------------------------------------------------------
    # VAE model
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
        logger.warning("No VAE checkpoint found – creating untrained model")
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
        logger.warning("Using untrained VAE – embeddings will be random")

    # ------------------------------------------------------------------
    # Expert registry
    # ------------------------------------------------------------------
    registry = build_expert_registry(cfg)

    # ------------------------------------------------------------------
    # Vector database
    # ------------------------------------------------------------------
    latent_dim = cfg.embedding.vae.latent_dim_tech + cfg.embedding.vae.latent_dim_mkt
    index_dir = Path("data/index")
    index_dir.mkdir(parents=True, exist_ok=True)

    db = VectorDatabase(
        dim=latent_dim,
        index_path=index_dir / "faiss.index",
        db_path=index_dir / "metadata.db",
    )
    logger.info("Vector database initialised (dim=%d)", latent_dim)

    # ------------------------------------------------------------------
    # State builder
    # ------------------------------------------------------------------
    state_builder = StateBuilder(window_length=L)

    # ------------------------------------------------------------------
    # Indexing engine
    # ------------------------------------------------------------------
    engine = IndexingEngine(
        vae=vae,
        expert_registry=registry,
        state_builder=state_builder,
        db=db,
        config={
            "H": cfg.window.H,
            "L": L,
            "mock_llm": args.mock_llm,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "llm": {
                "model_name": cfg.llm.model_name,
                "api_base": cfg.llm.api_base,
                "api_key": cfg.llm.api_key,
                "temperature": cfg.llm.temperature,
                "max_tokens": cfg.llm.max_tokens,
            },
        },
    )

    n_records = engine.run_indexing(
        df=df,
        df_tech=df_tech,
        df_mkt=df_mkt,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    db.save(
        index_path=index_dir / "faiss.index",
        db_path=index_dir / "metadata.db",
    )

    # Quick sanity query
    if n_records > 0:
        sample_emb = np.random.randn(latent_dim).astype(np.float32)
        results = db.query_similar(sample_emb, k=3)
        logger.info("Sanity query returned %d results", len(results))
        for r in results[:2]:
            logger.info(
                "  ts=%s expert=%s ret=%.4f sharpe=%.2f dd=%.4f",
                r["timestamp"],
                r["expert_id"],
                r["cumulative_return"],
                r["sharpe"],
                r["drawdown"],
            )

    db.close()
    logger.info("Indexing pipeline complete – %d records stored", n_records)


if __name__ == "__main__":
    main()
