"""Utility functions for the Dual-Stream Transformer-VAE embedding module."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig

from src.embedding.vae import DualStreamVAE
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_vae_model(
    cfg: DictConfig,
    checkpoint_path: str | Path,
    tech_dim: int,
    mkt_dim: int,
    L: int = 60,
    device: str | None = None,
) -> DualStreamVAE:
    """Load a pretrained ``DualStreamVAE`` from a checkpoint file.

    Parameters
    ----------
    cfg :
        Hydra configuration (must contain ``embedding.vae`` key).
    checkpoint_path :
        Path to the ``.pt`` checkpoint.
    tech_dim :
        Flattened technical feature dimension ``A * F1``.
    mkt_dim :
        Market feature dimension ``F2``.
    L :
        Window length used during training.
    device :
        Target device (default: auto-detect CUDA / CPU).

    Returns
    -------
    DualStreamVAE
        Model in evaluation mode with loaded weights.
    """
    model = DualStreamVAE.from_config(cfg, tech_dim, mkt_dim, L=L)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded VAE checkpoint from %s (epoch %d, loss=%.4f)",
        checkpoint_path,
        state.get("epoch", -1),
        state.get("loss", float("nan")),
    )
    return model


@torch.inference_mode()
def compute_embeddings(
    model: DualStreamVAE,
    tech_states: List[torch.Tensor],
    mkt_states: List[torch.Tensor],
    batch_size: int = 256,
    device: str | None = None,
) -> torch.Tensor:
    """Pre-compute and cache embeddings for a list of historical states.

    Parameters
    ----------
    model :
        Trained ``DualStreamVAE`` in evaluation mode.
    tech_states :
        List of tech tensors, each shape ``(L, A*F1)``.
    mkt_states :
        List of market tensors, each shape ``(L, F2)``.
    batch_size :
        Batch size for batched inference.
    device :
        Compute device (default: auto-detect).

    Returns
    -------
    Tensor of shape ``(N, latent_dim_tech + latent_dim_mkt)`` with
    all embeddings stacked.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    N = len(tech_states)
    embeddings = []

    for i in range(0, N, batch_size):
        tech_batch = torch.stack(tech_states[i : i + batch_size]).to(device)
        mkt_batch = torch.stack(mkt_states[i : i + batch_size]).to(device)
        h = model.get_embedding(tech_batch, mkt_batch)
        embeddings.append(h.cpu())

    logger.info("Computed %d embeddings", N)
    return torch.cat(embeddings, dim=0)


def cache_embeddings(
    embeddings: torch.Tensor,
    save_path: str | Path,
    timestamps: Optional[List[str]] = None,
) -> None:
    """Save pre-computed embeddings to disk for later use by the indexer.

    Parameters
    ----------
    embeddings :
        Tensor of shape ``(N, D)``.
    save_path :
        Destination file path (saved as ``.pt``).
    timestamps :
        Optional list of timestamp strings to include in the saved dict.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict = {"embeddings": embeddings}
    if timestamps is not None:
        save_dict["timestamps"] = timestamps

    torch.save(save_dict, str(path))
    logger.info("Cached %d embeddings to %s", embeddings.size(0), path)
