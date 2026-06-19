"""Dual-Stream Transformer-VAE embedding module.

Provides:
- ``DualStreamVAE`` – the Transformer-VAE model
- ``vae_loss``      – reconstruction + KL divergence loss
- ``load_vae_model``        – load a pretrained VAE from checkpoint
- ``compute_embeddings``    – batch inference to produce embeddings
- ``cache_embeddings``      – save embeddings to disk for indexing
"""

from src.embedding.vae import DualStreamVAE, vae_loss
from src.embedding.utils import cache_embeddings, compute_embeddings, load_vae_model

__all__ = [
    "DualStreamVAE",
    "vae_loss",
    "load_vae_model",
    "compute_embeddings",
    "cache_embeddings",
]
