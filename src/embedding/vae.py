"""Dual-Stream Transformer-VAE for joint embedding of technical and market state streams."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class StreamTransformerEncoder(nn.Module):
    """Transformer encoder for a single input stream.

    Projects input to ``d_model``, applies positional encoding,
    runs a stacked Transformer encoder, and pools over the time
    dimension via mean to yield a fixed-size vector.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x.mean(dim=1)


class DualStreamVAE(nn.Module):
    """Dual-Stream Transformer-VAE embedding module.

    Encodes a technical state stream ``S_t^tech`` (shape ``[L, A*F1]``)
    and a market-wide stream ``S_t^mkt`` (shape ``[L, F2]``) into
    separate latent codes via Transformer encoders + variational layers.
    The decoder reconstructs both streams from the latent codes.

    After training, :meth:`get_embedding` returns the joint embedding
    ``h_t = [z_tech | z_mkt]`` of dimension ``latent_dim_tech + latent_dim_mkt``.
    """

    def __init__(
        self,
        input_dim_tech: int,
        input_dim_mkt: int,
        latent_dim_tech: int = 16,
        latent_dim_mkt: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        L: int = 60,
    ):
        super().__init__()
        self.input_dim_tech = input_dim_tech
        self.input_dim_mkt = input_dim_mkt
        self.latent_dim_tech = latent_dim_tech
        self.latent_dim_mkt = latent_dim_mkt
        self.latent_dim = latent_dim_tech + latent_dim_mkt
        self.L = L

        # --- Encoders ---
        self.tech_encoder = StreamTransformerEncoder(
            input_dim_tech, d_model, nhead, num_layers, dropout,
        )
        self.mkt_encoder = StreamTransformerEncoder(
            input_dim_mkt, d_model, nhead, num_layers, dropout,
        )

        # --- Variational projections ---
        self.mu_tech = nn.Linear(d_model, latent_dim_tech)
        self.logvar_tech = nn.Linear(d_model, latent_dim_tech)
        self.mu_mkt = nn.Linear(d_model, latent_dim_mkt)
        self.logvar_mkt = nn.Linear(d_model, latent_dim_mkt)

        # --- Decoders (MLP) ---
        self.tech_decoder = nn.Sequential(
            nn.Linear(latent_dim_tech, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, L * input_dim_tech),
        )
        self.mkt_decoder = nn.Sequential(
            nn.Linear(latent_dim_mkt, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, L * input_dim_mkt),
        )

    def encode(
        self, tech: torch.Tensor, mkt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if tech.dim() == 2:
            tech = tech.unsqueeze(0)
        if mkt.dim() == 2:
            mkt = mkt.unsqueeze(0)

        h_tech = self.tech_encoder(tech)
        h_mkt = self.mkt_encoder(mkt)

        mu_t = self.mu_tech(h_tech)
        logvar_t = self.logvar_tech(h_tech)
        mu_m = self.mu_mkt(h_mkt)
        logvar_m = self.logvar_mkt(h_mkt)

        return mu_t, logvar_t, mu_m, logvar_m

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self, z_tech: torch.Tensor, z_mkt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        recon_tech = self.tech_decoder(z_tech).view(
            -1, self.L, self.input_dim_tech
        )
        recon_mkt = self.mkt_decoder(z_mkt).view(
            -1, self.L, self.input_dim_mkt
        )
        return recon_tech, recon_mkt

    def forward(
        self, tech: torch.Tensor, mkt: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu_t, logvar_t, mu_m, logvar_m = self.encode(tech, mkt)
        z_tech = self.reparameterize(mu_t, logvar_t)
        z_mkt = self.reparameterize(mu_m, logvar_m)
        recon_tech, recon_mkt = self.decode(z_tech, z_mkt)

        mu = torch.cat([mu_t, mu_m], dim=-1)
        logvar = torch.cat([logvar_t, logvar_m], dim=-1)

        return (recon_tech, recon_mkt), mu, logvar

    def get_embedding(self, tech: torch.Tensor, mkt: torch.Tensor) -> torch.Tensor:
        mu_t, logvar_t, mu_m, logvar_m = self.encode(tech, mkt)
        z_tech = self.reparameterize(mu_t, logvar_t)
        z_mkt = self.reparameterize(mu_m, logvar_m)
        return torch.cat([z_tech, z_mkt], dim=-1)

    @classmethod
    def from_config(
        cls, cfg, tech_dim: int, mkt_dim: int, L: int = 60
    ) -> "DualStreamVAE":
        vae_cfg = cfg.embedding.vae
        return cls(
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


def vae_loss(
    recon_tech: torch.Tensor,
    recon_mkt: torch.Tensor,
    tech: torch.Tensor,
    mkt: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss: MSE reconstruction + β-weighted KL divergence."""
    recon_loss = F.mse_loss(recon_tech, tech) + F.mse_loss(recon_mkt, mkt)

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
    ).mean()

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
