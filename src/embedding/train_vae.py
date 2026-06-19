"""Training script for the Dual-Stream Transformer-VAE embedding module.

Usage
-----
    python src/embedding/train_vae.py

The script loads the Hydra config from ``src/config/config.yaml``,
builds dummy training data (replace with real data loading for production),
trains the VAE and saves checkpoints to ``checkpoints/vae/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.embedding.vae import DualStreamVAE, vae_loss
from src.utils.helpers import load_config, save_checkpoint, set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StateDataset(Dataset):
    """PyTorch dataset wrapping pre-built tech/market state tensors."""

    def __init__(
        self, tech_states: list[torch.Tensor], mkt_states: list[torch.Tensor]
    ):
        assert len(tech_states) == len(mkt_states)
        self.tech_states = tech_states
        self.mkt_states = mkt_states

    def __len__(self) -> int:
        return len(self.tech_states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tech_states[idx], self.mkt_states[idx]


def generate_dummy_data(
    n_samples: int = 2000,
    L: int = 60,
    A: int = 5,
    F1: int = 8,
    F2: int = 3,
    seed: int = 42,
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Generate synthetic state data for testing / development."""
    torch.manual_seed(seed)
    tech_dim = A * F1
    tech_states = [torch.randn(L, tech_dim) for _ in range(n_samples)]
    mkt_states = [torch.randn(L, F2) for _ in range(n_samples)]
    return tech_states, mkt_states


def train_epoch(
    model: DualStreamVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch and return average losses."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for tech, mkt in loader:
        tech = tech.to(device)
        mkt = mkt.to(device)

        optimizer.zero_grad()
        (recon_tech, recon_mkt), mu, logvar = model(tech, mkt)
        loss, recon_loss, kl_loss = vae_loss(
            recon_tech, recon_mkt, tech, mkt, mu, logvar, beta=beta
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


def main() -> None:
    cfg = load_config()

    # --- Settings ---
    train_cfg = cfg.embedding.train
    vae_cfg = cfg.embedding.vae
    L = cfg.window.L
    A = len(cfg.data.coin_list)
    F1 = len(cfg.get("tech_feature_names", [
        "log_return", "rsi", "macd", "macd_signal", "macd_hist",
        "atr", "realized_vol", "volume_change",
    ]))
    F2 = len(cfg.get("market_feature_names", [
        "mkt_return", "mkt_volatility", "mkt_liquidity",
    ]))

    tech_dim = A * F1
    mkt_dim = F2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    logger.info(
        "Config — L=%d, A=%d, F1=%d, F2=%d, tech_dim=%d, mkt_dim=%d, device=%s",
        L, A, F1, F2, tech_dim, mkt_dim, device,
    )

    # --- Data ---
    logger.info("Generating dummy training data (%d samples) ...", train_cfg.n_samples)
    tech_states, mkt_states = generate_dummy_data(
        n_samples=train_cfg.n_samples,
        L=L, A=A, F1=F1, F2=F2, seed=cfg.seed,
    )
    dataset = StateDataset(tech_states, mkt_states)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # --- Model ---
    model = DualStreamVAE(
        input_dim_tech=tech_dim,
        input_dim_mkt=mkt_dim,
        latent_dim_tech=vae_cfg.latent_dim_tech,
        latent_dim_mkt=vae_cfg.latent_dim_mkt,
        d_model=vae_cfg.d_model,
        nhead=vae_cfg.nhead,
        num_layers=vae_cfg.num_layers,
        dropout=vae_cfg.dropout,
        L=L,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    beta = vae_cfg.beta

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model has %d trainable parameters", n_params)

    # --- Training loop ---
    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg.epochs + 1):
        metrics = train_epoch(model, loader, optimizer, beta, device)

        if epoch % max(1, train_cfg.epochs // 10) == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d — loss=%.4f  recon=%.4f  kl=%.4f",
                epoch, train_cfg.epochs,
                metrics["loss"], metrics["recon_loss"], metrics["kl_loss"],
            )

        if epoch % train_cfg.save_every == 0 or epoch == train_cfg.epochs:
            ckpt_path = checkpoint_dir / f"vae_epoch_{epoch:03d}.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": metrics["loss"],
                    "config": {
                        "input_dim_tech": tech_dim,
                        "input_dim_mkt": mkt_dim,
                        "latent_dim_tech": vae_cfg.latent_dim_tech,
                        "latent_dim_mkt": vae_cfg.latent_dim_mkt,
                        "d_model": vae_cfg.d_model,
                        "nhead": vae_cfg.nhead,
                        "num_layers": vae_cfg.num_layers,
                        "dropout": vae_cfg.dropout,
                        "L": L,
                    },
                },
                str(ckpt_path),
            )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
