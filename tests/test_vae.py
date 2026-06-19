"""Tests for the Dual-Stream Transformer-VAE embedding module."""

from __future__ import annotations

import pytest
import torch

from src.embedding.vae import DualStreamVAE, vae_loss

TOL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dummy_batch(
    L: int = 60,
    A: int = 5,
    F1: int = 8,
    F2: int = 3,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    tech = torch.randn(batch_size, L, A * F1)
    mkt = torch.randn(batch_size, L, F2)
    return tech, mkt


def make_model(
    L: int = 60,
    A: int = 5,
    F1: int = 8,
    F2: int = 3,
    latent_dim_tech: int = 16,
    latent_dim_mkt: int = 16,
) -> DualStreamVAE:
    return DualStreamVAE(
        input_dim_tech=A * F1,
        input_dim_mkt=F2,
        latent_dim_tech=latent_dim_tech,
        latent_dim_mkt=latent_dim_mkt,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        L=L,
    )


# ---------------------------------------------------------------------------
# Test DualStreamVAE
# ---------------------------------------------------------------------------

class TestDualStreamVAE:
    """Verify forward pass shape contracts."""

    def test_forward_output_shapes(self):
        L, A, F1, F2 = 60, 5, 8, 3
        model = make_model(L=L, A=A, F1=F1, F2=F2)
        tech, mkt = dummy_batch(L=L, A=A, F1=F1, F2=F2)

        (recon_tech, recon_mkt), mu, logvar = model(tech, mkt)

        assert recon_tech.shape == tech.shape, (
            f"recon_tech {recon_tech.shape} != tech {tech.shape}"
        )
        assert recon_mkt.shape == mkt.shape, (
            f"recon_mkt {recon_mkt.shape} != mkt {mkt.shape}"
        )
        latent_dim = model.latent_dim_tech + model.latent_dim_mkt
        assert mu.shape == (tech.size(0), latent_dim)
        assert logvar.shape == (tech.size(0), latent_dim)

    def test_forward_single_sample(self):
        """Should handle 2D inputs without batch dim."""
        L, A, F1, F2 = 60, 5, 8, 3
        model = make_model(L=L, A=A, F1=F1, F2=F2)
        tech = torch.randn(L, A * F1)
        mkt = torch.randn(L, F2)

        (recon_tech, recon_mkt), mu, logvar = model(tech, mkt)

        assert recon_tech.shape == (1, L, A * F1)
        assert recon_mkt.shape == (1, L, F2)
        assert mu.shape == (1, model.latent_dim)
        assert logvar.shape == (1, model.latent_dim)

    def test_get_embedding_shape(self):
        model = make_model()
        tech, mkt = dummy_batch()

        h = model.get_embedding(tech, mkt)
        expected_dim = model.latent_dim_tech + model.latent_dim_mkt
        assert h.shape == (tech.size(0), expected_dim), (
            f"embedding {h.shape} != (B, {expected_dim})"
        )

    def test_get_embedding_single_sample(self):
        model = make_model()
        tech = torch.randn(60, 40)
        mkt = torch.randn(60, 3)

        h = model.get_embedding(tech, mkt)
        expected_dim = model.latent_dim
        assert h.shape == (1, expected_dim)

    def test_encode_output_shapes(self):
        model = make_model()
        tech, mkt = dummy_batch()

        mu_t, lv_t, mu_m, lv_m = model.encode(tech, mkt)
        assert mu_t.shape == (tech.size(0), model.latent_dim_tech)
        assert lv_t.shape == (tech.size(0), model.latent_dim_tech)
        assert mu_m.shape == (tech.size(0), model.latent_dim_mkt)
        assert lv_m.shape == (tech.size(0), model.latent_dim_mkt)

    def test_reparameterize_produces_correct_shape(self):
        model = make_model()
        mu = torch.randn(4, 16)
        logvar = torch.randn(4, 16)
        z = model.reparameterize(mu, logvar)
        assert z.shape == mu.shape
        # stochastic: verify at least not all the same
        assert not torch.allclose(z, mu, atol=TOL)

    def test_vae_loss_shapes_and_grad(self):
        model = make_model()
        tech, mkt = dummy_batch()

        (recon_tech, recon_mkt), mu, logvar = model(tech, mkt)
        loss, recon_loss, kl_loss = vae_loss(
            recon_tech, recon_mkt, tech, mkt, mu, logvar, beta=0.001
        )

        assert loss.ndim == 0  # scalar
        assert recon_loss.ndim == 0
        assert kl_loss.ndim == 0

        # Should be backprop-able
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"

    def test_different_latent_dims(self):
        """Verify model works with different latent dimensions per stream."""
        model = DualStreamVAE(
            input_dim_tech=40,
            input_dim_mkt=3,
            latent_dim_tech=32,
            latent_dim_mkt=8,
            d_model=64,
            nhead=4,
            num_layers=2,
            L=60,
        )
        tech, mkt = dummy_batch()
        (recon_tech, recon_mkt), mu, logvar = model(tech, mkt)

        assert recon_tech.shape == tech.shape
        assert recon_mkt.shape == mkt.shape
        assert mu.shape == (tech.size(0), 40)
        assert logvar.shape == (tech.size(0), 40)

        h = model.get_embedding(tech, mkt)
        assert h.shape == (tech.size(0), 40)


# ---------------------------------------------------------------------------
# Test training sanity (loss decreases)
# ---------------------------------------------------------------------------

class TestTrainingSanity:
    """Verify that over a few iterations the loss decreases."""

    def test_loss_decreases_over_mini_training(self):
        torch.manual_seed(42)
        model = make_model()
        tech, mkt = dummy_batch(batch_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            (r_tech, r_mkt), mu, lv = model(tech, mkt)
            loss, _, _ = vae_loss(r_tech, r_mkt, tech, mkt, mu, lv, beta=0.001)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )
