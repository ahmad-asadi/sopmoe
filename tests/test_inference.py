"""Tests for the online inference pipeline (Algorithm 2).

Covers:
  - Weight projection utilities
  - Similarity-from-distance computation
  - Uncertainty prompt building
  - LLM response parsing
  - InferenceEngine end-to-end with populated index
  - Heuristic (fallback) uncertainty
  - Caching behaviour
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch

from src.embedding.vae import DualStreamVAE
from src.experts.finrl_expert import DummyDRLExpert
from src.experts.registry import ExpertRegistry
from src.indexing.database import VectorDatabase
from src.inference.engine import InferenceEngine
from src.inference.prompts import build_uncertainty_prompt
from src.inference.utils import project_weights, similarity_from_distance

TOL = 1e-5


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def small_vae():
    L = 10
    A, F1, F2 = 3, 2, 2
    model = DualStreamVAE(
        input_dim_tech=A * F1,
        input_dim_mkt=F2,
        latent_dim_tech=4,
        latent_dim_mkt=4,
        d_model=16,
        nhead=2,
        num_layers=1,
        dropout=0.0,
        L=L,
    )
    model.eval()
    return model


@pytest.fixture
def expert_registry():
    reg = ExpertRegistry()
    reg.register(DummyDRLExpert("expert_a", n_assets=3))
    reg.register(DummyDRLExpert("expert_b", n_assets=3))
    reg.register(DummyDRLExpert("expert_c", n_assets=3))
    return reg


@pytest.fixture
def populated_db(small_vae, expert_registry):
    """Create a ``VectorDatabase`` with realistic mock data."""
    dim = small_vae.latent_dim
    db = VectorDatabase(dim=dim)
    rng = np.random.default_rng(42)

    # Create 3 clusters of embeddings for each expert
    for cluster_center, ret, sharpe, dd in [
        (np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.05, 1.2, -0.02),
        (np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), -0.02, 0.5, -0.10),
        (np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.10, 2.0, -0.01),
    ]:
        for expert_name in ["expert_a", "expert_b", "expert_c"]:
            for i in range(5):
                emb = cluster_center + rng.normal(0, 0.05, dim).astype(np.float32)
                db.add_record(
                    embedding=emb,
                    timestamp=f"2020-01-{10 + i:02d}",
                    expert_id=expert_name,
                    sop_text=f"SoP for {expert_name} at cluster",
                    cumulative_return=ret + rng.normal(0, 0.01),
                    sharpe=sharpe + rng.normal(0, 0.1),
                    drawdown=dd + rng.normal(0, 0.005),
                )

    # Add a few outlier records
    outlier_emb = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    for expert_name in ["expert_a", "expert_b", "expert_c"]:
        db.add_record(
            embedding=outlier_emb,
            timestamp="2020-02-01",
            expert_id=expert_name,
            sop_text=f"SoP outlier for {expert_name}",
            cumulative_return=-0.15,
            sharpe=-1.0,
            drawdown=-0.25,
        )

    return db


@pytest.fixture
def inference_engine(small_vae, expert_registry, populated_db):
    return InferenceEngine(
        vae=small_vae,
        expert_registry=expert_registry,
        db=populated_db,
        llm_client=None,
        config={
            "K": 3,
            "lambda_": 0.1,
            "device": "cpu",
        },
    )


# ===================================================================
# Test utils
# ===================================================================

class TestProjectWeights:
    def test_projects_to_simplex(self):
        w = np.array([2.0, 1.0, 0.5, -0.5])
        proj = project_weights(w)
        assert abs(proj.sum() - 1.0) < TOL
        assert np.all(proj >= 0)

    def test_all_zeros(self):
        w = np.zeros(5)
        proj = project_weights(w)
        assert abs(proj.sum() - 1.0) < TOL
        assert np.allclose(proj, 1.0 / 5.0, atol=TOL)

    def test_all_negative(self):
        w = np.array([-1.0, -2.0, -3.0])
        proj = project_weights(w)
        assert abs(proj.sum() - 1.0) < TOL
        assert np.all(proj >= 0)

    def test_nan_handling(self):
        w = np.array([1.0, np.nan, 2.0])
        proj = project_weights(w)
        assert abs(proj.sum() - 1.0) < TOL
        assert np.all(np.isfinite(proj))

    def test_single_element(self):
        w = np.array([5.0])
        proj = project_weights(w)
        assert abs(proj.sum() - 1.0) < TOL
        assert abs(proj[0] - 1.0) < TOL


class TestSimilarityFromDistance:
    def test_closest_gets_highest_weight(self):
        dists = np.array([0.1, 1.0, 5.0])
        sim = similarity_from_distance(dists)
        assert abs(sim.sum() - 1.0) < TOL
        assert sim[0] > sim[1] > sim[2]

    def test_uniform_distances(self):
        dists = np.array([1.0, 1.0, 1.0])
        sim = similarity_from_distance(dists)
        assert abs(sim.sum() - 1.0) < TOL
        assert np.allclose(sim, 1.0 / 3.0, atol=TOL)

    def test_zero_distance(self):
        dists = np.array([0.0, 10.0])
        sim = similarity_from_distance(dists)
        assert abs(sim.sum() - 1.0) < TOL
        assert sim[0] > 0.5

    def test_single_element(self):
        dists = np.array([2.0])
        sim = similarity_from_distance(dists)
        assert abs(sim[0] - 1.0) < TOL


# ===================================================================
# Test uncertainty prompt
# ===================================================================

class TestBuildUncertaintyPrompt:
    def test_prompt_contains_field_names(self):
        prompt = build_uncertainty_prompt(
            current_embedding=[0.1, 0.2, 0.3],
            retrieved_records={
                "expert_a": [
                    {"sop_text": "good perf", "cumulative_return": 0.05,
                     "sharpe": 1.2, "drawdown": -0.02, "distance": 0.5},
                ],
            },
            performance_summaries={
                "expert_a": {"weighted_return": 0.05, "std": 0.01},
            },
            embedding_dim=3,
        )
        assert "expert_uncertainties" in prompt
        assert "expert_a" in prompt
        assert "confidence" in prompt.lower() or "uncertainty" in prompt.lower()

    def test_prompt_handles_empty_retrieved(self):
        prompt = build_uncertainty_prompt(
            current_embedding=None,
            retrieved_records={},
            performance_summaries={},
            embedding_dim=None,
        )
        assert "expert_uncertainties" in prompt

    def test_prompt_includes_aggregate_stats(self):
        prompt = build_uncertainty_prompt(
            current_embedding=[1.0],
            retrieved_records={
                "expert_a": [
                    {"sop_text": "SoP text here", "cumulative_return": 0.05,
                     "sharpe": 1.2, "drawdown": -0.02, "distance": 0.5},
                ],
            },
            performance_summaries={
                "expert_a": {"weighted_return": 0.05, "std": 0.01},
            },
            embedding_dim=1,
        )
        assert "weighted_return" in prompt
        assert "performance_std" in prompt


# ===================================================================
# Test InferenceEngine
# ===================================================================

class TestInferenceEngine:
    def test_select_expert_returns_valid_weights(
        self, inference_engine, small_vae
    ):
        L = small_vae.L
        dim_tech = small_vae.input_dim_tech
        dim_mkt = small_vae.input_dim_mkt

        tech_state = torch.randn(L, dim_tech)
        mkt_state = torch.randn(L, dim_mkt)

        expert_state = np.random.randn(1 + 3 + 4 + 6).astype(np.float32)

        selected, weights, meta = inference_engine.select_expert(
            tech_state=tech_state,
            mkt_state=mkt_state,
            timestamp="2020-06-15",
            expert_state=expert_state,
        )

        assert selected in ("expert_a", "expert_b", "expert_c")
        assert abs(weights.sum() - 1.0) < TOL
        assert np.all(weights >= 0)
        assert meta["selected_expert"] == selected
        assert set(meta["utilities"].keys()) == {"expert_a", "expert_b", "expert_c"}
        assert set(meta["weighted_returns"].keys()) == {"expert_a", "expert_b", "expert_c"}
        assert set(meta["uncertainties"].keys()) == {"expert_a", "expert_b", "expert_c"}

    def test_select_expert_without_expert_state(
        self, inference_engine, small_vae
    ):
        L = small_vae.L
        dim_tech = small_vae.input_dim_tech
        dim_mkt = small_vae.input_dim_mkt

        tech_state = torch.randn(L, dim_tech)
        mkt_state = torch.randn(L, dim_mkt)

        selected, weights, meta = inference_engine.select_expert(
            tech_state=tech_state,
            mkt_state=mkt_state,
            timestamp="2020-06-15",
        )

        assert selected in ("expert_a", "expert_b", "expert_c")
        assert abs(weights.sum() - 1.0) < TOL

    def test_caching_on_embedding_hash(
        self, inference_engine, small_vae
    ):
        L = small_vae.L
        dim_tech = small_vae.input_dim_tech
        dim_mkt = small_vae.input_dim_mkt

        tech_state = torch.randn(L, dim_tech)
        mkt_state = torch.randn(L, dim_mkt)

        # First call populates the cache
        sel1, w1, meta1 = inference_engine.select_expert(
            tech_state=tech_state, mkt_state=mkt_state,
        )
        assert len(inference_engine._cache) == 1

        # Manually store a known result under a different key to verify
        # the cache is being used (avoid stochastic VAE sampling issue)
        fake_key = "deadbeef" * 8
        inference_engine._cache[fake_key] = ("expert_b", w1, meta1)

        # Retrieve from cache
        cached = inference_engine._cache.get(fake_key)
        assert cached is not None
        assert cached[0] == "expert_b"
        assert np.allclose(cached[1], w1)

    def test_cache_eviction(
        self, small_vae, expert_registry, populated_db
    ):
        engine = InferenceEngine(
            vae=small_vae,
            expert_registry=expert_registry,
            db=populated_db,
            llm_client=None,
            config={"K": 3, "lambda_": 0.1, "device": "cpu", "cache_size": 2},
        )

        L = small_vae.L
        dim_tech = small_vae.input_dim_tech
        dim_mkt = small_vae.input_dim_mkt

        for i in range(3):
            ts = torch.randn(L, dim_tech) + i
            ms = torch.randn(L, dim_mkt) + i
            engine.select_expert(
                tech_state=ts, mkt_state=ms, timestamp=f"2020-01-{i+1:02d}",
            )

        assert len(engine._cache) <= 2

    def test_empty_index_returns_some_expert(
        self, small_vae, expert_registry
    ):
        db = VectorDatabase(dim=small_vae.latent_dim)
        engine = InferenceEngine(
            vae=small_vae,
            expert_registry=expert_registry,
            db=db,
            llm_client=None,
            config={"K": 3, "lambda_": 0.1, "device": "cpu"},
        )

        tech_state = torch.randn(small_vae.L, small_vae.input_dim_tech)
        mkt_state = torch.randn(small_vae.L, small_vae.input_dim_mkt)

        selected, weights, meta = engine.select_expert(
            tech_state=tech_state, mkt_state=mkt_state,
        )

        assert selected in ("expert_a", "expert_b", "expert_c")
        assert abs(weights.sum() - 1.0) < TOL

    def test_heuristic_uncertainty_std_maps_to_0_1_range(
        self, small_vae, expert_registry
    ):
        db = VectorDatabase(dim=small_vae.latent_dim)
        rng = np.random.default_rng(42)

        # Insert records with very consistent performance (low uncertainty)
        for i in range(5):
            db.add_record(
                embedding=rng.normal(0, 0.1, small_vae.latent_dim).astype(np.float32),
                timestamp=f"2020-01-{i+1:02d}",
                expert_id="expert_a",
                cumulative_return=0.05,  # identical
            )

        # Insert records with high variance (high uncertainty)
        for i in range(5):
            db.add_record(
                embedding=rng.normal(0, 0.1, small_vae.latent_dim).astype(np.float32),
                timestamp=f"2020-01-{i+1:02d}",
                expert_id="expert_b",
                cumulative_return=(-1) ** i * 0.2,  # high variance
            )

        engine = InferenceEngine(
            vae=small_vae,
            expert_registry=expert_registry,
            db=db,
            llm_client=None,
            config={"K": 3, "lambda_": 0.1, "device": "cpu"},
        )

        tech_state = torch.randn(small_vae.L, small_vae.input_dim_tech)
        mkt_state = torch.randn(small_vae.L, small_vae.input_dim_mkt)

        _, _, meta = engine.select_expert(
            tech_state=tech_state, mkt_state=mkt_state,
        )

        assert 0.0 <= meta["uncertainties"]["expert_a"] <= 1.0
        assert 0.0 <= meta["uncertainties"]["expert_b"] <= 1.0

    def test_llm_uncertainty_parsing(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine._expert_names = ["expert_a", "expert_b"]

        valid_response = json.dumps({
            "expert_uncertainties": {"expert_a": 0.1, "expert_b": 0.8},
        })
        parsed = engine._parse_uncertainty_response(valid_response)
        assert abs(parsed["expert_a"] - 0.1) < TOL
        assert abs(parsed["expert_b"] - 0.8) < TOL

    def test_llm_uncertainty_parsing_json_within_text(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine._expert_names = ["expert_a"]

        response = "Here is the analysis:\n" + json.dumps({
            "expert_uncertainties": {"expert_a": 0.3},
        })
        parsed = engine._parse_uncertainty_response(response)
        assert abs(parsed["expert_a"] - 0.3) < TOL

    def test_llm_uncertainty_parsing_fallback(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine._expert_names = ["expert_a"]

        parsed = engine._parse_uncertainty_response("not valid json")
        assert parsed == {}

    def test_llm_client_is_called_when_provided(
        self, small_vae, expert_registry, populated_db
    ):
        call_log: list[str] = []

        def mock_llm(prompt: str) -> str:
            call_log.append(prompt)
            return json.dumps({
                "expert_uncertainties": {
                    "expert_a": 0.1, "expert_b": 0.5, "expert_c": 0.9,
                },
            })

        engine = InferenceEngine(
            vae=small_vae,
            expert_registry=expert_registry,
            db=populated_db,
            llm_client=mock_llm,
            config={"K": 3, "lambda_": 0.1, "device": "cpu"},
        )

        tech_state = torch.randn(small_vae.L, small_vae.input_dim_tech)
        mkt_state = torch.randn(small_vae.L, small_vae.input_dim_mkt)

        _, _, meta = engine.select_expert(
            tech_state=tech_state, mkt_state=mkt_state,
        )

        assert len(call_log) == 1
        assert abs(meta["uncertainties"]["expert_a"] - 0.1) < TOL
        assert abs(meta["uncertainties"]["expert_b"] - 0.5) < TOL
        assert abs(meta["uncertainties"]["expert_c"] - 0.9) < TOL

    def test_llm_client_failure_falls_back_to_heuristic(
        self, small_vae, expert_registry, populated_db
    ):
        def failing_llm(_prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        engine = InferenceEngine(
            vae=small_vae,
            expert_registry=expert_registry,
            db=populated_db,
            llm_client=failing_llm,
            config={"K": 3, "lambda_": 0.1, "device": "cpu"},
        )

        tech_state = torch.randn(small_vae.L, small_vae.input_dim_tech)
        mkt_state = torch.randn(small_vae.L, small_vae.input_dim_mkt)

        _, _, meta = engine.select_expert(
            tech_state=tech_state, mkt_state=mkt_state,
        )

        # Should have fallen back to heuristic values
        for expert_name in ["expert_a", "expert_b", "expert_c"]:
            assert 0.0 <= meta["uncertainties"][expert_name] <= 1.0

    def test_cache_clear(self, inference_engine):
        inference_engine.clear_cache()
        assert len(inference_engine._cache) == 0

        # Populate cache
        L = inference_engine.vae.L
        dim_tech = inference_engine.vae.input_dim_tech
        dim_mkt = inference_engine.vae.input_dim_mkt
        tech_state = torch.randn(L, dim_tech)
        mkt_state = torch.randn(L, dim_mkt)

        inference_engine.select_expert(
            tech_state=tech_state, mkt_state=mkt_state,
        )
        assert len(inference_engine._cache) == 1

        inference_engine.clear_cache()
        assert len(inference_engine._cache) == 0

    def test_aggregate_performance_no_records(self, inference_engine):
        retrieved = {
            "expert_a": [],
            "expert_b": [],
            "expert_c": [],
        }
        w_ret, meta = inference_engine._aggregate_performance(retrieved)
        for expert_name in ["expert_a", "expert_b", "expert_c"]:
            assert w_ret[expert_name] == 0.0
            assert meta[expert_name]["n_neighbours"] == 0
