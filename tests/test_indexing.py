"""Tests for the offline indexing pipeline (Algorithm 1).

Covers:
  - SoP prompt template
  - SoP response parsing
  - VectorDatabase (FAISS + SQLite) CRUD
  - IndexingEngine end-to-end on a small dataset with mock LLM
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest
import torch

from src.embedding.vae import DualStreamVAE
from src.experts.finrl_expert import DummyDRLExpert
from src.experts.registry import ExpertRegistry
from src.indexing.database import VectorDatabase
from src.indexing.engine import IndexingEngine
from src.indexing.prompts import build_sop_prompt, parse_sop_response
from src.data.state_builder import StateBuilder

TOL = 1e-5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    reg.register(DummyDRLExpert("test_expert", n_assets=3))
    return reg


@pytest.fixture
def small_dataframe():
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    symbols = ["A", "B", "C"]
    rows = []
    rng = np.random.default_rng(42)
    # Generate a baseline price path
    prices = {s: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))) for s in symbols}
    for i, d in enumerate(dates):
        for s in symbols:
            rows.append({
                "date": d,
                "symbol": s,
                "open": prices[s][i],
                "high": prices[s][i] * 1.01,
                "low": prices[s][i] * 0.99,
                "close": prices[s][i],
                "volume": rng.lognormal(15, 0.5),
                "log_return": np.log(prices[s][i] / (prices[s][i - 1] if i > 0 else prices[s][i])),
                "rsi": rng.uniform(30, 70),
                "atr": rng.uniform(0.5, 2.0),
            })
    df = pd.DataFrame(rows).set_index(["date", "symbol"])
    return df


@pytest.fixture
def features(small_dataframe):
    df = small_dataframe.copy()
    symbols = df.index.get_level_values("symbol").unique()
    tech_cols = ["log_return", "rsi", "atr"]
    df_tech = df[tech_cols].copy()
    df_mkt = df.groupby("date")["log_return"].agg(
        mkt_return="mean",
        mkt_volatility="std",
    ).fillna(0)
    return df_tech, df_mkt


@pytest.fixture
def state_builder():
    return StateBuilder(
        window_length=10,
        tech_feature_names=["log_return", "rsi"],
        market_feature_names=["mkt_return", "mkt_volatility"],
    )


# ---------------------------------------------------------------------------
# Test prompt building
# ---------------------------------------------------------------------------

class TestBuildSopPrompt:
    def test_prompt_contains_axioms(self):
        prompt = build_sop_prompt(
            timestamp="2020-06-15",
            market_context={"volatility": "0.25", "trend": "bearish"},
            expert_name="ppo",
            allocation=[0.2, 0.5, 0.3],
            performance_metrics={
                "cumulative_return": 0.12,
                "sharpe": 1.5,
                "max_drawdown": -0.08,
            },
            uncertainty_estimate=0.15,
        )
        assert "Empirical Alignment" in prompt
        assert "Conservative Margining" in prompt
        assert "Tail Separation" in prompt
        assert "calculated_rho" in prompt
        assert "confidence_bound_epsilon" in prompt
        assert "regime_separation_margin_tau" in prompt
        assert "tail_optimal_flag" in prompt

    def test_prompt_contains_values(self):
        prompt = build_sop_prompt(
            timestamp="2020-06-15",
            market_context={"volatility": "0.25", "trend": "bearish"},
            expert_name="ppo",
            allocation={"BTC": 0.5, "ETH": 0.5},
            performance_metrics={
                "cumulative_return": 0.12,
                "sharpe": 1.5,
                "max_drawdown": -0.08,
            },
            uncertainty_estimate=0.15,
        )
        assert "2020-06-15" in prompt
        assert "ppo" in prompt
        assert "0.12" in prompt
        assert "1.5" in prompt
        assert "-0.08" in prompt
        assert "0.15" in prompt


# ---------------------------------------------------------------------------
# Test SoP response parsing
# ---------------------------------------------------------------------------

class TestParseSopResponse:
    def test_parse_valid_json(self):
        response = json.dumps({
            "calculated_rho": 0.8,
            "confidence_bound_epsilon": 0.05,
            "regime_separation_margin_tau": 0.1,
            "tail_optimal_flag": True,
        })
        parsed = parse_sop_response(response)
        assert parsed["calculated_rho"] == 0.8
        assert parsed["confidence_bound_epsilon"] == 0.05
        assert parsed["regime_separation_margin_tau"] == 0.1
        assert parsed["tail_optimal_flag"] is True

    def test_parse_json_within_text(self):
        response = "Here is the SoP:\n" + json.dumps({
            "calculated_rho": 0.75,
            "confidence_bound_epsilon": 0.08,
            "regime_separation_margin_tau": 0.06,
            "tail_optimal_flag": False,
        })
        parsed = parse_sop_response(response)
        assert parsed["calculated_rho"] == 0.75
        assert parsed["tail_optimal_flag"] is False

    def test_parse_invalid_falls_back_to_defaults(self):
        parsed = parse_sop_response("Not a JSON response at all")
        assert parsed["calculated_rho"] == 0.5
        assert parsed["confidence_bound_epsilon"] == 0.1

    def test_parse_empty_falls_back(self):
        parsed = parse_sop_response("")
        assert parsed["calculated_rho"] == 0.5


# ---------------------------------------------------------------------------
# Test VectorDatabase
# ---------------------------------------------------------------------------

class TestVectorDatabase:
    def test_add_and_query(self):
        db = VectorDatabase(dim=4)
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        db.add_record(
            embedding=emb,
            timestamp="2020-01-01",
            expert_id="test",
            sop_text="test sop",
            cumulative_return=0.1,
            sharpe=1.2,
            drawdown=-0.05,
            uncertainty_score=0.1,
        )
        assert len(db) == 1
        results = db.query_similar(emb, k=5)
        assert len(results) == 1
        assert results[0]["expert_id"] == "test"
        assert results[0]["sop_text"] == "test sop"
        assert abs(results[0]["cumulative_return"] - 0.1) < TOL

    def test_query_multiple(self):
        db = VectorDatabase(dim=2)
        for i in range(5):
            emb = np.array([float(i), 0.0], dtype=np.float32)
            db.add_record(
                embedding=emb,
                timestamp=f"2020-01-0{i+1}",
                expert_id=f"expert_{i}",
            )
        # Query closest to center
        query = np.array([2.0, 0.0], dtype=np.float32)
        results = db.query_similar(query, k=3)
        assert len(results) == 3
        # Should return experts 2, 1, 3 (or similar by distance)
        ids = [r["expert_id"] for r in results]
        assert "expert_2" in ids

    def test_query_with_expert_filter(self):
        db = VectorDatabase(dim=2)
        for i in range(3):
            db.add_record(
                embedding=np.array([float(i), 0.0], dtype=np.float32),
                timestamp="2020-01-01",
                expert_id="common" if i < 2 else "other",
            )
        query = np.array([0.0, 0.0], dtype=np.float32)
        results = db.query_similar(query, k=5, expert_id="common")
        assert len(results) == 2
        for r in results:
            assert r["expert_id"] == "common"

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = VectorDatabase(dim=4)
            emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            db.add_record(embedding=emb, timestamp="t", expert_id="e")
            db.save(
                index_path=Path(tmp) / "index.faiss",
                db_path=Path(tmp) / "meta.db",
            )

            db2 = VectorDatabase(dim=4)
            db2.load(
                index_path=Path(tmp) / "index.faiss",
                db_path=Path(tmp) / "meta.db",
            )
            assert len(db2) == 1
            results = db2.query_similar(emb, k=5)
            assert len(results) == 1
            assert results[0]["expert_id"] == "e"

    def test_query_empty_db(self):
        db = VectorDatabase(dim=4)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = db.query_similar(query, k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Test IndexingEngine end-to-end (with mock LLM)
# ---------------------------------------------------------------------------

class TestIndexingEngine:
    def test_index_small_dataset(
        self, small_vae, expert_registry, small_dataframe, features, state_builder
    ):
        df_tech, df_mkt = features
        dim = small_vae.latent_dim
        db = VectorDatabase(dim=dim)

        engine = IndexingEngine(
            vae=small_vae,
            expert_registry=expert_registry,
            state_builder=state_builder,
            db=db,
            config={
                "H": 5,
                "L": 10,
                "mock_llm": True,
                "device": "cpu",
                "llm": {},
            },
        )

        n = engine.run_indexing(
            df=small_dataframe,
            df_tech=df_tech,
            df_mkt=df_mkt,
        )

        assert n > 0, "Indexing should produce records"
        assert len(db) == n

        # Verify records have valid SoP text and performance numbers
        for expert_name in engine.expert_registry.list_experts():
            query_emb = np.random.randn(dim).astype(np.float32)
            results = db.query_similar(query_emb, k=5, expert_id=expert_name)
            if results:
                r = results[0]
                assert isinstance(r["sop_text"], str)
                assert len(r["sop_text"]) > 0
                assert isinstance(r["cumulative_return"], float)
                assert isinstance(r["sharpe"], float)

    def test_index_saves_and_reloads(
        self, small_vae, expert_registry, small_dataframe, features, state_builder
    ):
        df_tech, df_mkt = features
        dim = small_vae.latent_dim

        with tempfile.TemporaryDirectory() as tmp:
            db = VectorDatabase(dim=dim)
            engine = IndexingEngine(
                vae=small_vae,
                expert_registry=expert_registry,
                state_builder=state_builder,
                db=db,
                config={
                    "H": 5,
                    "L": 10,
                    "mock_llm": True,
                    "device": "cpu",
                    "llm": {},
                },
            )
            engine.run_indexing(df=small_dataframe, df_tech=df_tech, df_mkt=df_mkt)

            index_path = Path(tmp) / "faiss.index"
            db_path = Path(tmp) / "meta.db"
            db.save(index_path=index_path, db_path=db_path)

            db2 = VectorDatabase(dim=dim)
            db2.load(index_path=index_path, db_path=db_path)
            assert len(db2) == len(db)

            query_emb = np.random.randn(dim).astype(np.float32)
            results = db2.query_similar(query_emb, k=3)
            assert len(results) > 0

    def test_performance_metrics_valid(
        self, small_vae, expert_registry, small_dataframe, features, state_builder
    ):
        df_tech, df_mkt = features
        dim = small_vae.latent_dim
        db = VectorDatabase(dim=dim)

        engine = IndexingEngine(
            vae=small_vae,
            expert_registry=expert_registry,
            state_builder=state_builder,
            db=db,
            config={
                "H": 5,
                "L": 10,
                "mock_llm": True,
                "device": "cpu",
                "llm": {},
            },
        )

        engine.run_indexing(df=small_dataframe, df_tech=df_tech, df_mkt=df_mkt)

        # Verify that cumulative_return, sharpe, drawdown are finite numbers
        for faiss_id in range(len(db)):
            # Query the exact match
            results = db.query_similar(
                embedding=np.random.randn(dim).astype(np.float32),
                k=len(db),
            )
            for r in results:
                assert np.isfinite(r["cumulative_return"]), "cumulative_return is not finite"
                assert np.isfinite(r["sharpe"]), "sharpe is not finite"
                assert np.isfinite(r["drawdown"]), "drawdown is not finite"
                assert r["drawdown"] <= 0.0 or abs(r["drawdown"]) < TOL, (
                    f"drawdown should be <= 0, got {r['drawdown']}"
                )

    def test_empty_db_when_no_data(self, small_vae, expert_registry, state_builder):
        dim = small_vae.latent_dim
        db = VectorDatabase(dim=dim)

        empty_df = pd.DataFrame(
            columns=["date", "symbol", "close", "open", "high", "low", "volume"],
        ).set_index(["date", "symbol"])

        empty_tech = pd.DataFrame(
            columns=["log_return", "rsi", "atr"],
            index=pd.MultiIndex.from_tuples([], names=["date", "symbol"]),
        )
        empty_mkt = pd.DataFrame(
            columns=["mkt_return", "mkt_volatility", "mkt_liquidity"],
            index=pd.DatetimeIndex([]),
        )

        engine = IndexingEngine(
            vae=small_vae,
            expert_registry=expert_registry,
            state_builder=state_builder,
            db=db,
            config={
                "H": 5,
                "L": 10,
                "mock_llm": True,
                "device": "cpu",
                "llm": {},
            },
        )

        n = engine.run_indexing(
            df=empty_df,
            df_tech=empty_tech,
            df_mkt=empty_mkt,
        )
        assert n == 0


# ---------------------------------------------------------------------------
# Test mock SoP in engine
# ---------------------------------------------------------------------------

class TestMockLLM:
    def test_mock_generates_valid_sop(self):
        from src.indexing.engine import _MOCK_SOP
        parsed = json.loads(_MOCK_SOP)
        assert "calculated_rho" in parsed
        assert "confidence_bound_epsilon" in parsed
        assert "regime_separation_margin_tau" in parsed
        assert "tail_optimal_flag" in parsed
        assert "sop_text" in parsed
        assert isinstance(parsed["calculated_rho"], float)
        assert isinstance(parsed["tail_optimal_flag"], bool)
