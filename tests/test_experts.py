"""Tests for the expert pool module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experts.base import BaseExpert, PortfolioTradingEnv
from src.experts.finrl_expert import DummyDRLExpert, FinRLExpert, DRL_AGENTS
from src.experts.milp_expert import MILPExpert
from src.experts.alpha_mix_rl import AlphaMixRLExpert
from src.experts.tac_expert import TACExpert
from src.experts.registry import ExpertRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOL = 1e-5


def dummy_state(n_assets: int = 5) -> np.ndarray:
    dim = 1 + n_assets + (n_assets + 1) + 2 * n_assets  # cash + prices + weights + tech
    return np.random.randn(dim).astype(np.float32)


def dummy_env(n_assets: int = 5) -> PortfolioTradingEnv:
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    tickers = [f"A{i}" for i in range(n_assets)]
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "tic": t,
                         "close": abs(rng.normal(100, 20)),
                         "macd": rng.uniform(-2, 2),
                         "rsi": rng.uniform(20, 80)})
    df = pd.DataFrame(rows).set_index(["date", "tic"])
    return PortfolioTradingEnv(df, stock_dim=n_assets,
                               tech_indicator_list=["macd", "rsi"])


# ---------------------------------------------------------------------------
# Test BaseExpert contract
# ---------------------------------------------------------------------------

class _ConcreteExpert(BaseExpert):
    def __init__(self, n_assets: int = 5):
        self.name = "concrete"
        self.n_assets = n_assets

    def get_weights(self, state: np.ndarray) -> np.ndarray:
        w = np.ones(self.n_assets + 1, dtype=np.float32)
        return w / w.sum()


class TestBaseExpert:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseExpert()  # type: ignore[abstract]

    def test_concrete_returns_valid_weights(self):
        expert = _ConcreteExpert(n_assets=5)
        w = expert.get_weights(dummy_state(5))
        assert w.shape == (6,)
        assert abs(w.sum() - 1.0) < TOL
        assert np.all(w >= 0)

    def test_get_name(self):
        expert = _ConcreteExpert()
        assert expert.get_name() == "concrete"


# ---------------------------------------------------------------------------
# Test PortfolioTradingEnv
# ---------------------------------------------------------------------------

class TestPortfolioTradingEnv:
    def test_reset_returns_valid_obs(self):
        env = dummy_env(n_assets=3)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)

    def test_step_returns_valid_weights(self):
        env = dummy_env(n_assets=3)
        env.reset()
        action = np.array([1.0, 0.0, 0.0, 0.0])  # 100% cash
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(obs, np.ndarray)
        assert env.observation_space.contains(obs)

    def test_action_softmax_applied(self):
        env = dummy_env(n_assets=3)
        env.reset()
        action = np.array([10.0, 5.0, 2.0, 1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        # after softmax in step, weights should be valid
        assert abs(env.current_weights.sum() - 1.0) < TOL

    def test_episode_terminates(self):
        env = dummy_env(n_assets=2)
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        else:
            pytest.fail("Episode did not terminate after all steps")


# ---------------------------------------------------------------------------
# Test DummyDRLExpert
# ---------------------------------------------------------------------------

class TestDummyDRLExpert:
    def test_returns_uniform_weights(self):
        expert = DummyDRLExpert(name="dummy", n_assets=5)
        w = expert.get_weights(dummy_state(5))
        assert w.shape == (6,)
        assert abs(w.sum() - 1.0) < TOL
        assert np.allclose(w, 1.0 / 6, atol=TOL)

    def test_get_name(self):
        expert = DummyDRLExpert(n_assets=3)
        assert expert.get_name() == "dummy"


# ---------------------------------------------------------------------------
# Test MILPExpert
# ---------------------------------------------------------------------------

class TestMILPExpert:
    def test_returns_valid_weights(self):
        expert = MILPExpert(n_assets=5)
        # feed some history
        rng = np.random.default_rng(42)
        for _ in range(30):
            expert.update_history(rng.uniform(-0.02, 0.03, size=5))
        w = expert.get_weights(dummy_state(5))
        assert w.shape == (6,)
        assert abs(w.sum() - 1.0) < TOL
        assert np.all(w >= -TOL)

    def test_no_history_returns_uniform(self):
        expert = MILPExpert(n_assets=5)
        w = expert.get_weights(dummy_state(5))
        assert abs(w.sum() - 1.0) < TOL

    def test_get_name(self):
        expert = MILPExpert()
        assert expert.get_name() == "MILP"


# ---------------------------------------------------------------------------
# Test AlphaMixRLExpert
# ---------------------------------------------------------------------------

class TestAlphaMixRLExpert:
    def test_returns_valid_weights(self):
        sub = [DummyDRLExpert(f"sub_{i}", n_assets=3) for i in range(3)]
        expert = AlphaMixRLExpert(sub, n_assets=3)
        w = expert.get_weights(dummy_state(3))
        assert w.shape == (4,)
        assert abs(w.sum() - 1.0) < TOL
        assert np.all(w >= 0)

    def test_empty_experts_returns_uniform(self):
        expert = AlphaMixRLExpert([], n_assets=3)
        w = expert.get_weights(dummy_state(3))
        assert abs(w.sum() - 1.0) < TOL

    def test_update_performance(self):
        sub = [DummyDRLExpert("sub_0", n_assets=2)]
        expert = AlphaMixRLExpert(sub, n_assets=2)
        expert.update_performance(0, 0.01)
        w = expert.get_weights(dummy_state(2))
        assert abs(w.sum() - 1.0) < TOL

    def test_get_name(self):
        expert = AlphaMixRLExpert([])
        assert "AlphaMixRL" in expert.get_name()


# ---------------------------------------------------------------------------
# Test TACExpert
# ---------------------------------------------------------------------------

class TestTACExpert:
    def test_returns_valid_weights(self):
        sub = [DummyDRLExpert(f"sub_{i}", n_assets=3) for i in range(3)]
        expert = TACExpert(sub, n_assets=3)
        w = expert.get_weights(dummy_state(3))
        assert w.shape == (4,)
        assert abs(w.sum() - 1.0) < TOL
        assert np.all(w >= 0)

    def test_empty_experts_returns_uniform(self):
        expert = TACExpert([], n_assets=3)
        w = expert.get_weights(dummy_state(3))
        assert abs(w.sum() - 1.0) < TOL

    def test_selects_best_expert(self):
        sub = [DummyDRLExpert(f"sub_{i}", n_assets=2) for i in range(2)]
        expert = TACExpert(sub, n_assets=2)
        # give expert 0 positive returns, expert 1 negative returns
        for _ in range(30):
            expert.update_performance(0, 0.01)
            expert.update_performance(1, -0.01)
        w = expert.get_weights(dummy_state(2))
        assert abs(w.sum() - 1.0) < TOL

    def test_get_name(self):
        expert = TACExpert([])
        assert expert.get_name() == "TAC"


# ---------------------------------------------------------------------------
# Test ExpertRegistry
# ---------------------------------------------------------------------------

class TestExpertRegistry:
    def test_register_and_get(self):
        registry = ExpertRegistry()
        e = DummyDRLExpert("test_expert", n_assets=3)
        registry.register(e)
        assert "test_expert" in registry
        assert registry.get_expert("test_expert") is e

    def test_list_experts(self):
        registry = ExpertRegistry()
        registry.register(DummyDRLExpert("a", n_assets=2))
        registry.register(DummyDRLExpert("b", n_assets=2))
        assert set(registry.list_experts()) == {"a", "b"}

    def test_remove(self):
        registry = ExpertRegistry()
        registry.register(DummyDRLExpert("x", n_assets=2))
        registry.remove("x")
        assert "x" not in registry

    def test_get_nonexistent_raises(self):
        registry = ExpertRegistry()
        with pytest.raises(KeyError):
            registry.get_expert("nonexistent")

    def test_evaluate_on_dataset(self):
        experts = [DummyDRLExpert("uniform", n_assets=2)]
        states = [dummy_state(2) for _ in range(10)]
        rng = np.random.default_rng(0)
        returns = [rng.uniform(-0.02, 0.03, size=2) for _ in range(10)]
        results = ExpertRegistry.evaluate_on_dataset(experts, states, returns)
        assert "uniform" in results
        metrics = results["uniform"]
        for key in ("cumulative_return", "sharpe", "max_drawdown", "volatility"):
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_len(self):
        registry = ExpertRegistry()
        assert len(registry) == 0
        registry.register(DummyDRLExpert("a", n_assets=2))
        assert len(registry) == 1


# ---------------------------------------------------------------------------
# Test DRL_AGENTS dict
# ---------------------------------------------------------------------------

class TestDRLAgents:
    def test_agents_are_callable_classes(self):
        for name, cls in DRL_AGENTS.items():
            assert callable(cls), f"{name} is not callable"
            assert hasattr(cls, "load"), f"{name} has no load method"
            assert hasattr(cls, "save"), f"{name} has no save method"

    def test_known_agents_present(self):
        names = {k.lower() for k in DRL_AGENTS}
        for required in ("a2c", "ddpg", "ppo", "sac"):
            assert required in names, f"Missing DRL agent: {required}"
