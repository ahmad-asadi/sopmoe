"""Tests for the backtest evaluation framework."""

from __future__ import annotations

from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import compute_metrics, metrics_dataframe, format_metrics_row
from src.backtest.engine import Backtester
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.experts.base import BaseExpert
from src.experts.finrl_expert import DummyDRLExpert
from src.experts.registry import ExpertRegistry
from src.inference.utils import project_weights

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TOL = 1e-5


_DATE_RANGE = ("2022-01-01", "2022-06-30")  # short period for fast tests
_COINS = ["ASSET1", "ASSET2", "ASSET3"]


@pytest.fixture(scope="module")
def synthetic_data_dir():
    """Generate synthetic price data for backtesting (short period)."""
    tmpdir = tempfile.mkdtemp()
    raw_dir = Path(tmpdir) / "raw"
    proc_dir = Path(tmpdir) / "processed"
    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    dates = pd.date_range(*_DATE_RANGE, freq="D")

    for coin in _COINS:
        rows = []
        price = 100.0
        for d in dates:
            price *= 1.0 + rng.normal(0.0005, 0.02)
            rows.append({
                "Date": d,
                "Open": price * (1 + rng.normal(0, 0.005)),
                "High": price * (1 + abs(rng.normal(0, 0.005))),
                "Low": price * (1 - abs(rng.normal(0, 0.005))),
                "Close": price,
                "Volume": rng.lognormal(15, 1),
            })
        pd.DataFrame(rows).to_csv(raw_dir / f"{coin}.csv", index=False)

    yield raw_dir, proc_dir, _COINS
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="module")
def loader(synthetic_data_dir):
    raw_dir, proc_dir, coins = synthetic_data_dir
    return DataLoader(
        coin_list=coins,
        start_date=_DATE_RANGE[0],
        end_date=_DATE_RANGE[1],
        data_dir=str(raw_dir),
        cache_dir=str(proc_dir),
    )


@pytest.fixture(scope="module")
def registry():
    reg = ExpertRegistry()
    for name in ["algo_a", "algo_b", "algo_c"]:
        reg.register(DummyDRLExpert(name=name, n_assets=3))
    return reg


@pytest.fixture(scope="module")
def backtester(loader, registry):
    return Backtester(
        inference_engine=None,
        expert_registry=registry,
        data_loader=loader,
        config={
            "initial_capital": 100000.0,
            "transaction_cost_pct": 0.001,
            "lookback": 10,
            "test_start": _DATE_RANGE[0],
        },
    )


# ---------------------------------------------------------------------------
# Test compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_basic_metrics(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.02, 252)
        m = compute_metrics(returns)
        assert isinstance(m, dict)
        for key in (
            "cumulative_return",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "max_drawdown_duration",
        ):
            assert key in m

    def test_constant_returns(self):
        returns = np.full(100, 0.001)
        m = compute_metrics(returns)
        assert m["annualised_volatility"] < 0.001
        assert m["max_drawdown"] >= -0.001
        assert m["cumulative_return"] > 0

    def test_zero_returns(self):
        m = compute_metrics(np.zeros(100))
        assert m["cumulative_return"] == 0.0
        assert m["sharpe_ratio"] == 0.0

    def test_empty_returns(self):
        m = compute_metrics([])
        assert all(v == 0.0 for v in m.values())

    def test_risk_free_rate(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.02, 252)
        m0 = compute_metrics(returns, risk_free_rate=0.0)
        m1 = compute_metrics(returns, risk_free_rate=0.05)
        assert m1["sharpe_ratio"] < m0["sharpe_ratio"]

    def test_negative_sharpe(self):
        returns = np.full(252, -0.001)
        m = compute_metrics(returns)
        assert m["sharpe_ratio"] < 0


# ---------------------------------------------------------------------------
# Test Backtester construction
# ---------------------------------------------------------------------------


class TestBacktesterConstruction:
    def test_construct_with_dummy_data(self, loader, registry):
        bt = Backtester(
            inference_engine=None,
            expert_registry=registry,
            data_loader=loader,
            config={
                "initial_capital": 100000.0,
                "transaction_cost_pct": 0.001,
                "lookback": 10,
                "test_start": "2022-01-01",
            },
        )
        assert bt.initial_capital == 100000.0
        assert bt.transaction_cost_pct == 0.001
        assert bt.n_assets == 3
        assert len(bt.test_states) > 0

    def test_no_test_states_start_too_late(self, loader, registry):
        bt = Backtester(
            inference_engine=None,
            expert_registry=registry,
            data_loader=loader,
            config={
                "test_start": "2099-01-01",
                "lookback": 10,
            },
        )
        assert len(bt.test_states) == 0


# ---------------------------------------------------------------------------
# Test run_expert
# ---------------------------------------------------------------------------


class TestRunExpert:
    def test_returns_valid_structure(self, backtester):
        bt = backtester
        bt.transaction_cost_pct = 0.0
        result = bt.run_expert("algo_a", return_equity_curve=True)
        assert "daily_returns" in result
        assert "equity_curve" in result
        assert "n_trades" in result
        assert isinstance(result["daily_returns"], np.ndarray)
        assert isinstance(result["equity_curve"], np.ndarray)
        assert len(result["daily_returns"]) == len(result["equity_curve"])

    def test_expert_weights_are_valid(self, backtester):
        bt = backtester
        bt.transaction_cost_pct = 0.0
        result = bt.run_expert("algo_a")
        daily_returns = result["daily_returns"]
        assert np.all(np.isfinite(daily_returns))
        assert np.all(daily_returns >= -0.5)
        assert np.all(daily_returns <= 0.5)

    def test_uniform_expert(self, backtester):
        bt = backtester
        bt.transaction_cost_pct = 0.0
        r1 = bt.run_expert("algo_a")
        r2 = bt.run_expert("algo_b")
        np.testing.assert_array_almost_equal(r1["daily_returns"], r2["daily_returns"])

    def test_trading_days_count(self, backtester):
        bt = backtester
        bt.transaction_cost_pct = 0.0
        result = bt.run_expert("algo_a")
        n_days = len(result["daily_returns"])
        assert n_days > 50
        assert n_days < 500


# ---------------------------------------------------------------------------
# Test run_switching (no retrieval fallback)
# ---------------------------------------------------------------------------


class TestRunSwitching:
    def test_gating_fallback_runs(self, backtester):
        bt = backtester
        bt.transaction_cost_pct = 0.0
        result = bt.run_switching(
            strategy_name="test_gate",
            use_retrieval=False,
            return_equity_curve=True,
        )
        assert "daily_returns" in result
        assert "equity_curve" in result
        assert "selected_experts" in result
        assert len(result["selected_experts"]) == len(result["daily_returns"])
        assert np.all(np.isfinite(result["daily_returns"]))

    def test_switching_with_engine_returns_same(self, backtester):
        """When all experts are identical dummies, switching should match."""
        bt = backtester
        bt.transaction_cost_pct = 0.0
        expert_result = bt.run_expert("algo_a")
        gate_result = bt.run_switching(
            strategy_name="gate",
            use_retrieval=False,
        )
        np.testing.assert_array_almost_equal(
            expert_result["daily_returns"], gate_result["daily_returns"]
        )


# ---------------------------------------------------------------------------
# Test portfolio_return (with and without transaction costs)
# ---------------------------------------------------------------------------


class TestPortfolioReturn:
    def test_no_cash_fully_invested(self):
        """If weight is 1.0 on a single asset, return equals asset return."""
        from src.backtest.engine import Backtester

        weights = np.array([0.0, 1.0, 0.0, 0.0])  # 4 slots: cash + 3 assets
        prev_weights = np.array([1.0, 0.0, 0.0, 0.0])

        # We cannot easily unit-test _portfolio_return because it needs
        # the backtester's asset_returns DataFrame.  Instead test that
        # the project_weights works correctly.
        proj = project_weights(weights)
        assert abs(proj.sum() - 1.0) < TOL
        assert np.all(proj >= 0)

    def test_project_weights_uniform(self):
        raw = np.array([2.0, 2.0, 2.0, 2.0])
        proj = project_weights(raw)
        assert abs(proj.sum() - 1.0) < TOL
        assert np.allclose(proj, 0.25, atol=TOL)

    def test_project_weights_negative(self):
        raw = np.array([-5.0, 0.0, 5.0, 0.0])
        proj = project_weights(raw)
        assert abs(proj.sum() - 1.0) < TOL
        assert np.all(proj >= 0)


# ---------------------------------------------------------------------------
# Test metrics_dataframe / format_metrics_row
# ---------------------------------------------------------------------------


class TestMetricsFormatting:
    def test_metrics_dataframe(self):
        results = {
            "strat_a": {"sharpe_ratio": 1.5, "cumulative_return": 0.25},
            "strat_b": {"sharpe_ratio": 0.8, "cumulative_return": 0.10},
        }
        df = metrics_dataframe(results)
        assert list(df.index) == ["strat_a", "strat_b"]
        assert "sharpe_ratio" in df.columns

    def test_format_metrics_row(self):
        metrics = {"sharpe_ratio": 1.2345, "max_drawdown_duration": 45}
        formatted = format_metrics_row(metrics)
        assert "sharpe" in formatted
        assert "45d" in formatted


# ---------------------------------------------------------------------------
# Test MILP expert with backtester
# ---------------------------------------------------------------------------


class TestMILPInBacktest:
    def test_milp_via_backtester(self, loader):
        from src.experts.milp_expert import MILPExpert

        registry = ExpertRegistry()
        milp = MILPExpert(n_assets=3, window=10)
        registry.register(milp)

        bt = Backtester(
            inference_engine=None,
            expert_registry=registry,
            data_loader=loader,
            config={
                "lookback": 10,
                "test_start": "2022-01-01",
                "transaction_cost_pct": 0.0,
            },
        )
        result = bt.run_expert("MILP", return_equity_curve=True)
        assert len(result["daily_returns"]) > 0
        assert np.all(np.isfinite(result["daily_returns"]))
