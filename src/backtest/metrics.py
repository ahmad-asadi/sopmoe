# -*- coding: utf-8 -*-
"""Performance metrics for portfolio backtesting.

All metrics are computed from a series of daily portfolio returns and
reported in annualised form following standard finance conventions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_metrics(
    returns_series: np.ndarray | list[float] | pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute a full suite of annualised performance metrics.

    Parameters
    ----------
    returns_series :
        Daily portfolio returns (e.g. 0.01 for 1 %).
    risk_free_rate :
        Annualised risk-free rate (default 0.0).

    Returns
    -------
    dict[str, float]
        Keys:
        - ``cumulative_return``   - total ROI over the period
        - ``annualised_return``   - geometric annualised return
        - ``annualised_volatility``
        - ``sharpe_ratio``        - annualised
        - ``sortino_ratio``       - annualised
        - ``max_drawdown``        - deepest peak-to-trough decline
        - ``max_drawdown_duration`` - longest drawdown in trading days
    """
    returns = np.asarray(returns_series, dtype=np.float64)
    if len(returns) == 0:
        return {k: 0.0 for k in _METRIC_KEYS}

    n_days = len(returns)
    ann_factor = 252.0

    # --- Cumulative return (ROI) ---
    cumulative_return = float(np.prod(1.0 + returns)) - 1.0

    # --- Annualised return (geometric) ---
    annualised_return = float(np.expm1(np.log1p(cumulative_return) * ann_factor / n_days))

    # --- Annualised volatility ---
    annualised_vol = float(np.std(returns, ddof=1)) * np.sqrt(ann_factor)
    if annualised_vol < 1e-12:
        annualised_vol = 1e-12

    # --- Daily risk-free rate ---
    daily_rf = risk_free_rate / ann_factor

    # --- Sharpe ratio ---
    excess = returns - daily_rf
    sharpe = float(np.mean(excess) / (np.std(returns, ddof=1) + 1e-12)) * np.sqrt(ann_factor)

    # --- Sortino ratio (downside deviation) ---
    downside = excess[excess < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    if downside_std < 1e-12:
        downside_std = 1e-12
    sortino = float(np.mean(excess) / downside_std) * np.sqrt(ann_factor)

    # --- Maximum drawdown ---
    cum = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / (running_max + 1e-12)
    max_dd = float(np.min(dd))

    # --- Max drawdown duration (trading days) ---
    dd_series = pd.Series(dd)
    in_dd = dd_series < -1e-12
    durations = []
    current = 0
    for flag in in_dd:
        if flag:
            current += 1
        else:
            if current > 0:
                durations.append(current)
            current = 0
    if current > 0:
        durations.append(current)
    max_dd_duration = max(durations) if durations else 0

    return {
        "cumulative_return": round(cumulative_return, 6),
        "annualised_return": round(annualised_return, 6),
        "annualised_volatility": round(annualised_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_duration": max_dd_duration,
    }


_METRIC_KEYS = [
    "cumulative_return",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "max_drawdown_duration",
]


def metrics_dataframe(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Convert a nested ``{strategy_name: {metric: value}}`` dict to a DataFrame."""
    df = pd.DataFrame(results).T
    df.index.name = "strategy"
    return df


def format_metrics_row(
    metrics: dict[str, Any],
    decimals: dict[str, int] | None = None,
) -> str:
    """Format a single row of metrics for console output."""
    decimals = decimals or {
        "cumulative_return": 4,
        "annualised_return": 4,
        "annualised_volatility": 4,
        "sharpe_ratio": 3,
        "sortino_ratio": 3,
        "max_drawdown": 4,
    }
    parts = []
    for k, v in metrics.items():
        if k == "max_drawdown_duration":
            parts.append(f"{k}={int(v)}d")
        elif isinstance(v, float):
            nd = decimals.get(k, 4)
            parts.append(f"{k}={v:.{nd}f}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)
