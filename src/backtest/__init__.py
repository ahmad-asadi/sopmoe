# -*- coding: utf-8 -*-
"""Backtesting and performance evaluation.

Provides:
- ``Backtester``      - full backtesting loop for experts and switching strategies
- ``compute_metrics`` - annualised performance metrics (Sharpe, Sortino, MDD, ...)
- ``plot_equity_curves``, ``plot_drawdowns``, etc.
"""

from src.backtest.engine import Backtester
from src.backtest.metrics import compute_metrics, metrics_dataframe, format_metrics_row
from src.backtest.plotting import (
    plot_equity_curves,
    plot_drawdowns,
    plot_weights_heatmap,
    plot_metrics_comparison,
)

__all__ = [
    "Backtester",
    "compute_metrics",
    "metrics_dataframe",
    "format_metrics_row",
    "plot_equity_curves",
    "plot_drawdowns",
    "plot_weights_heatmap",
    "plot_metrics_comparison",
]
