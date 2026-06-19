# -*- coding: utf-8 -*-
"""Plotting utilities for backtest results - equity curves, drawdowns, tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Consistent style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def plot_equity_curves(
    results: dict[str, np.ndarray],
    title: str = "Equity Curves",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    initial_capital: float = 100_000.0,
) -> plt.Figure:
    """Plot equity curves for one or more strategies.

    Parameters
    ----------
    results :
        ``{strategy_name: daily_returns_array}``
    title :
        Plot title.
    save_path :
        If given, save the figure to this path.
    figsize :
        Figure dimensions ``(width, height)``.
    initial_capital :
        Starting portfolio value for scaling the equity curve.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", n_colors=len(results))
    for (name, returns), color in zip(results.items(), colors):
        equity = initial_capital * np.cumprod(1.0 + returns)
        ax.plot(equity, label=name, color=color, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Equity curves saved to %s", save_path)

    return fig


def plot_drawdowns(
    returns: np.ndarray,
    strategy_name: str = "Strategy",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plot the drawdown curve for a single strategy.

    Parameters
    ----------
    returns :
        Daily returns array.
    strategy_name :
        Label for the plot title.
    save_path :
        Optional save destination.
    figsize :
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cum = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / (running_max + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(range(len(dd)), dd * 100, 0, color="crimson", alpha=0.4)
    ax.set_title(f"Drawdown - {strategy_name}")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Drawdown plot saved to %s", save_path)

    return fig


def plot_weights_heatmap(
    weights_history: list[np.ndarray] | np.ndarray,
    expert_names: list[str] | None = None,
    title: str = "Expert Selection Over Time",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 3),
) -> plt.Figure:
    """Heatmap of the selected expert at each timestep (one-hot encoded).

    Parameters
    ----------
    weights_history :
        Array of shape ``(n_steps, n_experts)`` or list thereof.
    expert_names :
        Labels for each column (expert).  Auto-generated if ``None``.
    title :
        Plot title.
    save_path :
        Optional save destination.
    figsize :
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.asarray(weights_history)
    if data.ndim == 1:
        data = np.eye(len(np.unique(data)))[data]

    if expert_names is None:
        expert_names = [f"E{i}" for i in range(data.shape[1])]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.T, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(len(expert_names)))
    ax.set_yticklabels(expert_names)
    ax.set_xlabel("Trading Day")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Weights heatmap saved to %s", save_path)

    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    metric_cols: list[str] | None = None,
    title: str = "Strategy Comparison",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Grouped bar chart comparing strategies across selected metrics.

    Parameters
    ----------
    metrics_df :
        DataFrame with strategies as rows and metrics as columns.
    metric_cols :
        Metric columns to plot.  Defaults to Sharpe, Sortino, Vol, MDD.
    title :
        Plot title.
    save_path :
        Optional save destination.
    figsize :
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metric_cols is None:
        metric_cols = ["sharpe_ratio", "sortino_ratio", "annualised_volatility", "max_drawdown"]

    available = [c for c in metric_cols if c in metrics_df.columns]
    if not available:
        logger.warning("No matching metric columns found in %s", list(metrics_df.columns))
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    df = metrics_df[available].copy()
    df.index = df.index.str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind="bar", ax=ax, rot=30, colormap="viridis", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Metrics comparison saved to %s", save_path)

    return fig
