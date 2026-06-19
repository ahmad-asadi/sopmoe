"""Utility functions for the online inference pipeline."""

from __future__ import annotations

import numpy as np


def project_weights(weights: np.ndarray) -> np.ndarray:
    """Project a weight vector onto the probability simplex via softmax.

    Handles edge cases: all-zero inputs, negative values, and NaN.
    Returns a non-negative vector that sums to 1 (within numerical tolerance).

    Parameters
    ----------
    weights :
        Raw score vector of shape ``(n_assets,)``.

    Returns
    -------
    np.ndarray
        Projected weights of the same shape, summing to 1.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(np.isnan(weights)):
        weights = np.nan_to_num(weights, nan=0.0)
    weights = np.clip(weights, -1e10, 1e10)
    exp_w = np.exp(weights - np.max(weights))
    proj = exp_w / (np.sum(exp_w) + 1e-10)
    return proj.astype(np.float32)


def similarity_from_distance(distances: np.ndarray) -> np.ndarray:
    """Convert L2 distances to normalised similarity weights.

    Uses ``sim = 1 / (1 + d)`` so that closer neighbours get higher weight.
    The result is normalised to sum to 1.

    Parameters
    ----------
    distances :
        Array of L2 distances from FAISS.

    Returns
    -------
    np.ndarray
        Normalised similarity weights, same shape as ``distances``.
    """
    distances = np.asarray(distances, dtype=np.float64)
    sim = 1.0 / (1.0 + distances)
    sim = np.where(np.isfinite(sim), sim, 0.0)
    total = np.sum(sim) + 1e-10
    return (sim / total).astype(np.float32)
