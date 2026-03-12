"""Lebesgue constant estimation via the barycentric formula."""

from __future__ import annotations

import numpy as np

from ..core.barycentric import barycentric_weights


def lebesgue_function(
    x_nodes: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """Compute the Lebesgue function Λ(x) = Σ_k |l_k(x)|.

    Uses barycentric weights to compute the Lagrange basis functions in a
    numerically stable manner.

    Parameters
    ----------
    x_nodes : np.ndarray
        Interpolation nodes (n,).
    x_eval : np.ndarray
        Evaluation grid (m,).

    Returns
    -------
    np.ndarray
        Lebesgue function values at x_eval (m,).
    """
    x_eval = np.asarray(x_eval, dtype=np.float64)
    weights = barycentric_weights(x_nodes)

    # diff[i, k] = x_eval[i] - x_nodes[k], shape (m, n)
    diff = x_eval[:, np.newaxis] - x_nodes[np.newaxis, :]

    # Identify evaluation points that coincide with nodes
    eps = np.finfo(np.float64).eps * (np.max(np.abs(x_nodes)) + 1.0)
    coincident_mask = np.abs(diff) < eps  # (m, n)

    leb = np.empty(x_eval.size, dtype=np.float64)

    # Points not coinciding with any node
    non_coincident = ~np.any(coincident_mask, axis=1)
    if np.any(non_coincident):
        d = diff[non_coincident, :]
        w_over_d = weights[np.newaxis, :] / d          # (m', n)
        denom = np.abs(w_over_d.sum(axis=1))           # (m',)
        numerator = np.sum(np.abs(w_over_d), axis=1)   # (m',)
        with np.errstate(invalid="ignore", divide="ignore"):
            leb[non_coincident] = np.where(denom == 0.0, 0.0, numerator / denom)

    # Points coinciding with a node: l_k = 1, all others = 0
    if np.any(~non_coincident):
        leb[~non_coincident] = 1.0

    return leb


def lebesgue_constant(
    x_nodes: np.ndarray,
    n_eval: int = 10000,
) -> float:
    """Estimate the Lebesgue constant Λ_n = max_x Λ(x).

    Uses a fine uniform grid to approximate the supremum of the Lebesgue function.

    Parameters
    ----------
    x_nodes : np.ndarray
        Interpolation nodes (n,).
    n_eval : int
        Number of evaluation points (>= 10000 recommended).

    Returns
    -------
    float
        Estimated Lebesgue constant.
    """
    if n_eval < 2:
        raise ValueError(f"n_eval must be >= 2, got {n_eval}.")

    a, b = x_nodes.min(), x_nodes.max()
    x_eval = np.linspace(a, b, n_eval)
    leb = lebesgue_function(x_nodes, x_eval)
    return float(np.max(leb))
