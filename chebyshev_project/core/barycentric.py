"""Barycentric interpolation weights and evaluation."""

import numpy as np


def barycentric_weights(x: np.ndarray) -> np.ndarray:
    """Compute barycentric interpolation weights for general nodes.

    Uses log-domain computation to prevent overflow/underflow for large n.
    Weights are normalised so that max|w_k| = 1.

    Parameters
    ----------
    x : np.ndarray
        Array of n distinct interpolation nodes.

    Returns
    -------
    np.ndarray
        Array of n barycentric weights.

    Complexity: O(n^2) time, O(n) space.
    """
    n = x.size
    if n < 1:
        raise ValueError("Node array must be non-empty.")

    w = np.empty(n, dtype=np.float64)
    for k in range(n):
        diff = x[k] - np.delete(x, k)
        if np.any(diff == 0.0):
            raise ValueError("Nodes must be distinct.")
        log_abs = np.sum(np.log(np.abs(diff)))
        sign = np.prod(np.sign(diff))
        w[k] = sign * np.exp(-log_abs)

    # Normalise to prevent scale issues
    w /= np.max(np.abs(w))
    return w


def barycentric_interpolate(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_eval: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the polynomial interpolant via the second barycentric formula.

    p(x) = (Σ_k w_k * f_k / (x - x_k)) / (Σ_k w_k / (x - x_k))

    Points in x_eval that coincide with a node return the node value directly,
    avoiding division by zero.

    Parameters
    ----------
    x_nodes : np.ndarray
        Interpolation nodes (n,).
    y_nodes : np.ndarray
        Function values at nodes (n,).
    x_eval : np.ndarray
        Evaluation points (m,).
    weights : np.ndarray or None
        Pre-computed barycentric weights; computed if None.

    Returns
    -------
    np.ndarray
        Interpolated values at x_eval (m,).

    Complexity: O(n*m) time, O(n+m) space (after weight computation).
    """
    if x_nodes.size != y_nodes.size:
        raise ValueError("x_nodes and y_nodes must have the same length.")
    if x_eval.ndim == 0:
        x_eval = x_eval.reshape(1)

    if weights is None:
        weights = barycentric_weights(x_nodes)

    x_eval = np.asarray(x_eval, dtype=np.float64)
    result = np.empty(x_eval.size, dtype=np.float64)

    # Difference matrix: (m, n)
    diff = x_eval[:, np.newaxis] - x_nodes[np.newaxis, :]  # (m, n)

    # Detect evaluation points that coincide with a node (within machine epsilon)
    coincident = np.abs(diff) < np.finfo(np.float64).eps * np.max(np.abs(x_nodes) + 1.0)
    node_match = np.any(coincident, axis=1)

    # Non-coincident points: standard barycentric formula
    mask = ~node_match
    if np.any(mask):
        d = diff[mask, :]  # (m', n)
        w_over_d = weights[np.newaxis, :] / d  # (m', n)
        result[mask] = (w_over_d * y_nodes[np.newaxis, :]).sum(axis=1) / w_over_d.sum(axis=1)

    # Coincident points: return node value directly
    if np.any(node_match):
        idx_eval = np.where(node_match)[0]
        for ie in idx_eval:
            k = np.argmin(np.abs(diff[ie, :]))
            result[ie] = y_nodes[k]

    return result
