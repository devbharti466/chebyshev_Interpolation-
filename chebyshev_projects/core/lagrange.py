"""Classical Lagrange interpolation (educational reference)."""

import numpy as np


def lagrange_interpolate(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """Classical Lagrange interpolation.

    Computes p(x) = Σ_k y_k * l_k(x), where
    l_k(x) = Π_{j≠k} (x - x_j) / (x_k - x_j).

    WARNING: O(n^2 * m) complexity. Numerically unstable for large n.
    Provided as an educational reference only; prefer barycentric_interpolate
    for production use.

    Parameters
    ----------
    x_nodes : np.ndarray
        Interpolation nodes (n,).
    y_nodes : np.ndarray
        Function values at nodes (n,).
    x_eval : np.ndarray
        Evaluation points (m,).

    Returns
    -------
    np.ndarray
        Interpolated values at x_eval (m,).

    Complexity: O(n^2 * m) time, O(n * m) space.
    """
    if x_nodes.size != y_nodes.size:
        raise ValueError("x_nodes and y_nodes must have the same length.")

    x_eval = np.asarray(x_eval, dtype=np.float64)
    n = x_nodes.size
    m = x_eval.size

    # diff[i, k] = x_eval[i] - x_nodes[k]
    diff = x_eval[:, np.newaxis] - x_nodes[np.newaxis, :]  # (m, n)

    result = np.zeros(m, dtype=np.float64)
    for k in range(n):
        # Basis polynomial l_k at all eval points
        num = np.delete(diff, k, axis=1).prod(axis=1)  # (m,)
        den = np.prod(x_nodes[k] - np.delete(x_nodes, k))
        if den == 0.0:
            raise ValueError(f"Duplicate node detected at index {k}.")
        result += y_nodes[k] * num / den

    return result
