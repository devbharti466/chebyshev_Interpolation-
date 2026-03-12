"""Newton interpolation via divided differences."""

import numpy as np


def divided_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Newton divided difference coefficients in-place.

    Returns the first row of the divided difference table, which contains
    the Newton coefficients [f[x_0], f[x_0,x_1], ..., f[x_0,...,x_{n-1}]].

    Parameters
    ----------
    x : np.ndarray
        Interpolation nodes (n,); must be distinct.
    y : np.ndarray
        Function values at nodes (n,).

    Returns
    -------
    np.ndarray
        Newton coefficients (n,).

    Complexity: O(n^2) time, O(n) space (in-place on a copy of y).
    """
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    n = x.size
    coeff = y.copy().astype(np.float64)

    for j in range(1, n):
        denom = x[j:] - x[: n - j]
        if np.any(denom == 0.0):
            raise ValueError("Nodes must be distinct for divided differences.")
        coeff[j:] = (coeff[j:] - coeff[j - 1 : n - 1]) / denom

    return coeff


def newton_interpolate(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """Newton form interpolation with Horner-like nested evaluation.

    Uses the Newton divided difference representation:
    p(x) = c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ...

    Evaluated efficiently via nested multiplication (Horner scheme).

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

    Complexity: O(n^2) setup + O(n*m) evaluation.
    """
    if x_nodes.size != y_nodes.size:
        raise ValueError("x_nodes and y_nodes must have the same length.")

    x_eval = np.asarray(x_eval, dtype=np.float64)
    coeff = divided_differences(x_nodes, y_nodes)
    n = coeff.size

    # Horner-like nested evaluation: traverse coefficients from last to first
    result = np.full(x_eval.shape, coeff[n - 1], dtype=np.float64)
    for k in range(n - 2, -1, -1):
        result = result * (x_eval - x_nodes[k]) + coeff[k]

    return result
