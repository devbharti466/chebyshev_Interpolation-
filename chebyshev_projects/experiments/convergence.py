"""Convergence study engine for interpolation methods."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.nodes import chebyshev_nodes, equally_spaced_nodes
from ..core.barycentric import barycentric_weights, barycentric_interpolate


def max_norm_error(
    f: Callable[[np.ndarray], np.ndarray],
    interpolant_values: np.ndarray,
    grid: np.ndarray,
) -> float:
    """Compute the maximum (infinity) norm error on a grid.

    ||f - p||_∞ = max_x |f(x) - p(x)|

    Parameters
    ----------
    f : Callable
        The target function.
    interpolant_values : np.ndarray
        Interpolant evaluated at grid points.
    grid : np.ndarray
        Evaluation grid.

    Returns
    -------
    float
        Maximum absolute error.
    """
    return float(np.max(np.abs(f(grid) - interpolant_values)))


def convergence_study(
    function_list: list[tuple[str, Callable, str]],
    n_values: list[int],
    node_types: list[str] | None = None,
    a: float = -1.0,
    b: float = 1.0,
    n_eval: int = 5000,
) -> dict[tuple[str, str], np.ndarray]:
    """Systematic convergence study for multiple functions and node types.

    For each (function, node_type) combination, evaluates the maximum interpolation
    error for each value of n in n_values using barycentric interpolation.

    Parameters
    ----------
    function_list : list of (name, callable, description)
        Test functions; each entry is (name, f, description).
    n_values : list of int
        Node counts to test.
    node_types : list of str or None
        Node types to test; default ["chebyshev", "equispaced"].
    a, b : float
        Interpolation interval.
    n_eval : int
        Number of evaluation points for error estimation.

    Returns
    -------
    dict
        Keys are (func_name, node_type) tuples; values are error arrays
        of shape (len(n_values),).
    """
    if node_types is None:
        node_types = ["chebyshev", "equispaced"]

    x_eval = np.linspace(a, b, n_eval)
    results: dict[tuple[str, str], np.ndarray] = {}

    node_generators = {
        "chebyshev": chebyshev_nodes,
        "equispaced": equally_spaced_nodes,
    }

    for node_type in node_types:
        if node_type not in node_generators:
            raise ValueError(f"Unknown node type '{node_type}'. Choose from {list(node_generators)}.")
        gen = node_generators[node_type]

        for name, f, _ in function_list:
            errors = np.empty(len(n_values), dtype=np.float64)
            for i, n in enumerate(n_values):
                x_nodes = gen(n, a, b)
                y_nodes = f(x_nodes)
                weights = barycentric_weights(x_nodes)
                p_eval = barycentric_interpolate(x_nodes, y_nodes, x_eval, weights)
                errors[i] = max_norm_error(f, p_eval, x_eval)
            results[(name, node_type)] = errors

    return results
