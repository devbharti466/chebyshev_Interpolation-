"""Stability and condition number analysis for interpolation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.nodes import chebyshev_nodes, equally_spaced_nodes
from ..core.barycentric import barycentric_weights, barycentric_interpolate
from ..applications.regression import chebyshev_design_matrix, vandermonde_matrix


def interpolation_condition_analysis(
    f: Callable[[np.ndarray], np.ndarray],
    n_values: list[int],
    n_trials: int = 50,
    noise_level: float = 1e-14,
    a: float = -1.0,
    b: float = 1.0,
    n_eval: int = 1000,
) -> dict[str, np.ndarray]:
    """Sensitivity to machine-epsilon-scale data perturbation.

    For each n in n_values, repeatedly perturbs the function values at
    Chebyshev nodes with uniform noise of magnitude noise_level and measures
    the maximum change in the interpolant over a fine evaluation grid.

    Parameters
    ----------
    f : Callable
        Target function.
    n_values : list of int
        Node counts to test.
    n_trials : int
        Number of random perturbation trials per n.
    noise_level : float
        Magnitude of data perturbation.
    a, b : float
        Interpolation interval.
    n_eval : int
        Number of evaluation points.

    Returns
    -------
    dict with keys "n_values", "mean_amplification", "max_amplification".
    Each value is an array of length len(n_values).
    """
    rng = np.random.default_rng(seed=0)
    x_eval = np.linspace(a, b, n_eval)

    mean_amp = np.empty(len(n_values), dtype=np.float64)
    max_amp = np.empty(len(n_values), dtype=np.float64)

    for i, n in enumerate(n_values):
        x_nodes = chebyshev_nodes(n, a, b)
        y_clean = f(x_nodes)
        weights = barycentric_weights(x_nodes)
        p_clean = barycentric_interpolate(x_nodes, y_clean, x_eval, weights)

        trial_max = np.empty(n_trials, dtype=np.float64)
        for t in range(n_trials):
            noise = rng.uniform(-noise_level, noise_level, size=n)
            y_noisy = y_clean + noise
            p_noisy = barycentric_interpolate(x_nodes, y_noisy, x_eval, weights)
            delta_p = np.max(np.abs(p_noisy - p_clean))
            trial_max[t] = delta_p / (noise_level + 1e-300)

        mean_amp[i] = float(np.mean(trial_max))
        max_amp[i] = float(np.max(trial_max))

    return {
        "n_values": np.array(n_values),
        "mean_amplification": mean_amp,
        "max_amplification": max_amp,
    }


def vandermonde_vs_chebyshev_condition(
    n_values: list[int],
    a: float = -1.0,
    b: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compare condition numbers of Vandermonde vs Chebyshev design matrices.

    For each n, constructs both matrices on the respective node sets and
    computes their 2-norm condition numbers.

    Parameters
    ----------
    n_values : list of int
        Node counts (= degree + 1 for square systems).
    a, b : float
        Interpolation interval.

    Returns
    -------
    dict with keys "n_values", "vandermonde_cond", "chebyshev_cond".
    """
    vand_cond = np.empty(len(n_values), dtype=np.float64)
    cheb_cond = np.empty(len(n_values), dtype=np.float64)

    for i, n in enumerate(n_values):
        # Vandermonde matrix on equispaced nodes
        x_eq = equally_spaced_nodes(n, a, b)
        V = vandermonde_matrix(x_eq, n - 1)
        vand_cond[i] = np.linalg.cond(V)

        # Chebyshev design matrix on Chebyshev nodes
        x_cheb = chebyshev_nodes(n, a, b)
        C = chebyshev_design_matrix(x_cheb, n - 1, a, b)
        cheb_cond[i] = np.linalg.cond(C)

    return {
        "n_values": np.array(n_values),
        "vandermonde_cond": vand_cond,
        "chebyshev_cond": cheb_cond,
    }
