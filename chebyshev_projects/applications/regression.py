"""Chebyshev-basis regression and conditioning analysis."""

from __future__ import annotations

import numpy as np


def chebyshev_design_matrix(
    x: np.ndarray,
    degree: int,
    a: float = -1.0,
    b: float = 1.0,
) -> np.ndarray:
    """Build the Chebyshev design matrix [T_0(t), T_1(t), …, T_d(t)].

    Uses the three-term recurrence T_{k+1}(t) = 2t*T_k(t) - T_{k-1}(t).

    Parameters
    ----------
    x : np.ndarray
        Data points (m,) in [a, b].
    degree : int
        Maximum Chebyshev degree d (matrix has d+1 columns).
    a, b : float
        Interval endpoints for affine mapping to [-1, 1].

    Returns
    -------
    np.ndarray
        Design matrix of shape (m, degree+1).
    """
    if degree < 0:
        raise ValueError(f"Degree must be >= 0, got {degree}.")
    if a >= b:
        raise ValueError(f"Interval endpoints must satisfy a < b, got a={a}, b={b}.")

    x = np.asarray(x, dtype=np.float64)
    t = 2.0 * (x - a) / (b - a) - 1.0  # map to [-1, 1]
    m = t.size

    Phi = np.empty((m, degree + 1), dtype=np.float64)
    Phi[:, 0] = 1.0
    if degree >= 1:
        Phi[:, 1] = t
    for k in range(2, degree + 1):
        Phi[:, k] = 2.0 * t * Phi[:, k - 1] - Phi[:, k - 2]

    return Phi


def vandermonde_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Build the Vandermonde design matrix [1, x, x², …, x^d].

    Parameters
    ----------
    x : np.ndarray
        Data points (m,).
    degree : int
        Maximum polynomial degree d (matrix has d+1 columns).

    Returns
    -------
    np.ndarray
        Vandermonde matrix of shape (m, degree+1).
    """
    if degree < 0:
        raise ValueError(f"Degree must be >= 0, got {degree}.")

    x = np.asarray(x, dtype=np.float64)
    return np.vander(x, degree + 1, increasing=True)


def chebyshev_regression(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int,
    a: float = -1.0,
    b: float = 1.0,
) -> np.ndarray:
    """Least-squares regression in the Chebyshev basis.

    Solves min_c ||Φc - y||₂ where Φ is the Chebyshev design matrix.

    Parameters
    ----------
    x_data : np.ndarray
        Input data points (m,).
    y_data : np.ndarray
        Target values (m,).
    degree : int
        Maximum Chebyshev degree.
    a, b : float
        Interval for Chebyshev basis.

    Returns
    -------
    np.ndarray
        Coefficient vector c of length degree+1.
    """
    if x_data.size != y_data.size:
        raise ValueError("x_data and y_data must have the same length.")

    Phi = chebyshev_design_matrix(x_data, degree, a, b)
    coeffs, _, _, _ = np.linalg.lstsq(Phi, y_data.astype(np.float64), rcond=None)
    return coeffs


def conditioning_comparison(
    n_values: list[int],
    degree: int,
    a: float = -1.0,
    b: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compare condition numbers and singular values for Chebyshev vs Vandermonde.

    For each n in n_values, generates n data points on [a, b] and builds
    both design matrices, computing their 2-norm condition numbers and
    minimum/maximum singular values.

    Parameters
    ----------
    n_values : list of int
        Number of data points per experiment.
    degree : int
        Polynomial/Chebyshev degree.
    a, b : float
        Interval.

    Returns
    -------
    dict with keys:
        "n_values", "vandermonde_cond", "chebyshev_cond",
        "vandermonde_sv_min", "vandermonde_sv_max",
        "chebyshev_sv_min", "chebyshev_sv_max".
    """
    vand_cond = np.empty(len(n_values), dtype=np.float64)
    cheb_cond = np.empty(len(n_values), dtype=np.float64)
    vand_sv_min = np.empty(len(n_values), dtype=np.float64)
    vand_sv_max = np.empty(len(n_values), dtype=np.float64)
    cheb_sv_min = np.empty(len(n_values), dtype=np.float64)
    cheb_sv_max = np.empty(len(n_values), dtype=np.float64)

    for i, n in enumerate(n_values):
        x = np.linspace(a, b, n)

        V = vandermonde_matrix(x, degree)
        _, sv_v, _ = np.linalg.svd(V, full_matrices=False)
        vand_cond[i] = sv_v[0] / (sv_v[-1] if sv_v[-1] > 0 else np.finfo(float).tiny)
        vand_sv_min[i] = sv_v[-1]
        vand_sv_max[i] = sv_v[0]

        C = chebyshev_design_matrix(x, degree, a, b)
        _, sv_c, _ = np.linalg.svd(C, full_matrices=False)
        cheb_cond[i] = sv_c[0] / (sv_c[-1] if sv_c[-1] > 0 else np.finfo(float).tiny)
        cheb_sv_min[i] = sv_c[-1]
        cheb_sv_max[i] = sv_c[0]

    return {
        "n_values": np.array(n_values),
        "vandermonde_cond": vand_cond,
        "chebyshev_cond": cheb_cond,
        "vandermonde_sv_min": vand_sv_min,
        "vandermonde_sv_max": vand_sv_max,
        "chebyshev_sv_min": cheb_sv_min,
        "chebyshev_sv_max": cheb_sv_max,
    }
