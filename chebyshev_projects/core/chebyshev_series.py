"""Chebyshev series expansion and Clenshaw evaluation algorithm."""

from __future__ import annotations

from typing import Callable

import numpy as np


def chebyshev_coefficients(f: Callable[[np.ndarray], np.ndarray], n: int) -> np.ndarray:
    """Compute the first n Chebyshev expansion coefficients using a DCT-like formula.

    c_k = (2/n) * Σ_{j=0}^{n-1} f(x_j) * cos(k * (2j+1)*π / (2n)),  k = 0, …, n-1

    where x_j are the Chebyshev nodes of the first kind on [-1, 1].
    The k=0 coefficient uses the standard factor (1/n) (half the general factor).

    Parameters
    ----------
    f : Callable
        Function to expand; must accept a NumPy array.
    n : int
        Number of coefficients (must be >= 1).

    Returns
    -------
    np.ndarray
        Chebyshev coefficients c_0, c_1, ..., c_{n-1}.

    Complexity: O(n^2) time via vectorised outer product, O(n^2) space.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    # Chebyshev nodes in natural cosine order (descending): x_j = cos((2j+1)π/(2n))
    j_arr = np.arange(n, dtype=np.float64)
    x = np.cos((2.0 * j_arr + 1.0) / (2.0 * n) * np.pi)  # descending, shape (n,)
    fvals = f(x)  # shape (n,)

    k = np.arange(n, dtype=np.float64)        # (n,)
    j = np.arange(n, dtype=np.float64)        # (n,)
    # Cosine matrix: cos[k, j] = cos(k * (2j+1)*pi / (2n))
    cos_matrix = np.cos(
        np.outer(k, (2.0 * j + 1.0)) * np.pi / (2.0 * n)
    )  # (n, n)

    coeffs = (2.0 / n) * (cos_matrix @ fvals)
    coeffs[0] *= 0.5  # standard normalisation for c_0

    return coeffs


def clenshaw_evaluate(
    coeffs: np.ndarray,
    x_eval: np.ndarray,
    a: float = -1.0,
    b: float = 1.0,
) -> np.ndarray:
    """Evaluate a Chebyshev series via the Clenshaw backward recurrence.

    p(x) = Σ_{k=0}^{n-1} c_k * T_k(t),  t = 2(x-a)/(b-a) - 1 ∈ [-1, 1]

    The recurrence is:
        b_{n}   = b_{n+1} = 0
        b_k     = c_k + 2t * b_{k+1} - b_{k+2},  k = n-1, …, 1
        p(x)    = c_0 + t * b_1 - b_2

    Unconditionally numerically stable.

    Parameters
    ----------
    coeffs : np.ndarray
        Chebyshev coefficients c_0, …, c_{n-1}.
    x_eval : np.ndarray
        Evaluation points in [a, b].
    a, b : float
        Interval endpoints.

    Returns
    -------
    np.ndarray
        Values of the Chebyshev series at x_eval.

    Complexity: O(n*m) time, O(m) auxiliary space.
    """
    if a >= b:
        raise ValueError(f"Interval endpoints must satisfy a < b, got a={a}, b={b}.")

    x_eval = np.asarray(x_eval, dtype=np.float64)
    scalar_input = x_eval.ndim == 0
    x_eval = np.atleast_1d(x_eval)

    # Map to [-1, 1]
    t = 2.0 * (x_eval - a) / (b - a) - 1.0

    n = coeffs.size
    if n == 0:
        return np.zeros_like(t)
    if n == 1:
        return np.full_like(t, coeffs[0])

    b_next2 = np.zeros_like(t)
    b_next1 = np.zeros_like(t)

    for k in range(n - 1, 0, -1):
        b_cur = coeffs[k] + 2.0 * t * b_next1 - b_next2
        b_next2 = b_next1
        b_next1 = b_cur

    result = coeffs[0] + t * b_next1 - b_next2

    return result[0] if scalar_input else result


def chebyshev_condition_number(
    coeffs: np.ndarray,
    x_eval: np.ndarray,
    a: float = -1.0,
    b: float = 1.0,
) -> np.ndarray:
    """Pointwise condition number of Chebyshev series evaluation.

    κ(x) = Σ_k |c_k * T_k(t)| / |Σ_k c_k * T_k(t)|

    A large κ(x) indicates potential cancellation at that point.

    Parameters
    ----------
    coeffs : np.ndarray
        Chebyshev coefficients.
    x_eval : np.ndarray
        Evaluation points in [a, b].
    a, b : float
        Interval endpoints.

    Returns
    -------
    np.ndarray
        Pointwise condition numbers at x_eval.
    """
    if a >= b:
        raise ValueError(f"Interval endpoints must satisfy a < b, got a={a}, b={b}.")

    x_eval = np.asarray(x_eval, dtype=np.float64)
    x_eval = np.atleast_1d(x_eval)
    t = 2.0 * (x_eval - a) / (b - a) - 1.0  # (m,)

    n = coeffs.size
    if n == 0:
        return np.zeros_like(t)

    # Build T_k(t) for k = 0, …, n-1 via three-term recurrence
    m = t.size
    T = np.empty((n, m), dtype=np.float64)
    T[0, :] = 1.0
    if n > 1:
        T[1, :] = t
    for k in range(2, n):
        T[k, :] = 2.0 * t * T[k - 1, :] - T[k - 2, :]

    # Weighted sum: (n,) broadcast over (n, m)
    terms = coeffs[:, np.newaxis] * T  # (n, m)
    numerator = np.sum(np.abs(terms), axis=0)
    denominator = np.abs(np.sum(terms, axis=0))

    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(denominator == 0.0, np.inf, numerator / denominator)

    return kappa
