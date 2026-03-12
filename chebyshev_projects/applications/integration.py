"""Numerical integration via interpolation-based quadrature."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.nodes import chebyshev_nodes
from ..core.chebyshev_series import chebyshev_coefficients


def chebyshev_quadrature(
    f: Callable[[np.ndarray], np.ndarray],
    n: int,
    a: float = -1.0,
    b: float = 1.0,
) -> float:
    """Clenshaw-Curtis-type quadrature: integrate the Chebyshev interpolant exactly.

    Integrates the degree-(n-1) Chebyshev interpolant of f over [a, b]:
    ∫_a^b p(x) dx = (b-a)/2 * Σ_k c_k * ∫_{-1}^{1} T_k(t) dt

    Using ∫_{-1}^{1} T_k(t) dt = 0 for odd k, and
    = 2/(1 - k²) for even k ≠ 0, and = 2 for k = 0.

    Parameters
    ----------
    f : Callable
        Integrand.
    n : int
        Number of Chebyshev nodes (interpolation degree n-1).
    a, b : float
        Integration interval.

    Returns
    -------
    float
        Approximation of ∫_a^b f(x) dx.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if a >= b:
        raise ValueError(f"Interval must satisfy a < b, got a={a}, b={b}.")

    # Map f to [-1, 1] via substitution
    def f_mapped(t: np.ndarray) -> np.ndarray:
        x = 0.5 * (a + b) + 0.5 * (b - a) * t
        return f(x)

    coeffs = chebyshev_coefficients(f_mapped, n)

    # Integration weights: w_k = ∫_{-1}^1 T_k(t) dt
    k = np.arange(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(
            k == 0,
            2.0,
            np.where(k % 2 == 0, 2.0 / (1.0 - k ** 2), 0.0),
        )

    integral_on_unit = float(np.dot(coeffs, w))
    # Scale back to [a, b]
    return 0.5 * (b - a) * integral_on_unit


def trapezoidal_rule(
    f: Callable[[np.ndarray], np.ndarray],
    n: int,
    a: float = -1.0,
    b: float = 1.0,
) -> float:
    """Composite trapezoidal rule with n equally spaced panels.

    Parameters
    ----------
    f : Callable
        Integrand.
    n : int
        Number of sub-intervals (must be >= 1).
    a, b : float
        Integration interval.

    Returns
    -------
    float
        Approximation of ∫_a^b f(x) dx.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if a >= b:
        raise ValueError(f"Interval must satisfy a < b, got a={a}, b={b}.")

    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def simpson_rule(
    f: Callable[[np.ndarray], np.ndarray],
    n: int,
    a: float = -1.0,
    b: float = 1.0,
) -> float:
    """Composite Simpson's 1/3 rule with n sub-intervals (n must be even).

    Parameters
    ----------
    f : Callable
        Integrand.
    n : int
        Number of sub-intervals (must be even and >= 2).
    a, b : float
        Integration interval.

    Returns
    -------
    float
        Approximation of ∫_a^b f(x) dx.
    """
    if n < 2 or n % 2 != 0:
        raise ValueError(f"n must be even and >= 2, got {n}.")
    if a >= b:
        raise ValueError(f"Interval must satisfy a < b, got a={a}, b={b}.")

    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    coeff = np.ones(n + 1)
    coeff[1:-1:2] = 4.0
    coeff[2:-2:2] = 2.0
    return float(h / 3.0 * np.dot(coeff, y))


def integration_convergence_study(
    f: Callable[[np.ndarray], np.ndarray],
    exact_integral: float,
    n_values: list[int],
    a: float = -1.0,
    b: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compare convergence of Chebyshev quadrature vs trapezoidal vs Simpson.

    Parameters
    ----------
    f : Callable
        Integrand.
    exact_integral : float
        Analytically known value of ∫_a^b f(x) dx.
    n_values : list of int
        Node/panel counts to test.
    a, b : float
        Integration interval.

    Returns
    -------
    dict with keys "n_values", "chebyshev", "trapezoidal", "simpson".
    Each value is a float array of absolute errors.
    """
    n_arr = np.array(n_values, dtype=int)
    cheb_err = np.empty(len(n_values), dtype=np.float64)
    trap_err = np.empty(len(n_values), dtype=np.float64)
    simp_err = np.empty(len(n_values), dtype=np.float64)

    for i, n in enumerate(n_values):
        cheb_err[i] = abs(chebyshev_quadrature(f, n, a, b) - exact_integral)
        trap_err[i] = abs(trapezoidal_rule(f, n, a, b) - exact_integral)
        # Simpson requires even n; use n if even, n+1 if odd
        n_simp = n if n % 2 == 0 else n + 1
        simp_err[i] = abs(simpson_rule(f, n_simp, a, b) - exact_integral)

    return {
        "n_values": n_arr,
        "chebyshev": cheb_err,
        "trapezoidal": trap_err,
        "simpson": simp_err,
    }
