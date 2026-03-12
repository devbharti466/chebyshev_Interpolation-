"""Tests for numerical integration routines."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chebyshev_project.applications.integration import (
    chebyshev_quadrature,
    trapezoidal_rule,
    simpson_rule,
)


def test_chebyshev_polynomial_exactness() -> None:
    """Chebyshev quadrature should be exact for polynomials up to degree n-1."""
    # ∫_{-1}^{1} (1 + x + x²) dx = 2 + 0 + 2/3 = 8/3
    exact = 8.0 / 3.0
    for n in [4, 8, 16]:
        result = chebyshev_quadrature(lambda x: 1.0 + x + x ** 2, n)
        assert abs(result - exact) < 1e-12, \
            f"Chebyshev quadrature polynomial exactness failed for n={n}: {result}"


def test_trapezoidal_linear() -> None:
    """Trapezoidal rule is exact for linear functions."""
    # ∫_{-1}^{1} (2 + 3x) dx = 4
    exact = 4.0
    result = trapezoidal_rule(lambda x: 2.0 + 3.0 * x, 100)
    assert abs(result - exact) < 1e-11, \
        f"Trapezoidal rule failed for linear function: {result}"


def test_simpson_cubic() -> None:
    """Simpson's rule is exact for polynomials up to degree 3."""
    # ∫_{-1}^{1} x³ dx = 0
    result = simpson_rule(lambda x: x ** 3, 4)
    assert abs(result) < 1e-14, \
        f"Simpson's rule failed for cubic: {result}"


def test_chebyshev_exp() -> None:
    """Chebyshev quadrature of exp(x) over [-1,1] should converge rapidly."""
    exact = np.e - 1.0 / np.e
    result = chebyshev_quadrature(np.exp, 20)
    assert abs(result - exact) < 1e-13, \
        f"Chebyshev quadrature of exp(x) failed: error={abs(result-exact)}"


def test_trapezoidal_convergence() -> None:
    """Trapezoidal rule error should decrease as O(n^-2) for smooth functions."""
    exact = np.e - 1.0 / np.e
    err_n  = abs(trapezoidal_rule(np.exp, 20) - exact)
    err_2n = abs(trapezoidal_rule(np.exp, 40) - exact)
    # Ratio should be ~4
    assert err_n / err_2n > 3.0, \
        "Trapezoidal rule does not show expected O(n^-2) convergence."


def test_simpson_convergence() -> None:
    """Simpson's rule error should decrease as O(n^-4) for smooth functions."""
    exact = np.e - 1.0 / np.e
    err_n  = abs(simpson_rule(np.exp, 20) - exact)
    err_2n = abs(simpson_rule(np.exp, 40) - exact)
    # Ratio should be ~16
    assert err_n / err_2n > 10.0, \
        "Simpson's rule does not show expected O(n^-4) convergence."


def run_all() -> None:
    test_chebyshev_polynomial_exactness()
    test_trapezoidal_linear()
    test_simpson_cubic()
    test_chebyshev_exp()
    test_trapezoidal_convergence()
    test_simpson_convergence()
    print("test_integration: ALL PASSED")


if __name__ == "__main__":
    run_all()
