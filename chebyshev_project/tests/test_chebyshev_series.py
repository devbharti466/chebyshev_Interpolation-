"""Tests for Chebyshev series expansion and Clenshaw evaluation."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chebyshev_project.core.chebyshev_series import (
    chebyshev_coefficients,
    clenshaw_evaluate,
    chebyshev_condition_number,
)


def test_constant_coefficients() -> None:
    """For f(x) = c, only c_0 should be non-zero."""
    c = 5.0
    coeffs = chebyshev_coefficients(lambda x: np.full_like(x, c), 10)
    assert abs(coeffs[0] - c) < 1e-12, \
        f"Constant: c_0 should be {c}, got {coeffs[0]}."
    assert np.all(np.abs(coeffs[1:]) < 1e-12), \
        "Constant: coefficients c_k (k>0) should be zero."


def test_identity_coefficients() -> None:
    """For f(x) = x, only c_1 should be 1 (rest near 0)."""
    coeffs = chebyshev_coefficients(lambda x: x, 10)
    assert abs(coeffs[1] - 1.0) < 1e-12, \
        f"Identity: c_1 should be 1.0, got {coeffs[1]}."
    mask = np.ones(10, dtype=bool)
    mask[1] = False
    assert np.all(np.abs(coeffs[mask]) < 1e-12), \
        "Identity: c_k (k≠1) should be zero."


def test_clenshaw_accuracy_exp() -> None:
    """Clenshaw evaluation of exp(x) should be accurate for moderate n."""
    n = 20
    coeffs = chebyshev_coefficients(np.exp, n)
    x_eval = np.linspace(-1, 1, 200)
    p = clenshaw_evaluate(coeffs, x_eval)
    err = np.max(np.abs(p - np.exp(x_eval)))
    assert err < 1e-13, f"Clenshaw accuracy for exp(x): error={err}"


def test_clenshaw_mapped_interval() -> None:
    """Clenshaw should work correctly on mapped intervals."""
    a, b = 0.0, 2.0
    n = 15

    def f(x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    # Map f to [-1, 1] for coefficient computation
    def f_mapped(t: np.ndarray) -> np.ndarray:
        return f(0.5 * (a + b) + 0.5 * (b - a) * t)

    coeffs = chebyshev_coefficients(f_mapped, n)
    x_eval = np.linspace(a, b, 100)
    p = clenshaw_evaluate(coeffs, x_eval, a, b)
    err = np.max(np.abs(p - f(x_eval)))
    assert err < 1e-10, f"Clenshaw mapped interval error: {err}"


def test_clenshaw_single_coeff() -> None:
    """Clenshaw with one coefficient should return that constant."""
    coeffs = np.array([3.7])
    x_eval = np.linspace(-1, 1, 10)
    p = clenshaw_evaluate(coeffs, x_eval)
    assert np.allclose(p, 3.7, atol=1e-14), "Single-coefficient Clenshaw failed."


def test_condition_number_well_conditioned() -> None:
    """Condition number for a constant series should be 1."""
    coeffs = np.array([5.0, 0.0, 0.0, 0.0])
    x_eval = np.array([0.0, 0.5])
    kappa = chebyshev_condition_number(coeffs, x_eval)
    assert np.allclose(kappa, 1.0, atol=1e-12), \
        f"Constant series condition number should be 1, got {kappa}."


def run_all() -> None:
    test_constant_coefficients()
    test_identity_coefficients()
    test_clenshaw_accuracy_exp()
    test_clenshaw_mapped_interval()
    test_clenshaw_single_coeff()
    test_condition_number_well_conditioned()
    print("test_chebyshev_series: ALL PASSED")


if __name__ == "__main__":
    run_all()
