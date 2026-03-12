"""Tests for interpolation methods: barycentric, Lagrange, Newton."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chebyshev_project.core.nodes import chebyshev_nodes, equally_spaced_nodes
from chebyshev_project.core.barycentric import barycentric_weights, barycentric_interpolate
from chebyshev_project.core.lagrange import lagrange_interpolate
from chebyshev_project.core.newton import newton_interpolate


def _poly(x: np.ndarray) -> np.ndarray:
    """Degree-3 polynomial: p(x) = 2 - x + 3x² - x³."""
    return 2.0 - x + 3.0 * x ** 2 - x ** 3


def test_polynomial_reproduction_barycentric() -> None:
    """Interpolation of a degree-3 poly through 4 nodes should be exact."""
    x_nodes = chebyshev_nodes(4)
    y_nodes = _poly(x_nodes)
    x_eval = np.linspace(-1, 1, 200)
    p = barycentric_interpolate(x_nodes, y_nodes, x_eval)
    assert np.allclose(p, _poly(x_eval), atol=1e-12), \
        "Barycentric: polynomial reproduction failed."


def test_polynomial_reproduction_lagrange() -> None:
    x_nodes = chebyshev_nodes(4)
    y_nodes = _poly(x_nodes)
    x_eval = np.linspace(-1, 1, 100)
    p = lagrange_interpolate(x_nodes, y_nodes, x_eval)
    assert np.allclose(p, _poly(x_eval), atol=1e-10), \
        "Lagrange: polynomial reproduction failed."


def test_polynomial_reproduction_newton() -> None:
    x_nodes = chebyshev_nodes(4)
    y_nodes = _poly(x_nodes)
    x_eval = np.linspace(-1, 1, 100)
    p = newton_interpolate(x_nodes, y_nodes, x_eval)
    assert np.allclose(p, _poly(x_eval), atol=1e-11), \
        "Newton: polynomial reproduction failed."


def test_coincident_points_barycentric() -> None:
    """Evaluating exactly at nodes must return node values."""
    x_nodes = chebyshev_nodes(8)
    y_nodes = np.sin(x_nodes)
    p = barycentric_interpolate(x_nodes, y_nodes, x_nodes)
    assert np.allclose(p, y_nodes, atol=1e-12), \
        "Barycentric: failed to return node values at coincident points."


def test_cross_method_agreement() -> None:
    """All three methods should agree to ~1e-9 on a non-polynomial function."""
    x_nodes = chebyshev_nodes(10)
    y_nodes = np.exp(x_nodes)
    x_eval = np.linspace(-0.99, 0.99, 300)

    p_bary  = barycentric_interpolate(x_nodes, y_nodes, x_eval)
    p_lag   = lagrange_interpolate(x_nodes, y_nodes, x_eval)
    p_newt  = newton_interpolate(x_nodes, y_nodes, x_eval)

    assert np.allclose(p_bary, p_lag,  atol=1e-9), "Barycentric vs Lagrange mismatch."
    assert np.allclose(p_bary, p_newt, atol=1e-9), "Barycentric vs Newton mismatch."


def test_constant_function() -> None:
    """Interpolating a constant should return that constant everywhere."""
    x_nodes = chebyshev_nodes(6)
    y_nodes = np.full_like(x_nodes, 3.14)
    x_eval = np.linspace(-1, 1, 50)
    for interp in [barycentric_interpolate, lagrange_interpolate, newton_interpolate]:
        p = interp(x_nodes, y_nodes, x_eval)
        assert np.allclose(p, 3.14, atol=1e-12), \
            f"{interp.__name__}: constant function reproduction failed."


def run_all() -> None:
    test_polynomial_reproduction_barycentric()
    test_polynomial_reproduction_lagrange()
    test_polynomial_reproduction_newton()
    test_coincident_points_barycentric()
    test_cross_method_agreement()
    test_constant_function()
    print("test_interpolation: ALL PASSED")


if __name__ == "__main__":
    run_all()
