"""Standard test functions for interpolation experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np

# Exact integrals over [-1, 1]
_E = np.e
_PI = np.pi

TEST_FUNCTIONS: list[tuple[str, Callable[[np.ndarray], np.ndarray], str]] = [
    ("Runge",    lambda x: 1.0 / (1.0 + 25.0 * x ** 2),  "analytic, Runge phenomenon"),
    ("|x|",      lambda x: np.abs(x),                      "Lipschitz, not differentiable at 0"),
    ("sin(5x)",  lambda x: np.sin(5.0 * x),                "smooth, oscillatory"),
    ("exp(-x²)", lambda x: np.exp(-(x ** 2)),              "entire, Gaussian"),
    ("step",     lambda x: np.where(x >= 0.0, 1.0, 0.0),  "discontinuous"),
    ("|x|³",     lambda x: np.abs(x) ** 3,                 "C² but not C³"),
]

INTEGRATION_TEST_FUNCTIONS: list[tuple[str, Callable[[np.ndarray], np.ndarray], float]] = [
    ("exp(x)",    lambda x: np.exp(x),              _E - 1.0 / _E),
    ("x⁴",        lambda x: x ** 4,                 2.0 / 5.0),
    ("cos(πx)",   lambda x: np.cos(_PI * x),        0.0),
    ("1/(1+x²)",  lambda x: 1.0 / (1.0 + x ** 2),  _PI / 2.0),
]
