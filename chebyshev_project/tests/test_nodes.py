"""Tests for core/nodes.py."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chebyshev_project.core.nodes import chebyshev_nodes, equally_spaced_nodes


def test_chebyshev_count() -> None:
    for n in [1, 5, 10, 25]:
        x = chebyshev_nodes(n)
        assert x.size == n, f"Expected {n} nodes, got {x.size}"


def test_chebyshev_interval() -> None:
    for n in [5, 20]:
        x = chebyshev_nodes(n, -1.0, 1.0)
        assert np.all(x >= -1.0 - 1e-12) and np.all(x <= 1.0 + 1e-12), \
            "Nodes outside [-1, 1]."


def test_chebyshev_sorted() -> None:
    x = chebyshev_nodes(20)
    assert np.all(np.diff(x) > 0), "Chebyshev nodes not sorted ascending."


def test_chebyshev_symmetry() -> None:
    for n in [6, 11]:
        x = chebyshev_nodes(n)
        assert np.allclose(x + x[::-1], 0.0, atol=1e-12), \
            "Chebyshev nodes not symmetric about 0."


def test_chebyshev_mapped_interval() -> None:
    a, b = -3.0, 7.0
    x = chebyshev_nodes(15, a, b)
    assert np.all(x >= a - 1e-12) and np.all(x <= b + 1e-12), \
        f"Mapped nodes outside [{a}, {b}]."


def test_equispaced_count() -> None:
    for n in [1, 5, 10]:
        x = equally_spaced_nodes(n)
        assert x.size == n, f"Expected {n} nodes, got {x.size}"


def test_equispaced_endpoints() -> None:
    a, b = -2.0, 3.0
    x = equally_spaced_nodes(10, a, b)
    assert abs(x[0] - a) < 1e-14 and abs(x[-1] - b) < 1e-14, \
        "Equispaced nodes do not match endpoints."


def test_equispaced_sorted() -> None:
    x = equally_spaced_nodes(20)
    assert np.all(np.diff(x) > 0), "Equispaced nodes not sorted ascending."


def test_invalid_inputs() -> None:
    for func in [chebyshev_nodes, equally_spaced_nodes]:
        try:
            func(0)
            assert False, "Should raise ValueError for n=0"
        except ValueError:
            pass
        try:
            func(5, 1.0, 0.0)
            assert False, "Should raise ValueError for a >= b"
        except ValueError:
            pass


def run_all() -> None:
    test_chebyshev_count()
    test_chebyshev_interval()
    test_chebyshev_sorted()
    test_chebyshev_symmetry()
    test_chebyshev_mapped_interval()
    test_equispaced_count()
    test_equispaced_endpoints()
    test_equispaced_sorted()
    test_invalid_inputs()
    print("test_nodes: ALL PASSED")


if __name__ == "__main__":
    run_all()
