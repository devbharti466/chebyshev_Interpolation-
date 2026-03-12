"""Chebyshev and equispaced node generation."""

import numpy as np


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Generate n Chebyshev nodes of the first kind on [a, b].

    Nodes: x_k = cos((2k+1)/(2n) * pi), k = 0, ..., n-1
    Mapped to [a, b] via affine transformation.

    Parameters
    ----------
    n : int
        Number of nodes (must be >= 1).
    a, b : float
        Interval endpoints; must satisfy a < b.

    Returns
    -------
    np.ndarray
        Array of n Chebyshev nodes sorted in ascending order.

    Complexity: O(n) time, O(n) space.
    """
    if n < 1:
        raise ValueError(f"Number of nodes must be >= 1, got {n}.")
    if a >= b:
        raise ValueError(f"Interval endpoints must satisfy a < b, got a={a}, b={b}.")

    k = np.arange(n, dtype=np.float64)
    # Nodes on [-1, 1] in descending cosine order; sort ascending
    x = np.cos((2.0 * k + 1.0) / (2.0 * n) * np.pi)
    x = np.sort(x)
    # Affine map to [a, b]
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def equally_spaced_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Generate n equally spaced nodes on [a, b].

    Parameters
    ----------
    n : int
        Number of nodes (must be >= 1).
    a, b : float
        Interval endpoints; must satisfy a < b.

    Returns
    -------
    np.ndarray
        Array of n equispaced nodes on [a, b].

    Complexity: O(n) time, O(n) space.
    """
    if n < 1:
        raise ValueError(f"Number of nodes must be >= 1, got {n}.")
    if a >= b:
        raise ValueError(f"Interval endpoints must satisfy a < b, got a={a}, b={b}.")

    return np.linspace(a, b, n)
