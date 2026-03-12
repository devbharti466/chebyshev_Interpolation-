"""Publication-quality plotting utilities (non-interactive, Agg backend)."""

from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(
    n_values: list[int],
    errors: dict[tuple[str, str], np.ndarray],
    save_path: str,
    title: str = "Interpolation Convergence",
) -> None:
    """Log-scale convergence plot for multiple (function, node_type) pairs.

    Parameters
    ----------
    n_values : list of int
        Node counts.
    errors : dict
        Keys (func_name, node_type) → error arrays.
    save_path : str
        Output file path.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    linestyles = {"chebyshev": "-", "equispaced": "--"}
    markers = {"chebyshev": "o", "equispaced": "s"}

    for idx, ((fname, ntype), err) in enumerate(errors.items()):
        ls = linestyles.get(ntype, "-")
        mk = markers.get(ntype, "^")
        color = _COLORS[idx % len(_COLORS)]
        ax.semilogy(
            n_values,
            np.maximum(err, 1e-17),
            ls=ls,
            marker=mk,
            color=color,
            markersize=4,
            label=f"{fname} ({ntype})",
        )

    ax.set_xlabel("Number of nodes $n$")
    ax.set_ylabel(r"$\|\,f - p_n\,\|_\infty$")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)


def plot_lebesgue_function(
    x_eval: np.ndarray,
    leb_values: dict[str, np.ndarray],
    save_path: str,
    title: str = "Lebesgue Function",
) -> None:
    """Plot the Lebesgue function for multiple node sets.

    Parameters
    ----------
    x_eval : np.ndarray
        Evaluation grid.
    leb_values : dict
        label → Lebesgue function array.
    save_path : str
        Output file path.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, lv in leb_values.items():
        ax.plot(x_eval, lv, label=label)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\Lambda(x)$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)


def plot_lebesgue_constants(
    n_values: list[int],
    constants: dict[str, np.ndarray],
    save_path: str,
    title: str = "Lebesgue Constants vs $n$",
) -> None:
    """Plot Lebesgue constants Λ_n vs n.

    Parameters
    ----------
    n_values : list of int
        Node counts.
    constants : dict
        label → array of Lebesgue constants.
    save_path : str
        Output file path.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, lc in constants.items():
        ax.semilogy(n_values, lc, marker="o", markersize=4, label=label)
    ax.set_xlabel("Number of nodes $n$")
    ax.set_ylabel(r"$\Lambda_n$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)


def plot_interpolation_comparison(
    x_fine: np.ndarray,
    f_fine: np.ndarray,
    interp_values: dict[str, np.ndarray],
    node_sets: dict[str, np.ndarray],
    func_name: str,
    save_path: str,
    n: int = 0,
) -> None:
    """Side-by-side comparison of Chebyshev vs equispaced interpolants.

    Parameters
    ----------
    x_fine : np.ndarray
        Fine evaluation grid.
    f_fine : np.ndarray
        True function values on grid.
    interp_values : dict
        label → interpolant values on x_fine.
    node_sets : dict
        label → node positions (for scatter).
    func_name : str
        Function name for the title.
    save_path : str
        Output file path.
    n : int
        Degree (for title).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    labels = list(interp_values.keys())

    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(x_fine, f_fine, "k-", lw=1.0, label="$f$", zorder=5)
        ax.plot(x_fine, interp_values[label], "--", lw=1.4, label=f"$p_{{{n}}}$ ({label})")
        if label in node_sets:
            xn = node_sets[label]
            ax.scatter(xn, np.zeros_like(xn), s=20, zorder=6, color="red")
        ax.set_title(f"{func_name}: {label} nodes, $n={n}$")
        ax.set_xlabel("$x$")
        ax.legend()

    fig.tight_layout()
    _save(fig, save_path)


def plot_integration_convergence(
    results: dict[str, Any],
    func_name: str,
    save_path: str,
) -> None:
    """Log-log or semilogy convergence plot for integration methods.

    Parameters
    ----------
    results : dict
        Output of integration_convergence_study; must contain
        "n_values", "chebyshev", "trapezoidal", "simpson".
    func_name : str
        Function name for the title.
    save_path : str
        Output file path.
    """
    n = results["n_values"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, ls, mk in [("chebyshev", "-", "o"), ("trapezoidal", "--", "s"), ("simpson", "-.", "^")]:
        err = np.maximum(results[key], 1e-17)
        ax.semilogy(n, err, ls=ls, marker=mk, markersize=4, label=key.capitalize())
    ax.set_xlabel("$n$")
    ax.set_ylabel("Absolute error")
    ax.set_title(f"Integration convergence: {func_name}")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)


def plot_condition_numbers(
    n_values: np.ndarray,
    vandermonde_cond: np.ndarray,
    chebyshev_cond: np.ndarray,
    save_path: str,
    title: str = "Condition Numbers: Vandermonde vs Chebyshev",
) -> None:
    """Log-scale condition number comparison.

    Parameters
    ----------
    n_values : np.ndarray
        Node counts.
    vandermonde_cond, chebyshev_cond : np.ndarray
        Condition numbers.
    save_path : str
        Output file path.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(n_values, vandermonde_cond, "r--o", markersize=4, label="Vandermonde")
    ax.semilogy(n_values, chebyshev_cond,   "b-o",  markersize=4, label="Chebyshev")
    ax.set_xlabel("$n$")
    ax.set_ylabel("Condition number (2-norm)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)


def plot_chebyshev_vs_equispaced_error(
    x_eval: np.ndarray,
    cheb_error: np.ndarray,
    equi_error: np.ndarray,
    func_name: str,
    n: int,
    save_path: str,
) -> None:
    """Pointwise error comparison between Chebyshev and equispaced nodes.

    Parameters
    ----------
    x_eval : np.ndarray
        Evaluation grid.
    cheb_error, equi_error : np.ndarray
        Pointwise absolute errors.
    func_name : str
        Function name.
    n : int
        Number of nodes.
    save_path : str
        Output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(x_eval, np.maximum(np.abs(cheb_error), 1e-17), label="Chebyshev nodes")
    ax.semilogy(x_eval, np.maximum(np.abs(equi_error), 1e-17), "--", label="Equispaced nodes")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|f(x) - p_n(x)|$")
    ax.set_title(f"Pointwise error: {func_name}, $n={n}$")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)


def plot_stability_analysis(
    results: dict[str, np.ndarray],
    save_path: str,
    title: str = "Noise Amplification vs n",
) -> None:
    """Plot noise amplification factor vs n.

    Parameters
    ----------
    results : dict
        Output of interpolation_condition_analysis; must contain
        "n_values", "mean_amplification", "max_amplification".
    save_path : str
        Output file path.
    title : str
        Plot title.
    """
    n = results["n_values"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(n, results["mean_amplification"], "-o", markersize=4, label="Mean amplification")
    ax.semilogy(n, results["max_amplification"],  "--s", markersize=4, label="Max amplification")
    ax.set_xlabel("Number of nodes $n$")
    ax.set_ylabel("$\\|\\Delta p\\|_\\infty / \\|\\Delta f\\|_\\infty$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
