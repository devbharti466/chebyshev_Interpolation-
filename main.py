"""Automated execution script for the Chebyshev Interpolation project.

Runs the full pipeline:
  1. Creates results/ directory
  2. Executes all unit tests
  3. Convergence study
  4. Lebesgue constants
  5. Interpolation comparison plots
  6. Integration convergence
  7. Regression conditioning
  8. Stability experiments
  9. Chebyshev series demo
  10. Summary report with timing
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

# Deterministic seed
np.random.seed(42)

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from chebyshev_project.core.nodes import chebyshev_nodes, equally_spaced_nodes
from chebyshev_project.core.barycentric import barycentric_weights, barycentric_interpolate
from chebyshev_project.core.chebyshev_series import chebyshev_coefficients, clenshaw_evaluate

from chebyshev_project.experiments.convergence import convergence_study
from chebyshev_project.experiments.lebesgue import lebesgue_function, lebesgue_constant
from chebyshev_project.experiments.stability import (
    interpolation_condition_analysis,
    vandermonde_vs_chebyshev_condition,
)

from chebyshev_project.applications.integration import (
    chebyshev_quadrature,
    integration_convergence_study,
)
from chebyshev_project.applications.regression import conditioning_comparison

from chebyshev_project.utils.test_functions import TEST_FUNCTIONS, INTEGRATION_TEST_FUNCTIONS
from chebyshev_project.utils.plotting import (
    plot_convergence,
    plot_lebesgue_function,
    plot_lebesgue_constants,
    plot_interpolation_comparison,
    plot_integration_convergence,
    plot_condition_numbers,
    plot_chebyshev_vs_equispaced_error,
    plot_stability_analysis,
)
from chebyshev_project.utils.tables import results_to_csv, print_summary_table

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _path(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _safe_name(name: str) -> str:
    """Sanitize a function name for use in filenames."""
    replacements = [("/", "_"), (" ", "_"), ("²", "2"), ("⁴", "4"), ("(", ""), (")", "")]
    result = name
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def step1_create_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[1] Results directory ready: {RESULTS_DIR}")


def step2_unit_tests() -> None:
    _section("Unit Tests")
    from chebyshev_project.tests.test_nodes import run_all as run_nodes
    from chebyshev_project.tests.test_interpolation import run_all as run_interp
    from chebyshev_project.tests.test_integration import run_all as run_integ
    from chebyshev_project.tests.test_chebyshev_series import run_all as run_series

    run_nodes()
    run_interp()
    run_integ()
    run_series()
    print("[2] All unit tests passed.")


def step3_convergence_study() -> None:
    _section("Convergence Study")
    n_values = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    results = convergence_study(TEST_FUNCTIONS, n_values)

    plot_convergence(
        n_values,
        results,
        _path("convergence_all.png"),
        title="Convergence: All Test Functions",
    )

    # CSV output
    headers = ["function", "node_type"] + [f"n={n}" for n in n_values]
    rows = []
    for (fname, ntype), errs in results.items():
        rows.append([fname, ntype] + [f"{e:.3e}" for e in errs])
    results_to_csv(headers, rows, _path("convergence.csv"))

    print_summary_table(
        ["Function", "Node Type", "n=5 error", "n=50 error", "n=100 error"],
        [[fname, ntype, f"{errs[0]:.2e}", f"{errs[6]:.2e}", f"{errs[9]:.2e}"]
         for (fname, ntype), errs in results.items()],
        title="Convergence Summary",
    )
    print("[3] Convergence study complete.")


def step4_lebesgue_constants() -> None:
    _section("Lebesgue Constants")
    n_values = list(range(5, 65, 5))

    cheb_lcs = np.array([lebesgue_constant(chebyshev_nodes(n)) for n in n_values])
    equi_lcs = np.array([lebesgue_constant(equally_spaced_nodes(n)) for n in n_values])

    plot_lebesgue_constants(
        n_values,
        {"Chebyshev": cheb_lcs, "Equispaced": equi_lcs},
        _path("lebesgue_constants.png"),
    )

    # Lebesgue function at n=20
    x_eval = np.linspace(-1, 1, 5000)
    plot_lebesgue_function(
        x_eval,
        {
            "Chebyshev n=20": lebesgue_function(chebyshev_nodes(20), x_eval),
            "Equispaced n=20": lebesgue_function(equally_spaced_nodes(20), x_eval),
        },
        _path("lebesgue_function_n20.png"),
        title="Lebesgue Function at n=20",
    )

    headers = ["n", "Chebyshev Λ_n", "Equispaced Λ_n"]
    rows = [[n, f"{c:.4f}", f"{e:.4f}"] for n, c, e in zip(n_values, cheb_lcs, equi_lcs)]
    results_to_csv(headers, rows, _path("lebesgue_constants.csv"))
    print_summary_table(headers, rows, "Lebesgue Constants")
    print("[4] Lebesgue constants computed.")


def step5_interpolation_comparison() -> None:
    _section("Interpolation Comparison Plots")
    n = 25
    x_fine = np.linspace(-1, 1, 1000)

    for fname, f, _ in TEST_FUNCTIONS[:4]:
        f_fine = f(x_fine)
        xn_cheb = chebyshev_nodes(n)
        xn_equi = equally_spaced_nodes(n)

        p_cheb = barycentric_interpolate(xn_cheb, f(xn_cheb), x_fine)
        p_equi = barycentric_interpolate(xn_equi, f(xn_equi), x_fine)

        safe_name = _safe_name(fname)
        plot_interpolation_comparison(
            x_fine, f_fine,
            {"Chebyshev": p_cheb, "Equispaced": p_equi},
            {"Chebyshev": xn_cheb, "Equispaced": xn_equi},
            fname,
            _path(f"comparison_{safe_name}_n{n}.png"),
            n=n,
        )
        plot_chebyshev_vs_equispaced_error(
            x_fine,
            p_cheb - f_fine,
            p_equi - f_fine,
            fname,
            n,
            _path(f"error_{safe_name}_n{n}.png"),
        )

    print("[5] Interpolation comparison plots complete.")


def step6_integration_convergence() -> None:
    _section("Integration Convergence")
    n_values = [4, 6, 8, 10, 15, 20, 30, 40]

    for fname, f, exact in INTEGRATION_TEST_FUNCTIONS:
        res = integration_convergence_study(f, exact, n_values)
        safe_name = _safe_name(fname)
        plot_integration_convergence(res, fname, _path(f"integration_{safe_name}.png"))

    print("[6] Integration convergence plots complete.")


def step7_regression_conditioning() -> None:
    _section("Regression Conditioning")
    n_values = [10, 20, 30, 50, 80, 100]
    degree = 15
    res = conditioning_comparison(n_values, degree)

    plot_condition_numbers(
        res["n_values"],
        res["vandermonde_cond"],
        res["chebyshev_cond"],
        _path("conditioning_comparison.png"),
        title=f"Condition Numbers: Vandermonde vs Chebyshev (degree={degree})",
    )

    headers = ["n", "Vandermonde κ", "Chebyshev κ"]
    rows = [[n, f"{vc:.2e}", f"{cc:.2e}"]
            for n, vc, cc in zip(res["n_values"], res["vandermonde_cond"], res["chebyshev_cond"])]
    results_to_csv(headers, rows, _path("conditioning.csv"))
    print_summary_table(headers, rows, "Conditioning Comparison")
    print("[7] Regression conditioning complete.")


def step8_stability() -> None:
    _section("Stability Experiments")
    runge = TEST_FUNCTIONS[0][1]  # Runge function
    n_values = [10, 20, 30, 40, 50]

    stab_res = interpolation_condition_analysis(runge, n_values)
    plot_stability_analysis(stab_res, _path("stability_runge.png"),
                            title="Noise Amplification: Runge Function")

    cond_res = vandermonde_vs_chebyshev_condition(n_values)
    plot_condition_numbers(
        cond_res["n_values"],
        cond_res["vandermonde_cond"],
        cond_res["chebyshev_cond"],
        _path("vand_vs_cheb_cond.png"),
        title="Square System Condition Numbers",
    )
    print("[8] Stability experiments complete.")


def step9_chebyshev_series_demo() -> None:
    _section("Chebyshev Series Demo")
    n = 20
    coeffs = chebyshev_coefficients(np.exp, n)
    x_eval = np.linspace(-1, 1, 500)
    p = clenshaw_evaluate(coeffs, x_eval)
    err = np.max(np.abs(p - np.exp(x_eval)))

    print(f"  exp(x): n={n} Chebyshev coefficients, max error = {err:.3e}")
    print(f"  Coefficients (first 8): {coeffs[:8]}")
    print("[9] Chebyshev series demo complete.")


def main() -> None:
    total_start = time.perf_counter()
    print("=" * 60)
    print("  CHEBYSHEV INTERPOLATION — AUTOMATED EXECUTION")
    print("=" * 60)

    timings: list[tuple[str, float]] = []

    steps = [
        ("Create results dir",         step1_create_results_dir),
        ("Unit tests",                  step2_unit_tests),
        ("Convergence study",           step3_convergence_study),
        ("Lebesgue constants",          step4_lebesgue_constants),
        ("Interpolation comparisons",   step5_interpolation_comparison),
        ("Integration convergence",     step6_integration_convergence),
        ("Regression conditioning",     step7_regression_conditioning),
        ("Stability experiments",       step8_stability),
        ("Chebyshev series demo",       step9_chebyshev_series_demo),
    ]

    for name, fn in steps:
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        timings.append((name, elapsed))

    total = time.perf_counter() - total_start

    print_summary_table(
        ["Step", "Time (s)"],
        [[name, f"{t:.2f}"] for name, t in timings] + [["TOTAL", f"{total:.2f}"]],
        title="Execution Timing Summary",
    )

    print(f"\nAll outputs written to: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
