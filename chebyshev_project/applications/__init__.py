from .integration import chebyshev_quadrature, trapezoidal_rule, simpson_rule, integration_convergence_study
from .regression import (
    chebyshev_design_matrix,
    vandermonde_matrix,
    chebyshev_regression,
    conditioning_comparison,
)

__all__ = [
    "chebyshev_quadrature",
    "trapezoidal_rule",
    "simpson_rule",
    "integration_convergence_study",
    "chebyshev_design_matrix",
    "vandermonde_matrix",
    "chebyshev_regression",
    "conditioning_comparison",
]
