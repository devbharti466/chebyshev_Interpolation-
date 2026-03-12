# Chebyshev Interpolation: Accuracy, Convergence, and Applications

A postgraduate-level **Numerical Analysis** project implementing Chebyshev interpolation theory and algorithms in pure NumPy, suitable for evaluation in a Scientific Programming Lab.

---

## Mathematical Background

### Chebyshev Nodes of the First Kind

The optimal interpolation nodes on $[-1,1]$ are the zeros of the Chebyshev polynomial $T_n$:

$$x_k = \cos\!\left(\frac{2k+1}{2n}\pi\right), \quad k = 0, 1, \ldots, n-1$$

Mapped to $[a,b]$ via the affine transform $x \mapsto \tfrac{a+b}{2} + \tfrac{b-a}{2}\,x$.

### Barycentric Interpolation Formula

The **second barycentric form** evaluates the degree-$(n-1)$ interpolant stably in $O(nm)$:

$$p(x) = \frac{\displaystyle\sum_{k=0}^{n-1} \frac{w_k}{x - x_k} f_k}{\displaystyle\sum_{k=0}^{n-1} \frac{w_k}{x - x_k}}, \qquad w_k = \frac{1}{\displaystyle\prod_{j \neq k}(x_k - x_j)}$$

Weights are computed in log-domain to prevent overflow. Division by zero at node locations is handled by returning $f_k$ directly.

### Convergence Theory

| Function class | Chebyshev convergence | Equispaced |
|---|---|---|
| Entire (e.g. $e^x$) | Geometric: $\|f - p_n\|_\infty = O(\rho^{-n})$ | Diverges (Runge phenomenon) |
| Analytic on $[-1,1]$ | Geometric | Diverges |
| $C^k$ (k-times differentiable) | Algebraic: $O(n^{-k})$ | Diverges |
| Lipschitz only ($\|x\|$) | $O(n^{-1})$ | Diverges |
| Discontinuous (step) | Gibbs oscillations, $O(1)$ | Diverges |

### Near-Minimax Property and Equioscillation

The Chebyshev polynomial $T_n$ achieves the minimax property:

$$\min_{\text{monic poly } p} \max_{x \in [-1,1]} |p(x)| = \frac{1}{2^{n-1}}, \quad \text{achieved by } \frac{T_n(x)}{2^{n-1}}$$

The best approximation from $\Pi_n$ equioscillates at $n+2$ points (Chebyshev–Markov–Stieltjes theorem). The Chebyshev interpolant is near-minimax: $\|f - p_n\|_\infty \leq (1 + \Lambda_n)\,\|f - p_n^*\|_\infty$, where $\Lambda_n = O(\log n)$ for Chebyshev nodes.

### Clenshaw Backward Recurrence

For $p(x) = \sum_{k=0}^{n-1} c_k T_k(t)$, $t \in [-1,1]$, the Clenshaw algorithm avoids explicit computation of $T_k$:

$$b_n = b_{n+1} = 0, \qquad b_k = c_k + 2t\,b_{k+1} - b_{k+2}, \quad k = n-1, \ldots, 1$$
$$p(x) = c_0 + t\,b_1 - b_2$$

This is unconditionally stable in the backward direction.

### Chebyshev Series Coefficients (DCT-like)

$$c_k = \frac{2}{n} \sum_{j=0}^{n-1} f(x_j)\,\cos\!\left(\frac{k(2j+1)\pi}{2n}\right), \quad c_0 \text{ halved by convention}$$

---

## Complexity Table

| Method | Setup | Evaluation | Space |
|---|---|---|---|
| Barycentric | $O(n^2)$ weights | $O(nm)$ | $O(n)$ |
| Lagrange | — | $O(n^2 m)$ | $O(nm)$ |
| Newton | $O(n^2)$ div diff | $O(nm)$ | $O(n)$ |
| Clenshaw | $O(n^2)$ coeffs | $O(nm)$ | $O(m)$ |

---

## Method Selection Guide

| Scenario | Recommended method |
|---|---|
| General-purpose interpolation | Barycentric (stable, $O(nm)$) |
| Chebyshev nodes only | Clenshaw (optimal, unconditionally stable) |
| Educational reference | Lagrange (transparent, but $O(n^2 m)$) |
| Root-finding / iterative | Newton (incremental coefficient update) |
| Numerical integration | Chebyshev quadrature (spectral accuracy) |
| Regression / least squares | Chebyshev design matrix (well-conditioned) |

---

## Project Structure

```
chebyshev_project/
├── core/
│   ├── nodes.py              # Chebyshev and equispaced node generation
│   ├── barycentric.py        # Barycentric weights and interpolation
│   ├── lagrange.py           # Lagrange interpolation (educational reference)
│   ├── newton.py             # Newton interpolation with divided differences
│   └── chebyshev_series.py   # Chebyshev series expansion, Clenshaw algorithm
├── experiments/
│   ├── convergence.py        # Convergence study engine
│   ├── lebesgue.py           # Lebesgue constant estimation
│   └── stability.py          # Stability and condition number analysis
├── applications/
│   ├── integration.py        # Numerical integration via interpolant
│   └── regression.py         # Regression with Chebyshev basis
├── utils/
│   ├── plotting.py           # Publication-quality plots (Agg backend)
│   ├── test_functions.py     # Standard test functions (Runge, |x|, etc.)
│   └── tables.py             # CSV/tabulated output utilities
├── tests/
│   ├── test_nodes.py
│   ├── test_interpolation.py
│   ├── test_integration.py
│   └── test_chebyshev_series.py
├── results/                  # Auto-created: plots and CSVs
├── main.py                   # Automated execution script
├── requirements.txt
└── README.md
```

---

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Run unit tests individually
python chebyshev_project/tests/test_nodes.py
python chebyshev_project/tests/test_interpolation.py
python chebyshev_project/tests/test_integration.py
python chebyshev_project/tests/test_chebyshev_series.py
```

All generated plots and CSV tables are saved to `results/`.

---

## Key Numerical Features

- **Barycentric weights**: computed in log-domain to prevent overflow for large $n$
- **Clenshaw recurrence**: backward, unconditionally stable
- **Divided differences**: in-place $O(n^2)$, $O(n)$ space
- **Chebyshev coefficients**: DCT-like formula (pure NumPy, no FFT library)
- **Lebesgue constant**: estimated on grid of ≥ 10 000 points
- **Input validation**: `ValueError` with descriptive messages throughout
- **Division-by-zero protection**: barycentric formula handles $x = x_k$ exactly

---

## References

1. **Trefethen, L. N.** (2013). *Approximation Theory and Approximation Practice*. SIAM.
2. **Berrut, J.-P. & Trefethen, L. N.** (2004). Barycentric Lagrange interpolation. *SIAM Review*, 46(3), 501–517.
3. **Higham, N. J.** (2004). The numerical stability of barycentric Lagrange interpolation. *IMA J. Numer. Anal.*, 24(4), 547–556.
4. **Higham, N. J.** (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
5. **Gautschi, W.** (2012). *Numerical Analysis* (2nd ed.). Birkhäuser.
