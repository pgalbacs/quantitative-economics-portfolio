# Uhlig (2005) Penalty Function Approach

This directory contains a modern Python implementation of the penalty function approach for structural VAR identification, originally proposed by Harald Uhlig (2005).

## Methodology
The implementation utilizes **JAX** for high-performance automatic differentiation, replacing traditional finite-difference methods. The core optimization leverages the **SciPy BFGS** algorithm to find the structural shock that best satisfies sign restrictions while minimizing a penalty function.

## Key Features
- **High-performance optimization:** JAX-accelerated gradient computation.
- **Robustness:** Implicit unit-sphere projection for numerical stability.
- **Reproducibility:** Consistent with the identification schemes presented in Uhlig (2005).

## Keywords
Bayesian VAR, Penalty Function, Sign Restrictions, JAX, Autodiff, Structural VAR, Harald Uhlig, Macroeconometrics.

## Reference
Uhlig, H. (2005). "What are the effects of monetary policy on output? Results from an agnostic identification procedure." *Journal of Monetary Economics*, 52(2), 381-419.
