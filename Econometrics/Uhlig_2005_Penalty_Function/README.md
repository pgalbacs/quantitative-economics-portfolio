# Bayesian SVAR Identification: The Penalty Function Approach

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Autodiff%20Enabled-orange.svg)](https://github.com/google/jax)
[![SciPy](https://img.shields.io/badge/SciPy-BFGS%20Optimization-lightgrey.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A high-performance, analytically differentiated Python implementation of the **Penalty Function Approach (Algorithm 1)** for identifying Structural Vector Autoregression (SVAR) models, as introduced by **Harald Uhlig (2005)**.

While pure sign restrictions rely on discarding models that violate economic intuition, the penalty function approach treats identification as a formal numerical optimization problem. This repository leverages `JAX` (Reverse-Mode Automatic Differentiation) and `SciPy` to elegantly minimize an asymmetric loss function over the unit hypersphere, finding the structural shock that most closely aligns with the theorized sign restrictions.

## 📖 Overview

This codebase is designed for researchers who require robust, fast, and numerically stable SVAR optimizations. It replaces slow finite-difference gradient approximations with exact analytical gradients compiled at the C-level, reducing optimization times from hours to seconds.

### Key Highlights
* **Uhlig (2005) Asymmetric Loss Function:** Accurately replicates the penalty structure that rewards correct response directions while heavily penalizing violations.
* **JAX Autodiff Engine:** Utilizes `jax.value_and_grad` to compute exact gradients of the multidimensional penalty objective instantly.
* **Implicit Unit Sphere Optimization:** Employs a scale-invariant objective function architecture, allowing fast, unconstrained `BFGS` solvers to navigate the impulse vector space smoothly.
* **Pedagogical "Engineering Notes":** The script is heavily annotated with specialized engineering and econometric notes, detailing the mathematical decisions behind state-space companion forms, conjugate priors, and tensor normalization.

## 🗂️ Core Features

1. **Numerically Stabilized Posterior Draws:** Computes OLS estimators and Inverse Wishart scale matrices, incorporating minimal jitter for guaranteed positive-definiteness in finite samples.
2. **State-Space Companion Matrix:** Efficient vectorization of the base MA components ($h=0 \dots 59$) to compute structural IRFs without costly recursive loops.
3. **Multi-Start BFGS Optimization:** Mitigates the risk of local minima in non-convex penalty surfaces by initiating optimizations from multiple random vectors on the unit sphere.
4. **FEVD Computation:** Calculates the Forecast Error Variance Decomposition, precisely quantifying the macroeconomic impact of the identified shock.
5. **Publication-Ready Visualization:** Replicates the exact visual formatting of Uhlig's (2005) Figure 14, generating median responses and Bayesian confidence intervals (16th and 84th percentiles).

## 🚀 Getting Started

### Prerequisites
The optimization engine relies heavily on `JAX` for its gradient calculations. Ensure your environment is set up with:
```bash
pip install numpy pandas matplotlib scipy jax jaxlib
