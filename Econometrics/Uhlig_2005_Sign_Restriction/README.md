# Bayesian SVAR Identification with Sign Restrictions

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-High%20Performance-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A robust, numerically stabilized, and highly optimized Python implementation for estimating Structural Vector Autoregression (SVAR) models using Bayesian methods and sign restrictions. 

This repository successfully bridges the gap between macroeconomic theory and advanced matrix algebra, reproducing the seminal monetary policy shock identification methods of **Uhlig (2005)** and the unbiased, set-identified Haar-measure rotation approach of **Rubio-Ramírez, Waggoner, and Zha (2010)**.

## 📖 Overview

In empirical macroeconomics, identifying structural shocks without imposing strict zero-restrictions relies heavily on inequality (sign) constraints. This repository provides a complete, transparent pipeline from raw data to Forecast Error Variance Decomposition (FEVD), optimized for both pedagogical clarity and computational speed.

### Key Highlights
* **Uhlig (2005) Pure Sign Restrictions:** Exact replication of the Bayesian Monte Carlo algorithm utilizing unit-sphere impulse vector draws.
* **RWZ (2010) Unbiased Set-Identification:** Implements the algorithm using orthogonal matrices drawn from the Haar measure via QR decomposition, eliminating prior bias in vector direction.
* **High-Performance Backend:** Replaces sluggish nested loops with `JAX`-powered Einstein summations (`einsum`) for lightning-fast, multidimensional tensor rotations.
* **Engineering Notes:** The codebase features unique, detailed architectural comments designed to assist researchers in understanding the deep mathematical mechanics of companion matrices, posterior scaling, and tensor operations.

## 🗂️ Core Features

1. **Reduced-Form VAR & Posterior Draws:** Computes OLS estimates and applies Normal-Inverse Wishart priors with numerical stabilization (jitter) to handle potentially singular matrices in finite samples.
2. **Companion Form Transformation:** Efficiently computes base Impulse Response Functions (IRFs) using state-space representations.
3. **Sign Restriction Evaluation:** Logical masking algorithms to filter millions of simulated structural models, retaining only those that conform to standard economic theory (e.g., contractionary monetary policy shocks).
4. **FEVD Analysis:** Computes the fractional contribution of identified shocks to the total forecast error variance.
5. **Publication-Ready Visualizations:** Automated generation of median responses alongside 16th and 84th percentiles.

## 🚀 Getting Started

### Prerequisites
Ensure you have the following libraries installed in your local environment:
```bash
pip install numpy pandas matplotlib scipy jax jaxlib
