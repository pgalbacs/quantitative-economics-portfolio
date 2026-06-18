# Bayesian SVAR Identification: The RWZ (2010) Haar-Measure Approach

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Tensor%20Optimized-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art, high-performance Python implementation of the exact sign restriction algorithm introduced by **Rubio-Ramírez, Waggoner, and Zha (2010)**. 

While earlier sign restriction methods (e.g., Uhlig 2005) sometimes imposed unintended and asymmetric prior weights on the impulse vectors, RWZ (2010) revolutionized the field by proving that drawing orthogonal matrices from the **Haar measure** via QR decomposition is the *only* way to ensure a strictly uniform, unbiased prior over the space of all possible structural identification schemes. This repository implements that mathematical benchmark.

## 📖 Overview

This codebase transforms complex set-identification theory into lightning-fast execution. By replacing traditional, sluggish nested loops with multidimensional tensor operations using Google's `JAX` library, the algorithm processes hundreds of thousands of rotation matrices in seconds.

### Key Highlights
* **Unbiased Set-Identification:** Perfectly uniform draws of orthogonal rotation matrices ($Q$) from the Haar measure.
* **4D Tensor Engine:** Utilizes JAX's Einstein summation (`jnp.einsum`) to multiply base Impulse Response Functions (IRFs) by batches of 10,000 rotation matrices simultaneously, bypassing pure Python loops completely.
* **Logical Filtering Masks:** Applies rigorous Boolean masking algorithms across all structural shocks and time horizons to elegantly isolate the valid models out of the Haar-drawn parallel universes.
* **Pedagogical "Engineering Notes":** The codebase is thoroughly annotated with advanced econometric insights, bridging the gap between theoretical matrix algebra (companion forms, QR decomposition) and efficient code architecture.

## 🗂️ Core Features

1. **Numerically Stabilized Posterior Draws:** Computes the reduced-form VAR via OLS and efficiently draws from the Normal-Inverse Wishart posterior, utilizing minimal jitter to prevent singularities.
2. **State-Space Companion Matrix:** Stacks VAR(p) lags into a VAR(1) companion form to calculate dynamic IRF components iteratively.
3. **Haar-Measure Rotation Batching:** Generates massive batches of standard normal matrices and applies vectorized QR decompositions to extract the orthogonal $Q$ matrices.
4. **Instant Sign Restriction Evaluation:** Applies Uhlig's benchmark constraints (e.g., contractionary monetary policy) simultaneously across all draws and horizons using multi-axis `JAX` boolean reductions.
5. **Bayesian Confidence Bands:** Automatically aggregates the surviving set of empirically equivalent models to compute the median structural response alongside the 16th and 84th percentiles.

## 🚀 Getting Started

### Prerequisites
Due to the heavy reliance on advanced tensor operations, `JAX` is required. Ensure your environment is prepared:
```bash
pip install numpy pandas matplotlib scipy jax jaxlib
