# Uhlig (2005) Sign Restriction Approach

This directory contains the foundational implementation of Harald Uhlig’s (2005) "agnostic identification procedure" using sign restrictions. This is the baseline structural VAR identification approach, often referred to as the "rejection method."

## Methodology
The implementation follows the rejection sampling procedure:
1.  **Draws:** We draw from the posterior distribution of the VAR coefficients and the covariance matrix of the errors (using the Inverse Wishart distribution).
2.  **Orthogonalization:** We compute the Cholesky decomposition of the error covariance matrix.
3.  **Rotation:** We generate random orthogonal rotation matrices to span the space of possible impulse responses.
4.  **Rejection:** We test whether the resulting impulse response functions satisfy the pre-specified sign restrictions for the identification of structural shocks. If they do not, the draw is discarded.

## Key Features
- **Baseline Identification:** Implements the original rejection-based sign restriction method.
- **Structural Analysis:** Ideal for identifying macroeconomic shocks (e.g., monetary policy shocks) based on sign-based theoretical priors.
- **Comparability:** Serves as the benchmark for comparing more advanced identification schemes, such as the penalty function approach.

## Keywords
Sign Restrictions, Rejection Method, Bayesian VAR, Structural VAR, Harald Uhlig, Identification, Macroeconometrics.

## Reference
Uhlig, H. (2005). "What are the effects of monetary policy on output? Results from an agnostic identification procedure." *Journal of Monetary Economics*, 52(2), 381-419.
