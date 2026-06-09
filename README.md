# Quantitative Economics Portfolio

Welcome to my quantitative economics research repository. This codebase serves as a centralized portfolio featuring modern Python implementations of theoretical macroeconomic models, applied economic analyses, and advanced econometric estimations.

The primary objective of this repository is to bridge the gap between rigorous economic theory and computational execution, utilizing robust numerical methods, structural estimation techniques, and clean, reproducible coding practices.

---

## 📂 Repository Structure

The repository is organized into two primary pillars, reflecting the core dimensions of quantitative economic research:

```text
quantitative-economics-portfolio/
├── econometrics/          # Empirical research, structural & Bayesian estimation
│   ├── bvar-estimation/   # Bayesian VARs with sign restrictions & penalty functions
│   └── [project-folders]  # Applied micro & macro econometric pipelines
└── theory/                # Theoretical models, dynamic programming & numerical solutions
    ├── uzawa-lucas-taxes/ # Endogenous growth models solved via homotopy continuation
    └── [project-folders]  # Macroeconomic theory implementations

1. Econometrics (/econometrics)
This section contains frameworks for empirical analysis, with a specific focus on structural and Bayesian econometrics, as well as applied micro/macro estimation.

Bayesian Vector Autoregressions (BVAR): Custom Python implementations utilizing advanced identification schemes. Features include structural VAR analysis with sign restrictions implemented via penalty function approaches (e.g., Uhlig's formulation) to trace macroeconomic shocks.

Applied Estimation: Data-driven scripts handling complex data pipelines, specialized regression models, and empirical strategies designed for academic-grade precision.

2. Economic Theory & Computation (/theory)
This section focuses on the numerical solutions, simulations, and steady-state analysis of dynamic economic models.

Endogenous Growth Models: Computational scripts solving the transition paths and Balanced Growth Paths (BGP) of complex theoretical structures, including the Lucas-Uzawa model with taxation.

Numerical Methods: Implementation of specialized mathematical algorithms, such as the homotopy continuation method, to solve highly non-linear systems of economic equilibria where standard gradient-based solvers fall short.

Computational Stack
The entire portfolio is built on a modern, high-performance Python stack optimized for scientific computing:

Core Math & Data: numpy, scipy, pandas

Visualization: matplotlib, seaborn

Environments: Structured .py scripts for production-grade pipelines and Jupyter Notebooks for research exploration.

License & Usage
This repository is fully open-source. Feel free to explore, clone, or adapt the code for academic and research purposes. If you reference or utilize specific implementations in your own academic work, a link back to this repository is highly appreciated.
