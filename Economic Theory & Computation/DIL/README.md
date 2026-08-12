# Predictive Agronomy and Optimal Stopping: Computational Proof of Concept

This repository contains the computational implementation of the Dual-Engine Architecture discussed in the companion theoretical working paper. It operationalizes a Stochastic Dynamic Programming (SDP) model for agricultural decision-making under uncertainty, leveraging JAX for XLA-compiled, vectorized backward induction.

## Overview

While modern meteorological models predict environmental states, this codebase provides the normative decision layer. It demonstrates the translation of probabilistic meteorological ensembles into state-contingent policy rules, mathematically formalizing the economic "option value of waiting" for risk-averse subsistence farmers.

## Core Components

The execution pipeline consists of four primary modules:
1. **Synthetic Predictive Engine:** An autoregressive weather generator demonstrating the informational martingale property (the temporal collapse of forecast uncertainty).
2. **Physical State Transitions:** A discrete-time FAO-56 soil moisture model acting as an endogenous physical buffer.
3. **The Bellman Engine:** A vectorized dynamic programming algorithm solving the optimal stopping problem (Plant vs. Wait) via backward induction.
4. **CRRA Utility Demonstration:** A full stochastic Markov Decision Process (MDP) evaluation contrasting risk-neutral and risk-averse ($\gamma=2.0$) policy boundaries.

## Execution and Dependencies

This Proof of Concept is written in Python and is built on the JAX ecosystem to demonstrate the mechanics of the decision engine.

**Dependencies:**
- `jax`, `jaxlib`
- `numpy`
- `matplotlib`, `seaborn` (for policy surface and state-space visualizations)

**Execution Environment:** 
This PoC is explicitly designed for seamless execution on standard local hardware. By employing a certainty equivalence heuristic in the baseline Bellman engine, the computational footprint is optimized for real-time local JIT compilation. No external GPU/TPU clusters or cloud environments are required to run this demonstration—simply install the standard dependencies and execute.

## Methodological Note and Limitations

As explicitly detailed in the *Technical and Methodological Appendix*, this repository contains a localized Proof of Concept (PoC). It currently employs synthetic weather generation (IID noise) and a certainty equivalence heuristic for the baseline decision engine to facilitate real-time local compilation. It is designed to demonstrate structural economic mechanics and computational feasibility. Production scaling will involve integration with live Deep Learning ensemble APIs and full MDP evaluation across all transition states.
