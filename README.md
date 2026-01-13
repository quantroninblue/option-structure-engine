Option Structure Engine
A Regime-Conditioned Neural System for Market-Neutral Convex Option Structures

1. Overview

This repository implements a research-grade neural system designed to discover market-neutral option portfolio structures with structural convex alpha.

This system is not a price-prediction model.
It does not forecast spot direction, volatility levels, or returns.

Instead, it learns option payoff geometry under volatility-surface regimes and discovers multi-leg option structures whose convexity survives:

spot stress scenarios

volatility regime shifts

tail-risk evaluation (CVaR)

hard Greek constraints

The engine generates option structures, not trades.
It is intended for research and structure discovery, not execution.

2. Core Design Philosophy

The engine is built around four non-negotiable principles:

1. No directional prediction

The model never predicts spot or volatility.
All learning happens on payoff shape, risk geometry, and stress behavior.

2. Explicit financial physics

Option payoffs, Greeks, convexity, and stress behavior are computed using explicit, differentiable financial primitives, not learned implicitly.

3. Hard neutrality constraints

Delta neutrality, bounded exposure, and convexity regularization are enforced structurally, not statistically.

4. Regime conditioning, not memorization

The model conditions on volatility-surface regimes, not individual instruments or dates.

3. Problem Setting

Let:

S = underlying spot price

Π(S) = terminal payoff of a multi-leg option portfolio

The goal is to learn a mapping:
Volatility Regime  →  Option Portfolio Structure

such that the resulting payoff satisfies:

near-zero delta

controlled exposure size

robust convex geometry

favorable tail behavior under stress

4. Volatility Regime Representation
Single-Maturity Regimes

For a given maturity, the implied volatility smile is resampled on a fixed log-moneyness grid.

From this surface, three interpretable features are extracted:

Level: mean implied volatility

Slope: average first derivative w.r.t. log-moneyness

Curvature: average second derivative

These form a compact regime vector:
(level, slope, curvature)

Multi-Maturity Extension

For multiple maturities, the same features are computed per maturity and then aggregated across maturities using:

mean

standard deviation

Resulting in a 6-dimensional regime vector:
(mean_level, std_level,
 mean_slope, std_slope,
 mean_curvature, std_curvature)
This captures term-structure information without learning full volatility surfaces.

5. Neural Architecture
Regime Encoder

A small MLP maps the regime feature vector into a latent embedding:
regime_features → latent_regime
This latent space represents the volatility environment, not a market state.

Portfolio Generator

A second MLP maps the latent regime into a fixed-length portfolio tensor:
latent_regime → {option_type, strike, weight}_i
This tensor is decoded into an explicit multi-leg option structure.

The generator does not choose trades; it chooses payoff geometry.

6. Differentiable Financial Physics

All financial quantities are computed explicitly:

terminal payoff curves

delta via differentiation

gamma via finite differences

stress payoffs under spot shocks

tail metrics (CVaR)

No learned approximations are used for these quantities.

7. Constraints
Delta Neutrality

Portfolio weights are projected onto the nullspace of aggregate delta:
Σ w_i · Δ_i = 0
This is enforced as a hard constraint.

Convexity Regularization

Convexity is encouraged but not enforced absolutely.

A smooth barrier penalizes large negative gamma regions while allowing:

localized concavity

convexity suppression under skew

This reflects real risk-tradeoffs.

Exposure Control

An L1 penalty on portfolio weights discourages:

leverage amplification

trivial convexity exploitation

The engine prefers small, robust structures.

8. Stress Testing and Training Objective

The training objective combines:

convexity reward

CVaR penalty under spot stress

portfolio size penalty

Conceptually:
Objective =
  - Convexity_Penalty
  + CVaR_Penalty
  + Size_Penalty
This forces the emergence of structural convex alpha, not directional risk.

9. Real Data Support

The system includes robust CSV ingestion that:

tolerates messy column names

infers strikes, vols, and maturities

supports single or multi-maturity surfaces

decouples data sourcing from model logic

Training and inference share the same pipeline, ensuring reproducibility.

10. Inference Interface (API-Agnostic)

The engine exposes a single inference entry point:
infer_structure(vol_surface: dict) -> dict

Input:

a canonical volatility surface (single or multi-maturity)

Output:

decoded portfolio legs

terminal payoff curve

gamma profile

CVaR diagnostics

convexity metrics

The engine is completely agnostic to where the data came from.

Any external system (Bloomberg, IB, internal APIs, CSVs) can be connected via an adapter layer without modifying the engine.

11. What This System Is NOT

This engine is explicitly:

not a price predictor

not a volatility forecaster

not a trading signal generator

not an execution system

It does not tell you what to trade tomorrow.

12. Intended Use

This repository is intended for:

research into convex alpha mechanisms

regime-aware option structure design

stress-robust portfolio construction

exploratory quantitative research

13. Key Insight

Convex option structures should be treated as geometric objects, not directional bets.

This engine demonstrates that convexity can be learned, constrained, and stress-tested as a structural property of option portfolios.

14. Summary

The Option Structure Engine:

learns payoff geometry, not prices

enforces realistic financial constraints

adapts to real volatility regimes

remains robust under stress

exposes a clean inference boundary

It is a structure discovery engine, not a trading model.