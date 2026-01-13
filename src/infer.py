"""
Inference entry point for the Option Structure Engine.

This file defines a clean, API-agnostic inference contract.
"""

import torch

from grids import make_spot_grid, make_moneyness_grid
from regime_encoder import (
    RegimeEncoder,
    real_vol_features,
    multi_maturity_vol_features,
)
from portfolio_generator import PortfolioGenerator, decode_portfolio_tensor
from real_vol import resample_vol_surface, normalize_vol_surface
from physics import terminal_portfolio_payoff
from stress_engine import spot_shock, aggregate_cvar
from constraints import convexity_barrier


# -------------------------------------------------
# Load frozen models (once)
# -------------------------------------------------

_ENCODER_SINGLE = RegimeEncoder(input_dim=3, hidden_dim=16, latent_dim=8)
_ENCODER_MULTI = RegimeEncoder(input_dim=6, hidden_dim=16, latent_dim=8)

_GENERATOR = PortfolioGenerator(latent_dim=8, hidden_dim=64)
_GENERATOR.load_state_dict(
    torch.load("checkpoints/generator.pt", map_location="cpu")
)
_GENERATOR.eval()


# -------------------------------------------------
# Inference API
# -------------------------------------------------

def infer_structure(vol_surface: dict) -> dict:
    """
    Core inference function.

    Parameters
    ----------
    vol_surface : dict
        Either:
        Single maturity:
        {
            "spot": float,
            "strikes": Tensor,
            "implied_vol": Tensor
        }

        Multi-maturity:
        {
            "spot": float,
            "strikes": List[Tensor],
            "implied_vol": List[Tensor]
        }

    Returns
    -------
    dict with:
        - legs
        - payoff
        - gamma
        - cvar
        - convex_penalty
    """

    spot = vol_surface["spot"]

    k_grid = make_moneyness_grid(-1.0, 1.0, 81)
    spot_grid = make_spot_grid(
        spot=spot,
        sigma=0.2,
        n_std=3.0,
        n_points=81,
    )

    # ---------- Regime encoding ----------
    if isinstance(vol_surface["strikes"], list):
        # Multi-maturity
        vol_surfaces = []

        for strikes, vol in zip(
            vol_surface["strikes"],
            vol_surface["implied_vol"],
        ):
            vol_grid = resample_vol_surface(
                strikes=strikes,
                vol=vol,
                spot=spot,
                k_grid=k_grid,
            )
            vol_surfaces.append(normalize_vol_surface(vol_grid))

        features = multi_maturity_vol_features(k_grid, vol_surfaces)
        latent = _ENCODER_MULTI(features)

    else:
        # Single maturity
        vol_grid = resample_vol_surface(
            strikes=vol_surface["strikes"],
            vol=vol_surface["implied_vol"],
            spot=spot,
            k_grid=k_grid,
        )

        vol_norm = normalize_vol_surface(vol_grid)
        features = real_vol_features(k_grid, vol_norm)
        latent = _ENCODER_SINGLE(features)

    # ---------- Generate structure ----------
    portfolio_tensor = _GENERATOR(latent)

    legs = decode_portfolio_tensor(
        portfolio_tensor=portfolio_tensor.squeeze(0),
        spot=spot,
    )

    # ---------- Payoff ----------
    payoff = terminal_portfolio_payoff(spot_grid, legs)

    # ---------- Gamma ----------
    dS = spot_grid[1:] - spot_grid[:-1]
    gamma = (
        payoff[2:] - 2 * payoff[1:-1] + payoff[:-2]
    ) / (dS[1:] * dS[:-1])

    # ---------- Stress & CVaR ----------
    stressed = [
        terminal_portfolio_payoff(
            spot_shock(spot_grid, s),
            legs,
        )
        for s in (-0.4, -0.2, 0.2, 0.4)
    ]

    cvar = aggregate_cvar(stressed)

    # ---------- Convexity ----------
    convex_penalty = convexity_barrier(payoff, spot_grid)

    return {
        "legs": legs,
        "spot_grid": spot_grid,
        "payoff": payoff,
        "gamma": gamma,
        "cvar": cvar,
        "convex_penalty": convex_penalty,
    }
