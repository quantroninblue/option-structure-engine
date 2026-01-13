"""
Numerical grids for option payoff evaluation.

This module defines deterministic spot and log-moneyness grids
used throughout the project.
"""

import torch


def make_spot_grid(
    spot: float,
    sigma: float,
    n_std: float = 3.0,
    n_points: int = 101,
) -> torch.Tensor:
    """
    Creates a log-spaced spot grid centered at spot.

    Returns:
        Tensor of shape [n_points]
    """
    x = torch.linspace(-n_std * sigma, n_std * sigma, n_points)
    return spot * torch.exp(x)


def make_moneyness_grid(
    k_min: float = -2.0,
    k_max: float = 2.0,
    n_points: int = 101,
) -> torch.Tensor:
    """
    Creates a log-moneyness grid.

    Returns:
        Tensor of shape [n_points]
    """
    return torch.linspace(k_min, k_max, n_points)
