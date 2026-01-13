"""
Deterministic adversarial stress operators.
"""

import torch
from typing import Callable


# -------------------------------------------------
# Spot shocks
# -------------------------------------------------

def spot_shock(
    spot: torch.Tensor,
    shock: float,
) -> torch.Tensor:
    """
    Applies a multiplicative spot shock.

    Args:
        spot: Tensor of spot prices [N]
        shock: fractional shock (e.g. -0.2 or +0.2)

    Returns:
        Shocked spot tensor
    """
    return spot * (1.0 + shock)


# -------------------------------------------------
# Volatility stresses
# -------------------------------------------------

def vol_level_shift(
    vol: torch.Tensor,
    shift: float,
) -> torch.Tensor:
    """
    Applies an additive volatility level shift.

    Args:
        vol: Tensor of implied volatilities [N]
        shift: additive shift (e.g. +0.05)

    Returns:
        Shifted vol tensor
    """
    return torch.clamp(vol + shift, min=1e-4)

from typing import List


def aggregate_worst_case(
    stressed_payoffs: List[torch.Tensor],
) -> torch.Tensor:
    """
    Computes worst-case payoff across stresses and spot.

    Args:
        stressed_payoffs: list of payoff tensors [N]

    Returns:
        Scalar tensor (worst-case payoff)
    """
    worst = torch.stack(
        [p.min() for p in stressed_payoffs]
    ).min()

    return worst


def aggregate_cvar(
    stressed_payoffs: List[torch.Tensor],
    q: float = 0.1,
) -> torch.Tensor:
    """
    Computes CVaR-style tail average across stresses and spot.

    Args:
        stressed_payoffs: list of payoff tensors [N]
        q: tail fraction (e.g. 0.1 = worst 10%)

    Returns:
        Scalar tensor (CVaR)
    """
    all_payoffs = torch.cat(stressed_payoffs)
    k = max(1, int(q * all_payoffs.numel()))

    worst_k, _ = torch.topk(
        all_payoffs,
        k=k,
        largest=False,
    )

    return worst_k.mean()
