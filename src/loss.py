"""
Capital-aware convex structural objective.
"""

import torch

from src.stress_engine import aggregate_cvar
from src.constraints import (
    VirtualIBKRAccount,
    short_call_init_margin,
    short_put_init_margin,
)


# ============================================================
# CAPITAL PHYSICS LOSS
# ============================================================

def margin_penalty(legs, spot):
    """
    Deterministic Reg-T margin usage proxy.
    """
    total = 0.0
    for leg in legs:
        q = leg["weight"]
        if q >= 0:
            continue
        k = leg["strike"]

        if leg["option_type"] == "call":
            m = short_call_init_margin(0.0, spot, k)
        else:
            m = short_put_init_margin(0.0, spot, k)

        total += abs(q) * m

    return total


def convexity_reward(payoff: torch.Tensor, spot: torch.Tensor):
    """
    Rewards second-derivative convexity around ATM.
    """
    d2 = payoff[2:] - 2 * payoff[1:-1] + payoff[:-2]
    return torch.mean(torch.relu(d2))


# ============================================================
# FINAL STRUCTURAL OBJECTIVE
# ============================================================

def structural_objective(
    payoff: torch.Tensor,
    spot: torch.Tensor,
    stressed_payoffs: list,
    legs: list,
    alpha: float = 8.0,
    beta: float = 0.05,
    gamma: float = 0.02,
):
    """
    Capital-aware convex alpha objective.
    """

    # Convexity reward
    convex = convexity_reward(payoff, spot)

    # Tail risk penalty
    cvar = aggregate_cvar(stressed_payoffs)
    tail_penalty = torch.clamp(-cvar, min=0.0)

    # Margin consumption penalty
    margin_use = margin_penalty(legs, spot)

    # Final scalar objective
    return convex - alpha * tail_penalty - beta * margin_use - gamma * sum(abs(l["weight"]) for l in legs)

def differentiable_convex_proxy(raw_tensor, spot):
    """
    Smooth convexity proxy for backprop.
    """
    offsets = raw_tensor[..., 1]
    weights = raw_tensor[..., 2]

    # Encourage ATM concentration
    convex_mass = torch.exp(- (offsets ** 2) / 0.05)

    # Penalize large size
    size_penalty = torch.abs(weights)

    return - torch.mean(convex_mass * size_penalty)
