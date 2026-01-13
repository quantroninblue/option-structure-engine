"""
Hard constraint enforcement for option portfolios.
"""

import torch
import torch.nn.functional as F


# -------------------------------------------------
# Delta neutrality
# -------------------------------------------------

def project_delta_neutral(
    leg_deltas: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Projects portfolio weights to enforce delta neutrality.
    """
    numerator = torch.dot(weights, leg_deltas)
    denominator = torch.dot(leg_deltas, leg_deltas) + 1e-8
    return weights - (numerator / denominator) * leg_deltas


# -------------------------------------------------
# Vega bounding
# -------------------------------------------------

def bound_vega(
    vega: torch.Tensor,
    max_vega: float,
) -> torch.Tensor:
    """
    Scales vega to enforce ||vega|| <= max_vega.
    """
    norm = torch.linalg.norm(vega, ord=2)
    if norm <= max_vega:
        return vega

    scale = max_vega / (norm + 1e-8)
    return vega * scale


# -------------------------------------------------
# Terminal payoff convexity (non-uniform grid)
# -------------------------------------------------

def terminal_convexity_violation(
    payoff: torch.Tensor,
    spot: torch.Tensor,
) -> torch.Tensor:
    """
    Computes minimum second derivative of terminal payoff
    on a non-uniform spot grid.

    Negative values indicate convexity violation.
    """
    if payoff.ndim != 1 or spot.ndim != 1:
        raise ValueError("payoff and spot must be 1D tensors")

    if spot.shape[0] < 3:
        raise ValueError("need at least 3 grid points")

    # Grid spacings
    dS_forward = spot[2:] - spot[1:-1]
    dS_backward = spot[1:-1] - spot[:-2]

    # Payoff differences
    df_forward = payoff[2:] - payoff[1:-1]
    df_backward = payoff[1:-1] - payoff[:-2]

    # Second derivative on non-uniform grid
    gamma = (
        2.0
        * (df_forward / dS_forward - df_backward / dS_backward)
        / (dS_forward + dS_backward)
    )

    return gamma.min()


def convexity_barrier(
    payoff: torch.Tensor,
    spot: torch.Tensor,
    barrier_scale: float = 100.0,
) -> torch.Tensor:
    """
    Hard convexity barrier: positive penalty if payoff is concave.
    """
    min_gamma = terminal_convexity_violation(payoff, spot)
    return F.softplus(-barrier_scale * min_gamma)

# ============================================================
# IBKR REG-T CAPITAL PHYSICS (DETERMINISTIC, VIRTUAL ACCOUNT)
# ============================================================

IBKR_EQUITY = 25000.0   # USD
WARNING_RATIO = 1.30   # margin call boundary
LIQUIDATION_RATIO = 1.00

class VirtualIBKRAccount:
    def __init__(self, equity: float = IBKR_EQUITY):
        self.equity = equity
        self.init_margin_used = 0.0
        self.maint_margin = 0.0

    @property
    def buying_power(self) -> float:
        # Reg-T retail buying power (2x equity minus used initial margin)
        return max(0.0, self.equity * 2.0 - self.init_margin_used)

    def margin_ratio(self) -> float:
        return self.equity / max(self.maint_margin, 1.0)

def short_call_init_margin(premium: float, spot: float, strike: float) -> float:
    otm = max(0.0, strike - spot)
    return premium + max(0.20 * spot - otm, 0.10 * spot)

def short_put_init_margin(premium: float, spot: float, strike: float) -> float:
    otm = max(0.0, spot - strike)
    return premium + max(0.20 * spot - otm, 0.10 * strike)

def spread_init_margin(max_loss: float) -> float:
    # Reg-T: max loss for defined-risk spreads
    return max_loss

def capital_feasible(account: VirtualIBKRAccount, add_init: float, add_maint: float) -> bool:
    # Initial margin constraint (buying power)
    if account.init_margin_used + add_init > account.equity * 2.0:
        return False
    # Maintenance margin constraint
    if (account.maint_margin + add_maint) <= 0:
        return True
    return (account.equity / (account.maint_margin + add_maint)) >= LIQUIDATION_RATIO

def margin_call(account: VirtualIBKRAccount) -> bool:
    return account.margin_ratio() < WARNING_RATIO

def forced_liquidation(account: VirtualIBKRAccount) -> bool:
    return account.margin_ratio() < LIQUIDATION_RATIO

