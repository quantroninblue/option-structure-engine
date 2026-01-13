"""
Differentiable Black–Scholes payoff engine.

Implements call and put option prices over a spot grid.
Greeks will be added later via autograd.
"""

import torch
from torch.distributions.normal import Normal


def bs_price(
    spot: torch.Tensor,
    strike: float,
    vol: float,
    maturity: float,
    option_type: str,
    rate: float = 0.0,
) -> torch.Tensor:
    """
    Black–Scholes option price (call or put).

    Args:
        spot: Tensor of spot prices [N]
        strike: Strike price
        vol: Implied volatility (scalar)
        maturity: Time to maturity (years)
        option_type: "call" or "put"
        rate: Risk-free rate

    Returns:
        Tensor of option prices [N]
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    eps = 1e-8
    sqrt_t = torch.sqrt(torch.tensor(maturity))

    d1 = (
        torch.log(spot / strike)
        + (rate + 0.5 * vol ** 2) * maturity
    ) / (vol * sqrt_t + eps)

    d2 = d1 - vol * sqrt_t

    normal = Normal(0.0, 1.0)

    discount = torch.exp(torch.tensor(-rate * maturity))

    if option_type == "call":
        price = spot * normal.cdf(d1) - strike * discount * normal.cdf(d2)
    else:
        price = strike * discount * normal.cdf(-d2) - spot * normal.cdf(-d1)

    return price

def price_portfolio(
    spot: torch.Tensor,
    legs: list,
    vol: float,
    maturity: float,
    rate: float = 0.0,
) -> torch.Tensor:
    """
    Prices a multi-leg option portfolio.

    Args:
        spot: Tensor of spot prices [N]
        legs: list of dicts, each with keys:
              {"option_type", "strike", "weight"}
        vol: Implied volatility (scalar)
        maturity: Time to maturity
        rate: Risk-free rate

    Returns:
        Tensor of portfolio prices [N]
    """
    portfolio_price = torch.zeros_like(spot)

    for leg in legs:
        leg_price = bs_price(
            spot=spot,
            strike=leg["strike"],
            vol=vol,
            maturity=maturity,
            option_type=leg["option_type"],
            rate=rate,
        )

        portfolio_price += leg["weight"] * leg_price

    return portfolio_price

def portfolio_delta(
    spot: torch.Tensor,
    portfolio_price: torch.Tensor,
) -> torch.Tensor:
    """
    Computes portfolio delta via autograd.

    Args:
        spot: Tensor of spot prices [N] with requires_grad=True
        portfolio_price: Tensor of portfolio prices [N]

    Returns:
        Tensor of deltas [N]
    """
    if not spot.requires_grad:
        raise ValueError("spot tensor must have requires_grad=True")

    delta = torch.autograd.grad(
        outputs=portfolio_price,
        inputs=spot,
        grad_outputs=torch.ones_like(portfolio_price),
        create_graph=True,
    )[0]

    return delta

def portfolio_gamma(
    spot: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """
    Computes portfolio gamma via autograd.

    Args:
        spot: Tensor of spot prices [N] with requires_grad=True
        delta: Tensor of portfolio deltas [N]

    Returns:
        Tensor of gammas [N]
    """
    if not spot.requires_grad:
        raise ValueError("spot tensor must have requires_grad=True")

    gamma = torch.autograd.grad(
        outputs=delta,
        inputs=spot,
        grad_outputs=torch.ones_like(delta),
        create_graph=True,
    )[0]

    return gamma

def portfolio_vega(
    vol: torch.Tensor,
    portfolio_price_fn,
) -> torch.Tensor:
    """
    Computes portfolio vega per spot via Jacobian.

    Args:
        vol: Volatility tensor with requires_grad=True
        portfolio_price_fn: function that maps vol -> portfolio_price [N]

    Returns:
        Tensor of vegas [N]
    """
    if not vol.requires_grad:
        raise ValueError("vol tensor must have requires_grad=True")

    vega = torch.autograd.functional.jacobian(
        portfolio_price_fn,
        vol,
    )

    return vega

def terminal_leg_payoff(
    spot: torch.Tensor,
    option_type: str,
    strike: float,
) -> torch.Tensor:
    """
    Terminal payoff of a single option leg.
    """
    if option_type == "call":
        return torch.clamp(spot - strike, min=0.0)
    elif option_type == "put":
        return torch.clamp(strike - spot, min=0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def terminal_portfolio_payoff(
    spot: torch.Tensor,
    legs: list,
) -> torch.Tensor:
    """
    Terminal payoff of a multi-leg option portfolio.
    """
    payoff = torch.zeros_like(spot)

    for leg in legs:
        leg_payoff = terminal_leg_payoff(
            spot=spot,
            option_type=leg["option_type"],
            strike=leg["strike"],
        )
        payoff += leg["weight"] * leg_payoff

    return payoff

