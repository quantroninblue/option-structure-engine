import torch

from grids import make_spot_grid
from physics import price_portfolio, portfolio_vega
from constraints import bound_vega

# Spot grid
spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=41,
)

# Vol tensor
vol = torch.tensor(0.2, requires_grad=True)
maturity = 1.0

# Portfolio with strong vega
legs = [
    {"option_type": "call", "strike": 95.0, "weight": 2.0},
    {"option_type": "call", "strike": 105.0, "weight": -2.0},
]

def portfolio_price_fn(v):
    return price_portfolio(
        spot=spot,
        legs=legs,
        vol=v,
        maturity=maturity,
    )

vega = portfolio_vega(
    vol=vol,
    portfolio_price_fn=portfolio_price_fn,
)

norm_before = torch.linalg.norm(vega, ord=2)

max_vega = 20.0
bounded_vega = bound_vega(
    vega=vega,
    max_vega=max_vega,
)

norm_after = torch.linalg.norm(bounded_vega, ord=2)

print("Vega norm before:", norm_before.item())
print("Vega norm after:", norm_after.item())
tol = 1e-5
print("Bound respected:", norm_after.item() <= max_vega + tol)
print("Excess:", norm_after.item() - max_vega)

