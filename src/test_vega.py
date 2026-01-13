import torch

from grids import make_spot_grid
from physics import price_portfolio, portfolio_vega

# Spot grid
spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=41,
)

# Volatility tensor
vol = torch.tensor(0.2, requires_grad=True)
maturity = 1.0

legs = [
    {"option_type": "call", "strike": 90.0, "weight": 1.0},
    {"option_type": "call", "strike": 110.0, "weight": -1.0},
]

# Define price as a function of vol (IMPORTANT)
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

print("Spot:")
print(spot)

print("\nVega per spot:")
print(vega.detach())

print("\nShape:", vega.shape)
