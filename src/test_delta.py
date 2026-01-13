import torch

from grids import make_spot_grid
from physics import price_portfolio, portfolio_delta

# Spot grid (IMPORTANT: requires_grad=True)
spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=41,
)
spot.requires_grad_(True)

# Same call spread as before
legs = [
    {"option_type": "call", "strike": 90.0, "weight": 1.0},
    {"option_type": "call", "strike": 110.0, "weight": -1.0},
]

vol = 0.2
maturity = 1.0

portfolio_price = price_portfolio(
    spot=spot,
    legs=legs,
    vol=vol,
    maturity=maturity,
)

delta = portfolio_delta(
    spot=spot,
    portfolio_price=portfolio_price,
)

print("Spot:")
print(spot.detach())

print("\nPortfolio payoff:")
print(portfolio_price.detach())

print("\nDelta:")
print(delta.detach())

print("\nShapes:", portfolio_price.shape, delta.shape)
