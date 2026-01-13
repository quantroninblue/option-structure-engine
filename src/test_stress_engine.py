import torch

from grids import make_spot_grid
from physics import price_portfolio
from stress_engine import spot_shock, vol_level_shift

# Spot grid
spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=41,
)

# Simple portfolio: long call
legs = [
    {"option_type": "call", "strike": 100.0, "weight": 1.0},
]

vol = torch.tensor(0.2)
maturity = 1.0

# Baseline price
price_base = price_portfolio(
    spot=spot,
    legs=legs,
    vol=vol,
    maturity=maturity,
)

# Stress scenarios
spot_down = spot_shock(spot, shock=-0.3)
spot_up = spot_shock(spot, shock=+0.3)

vol_up = vol_level_shift(vol, shift=+0.05)

price_spot_down = price_portfolio(
    spot=spot_down,
    legs=legs,
    vol=vol,
    maturity=maturity,
)

price_spot_up = price_portfolio(
    spot=spot_up,
    legs=legs,
    vol=vol,
    maturity=maturity,
)

price_vol_up = price_portfolio(
    spot=spot,
    legs=legs,
    vol=vol_up,
    maturity=maturity,
)

print("Baseline price (min, max):",
      price_base.min().item(), price_base.max().item())

print("Spot down stress (min, max):",
      price_spot_down.min().item(), price_spot_down.max().item())

print("Spot up stress (min, max):",
      price_spot_up.min().item(), price_spot_up.max().item())

print("Vol up stress (min, max):",
      price_vol_up.min().item(), price_vol_up.max().item())
