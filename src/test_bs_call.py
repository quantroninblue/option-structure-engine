import torch

from grids import make_spot_grid
from physics import bs_call_price

# Create spot grid
spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=21,
)

# Call parameters
strike = 100.0
vol = 0.2
maturity = 1.0

call_price = bs_call_price(
    spot=spot,
    strike=strike,
    vol=vol,
    maturity=maturity,
)

print("Spot:")
print(spot)

print("\nCall price:")
print(call_price)

print("\nShape:", call_price.shape)
