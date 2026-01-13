from grids import make_spot_grid
from physics import bs_price

# Spot grid
spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=21,
)

strike = 100.0
vol = 0.2
maturity = 1.0

call = bs_price(
    spot=spot,
    strike=strike,
    vol=vol,
    maturity=maturity,
    option_type="call",
)

put = bs_price(
    spot=spot,
    strike=strike,
    vol=vol,
    maturity=maturity,
    option_type="put",
)

print("Spot:")
print(spot)

print("\nCall price:")
print(call)

print("\nPut price:")
print(put)

print("\nShapes:", call.shape, put.shape)
