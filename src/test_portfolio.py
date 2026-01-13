from grids import make_spot_grid
from physics import price_portfolio

# Spot grid
spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=41,
)

# Define a simple call spread
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

print("Spot:")
print(spot)

print("\nPortfolio payoff:")
print(portfolio_price)

print("\nShape:", portfolio_price.shape)
