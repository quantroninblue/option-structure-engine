from grids import make_spot_grid, make_moneyness_grid

spot_grid = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=11,
)

moneyness_grid = make_moneyness_grid(
    k_min=-2.0,
    k_max=2.0,
    n_points=11,
)

print("Spot grid:")
print(spot_grid)
print("Shape:", spot_grid.shape)

print("\nMoneyness grid:")
print(moneyness_grid)
print("Shape:", moneyness_grid.shape)
