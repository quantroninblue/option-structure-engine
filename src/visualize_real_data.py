"""
Option 1 — Visualization of structures learned from real vol data.
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt

from grids import make_spot_grid, make_moneyness_grid
from csv_adapter import load_vol_surface_from_csv
from real_vol import resample_vol_surface, normalize_vol_surface
from regime_encoder import RegimeEncoder, real_vol_features
from portfolio_generator import PortfolioGenerator, decode_portfolio_tensor
from physics import terminal_portfolio_payoff
from stress_engine import spot_shock


# -------------------------------------------------
# Load trained generator
# -------------------------------------------------

encoder = RegimeEncoder(input_dim=3, hidden_dim=16, latent_dim=8)
generator = PortfolioGenerator(latent_dim=8, hidden_dim=64)

generator.load_state_dict(
    torch.load("checkpoints/generator.pt", map_location="cpu")
)
generator.eval()


# -------------------------------------------------
# Load real vol surface from CSV
# -------------------------------------------------

surface = load_vol_surface_from_csv(
    path="example_vol.csv",
    maturity=0.5,
)

spot0 = surface["spot"]

k_grid = make_moneyness_grid(-1.0, 1.0, 81)
spot_grid = make_spot_grid(
    spot=spot0,
    sigma=0.2,
    n_std=3.0,
    n_points=81,
)

vol_grid = resample_vol_surface(
    strikes=surface["strikes"],
    vol=surface["implied_vol"],
    spot=spot0,
    k_grid=k_grid,
)

vol_norm = normalize_vol_surface(vol_grid)

features = real_vol_features(k_grid, vol_norm)
latent = encoder(features)


# -------------------------------------------------
# Generate portfolio
# -------------------------------------------------

portfolio_tensor = generator(latent)

legs = decode_portfolio_tensor(
    portfolio_tensor=portfolio_tensor.squeeze(0),
    spot=spot0,
)

print("Learned real-data portfolio:")
for leg in legs:
    print(leg)


# -------------------------------------------------
# Payoffs
# -------------------------------------------------

payoff = terminal_portfolio_payoff(spot_grid, legs)

payoff_down = terminal_portfolio_payoff(
    spot_shock(spot_grid, -0.3), legs
)

payoff_up = terminal_portfolio_payoff(
    spot_shock(spot_grid, 0.3), legs
)


# -------------------------------------------------
# Gamma (finite difference)
# -------------------------------------------------

dS = spot_grid[1:] - spot_grid[:-1]
gamma = (
    payoff[2:] - 2 * payoff[1:-1] + payoff[:-2]
) / (dS[1:] * dS[:-1])

spot_mid = spot_grid[1:-1]


# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(14, 10))

# Payoff
plt.subplot(2, 2, 1)
plt.plot(spot_grid, payoff, label="Base")
plt.plot(spot_grid, payoff_down, "--", label="Spot Down")
plt.plot(spot_grid, payoff_up, "--", label="Spot Up")
plt.axhline(0, color="black", linewidth=0.5)
plt.title("Real-Data Learned Terminal Payoff")
plt.xlabel("Spot")
plt.ylabel("Payoff")
plt.legend()

# Gamma
plt.subplot(2, 2, 2)
plt.plot(spot_mid, gamma)
plt.axhline(0, color="black", linewidth=0.5)
plt.title("Gamma (Real-Data Structure)")
plt.xlabel("Spot")
plt.ylabel("Gamma")

# Vol surface
plt.subplot(2, 2, 3)
plt.plot(k_grid, vol_grid)
plt.title("Real Implied Vol Surface")
plt.xlabel("Log-Moneyness")
plt.ylabel("Implied Vol")

# Weights
plt.subplot(2, 2, 4)
weights = [leg["weight"] for leg in legs]
plt.bar(range(len(weights)), weights)
plt.title("Portfolio Weights (Real Data)")
plt.xlabel("Leg Index")
plt.ylabel("Weight")

plt.tight_layout()
plt.show()
