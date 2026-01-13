"""
Option 2 — Visualization of multi-maturity real-vol learned structures.
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt

from grids import make_spot_grid, make_moneyness_grid
from surface_extractor import extract_multi_maturity_surface
from real_vol import resample_vol_surface, normalize_vol_surface
from regime_encoder import RegimeEncoder, multi_maturity_vol_features
from portfolio_generator import PortfolioGenerator, decode_portfolio_tensor
from physics import terminal_portfolio_payoff
from stress_engine import spot_shock


# -------------------------------------------------
# Load trained multi-maturity model
# -------------------------------------------------

encoder = RegimeEncoder(input_dim=6, hidden_dim=16, latent_dim=8)
generator = PortfolioGenerator(latent_dim=8, hidden_dim=64)

generator.load_state_dict(
    torch.load("checkpoints/generator.pt", map_location="cpu")
)
generator.eval()


# -------------------------------------------------
# Load multi-maturity vol data
# -------------------------------------------------

df = pd.read_csv("example_vol_multi.csv")

SPOT = 100.0

bundle = extract_multi_maturity_surface(df, spot=SPOT)

k_grid = make_moneyness_grid(-1.0, 1.0, 81)
spot_grid = make_spot_grid(
    spot=SPOT,
    sigma=0.2,
    n_std=3.0,
    n_points=81,
)

# -------------------------------------------------
# Resample all maturities
# -------------------------------------------------

vol_surfaces = []

for strikes, vol in zip(bundle["strikes"], bundle["implied_vol"]):
    vol_grid = resample_vol_surface(
        strikes=strikes,
        vol=vol,
        spot=bundle["spot"],
        k_grid=k_grid,
    )
    vol_surfaces.append(normalize_vol_surface(vol_grid))

features = multi_maturity_vol_features(k_grid, vol_surfaces)
latent = encoder(features)


# -------------------------------------------------
# Generate portfolio
# -------------------------------------------------

portfolio_tensor = generator(latent)

legs = decode_portfolio_tensor(
    portfolio_tensor=portfolio_tensor.squeeze(0),
    spot=SPOT,
)

print("Learned multi-maturity portfolio:")
for leg in legs:
    print(leg)


# -------------------------------------------------
# Payoffs & stress
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
plt.title("Multi-Maturity Learned Terminal Payoff")
plt.xlabel("Spot")
plt.ylabel("Payoff")
plt.legend()

# Gamma
plt.subplot(2, 2, 2)
plt.plot(spot_mid, gamma)
plt.axhline(0, color="black", linewidth=0.5)
plt.title("Gamma (Multi-Maturity Structure)")
plt.xlabel("Spot")
plt.ylabel("Gamma")

# Vol term structure
plt.subplot(2, 2, 3)
for i, vol in enumerate(vol_surfaces):
    plt.plot(k_grid, vol, label=f"T{i}")
plt.title("Normalized Vol Smiles (All Maturities)")
plt.xlabel("Log-Moneyness")
plt.ylabel("Normalized Vol")
plt.legend()

# Weights
plt.subplot(2, 2, 4)
weights = [leg["weight"] for leg in legs]
plt.bar(range(len(weights)), weights)
plt.title("Portfolio Weights (Multi-Maturity)")
plt.xlabel("Leg Index")
plt.ylabel("Weight")

plt.tight_layout()
plt.show()

