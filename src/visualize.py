"""
Phase 9.0 — Visualization of learned option structures.
"""

import torch
import matplotlib.pyplot as plt

from grids import make_moneyness_grid, make_spot_grid
from regime_encoder import vol_surface_features, RegimeEncoder
from portfolio_generator import PortfolioGenerator, decode_portfolio_tensor
from physics import terminal_portfolio_payoff
from stress_engine import spot_shock


# -------------------------------------------------
# Setup (match training)
# -------------------------------------------------

torch.manual_seed(0)

spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=81,
)

k = make_moneyness_grid(
    k_min=-1.0,
    k_max=1.0,
    n_points=81,
)

# Example regime (choose one)
level = 0.25
skew = -0.02
curvature = 0.10
vol = level + skew * k + curvature * k**2

# -------------------------------------------------
# Load models (fresh init = structure, not weights)
# -------------------------------------------------

encoder = RegimeEncoder(input_dim=3, hidden_dim=16, latent_dim=8)
generator = PortfolioGenerator(latent_dim=8, hidden_dim=64)

generator.load_state_dict(
    torch.load("checkpoints/generator.pt", map_location="cpu")
)
generator.eval()


# -------------------------------------------------
# Generate portfolio
# -------------------------------------------------

features = vol_surface_features(k, vol)
latent = encoder(features)

portfolio_tensor = generator(latent)

legs = decode_portfolio_tensor(
    portfolio_tensor=portfolio_tensor.squeeze(0),
    spot=100.0,
)

print("Learned portfolio legs:")
for leg in legs:
    print(leg)

# -------------------------------------------------
# Payoffs
# -------------------------------------------------

payoff = terminal_portfolio_payoff(spot, legs)

payoff_down = terminal_portfolio_payoff(
    spot_shock(spot, -0.3), legs
)

payoff_up = terminal_portfolio_payoff(
    spot_shock(spot, +0.3), legs
)

# -------------------------------------------------
# Gamma (finite difference)
# -------------------------------------------------

dS = spot[1:] - spot[:-1]
gamma = (
    payoff[2:] - 2 * payoff[1:-1] + payoff[:-2]
) / (dS[1:] * dS[:-1])

spot_mid = spot[1:-1]

# -------------------------------------------------
# Plotting
# -------------------------------------------------

plt.figure(figsize=(14, 10))

# Payoff
plt.subplot(2, 2, 1)
plt.plot(spot, payoff, label="Base")
plt.plot(spot, payoff_down, "--", label="Spot Down")
plt.plot(spot, payoff_up, "--", label="Spot Up")
plt.axhline(0, color="black", linewidth=0.5)
plt.title("Terminal Payoff")
plt.xlabel("Spot")
plt.ylabel("Payoff")
plt.legend()

# Gamma
plt.subplot(2, 2, 2)
plt.plot(spot_mid, gamma)
plt.axhline(0, color="black", linewidth=0.5)
plt.title("Gamma (Finite Difference)")
plt.xlabel("Spot")
plt.ylabel("Gamma")

# Vol smile
plt.subplot(2, 2, 3)
plt.plot(k, vol)
plt.title("Volatility Regime")
plt.xlabel("Log-Moneyness")
plt.ylabel("Vol")

# Weights summary
plt.subplot(2, 2, 4)
weights = [leg["weight"] for leg in legs]
strikes = [leg["strike"] for leg in legs]
plt.bar(range(len(weights)), weights)
plt.title("Portfolio Weights")
plt.xlabel("Leg Index")
plt.ylabel("Weight")

plt.tight_layout()
plt.show()
