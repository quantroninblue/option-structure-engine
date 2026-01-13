"""
Phase 9.2 — Regime sweep visualization.
"""

import torch
import matplotlib.pyplot as plt

from grids import make_moneyness_grid, make_spot_grid
from regime_encoder import vol_surface_features, RegimeEncoder
from portfolio_generator import PortfolioGenerator, decode_portfolio_tensor
from physics import terminal_portfolio_payoff


# -------------------------------------------------
# Setup
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

# Fixed regime components
LEVEL = 0.25
SKEW = -0.02

# Sweep curvature
CURVATURES = [0.02, 0.05, 0.10, 0.15]

# -------------------------------------------------
# Load trained models
# -------------------------------------------------

encoder = RegimeEncoder(input_dim=3, hidden_dim=16, latent_dim=8)
generator = PortfolioGenerator(latent_dim=8, hidden_dim=64)

generator.load_state_dict(
    torch.load("checkpoints/generator.pt", map_location="cpu")
)
generator.eval()

# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(10, 6))

for curv in CURVATURES:
    vol = LEVEL + SKEW * k + curv * k**2

    features = vol_surface_features(k, vol)
    latent = encoder(features)

    portfolio_tensor = generator(latent)

    legs = decode_portfolio_tensor(
        portfolio_tensor=portfolio_tensor.squeeze(0),
        spot=100.0,
    )

    payoff = terminal_portfolio_payoff(spot, legs)

    plt.plot(
        spot,
        payoff,
        label=f"curvature={curv:.2f}",
    )

plt.axhline(0, color="black", linewidth=0.5)
plt.xlabel("Spot")
plt.ylabel("Terminal Payoff")
plt.title("Regime Sweep: Curvature Sensitivity")
plt.legend()
plt.tight_layout()
plt.show()
