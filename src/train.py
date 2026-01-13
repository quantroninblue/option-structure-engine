"""
Differentiable capital-aware convexity training loop.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.optim as optim
import pandas as pd

from src.grids import make_moneyness_grid, make_spot_grid
from src.regime_encoder import RegimeEncoder, multi_maturity_vol_features
from src.portfolio_generator import PortfolioGenerator, decode_portfolio_tensor, capital_filter
from src.surface_extractor import extract_surfaces_from_df
from src.real_vol import resample_vol_surface, normalize_vol_surface
from src.loss import differentiable_convex_proxy


# -------------------------------------------------
# Config
# -------------------------------------------------

torch.manual_seed(0)

NUM_STEPS = 500
LR = 1e-3

CSV_PATH = "example_vol.csv"
SPOT_FALLBACK = 100.0


# -------------------------------------------------
# Grids
# -------------------------------------------------

k_grid = make_moneyness_grid(-1.0, 1.0, 41)


# -------------------------------------------------
# Models
# -------------------------------------------------

encoder = RegimeEncoder(input_dim=6)
generator = PortfolioGenerator(latent_dim=8)
optimizer = optim.Adam(generator.parameters(), lr=LR)


# -------------------------------------------------
# Load surfaces
# -------------------------------------------------

df = pd.read_csv(CSV_PATH)
surfaces = list(extract_surfaces_from_df(df, spot=SPOT_FALLBACK))
if not surfaces:
    raise RuntimeError("No valid vol surfaces extracted")


# -------------------------------------------------
# Training loop
# -------------------------------------------------

for step in range(NUM_STEPS):

    surface = surfaces[step % len(surfaces)]

    # --- Regime encoding ---
    vol_grid = resample_vol_surface(surface["strikes"], surface["implied_vol"], surface["spot"], k_grid)
    vol_norm = normalize_vol_surface(vol_grid)

    features = multi_maturity_vol_features(k_grid, [vol_norm]).unsqueeze(0)
    z = encoder(features)

    # --- Generate ---
    raw = generator(z)
    decoded = decode_portfolio_tensor(raw[0], surface["spot"])

    feasible, _ = capital_filter(decoded, surface["spot"])
    if not feasible:
        continue

    # --- Differentiable convex proxy loss ---
    loss = differentiable_convex_proxy(raw, surface["spot"])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step:03d} | Loss {loss.item():.4f}")

print("Training complete.")

os.makedirs("checkpoints", exist_ok=True)
torch.save(generator.state_dict(), "checkpoints/generator.pt")
print("\nSaved trained generator to checkpoints/generator.pt")
