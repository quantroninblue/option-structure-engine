import torch

from grids import make_moneyness_grid
from regime_encoder import vol_surface_features, RegimeEncoder

# Synthetic vol surface
k = make_moneyness_grid(
    k_min=-1.0,
    k_max=1.0,
    n_points=41,
)

vol = 0.2 + 0.05 * k**2 - 0.02 * k

features = vol_surface_features(k, vol)

encoder = RegimeEncoder(
    input_dim=3,
    hidden_dim=16,
    latent_dim=8,
)

latent = encoder(features)

print("Regime features:", features)
print("Latent embedding:", latent)
print("Latent shape:", latent.shape)
