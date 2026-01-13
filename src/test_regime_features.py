import torch

from grids import make_moneyness_grid
from regime_encoder import vol_surface_features

# Log-moneyness grid
k = make_moneyness_grid(
    k_min=-1.0,
    k_max=1.0,
    n_points=41,
)

# Synthetic vol smile
vol = 0.2 + 0.05 * k**2 - 0.02 * k

features = vol_surface_features(k, vol)

print("Level, Skew, Curvature:")
print(features)
print("Shape:", features.shape)
