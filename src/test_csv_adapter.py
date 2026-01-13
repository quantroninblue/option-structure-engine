import torch

from csv_adapter import load_vol_surface_from_csv
from real_vol import resample_vol_surface, normalize_vol_surface
from regime_encoder import real_vol_features, RegimeEncoder
from grids import make_moneyness_grid


# ---------------------------------------------
# Load CSV (you can replace path later)
# ---------------------------------------------

surface = load_vol_surface_from_csv(
    path="example_vol.csv",
    maturity=0.5,  # override for test
)

print("Loaded surface:")
for k, v in surface.items():
    print(k, v if not torch.is_tensor(v) else v.shape)


# ---------------------------------------------
# Resample & encode
# ---------------------------------------------

k_grid = make_moneyness_grid(-1.0, 1.0, 41)

vol_grid = resample_vol_surface(
    strikes=surface["strikes"],
    vol=surface["implied_vol"],
    spot=surface["spot"],
    k_grid=k_grid,
)

vol_norm = normalize_vol_surface(vol_grid)

features = real_vol_features(k_grid, vol_norm)

encoder = RegimeEncoder(input_dim=3, hidden_dim=16, latent_dim=8)
latent = encoder(features)

print("\nRegime features:", features)
print("Latent embedding:", latent)
