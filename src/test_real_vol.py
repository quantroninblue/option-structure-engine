import torch

from grids import make_moneyness_grid
from real_vol import resample_vol_surface, normalize_vol_surface
from regime_encoder import real_vol_features, RegimeEncoder


# Fake "real" market surface
spot = 100.0
strikes = torch.tensor([70, 80, 90, 100, 110, 120, 130], dtype=torch.float)
implied_vol = torch.tensor([0.38, 0.32, 0.28, 0.25, 0.27, 0.30, 0.34])

k_grid = make_moneyness_grid(-1.0, 1.0, 41)

vol_grid = resample_vol_surface(
    strikes=strikes,
    vol=implied_vol,
    spot=spot,
    k_grid=k_grid,
)

vol_norm = normalize_vol_surface(vol_grid)

features = real_vol_features(k_grid, vol_norm)

encoder = RegimeEncoder(input_dim=3, hidden_dim=16, latent_dim=8)
latent = encoder(features)

print("Real vol features:", features)
print("Latent embedding:", latent)
