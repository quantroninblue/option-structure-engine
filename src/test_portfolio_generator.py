import torch

from grids import make_moneyness_grid
from regime_encoder import vol_surface_features, RegimeEncoder
from portfolio_generator import PortfolioGenerator, decode_portfolio_tensor

# Synthetic vol surface
k = make_moneyness_grid(
    k_min=-1.0,
    k_max=1.0,
    n_points=41,
)

vol = 0.2 + 0.05 * k**2 - 0.02 * k

# Regime features + encoder
features = vol_surface_features(k, vol)
encoder = RegimeEncoder(input_dim=3, hidden_dim=16, latent_dim=8)
latent = encoder(features)

# Portfolio generator
generator = PortfolioGenerator(latent_dim=8, hidden_dim=64)

portfolio_tensor = generator(latent)

print("Raw portfolio tensor shape:", portfolio_tensor.shape)
print(portfolio_tensor)

# Decode to option legs
spot = 100.0
legs = decode_portfolio_tensor(
    portfolio_tensor=portfolio_tensor.squeeze(0),
    spot=spot,
)

print("\nDecoded portfolio legs:")
for leg in legs:
    print(leg)
