"""
Volatility surface regime representation.
Stable for both single-maturity and term-structure regimes.
"""

import torch
import torch.nn as nn


# ============================================================
# SINGLE SMILE FEATURES
# ============================================================

def vol_surface_features(log_moneyness: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
    if log_moneyness.ndim != 1 or vol.ndim != 1:
        raise ValueError("inputs must be 1D tensors")
    if log_moneyness.shape != vol.shape:
        raise ValueError("log_moneyness and vol must have same shape")

    dk = log_moneyness[1:] - log_moneyness[:-1]
    dv = vol[1:] - vol[:-1]

    d2v = vol[2:] - 2 * vol[1:-1] + vol[:-2]
    d2k = (dk[1:] + dk[:-1]) / 2.0

    level = vol.mean()
    skew = (dv / dk).mean()
    curvature = (d2v / (d2k ** 2)).mean()

    return torch.stack([level, skew, curvature])


# ============================================================
# TERM-STRUCTURE FEATURES (0-DTE SAFE)
# ============================================================

def multi_maturity_vol_features(k_grid: torch.Tensor, vol_surfaces: list) -> torch.Tensor:
    levels, slopes, curvatures = [], [], []

    for vol in vol_surfaces:
        dvol_dk = torch.gradient(vol, spacing=(k_grid,))[0]
        d2vol_dk2 = torch.gradient(dvol_dk, spacing=(k_grid,))[0]

        levels.append(vol.mean())
        slopes.append(dvol_dk.mean())
        curvatures.append(d2vol_dk2.mean())

    levels = torch.stack(levels)
    slopes = torch.stack(slopes)
    curvatures = torch.stack(curvatures)

    # unbiased=False makes single-slice regimes well-posed
    return torch.stack([
        levels.mean(),
        levels.std(unbiased=False),
        slopes.mean(),
        slopes.std(unbiased=False),
        curvatures.mean(),
        curvatures.std(unbiased=False),
    ])


# ============================================================
# REGIME ENCODER
# ============================================================

class RegimeEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16, latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
