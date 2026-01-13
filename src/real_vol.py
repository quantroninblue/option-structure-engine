"""
Real implied volatility surface ingestion utilities.
"""

import torch
import numpy as np


def strikes_to_log_moneyness(
    strikes: torch.Tensor,
    spot: float,
) -> torch.Tensor:
    return torch.log(strikes / spot)


def resample_vol_surface(
    strikes: torch.Tensor,
    vol: torch.Tensor,
    spot: float,
    k_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolates implied vol onto a fixed log-moneyness grid.
    Uses NumPy interpolation (safe for regime features).
    """
    k = strikes_to_log_moneyness(strikes, spot)

    # Move to CPU numpy (regime features are non-differentiable)
    k_np = k.detach().cpu().numpy()
    vol_np = vol.detach().cpu().numpy()
    k_grid_np = k_grid.detach().cpu().numpy()

    # Sort by k
    idx = np.argsort(k_np)
    k_np = k_np[idx]
    vol_np = vol_np[idx]

    # Linear interpolation
    vol_interp_np = np.interp(
        k_grid_np,
        k_np,
        vol_np,
        left=vol_np[0],
        right=vol_np[-1],
    )

    return torch.tensor(
        vol_interp_np,
        dtype=vol.dtype,
        device=vol.device,
    )


def normalize_vol_surface(vol: torch.Tensor) -> torch.Tensor:
    """
    Normalize vol surface for regime encoding.
    """
    mean = vol.mean()
    std = vol.std() + 1e-6
    return (vol - mean) / std
