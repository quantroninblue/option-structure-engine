"""
Robust CSV adapter for implied volatility surfaces.
"""

import pandas as pd
import torch
from typing import Dict, Optional


# ---------------------------------------------
# Column name heuristics
# ---------------------------------------------

STRIKE_ALIASES = [
    "strike", "strikes", "k", "K", "strike_price", "Strike", "STRIKE_PX"
]

VOL_ALIASES = [
    "iv", "vol", "implied_vol", "implied_volatility",
    "IV", "VOL", "ImpliedVol"
]

SPOT_ALIASES = [
    "spot", "underlying", "S", "Spot", "UNDERLYING_PX"
]

MATURITY_ALIASES = [
    "maturity", "tenor", "T", "expiry", "tau"
]


# ---------------------------------------------
# Helper utilities
# ---------------------------------------------

def _find_column(df: pd.DataFrame, aliases) -> Optional[str]:
    for col in df.columns:
        if col.lower() in [a.lower() for a in aliases]:
            return col
    return None


def _coerce_vol_units(vol: pd.Series) -> pd.Series:
    """
    Convert vol to decimal if given in percentage.
    """
    if vol.mean() > 1.5:
        return vol / 100.0
    return vol


# ---------------------------------------------
# Main adapter
# ---------------------------------------------

def load_vol_surface_from_csv(
    path: str,
    spot: Optional[float] = None,
    maturity: Optional[float] = None,
) -> Dict:
    """
    Load an implied vol surface from a messy CSV.

    Args:
        path: path to CSV file
        spot: optional override for spot
        maturity: optional override for maturity

    Returns:
        VolSurface dict
    """
    df = pd.read_csv(path)

    strike_col = _find_column(df, STRIKE_ALIASES)
    vol_col = _find_column(df, VOL_ALIASES)

    if strike_col is None:
        raise ValueError("Could not infer strike column from CSV")

    if vol_col is None:
        raise ValueError("Could not infer implied vol column from CSV")

    strikes = df[strike_col]
    vol = df[vol_col]

    vol = _coerce_vol_units(vol)

    # Drop invalid rows
    mask = strikes.notna() & vol.notna()
    strikes = strikes[mask]
    vol = vol[mask]

    # Convert to tensors
    strikes_t = torch.tensor(strikes.values, dtype=torch.float)
    vol_t = torch.tensor(vol.values, dtype=torch.float)

    # Infer spot if not provided
    if spot is None:
        spot_col = _find_column(df, SPOT_ALIASES)
        if spot_col is not None:
            spot = float(df[spot_col].dropna().iloc[0])
        else:
            # Fallback: ATM heuristic
            spot = float(strikes_t.median().item())

    # Infer maturity if not provided
    if maturity is None:
        mat_col = _find_column(df, MATURITY_ALIASES)
        if mat_col is not None:
            maturity = float(df[mat_col].dropna().iloc[0])
        else:
            raise ValueError("Maturity not found; must be provided")

    return {
        "spot": spot,
        "strikes": strikes_t,
        "implied_vol": vol_t,
        "maturity": maturity,
    }

# ============================================================
# LIVE SPX OPTION CHAIN (NO BROKER / NO KYC)
# ============================================================

import yfinance as yf

def load_live_spx_chain():
    """
    Loads the front (0-DTE / nearest) SPX option chain using free OPRA-delayed data.
    Returns a DataFrame compatible with the existing surface extractor.
    """
    spx = yf.Ticker("^SPX")

    expiries = spx.options
    if len(expiries) == 0:
        raise RuntimeError("No SPX option expiries available.")

    expiry = expiries[0]   # front expiry (0-DTE most days)
    chain = spx.option_chain(expiry)

    calls = chain.calls.copy()
    calls["option_type"] = "call"

    puts = chain.puts.copy()
    puts["option_type"] = "put"

    df = pd.concat([calls, puts], axis=0)

    # normalize column names to your surface extractor aliases
    df.rename(columns={
        "strike": "strike",
        "impliedVolatility": "implied_vol",
        "expiration": "maturity"
    }, inplace=True)

    return df
