"""
Surface extractor for mixed / messy vol CSVs.
"""

import pandas as pd
import torch
from typing import Iterator, Dict, Optional

from src.csv_adapter import load_live_spx_chain





from src.csv_adapter import (
    STRIKE_ALIASES,
    VOL_ALIASES,
    MATURITY_ALIASES,
    _find_column,
    _coerce_vol_units,
)


def extract_surfaces_from_df(
    df: pd.DataFrame,
    spot: float,
) -> Iterator[Dict]:
    """
    Yields VolSurface dicts from a mixed CSV DataFrame.
    """

    strike_col = _find_column(df, STRIKE_ALIASES)
    vol_col = _find_column(df, VOL_ALIASES)

    if strike_col is None or vol_col is None:
        return  # silently skip bad CSVs

    mat_col = _find_column(df, MATURITY_ALIASES)

    if mat_col is None:
        # Treat entire CSV as one surface
        yield _build_surface(
            df,
            strike_col,
            vol_col,
            spot,
            maturity=None,
        )
        return

    # Group by maturity / expiry
    for maturity, group in df.groupby(mat_col):
        surface = _build_surface(
            group,
            strike_col,
            vol_col,
            spot,
            maturity,
        )
        if surface is not None:
            yield surface


def _build_surface(
    df: pd.DataFrame,
    strike_col: str,
    vol_col: str,
    spot: float,
    maturity: Optional[float],
) -> Optional[Dict]:
    strikes = df[strike_col]
    vol = _coerce_vol_units(df[vol_col])

    mask = strikes.notna() & vol.notna()
    strikes = strikes[mask]
    vol = vol[mask]

    if len(strikes) < 5:
        return None  # too sparse

    try:
        strikes_t = torch.tensor(strikes.values, dtype=torch.float)
        vol_t = torch.tensor(vol.values, dtype=torch.float)
    except Exception:
        return None

    return {
        "spot": float(spot),
        "strikes": strikes_t,
        "implied_vol": vol_t,
        "maturity": float(maturity) if maturity is not None else None,
    }

def extract_multi_maturity_surface(
    df: pd.DataFrame,
    spot: float,
) -> dict:
    """
    Extract a multi-maturity vol surface bundle from a mixed CSV.
    """

    surfaces = list(extract_surfaces_from_df(df, spot))

    if len(surfaces) < 2:
        raise ValueError("Need at least 2 maturities for multi-maturity surface")

    maturities = []
    strikes = []
    vols = []

    for s in surfaces:
        maturities.append(s["maturity"])
        strikes.append(s["strikes"])
        vols.append(s["implied_vol"])

    return {
        "spot": spot,
        "maturities": maturities,
        "strikes": strikes,
        "implied_vol": vols,
    }



# ============================================================
# LIVE SPX GEOMETRY
# ============================================================

def extract_live_spx_surface():
    feed_df = load_live_spx_chain()
    spot = float(feed_df["strike"].median())

    try:
        return extract_multi_maturity_surface(feed_df, spot)
    except ValueError:
        # Fallback: single maturity smile geometry (0-DTE)
        surfaces = list(extract_surfaces_from_df(feed_df, spot))
        s = surfaces[0]

        return {
            "spot": s["spot"],
            "maturities": [s["maturity"]],
            "strikes": [s["strikes"]],
            "implied_vol": [s["implied_vol"]],
        }


