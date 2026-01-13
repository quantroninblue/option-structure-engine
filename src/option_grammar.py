"""
Convex option grammars for SPX 0-DTE:
- Iron Condor
- Butterfly

All structures are defined-risk by construction.
"""

import torch

CONTRACT_MULT = 100

def iron_condor(spot, center_offset, wing, width, size):
    """
    K1 < K2 < K3 < K4
    +1 long put, -1 short put, -1 short call, +1 long call
    """
    K2 = spot * torch.exp(center_offset - width/2)
    K3 = spot * torch.exp(center_offset + width/2)
    K1 = K2 * torch.exp(-wing)
    K4 = K3 * torch.exp( wing)

    return [
        {"option_type": "put",  "strike": float(K1), "weight": +size},
        {"option_type": "put",  "strike": float(K2), "weight": -size},
        {"option_type": "call", "strike": float(K3), "weight": -size},
        {"option_type": "call", "strike": float(K4), "weight": +size},
    ]


def butterfly(spot, center_offset, wing, size):
    """
    +1 long, -2 short, +1 long (all same type)
    Implemented as 4 legs with duplicated shorts.
    """
    K2 = spot * torch.exp(center_offset)
    K1 = K2 * torch.exp(-wing)
    K3 = K2 * torch.exp( wing)

    return [
        {"option_type": "call", "strike": float(K1), "weight": +size},
        {"option_type": "call", "strike": float(K2), "weight": -size},
        {"option_type": "call", "strike": float(K2), "weight": -size},
        {"option_type": "call", "strike": float(K3), "weight": +size},
    ]
