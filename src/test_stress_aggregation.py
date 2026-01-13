import torch

from grids import make_spot_grid
from physics import terminal_portfolio_payoff
from stress_engine import (
    spot_shock,
    aggregate_worst_case,
    aggregate_cvar,
)

# Spot grid
spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=41,
)

# Portfolio: long call spread
legs = [
    {"option_type": "call", "strike": 90.0, "weight": 1.0},
    {"option_type": "call", "strike": 110.0, "weight": -1.0},
]

# Stress scenarios (spot only, for simplicity)
shocks = [-0.4, -0.2, 0.0, 0.2, 0.4]

stressed_payoffs = []

for shock in shocks:
    spot_s = spot_shock(spot, shock)
    payoff = terminal_portfolio_payoff(
        spot_s,
        legs,
    )
    stressed_payoffs.append(payoff)

worst = aggregate_worst_case(stressed_payoffs)
cvar = aggregate_cvar(stressed_payoffs, q=0.1)

print("Worst-case payoff:", worst.item())
print("CVaR payoff:", cvar.item())
