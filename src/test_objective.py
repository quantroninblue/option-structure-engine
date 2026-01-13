import torch

from grids import make_spot_grid
from physics import terminal_portfolio_payoff
from stress_engine import spot_shock
from loss import structural_objective

# Spot grid
spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=41,
)

# Two portfolios
convex_legs = [
    {"option_type": "call", "strike": 90.0, "weight": 1.0},
    {"option_type": "call", "strike": 110.0, "weight": -1.0},
]

concave_legs = [
    {"option_type": "call", "strike": 100.0, "weight": -1.0},
]

# Stress scenarios
shocks = [-0.4, -0.2, 0.0, 0.2, 0.4]

def eval_portfolio(legs):
    payoff = terminal_portfolio_payoff(spot, legs)
    stressed = [
        terminal_portfolio_payoff(spot_shock(spot, s), legs)
        for s in shocks
    ]
    weights = torch.tensor([leg["weight"] for leg in legs])
    return structural_objective(
        payoff=payoff,
        spot=spot,
        stressed_payoffs=stressed,
        weights=weights,
    )

obj_convex = eval_portfolio(convex_legs)
obj_concave = eval_portfolio(concave_legs)

print("Convex objective:", obj_convex.item())
print("Concave objective:", obj_concave.item())
