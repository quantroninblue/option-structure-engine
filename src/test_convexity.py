from grids import make_spot_grid
from physics import terminal_portfolio_payoff
from constraints import terminal_convexity_violation, convexity_barrier

spot = make_spot_grid(
    spot=100.0,
    sigma=0.2,
    n_std=3.0,
    n_points=41,
)

# Convex portfolio: long call
convex_legs = [
    {"option_type": "call", "strike": 100.0, "weight": 1.0},
]

# Non-convex portfolio: short call
nonconvex_legs = [
    {"option_type": "call", "strike": 100.0, "weight": -1.0},
]

convex_payoff = terminal_portfolio_payoff(spot, convex_legs)
nonconvex_payoff = terminal_portfolio_payoff(spot, nonconvex_legs)

print("Convex min gamma:",
      terminal_convexity_violation(convex_payoff, spot).item())

print("Non-convex min gamma:",
      terminal_convexity_violation(nonconvex_payoff, spot).item())

print("Convex barrier:",
      convexity_barrier(convex_payoff, spot).item())

print("Non-convex barrier:",
      convexity_barrier(nonconvex_payoff, spot).item())
