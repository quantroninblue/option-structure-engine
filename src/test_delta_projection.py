import torch

from grids import make_spot_grid
from physics import bs_price
from constraints import project_delta_neutral

# Reference spot
S0 = 100.0

# Create a single-point spot tensor for delta evaluation
spot = torch.tensor([S0], requires_grad=True)

# Two-leg portfolio (intentionally NOT delta neutral)
legs = [
    {"option_type": "call", "strike": 90.0, "weight": 1.0},
    {"option_type": "call", "strike": 110.0, "weight": -0.3},
]

vol = 0.2
maturity = 1.0

# Compute individual leg deltas
leg_deltas = []
weights = []

for leg in legs:
    price = bs_price(
        spot=spot,
        strike=leg["strike"],
        vol=vol,
        maturity=maturity,
        option_type=leg["option_type"],
    )

    delta = torch.autograd.grad(
        outputs=price,
        inputs=spot,
        grad_outputs=torch.ones_like(price),
        create_graph=True,
    )[0]

    leg_deltas.append(delta.item())
    weights.append(leg["weight"])

leg_deltas = torch.tensor(leg_deltas)
weights = torch.tensor(weights)

print("Original leg deltas:", leg_deltas)
print("Original weights:", weights)
print("Original portfolio delta:", torch.dot(leg_deltas, weights))

# Project to delta neutrality
new_weights = project_delta_neutral(
    leg_deltas=leg_deltas,
    weights=weights,
)

print("\nProjected weights:", new_weights)
print("Projected portfolio delta:", torch.dot(leg_deltas, new_weights))
