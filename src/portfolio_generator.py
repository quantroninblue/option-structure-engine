import torch
import torch.nn as nn
from src.option_grammar import iron_condor, butterfly
from src.capital_physics import capital_feasible

ACCOUNT_EQUITY = 25_000.0


class PortfolioGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, z):
        return self.net(z)


def decode_portfolio_tensor(params, spot):
    type_logit, center, wing, width, size = params

    # Ultra-tight 25k retail clamps (guarantees feasibility)
    center = torch.clamp(center, -0.01, 0.01)
    wing   = torch.clamp(torch.abs(wing), 0.005, 0.01)
    width  = torch.clamp(torch.abs(width), 0.005, 0.01)
    size   = torch.tensor(1)

    if type_logit > 0:
        legs = iron_condor(spot, center, wing, width, 1)
    else:
        legs = butterfly(spot, center, wing, 1)

    for leg in legs:
        leg["strike"] = round(float(torch.tensor(leg["strike"]).detach()), 2)

    return legs


def capital_filter(legs, spot):
    feasible, used = capital_feasible(legs, spot)

    class Account:
        def __init__(self, used):
            self.init_margin_used = used
            self.buying_power = ACCOUNT_EQUITY - used

    return feasible, Account(used)
