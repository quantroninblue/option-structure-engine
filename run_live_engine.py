import torch
from src.surface_extractor import extract_live_spx_surface
from src.regime_encoder import RegimeEncoder, multi_maturity_vol_features
from src.portfolio_generator import PortfolioGenerator, decode_portfolio_tensor, capital_filter
from src.session_logger import LiveSessionLogger
from src.pnl_engine import portfolio_payoff

logger = LiveSessionLogger()
last_structure = None

def main_step():
    global last_structure

    surface = extract_live_spx_surface()
    spot = float(surface["spot"])

    # Build regime features
    k_grid = surface["strikes"][0]
    vols = surface["implied_vol"]
    feats = multi_maturity_vol_features(k_grid, vols).unsqueeze(0)

    enc = RegimeEncoder(input_dim=feats.shape[-1])
    z = enc(feats)

    gen = PortfolioGenerator(latent_dim=z.shape[-1])
    gen.load_state_dict(torch.load("checkpoints/generator.pt"))

    raw = gen(z)[0]
    legs = decode_portfolio_tensor(raw, spot)
    feasible, acct = capital_filter(legs, spot)

    # PnL tracking
    pnl = None
    if last_structure is not None:
        pnl = portfolio_payoff(spot, last_structure)

    if feasible:
        last_structure = legs

    logger.log({
        "spot": spot,
        "legs": legs,
        "margin_used": acct.init_margin_used,
        "buying_power": acct.buying_power,
        "feasible": feasible,
        "pnl": pnl
    })

    print("\nSpot:", spot)
    print("PnL:", pnl)
    print("Buying Power:", round(acct.buying_power, 2))
    for l in legs:
        print(l)
