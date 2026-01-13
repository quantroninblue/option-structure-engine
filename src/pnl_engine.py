def option_payoff(spot, strike, option_type):
    if option_type == "call":
        return max(spot - strike, 0)
    else:
        return max(strike - spot, 0)


def portfolio_payoff(spot, legs):
    total = 0.0
    for leg in legs:
        px = option_payoff(spot, leg["strike"], leg["option_type"])
        total += px * leg["weight"] * 100   # SPX multiplier
    return total
