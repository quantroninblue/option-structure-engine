"""
IBKR Reg-T margin physics for SPX 0-DTE
Supports verticals, butterflies, iron condors.
"""

ACCOUNT_EQUITY = 25_000.0
CONTRACT_MULT = 100


def net_side(shorts, longs, is_put):
    """
    Net one option side (puts or calls) into max-loss margin.
    shorts / longs = list of (strike, qty)
    """
    shorts = sorted(shorts, key=lambda x: x[0])
    longs = sorted(longs, key=lambda x: x[0])

    margin = 0.0

    for s_strike, s_qty in shorts:
        qty_left = s_qty

        # Find valid caps
        for i, (l_strike, l_qty) in enumerate(longs):
            if qty_left == 0:
                break

            valid = (l_strike < s_strike) if is_put else (l_strike > s_strike)
            if not valid:
                continue

            paired = min(qty_left, l_qty)
            wing = abs(s_strike - l_strike)

            margin += paired * wing * CONTRACT_MULT

            qty_left -= paired
            longs[i] = (l_strike, l_qty - paired)

        # Any leftover short = naked worst-case
        if qty_left > 0:
            # SPX worst-case loss approx full strike distance
            margin += qty_left * s_strike * CONTRACT_MULT

    return margin


def capital_feasible(legs, spot):
    put_shorts, put_longs = [], []
    call_shorts, call_longs = [], []

    for leg in legs:
        qty = leg["weight"]
        strike = leg["strike"]
        t = leg["option_type"]

        if t == "put":
            if qty < 0:
                put_shorts.append((strike, abs(qty)))
            else:
                put_longs.append((strike, qty))
        else:
            if qty < 0:
                call_shorts.append((strike, abs(qty)))
            else:
                call_longs.append((strike, qty))

    margin = 0.0
    margin += net_side(put_shorts, put_longs, is_put=True)
    margin += net_side(call_shorts, call_longs, is_put=False)

    return margin <= ACCOUNT_EQUITY, margin
