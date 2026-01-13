from ib_insync import *

class IBKRContractFactory:
    """
    Factory to create IBKR Option contracts for SPX options."""

    def __init__(self, ib):
        self.ib = ib
        self.spx = Index('SPX', 'CBOE')
        self.ib.qualifyContracts(self.spx)

    def make_contract(self, leg, expiry):
        """
        leg = { option_type, strike, weight }
        expiry = 'YYYYMMDD'
        """
        right = 'C' if leg['option_type'] == 'call' else 'P'

        return Option(
            symbol='SPX',
            lastTradeDateOrContractMonth=expiry,
            strike=round(leg['strike'], 0),
            right=right,
            exchange='CBOE',
            currency='USD'
        )
