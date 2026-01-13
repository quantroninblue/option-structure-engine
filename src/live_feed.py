from ib_insync import *
import pandas as pd

class IBKRLiveFeed:
    """
    Pulls live SPX spot + option chain + IV surface from IBKR.
    """

    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=19)

        self.spx = Index('SPX', 'CBOE')
        self.ib.qualifyContracts(self.spx)

    def get_spx_spot(self):
        ticker = self.ib.reqMktData(self.spx, '', False, False)
        self.ib.sleep(1)
        return ticker.marketPrice()

    def get_option_chain(self):
        chains = self.ib.reqSecDefOptParams(
            self.spx.symbol, '', self.spx.secType, self.spx.conId
        )
        return chains[0]

    def get_iv_surface(self, expiry):
        chain = self.get_option_chain()

        rows = []
        for strike in chain.strikes:
            for right in ['C', 'P']:
                opt = Option('SPX', expiry, strike, right, 'CBOE')
                self.ib.qualifyContracts(opt)
                ticker = self.ib.reqMktData(opt, '', False, False)
                self.ib.sleep(0.2)

                if ticker.modelGreeks:
                    rows.append({
                        "strike": strike,
                        "right": right,
                        "iv": ticker.modelGreeks.impliedVol
                    })

        return pd.DataFrame(rows)
