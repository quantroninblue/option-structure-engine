from ib_insync import *

class IBKRMarginEngine:
    """
    IBKR margin requirement for a list of option legs.
    """

    def __init__(self, ib):
        self.ib = ib

    def estimate_margin(self, contracts, qtys):
        """
        contracts : list of IBKR Option contracts
        qtys      : list of signed quantities (negative = short)

        Returns:
            Estimated initial margin change in USD.
        """
        total_margin = 0.0

        for contract, qty in zip(contracts, qtys):
            action = 'SELL' if qty < 0 else 'BUY'
            order = MarketOrder(action, abs(int(qty)))

            whatif = self.ib.whatIfOrder(contract, order)
            total_margin += whatif.initMarginChange

        return total_margin
