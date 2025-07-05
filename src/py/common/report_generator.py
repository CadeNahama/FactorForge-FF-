class ReportGenerator:
    """
    Base interface for trade logging and performance reporting.
    """
    def log_trade(self, trade):
        """
        Log a single trade event.
        Args:
            trade: Trade information (dict or object).
        """
        raise NotImplementedError

    def generate_summary(self):
        """
        Produce a summary report of performance.
        Returns:
            summary: Performance summary (e.g., dict, DataFrame, or string).
        """
        raise NotImplementedError

class BasicReportGenerator(ReportGenerator):
    """
    Basic trade logger and performance reporter.
    """
    def __init__(self):
        self.trades = []

    def log_trade(self, trade):
        """
        Log a single trade event.
        Args:
            trade: Trade information (dict or object).
        """
        self.trades.append(trade)

    def generate_summary(self):
        """
        Produce a summary report of performance.
        Returns:
            summary: Performance summary (string)
        """
        n_trades = len(self.trades)
        if n_trades == 0:
            return 'No trades executed.'
        buys = sum(1 for t in self.trades if t['action'] == 'buy')
        sells = sum(1 for t in self.trades if t['action'] == 'sell')
        summary = f"Total trades: {n_trades}\nBuys: {buys}\nSells: {sells}\n"
        tickers = set(t['ticker'] for t in self.trades)
        summary += f"Tickers traded: {', '.join(tickers)}\n"
        return summary 