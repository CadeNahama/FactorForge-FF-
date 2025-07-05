import pandas as pd

class ExecutionHandler:
    """
    Base interface for order execution (simulated or live).
    """
    def execute(self, orders):
        """
        Simulate or send orders to broker.
        Args:
            orders: List or dict of orders to execute.
        Returns:
            execution_results: Results of order execution (e.g., fills, status).
        """
        raise NotImplementedError

class BasicSimulatedExecutionHandler:
    """
    Basic simulated order execution handler.
    """
    def execute(self, orders):
        """
        Simulate order execution.
        Args:
            orders: Dict of {ticker: signal_strength}
        Returns:
            execution_results: List of trade dicts (mock fills)
        """
        execution_results = []
        for ticker, signal in orders.items():
            # Skip NaN or zero signals
            if signal == 0 or pd.isna(signal):
                continue
            trade = {
                'ticker': ticker,
                'action': 'buy' if signal > 0 else 'sell',
                'quantity': abs(int(signal * 100)),  # mock quantity
                'price': 100.0,  # mock price
                'pnl': 0.0,      # placeholder
            }
            execution_results.append(trade)
        return execution_results 