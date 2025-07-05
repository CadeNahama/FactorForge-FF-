class RiskEvaluator:
    """
    Base interface for risk evaluation and trade adjustment.
    """
    def __init__(self, max_position_size=0.005):
        self.max_position_size = max_position_size

    def evaluate(self, positions, signals, market_data):
        """
        Assess risk and adjust or block trades as needed.
        Args:
            positions: Current portfolio positions (dict)
            signals: Proposed trading signals (dict)
            market_data: DataFrame with market data
        Returns:
            adjusted_signals: Signals after risk checks/adjustments
        """
        adjusted_signals = {}
        portfolio_value = market_data.get('portfolio_value', 1_000_000)  # fallback
        for ticker, signal in signals.items():
            close_col = f"{ticker}_Close"
            if close_col not in market_data.columns:
                continue
            price = market_data[close_col].iloc[-1]
            if price <= 0 or price is None:
                continue
            # Enforce max position size (as fraction of portfolio)
            max_position_value = self.max_position_size * portfolio_value
            # For now, just pass through the signal (expand logic as needed)
            adjusted_signals[ticker] = signal
        return adjusted_signals

    def evaluate_single(self, positions, signals, market_data):
        """
        Assess risk and adjust or block trades as needed.
        Args:
            positions: Current portfolio positions (dict)
            signals: Proposed trading signals (dict)
            market_data: DataFrame with market data
        Returns:
            adjusted_signals: Signals after risk checks/adjustments
        """
        adjusted_signals = {}
        portfolio_value = market_data.get('portfolio_value', 1_000_000)  # fallback
        for ticker, signal in signals.items():
            close_col = f"{ticker}_Close"
            if close_col not in market_data.columns:
                continue
            price = market_data[close_col].iloc[-1]
            if price <= 0 or price is None:
                continue
            # Enforce max position size (as fraction of portfolio)
            max_position_value = self.max_position_size * portfolio_value
            # For now, just pass through the signal (expand logic as needed)
            adjusted_signals[ticker] = signal
        return adjusted_signals 