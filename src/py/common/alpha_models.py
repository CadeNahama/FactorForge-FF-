import numpy as np
import pandas as pd

class AlphaModel:
    """
    Base interface for alpha signal generation models.
    """
    def fit(self, data):
        """
        Train or calibrate the model on historical data.
        Args:
            data: Historical market data for training/calibration.
        """
        raise NotImplementedError

    def predict(self, data):
        """
        Generate trading signals from input data.
        Args:
            data: Market data for signal generation.
        Returns:
            signals: Trading signals (e.g., DataFrame or dict).
        """
        raise NotImplementedError

class AlphaModelRealistic(AlphaModel):
    """
    Alpha signal generation model with multi-factor logic: momentum, mean reversion, correlation, ML.
    """
    def __init__(self):
        self.current_model = None  # Placeholder for ML model
        self.tickers = None
        self.corr_window = 20  # Rolling window for correlation
        self.corr_index = 'SPY'  # Use SPY as the market index

    def fit(self, data):
        """
        Placeholder for ML model training. For now, does nothing.
        """
        # TODO: Implement ML model training if needed
        self.tickers = [col.split('_')[0] for col in data.columns if col.endswith('_Close') and col.split('_')[0] != self.corr_index]
        self.current_model = None

    def predict(self, data):
        """
        Generate trading signals for each ticker and date.
        Args:
            data: Market data DataFrame (indexed by date, columns per ticker)
        Returns:
            signals: Dict of {ticker: signal_strength}
        """
        signals = {}
        if self.tickers is None:
            self.tickers = [col.split('_')[0] for col in data.columns if col.endswith('_Close') and col.split('_')[0] != self.corr_index]
        for ticker in self.tickers:
            close_col = f"{ticker}_Close"
            if close_col not in data.columns:
                continue
            price_series = data[close_col].dropna()
            if len(price_series) < 50:
                continue
            returns = price_series.pct_change().dropna()
            # Multi-factor signals
            momentum_signal = self._calculate_momentum_signal(price_series)
            mean_reversion_signal = self._calculate_mean_reversion_signal(price_series)
            correlation_signal = self._calculate_correlation_signal(ticker, data)
            ml_signal = self._calculate_ml_signal(ticker, price_series, data)
            volatility = returns.std()
            volatility_adjustment = self._calculate_volatility_adjustment(volatility)
            # Weighted sum of all signals
            combined_signal = (
                0.35 * momentum_signal +
                0.35 * mean_reversion_signal +
                0.2 * correlation_signal +
                0.1 * ml_signal
            )
            combined_signal *= volatility_adjustment
            scaled_signal = np.tanh(combined_signal * 2) * 0.5  # Sigmoid scaling, reduce strength
            if abs(scaled_signal) < 0.03:
                scaled_signal = 0
            signals[ticker] = scaled_signal
        return signals

    def _calculate_momentum_signal(self, price_series):
        short_momentum = price_series.pct_change(5).iloc[-1]
        medium_momentum = price_series.pct_change(20).iloc[-1]
        long_momentum = price_series.pct_change(60).iloc[-1]
        rsi = self._calculate_rsi_series(price_series, 14).iloc[-1]
        rsi_signal = (rsi - 50) / 50
        momentum_signal = (
            0.3 * short_momentum +
            0.4 * medium_momentum +
            0.2 * long_momentum +
            0.1 * rsi_signal
        )
        return momentum_signal

    def _calculate_mean_reversion_signal(self, price_series):
        bb_features = self._calculate_bollinger_features(price_series)
        bb_signal = bb_features['bb_signal']
        z_score = self._calculate_z_score(price_series, 20)
        trend_strength = self._calculate_trend_strength(price_series)
        mean_reversion_signal = (
            0.5 * bb_signal +
            0.3 * z_score +
            0.2 * (1 - trend_strength)
        )
        return mean_reversion_signal

    def _calculate_correlation_signal(self, ticker, data):
        # Rolling correlation with SPY
        close_col = f"{ticker}_Close"
        spy_col = f"{self.corr_index}_Close"
        if spy_col not in data.columns or close_col not in data.columns:
            return 0
        ticker_returns = data[close_col].pct_change().dropna()
        spy_returns = data[spy_col].pct_change().dropna()
        # Align dates
        common_idx = ticker_returns.index.intersection(spy_returns.index)
        if len(common_idx) < self.corr_window:
            return 0
        ticker_series = ticker_returns.loc[common_idx]
        spy_series = spy_returns.loc[common_idx]
        rolling_corr = ticker_series.rolling(self.corr_window).corr(spy_series)
        if len(rolling_corr) == 0 or np.isnan(rolling_corr.iloc[-1]):
            return 0
        corr = rolling_corr.iloc[-1]
        # Low correlation = positive signal (diversification), high correlation = negative
        if corr < 0.3:
            return 0.3
        elif corr > 0.7:
            return -0.2
        else:
            return 0

    def _calculate_ml_signal(self, ticker, price_series, data):
        # Placeholder for ML-based signal
        return 0

    def _calculate_rsi_series(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_features(self, prices):
        sma = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        current_price = prices.iloc[-1]
        bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        if current_price > upper_band.iloc[-1]:
            bb_signal = -0.5
        elif current_price < lower_band.iloc[-1]:
            bb_signal = 0.5
        else:
            bb_signal = 0
        return {
            'bb_position': bb_position,
            'bb_signal': bb_signal,
            'upper_band': upper_band.iloc[-1],
            'lower_band': lower_band.iloc[-1]
        }

    def _calculate_z_score(self, prices, period):
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        z_score = (prices.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        return z_score

    def _calculate_trend_strength(self, prices):
        def rolling_slope(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]
        trend_strength = prices.rolling(window=20).apply(rolling_slope).iloc[-1]
        return abs(trend_strength)

    def _calculate_volatility_adjustment(self, volatility):
        if volatility > 0.05:
            return 0.5
        elif volatility > 0.03:
            return 0.8
        else:
            return 1.0 