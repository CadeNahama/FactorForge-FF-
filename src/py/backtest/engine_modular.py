"""
Modular Realistic Backtest System
Split from RealisticBacktestSystemV2 for world-class extensibility and maintainability!
Now functional using yfinance and real S&P500 data (SPY by default for demo).
"""

from typing import Optional, Dict, List, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AlphaModel:
    """
    Basic alpha model: generates buy-and-hold signals for all tickers.
    """
    def __init__(self, tickers):
        self.tickers = tickers
        self.current_model = None  # Placeholder for ML model

    def generate_signals(self, data, current_date):
        signals = {}
        debug_first_day = False
        if not hasattr(self, '_debug_signals_printed'):
            debug_first_day = True
            self._debug_signals_printed = True
            print(f"\n[DEBUG] Signal component breakdown for {current_date.strftime('%Y-%m-%d')}")
        for ticker in self.tickers:
            close_col = f"{ticker}_Close"
            if close_col not in data.columns:
                continue
            price_series = data[close_col].dropna()
            if len(price_series) < 50:
                continue
            returns = price_series.pct_change().dropna()
            momentum_signal = self._calculate_momentum_signal(price_series, returns)
            mean_reversion_signal = self._calculate_mean_reversion_signal(price_series)
            ml_signal = 0
            if self.current_model is not None:
                ml_signal = self._calculate_ml_signal(ticker, price_series, data)
            combined_signal = (
                0.4 * momentum_signal +
                0.3 * mean_reversion_signal +
                0.3 * ml_signal
            )
            volatility = returns.std()
            volatility_adjustment = self._calculate_volatility_adjustment(volatility)
            combined_signal *= volatility_adjustment
            scaled_signal = self._scale_signal_strength(combined_signal)
            signal_quality = self._calculate_signal_quality(scaled_signal, data)
            if debug_first_day:
                print(f"  {ticker}: momentum={momentum_signal:.6f}, meanrev={mean_reversion_signal:.6f}, ml={ml_signal:.6f}, combined={combined_signal:.6f}, scaled={scaled_signal:.6f}, volatility={volatility:.6f}, quality={signal_quality:.6f}")
            signals[ticker] = scaled_signal
        return signals

    def _calculate_momentum_signal(self, price_series, returns):
        short_momentum = price_series.pct_change(5).iloc[-1]
        medium_momentum = price_series.pct_change(20).iloc[-1]
        long_momentum = price_series.pct_change(60).iloc[-1]
        rsi = self._calculate_rsi_series(price_series, 14).iloc[-1]
        rsi_signal = (rsi - 50) / 50
        macd_signal = self._calculate_macd_features(price_series)['macd_signal']
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

    def _calculate_ml_signal(self, ticker, price_series, historical_data):
        try:
            if self.current_model is None:
                return 0
            features = self._create_ticker_features(price_series)
            prediction = self.current_model.predict([features])[0]
            ml_signal = (prediction - 0.5) * 2
            return ml_signal
        except Exception as e:
            print(f"ML signal calculation failed for {ticker}: {e}")
            return 0

    def _create_ticker_features(self, price_series):
        returns = price_series.pct_change().dropna()
        features = [
            returns.mean(),
            returns.std(),
            returns.skew(),
            returns.kurtosis(),
            price_series.pct_change(5).iloc[-1],
            price_series.pct_change(20).iloc[-1],
            self._calculate_rsi_series(price_series, 14).iloc[-1],
            self._calculate_bollinger_features(price_series)['bb_position'],
            self._calculate_macd_features(price_series)['macd'],
            self._calculate_trend_strength(price_series)
        ]
        return features

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

    def _calculate_macd_features(self, prices):
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'macd_signal': 1 if macd.iloc[-1] > signal.iloc[-1] else -1
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

    def _scale_signal_strength(self, signal):
        scaled_signal = np.tanh(signal * 2)
        return scaled_signal

    def _calculate_signal_quality(self, signal, historical_data):
        signal_strength = abs(signal)
        base_quality = 0.7
        if signal_strength > 0.8:
            quality = base_quality * 1.2
        elif signal_strength > 0.5:
            quality = base_quality * 1.0
        else:
            quality = base_quality * 0.8
        return min(quality, 1.0)

class ExecutionSimulator:
    """
    Executes buy-and-hold strategy: buys all assets at first bar, closes at the end.
    """
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def _get_price_col(self, price_df, ticker, price_type):
        # Helper to find the correct column in MultiIndex or single-level DataFrame
        for col in price_df.columns:
            if isinstance(col, tuple):
                if ticker in col and col[-1].lower() == price_type.lower():
                    return col
            else:
                if col.lower() == price_type.lower():
                    return col
        raise KeyError(f"Could not find column for {ticker} {price_type} in DataFrame columns: {price_df.columns}")

    def execute(self, price_df: pd.DataFrame, signals: dict) -> pd.DataFrame:
        # Robustly handle both single-level and MultiIndex DataFrames
        for ticker in signals:
            try:
                open_col = self._get_price_col(price_df, ticker, "Open")
                open_first = price_df[open_col].iloc[0]
            except Exception as e:
                print(f"❌ Could not find 'Open' price for {ticker}. Columns: {price_df.columns}")
                raise e
        # Start with initial capital distributed equally
        tickers = [ticker for ticker, signal in signals.items() if signal > 0]
        allocations = {ticker: self.initial_capital / len(tickers) for ticker in tickers} if tickers else {}
        portfolio_history = []
        for dt in price_df.index:
            total_value = 0
            for ticker in tickers:
                try:
                    open_col = self._get_price_col(price_df, ticker, "Open")
                    close_col = self._get_price_col(price_df, ticker, "Close")
                    open_first = price_df[open_col].iloc[0]
                    close_now = price_df[close_col].loc[dt] if hasattr(price_df[close_col], 'loc') else price_df[close_col][dt]
                    qty = allocations[ticker] / open_first if open_first > 0 else 0
                    value = qty * close_now
                    total_value += value
                except Exception as e:
                    print(f"❌ Could not compute value for {ticker} on {dt}: {e}")
            portfolio_history.append({
                "date": dt,
                "portfolio_value": total_value
            })
        return pd.DataFrame(portfolio_history).set_index("date")

class RiskEvaluator:
    """
    No-op for now; can implement max drawdown etc. later.
    """
    def __init__(self, **risk_params):
        pass

    def enforce(self, signals, positions, portfolio_value, daily_pnl):
        return signals

class ReportGenerator:
    """
    Plots and summarizes portfolio performance.
    """
    def __init__(self):
        pass

    def generate(self, results: pd.DataFrame, mc_results=None):
        print("Final portfolio value: $%.2f" % results['portfolio_value'].iloc[-1])
        print("Return: %.2f%%" % ((results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0] - 1) * 100))
        plt.figure(figsize=(10, 6))
        plt.plot(results.index, results['portfolio_value'], label='Portfolio Value')
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.show()

    def save(self, results):
        results.to_csv("backtest_results.csv")
        print("Saved results to backtest_results.csv")

class BacktestEngine:
    """
    Modular orchestrator for the full pipeline. Connects AlphaModel, ExecutionSimulator, RiskEvaluator, ReportGenerator.
    """
    def __init__(self, initial_capital: float, start_date: str, end_date: str, tickers: Optional[list]):
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        if tickers is None or len(tickers) == 0:
            self.tickers = ['SPY']
        else:
            self.tickers = tickers
        self.alpha_model = AlphaModel(self.tickers)
        self.execution_simulator = ExecutionSimulator(initial_capital)
        self.risk_evaluator = RiskEvaluator()
        self.report_generator = ReportGenerator()

    def run(self):
        # 1. Download data
        print(f"Downloading data for: {self.tickers}")
        data = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            group_by='ticker',
            auto_adjust=True,
            progress=True
        )
        # Ensure data is a valid DataFrame before concat and dropna
        if data is not None:
            data = pd.concat({self.tickers[0]: data}, axis=1)
            data = data.dropna()
        else:
            raise ValueError("No data to process after download.")
        # 2. Generate signals (buy-and-hold)
        signals = self.alpha_model.generate_signals(data, None)
        # 3. Run execution simulation:
        results = self.execution_simulator.execute(data, signals)
        # 4. Reporting
        self.report_generator.generate(results)
        self.report_generator.save(results)

if __name__ == "__main__":
    # SAMPLE USAGE: feel free to change tickers, dates, or capital
    engine = BacktestEngine(
        initial_capital=100000,
        start_date='2020-01-01',
        end_date='2024-01-01',
        tickers=['SPY']  # For demo; pass None or list of tickers for S&P 500 or subset
    )
    engine.run()