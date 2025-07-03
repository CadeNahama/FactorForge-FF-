#!/usr/bin/env python3
"""
Realistic Institutional-Grade Backtesting System v2
Python orchestration with C++ execution for HFT optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
from scipy import stats
import random

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced components
try:
    import quant_cpp  # C++ extension
    from ensemble_ml_system import EnsembleMLSystem
    from enhanced_data import EnhancedDataLoader, get_sp500_tickers
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

warnings.filterwarnings('ignore')

class RealisticBacktestSystemV2:
    """
    Realistic backtesting system v2 with Python orchestration and C++ execution
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,  # $1M starting capital
                 start_date: str = "2020-01-01",
                 end_date: str = "2024-12-31",
                 tickers: Optional[list] = None,
                 
                 # Transaction costs
                 commission_rate: float = 0.001,  # 0.1% commission
                 slippage_rate: float = 0.0005,   # 0.05% slippage
                 market_impact_rate: float = 0.0001,  # 0.01% per $100K traded
                 
                 # Risk controls
                 max_position_size: float = 0.05,  # 5% max position (more aggressive)
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
                 stop_loss_rate: float = 0.08,      # 8% stop loss
                 max_drawdown_limit: float = 0.08,  # 8% max drawdown
                 trailing_stop_rate: float = 0.05,  # 5% trailing stop
                 
                 # Validation settings
                 train_period_years: float = 2.0,  # 2 years for initial training
                 retrain_frequency_days: int = 63,  # Retrain every quarter
                 out_of_sample_start: str = "2022-01-01",
                 use_rolling_validation: bool = True, # rolling validation flag
                 use_advanced_signals: bool = False): # advanced signal flag
        
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        if tickers is not None:
            self.tickers = tickers
        else:
            self.tickers = get_sp500_tickers()
        
        # Transaction costs
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.market_impact_rate = market_impact_rate
        
        # Risk controls
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_rate = stop_loss_rate
        self.max_drawdown_limit = max_drawdown_limit
        self.trailing_stop_rate = trailing_stop_rate
        
        # Validation settings
        self.train_period_years = train_period_years
        self.retrain_frequency_days = retrain_frequency_days
        self.out_of_sample_start = out_of_sample_start
        self.use_rolling_validation = use_rolling_validation
        self.use_advanced_signals = use_advanced_signals
        
        # Initialize components
        self.data_loader = EnhancedDataLoader()
        self.execution_engine = quant_cpp.ExecutionEngine()  # C++ execution engine
        self.ml_system = EnsembleMLSystem()
        
        # Create risk parameters for C++ engine
        self.risk_params = quant_cpp.RiskParams()
        self.risk_params.max_portfolio_risk = max_portfolio_risk
        self.risk_params.max_position_size = max_position_size
        self.risk_params.commission_rate = commission_rate
        self.risk_params.slippage_rate = slippage_rate
        self.risk_params.market_impact_rate = market_impact_rate
        self.risk_params.initial_capital = initial_capital
        
        # Backtest state
        self.portfolio_value = initial_capital
        self.positions = {}
        self.cash = initial_capital
        self.portfolio_history = []
        self.trade_history = []
        self.risk_metrics_history = []
        
        # Performance tracking
        self.daily_returns = []
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        self.total_return = 0
        self.total_transaction_costs = 0
        
        # Model state
        self.current_model = None
        self.last_retrain_date = None
        
        # Stop loss and trailing stop tracking
        self.stop_loss_prices = {}
        self.trailing_stop_prices = {}
        self.peak_prices = {}
        
    def run_realistic_backtest(self):
        """Run realistic backtest with proper validation"""
        
        print("🏛️  REALISTIC INSTITUTIONAL BACKTEST v2")
        print("=" * 60)
        print("🚀 Python Orchestration + C++ Execution")
        print("=" * 60)
        
        # Step 1: Load data
        print("📊 Loading data...")
        data = self.data_loader.download_data(self.tickers, self.start_date, self.end_date)
        
        if data is None or data.empty:
            print("❌ Failed to load data")
            return None
        
        print(f"✓ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Step 2: Split data for proper validation
        train_data, test_data = self._split_data_for_validation(data)
        
        print(f"📈 Training period: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"🧪 Testing period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
        
        # Step 3: Run walk-forward backtest
        print("\n🔄 Running walk-forward backtest...")
        results = self._run_walk_forward_backtest(train_data, test_data)
        
        # Step 3b: Rolling window validation (if enabled)
        if self.use_rolling_validation:
            print("\n🔁 Running rolling window validation...")
            self._run_rolling_validation(data)
        
        # Step 4: Run Monte Carlo simulations
        print("\n🎲 Running Monte Carlo simulations...")
        mc_results = self._run_monte_carlo_simulations(results) if results else {}
        
        # Step 5: Generate comprehensive report
        if results:
            self._generate_realistic_report(results, mc_results)
        
        return results, mc_results
    
    def _split_data_for_validation(self, data):
        """Split data into training and out-of-sample testing periods"""
        
        # Convert out_of_sample_start to datetime
        oos_start = pd.to_datetime(self.out_of_sample_start)
        
        # Split data
        train_data = data[data.index < oos_start]
        test_data = data[data.index >= oos_start]
        
        return train_data, test_data
    
    def _run_walk_forward_backtest(self, train_data, test_data):
        """Run walk-forward backtest with proper data handling"""
        
        # Initialize tracking
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.total_transaction_costs = 0
        
        # Combine data for walk-forward
        all_data = pd.concat([train_data, test_data])
        
        # Calculate initial training period
        initial_train_end = train_data.index[0] + pd.DateOffset(years=self.train_period_years)
        
        # Walk-forward loop
        current_date = initial_train_end
        last_retrain = current_date - pd.DateOffset(days=self.retrain_frequency_days)
        
        while current_date <= all_data.index[-1]:
            # Check if we need to retrain
            if (current_date - last_retrain).days >= self.retrain_frequency_days:
                print(f"🔄 Retraining models on {current_date.strftime('%Y-%m-%d')}...")
                
                # Get training data (only data available up to current date)
                train_end = current_date - pd.DateOffset(days=1)
                train_start = train_end - pd.DateOffset(years=self.train_period_years)
                
                if train_start in all_data.index and train_end in all_data.index:
                    current_train_data = all_data.loc[train_start:train_end]
                    
                    # Train models
                    self._train_models(current_train_data)
                    last_retrain = current_date
            
            # Get current market data
            if current_date in all_data.index:
                current_prices = self._get_current_prices(all_data, current_date)
                
                # Generate signals using Python
                signals = self._generate_signals_realistic(all_data, current_date)
                
                # DEBUG: Print signals for the first 10 backtest days
                if len(self.portfolio_history) < 10:
                    print(f"\n[DEBUG] {current_date.strftime('%Y-%m-%d')} - Signals:")
                    if signals:
                        for t, s in signals.items():
                            print(f"  {t}: {s:.6f}")
                    else:
                        print("  No signals generated!")
                
                # Convert signals to C++ format
                cpp_signals = self._convert_signals_to_cpp(signals)
                
                # DEBUG: Print C++ signal quantities for the first 10 backtest days
                if len(self.portfolio_history) < 10:
                    print(f"[DEBUG] {current_date.strftime('%Y-%m-%d')} - C++ Signals (qty):")
                    if cpp_signals:
                        for sig in cpp_signals:
                            print(f"  {sig.ticker}: {sig.side} qty={sig.qty:.6f}")
                    else:
                        print("  No C++ signals generated!")
                
                # Create portfolio state for C++
                portfolio_state = quant_cpp.PortfolioState()
                portfolio_state.cash = self.cash
                portfolio_state.positions = self.positions
                portfolio_state.total_transaction_costs = self.total_transaction_costs
                
                # Execute trades using C++
                fills = self.execution_engine.executeSignals(cpp_signals, current_prices, portfolio_state, self.risk_params)
                
                # DEBUG: Print fills for the first 10 backtest days
                if len(self.portfolio_history) < 10:
                    print(f"[DEBUG] {current_date.strftime('%Y-%m-%d')} - Fills:")
                    if fills:
                        for fill in fills:
                            print(f"  {fill.ticker}: {fill.side} {fill.qty} @ {fill.price} (commission: {fill.commission}, slippage: {fill.slippage})")
                    else:
                        print("  No fills returned by C++ engine.")
                
                # Update portfolio state from C++
                self.cash = portfolio_state.cash
                self.positions = portfolio_state.positions
                self.total_transaction_costs = portfolio_state.total_transaction_costs
                
                # Calculate portfolio value
                self.portfolio_value = self.cash + sum(quantity * current_prices.get(ticker, 0) 
                                                     for ticker, quantity in self.positions.items())
                
                # Check stop losses and trailing stops (Python logic)
                self._check_stop_losses(current_prices, current_date)
                self._check_trailing_stops(current_prices, current_date)
                
                # Check drawdown limit
                if self._check_drawdown_limit():
                    print(f"⚠️  Drawdown limit reached on {current_date.strftime('%Y-%m-%d')}")
                    break
                
                # Record portfolio state
                self.portfolio_history.append({
                    'date': current_date,
                    'portfolio_value': self.portfolio_value,
                    'cash': self.cash,
                    'positions': self.positions.copy()
                })
                
                                        # Record trade results
                if fills and fills is not None:
                    for fill in fills:
                        self.trade_history.append({
                            'date': current_date,
                            'ticker': fill.ticker,
                            'action': fill.side,
                            'quantity': fill.qty,
                            'price': fill.price,
                            'pnl': 0,  # Will be calculated later
                            'commission': fill.commission,
                            'slippage': fill.slippage,
                            'market_impact': fill.market_impact
                        })
            
            # Move to next date
            current_date += pd.DateOffset(days=1)
        
        # Close all positions at the end
        if all_data.index[-1] in all_data.index:
            final_prices = self._get_current_prices(all_data, all_data.index[-1])
            self._close_all_positions(all_data.index[-1], final_prices)
        
        # Calculate final performance
        self._calculate_performance_realistic()
        
        return {
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history,
            'final_portfolio_value': self.portfolio_value,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown
        }
    
    def _train_models(self, train_data):
        """Train ML models for signal generation"""
        try:
            # Train ensemble model - use a simple model for now
            # In a real system, you'd implement proper ML training
            self.current_model = None  # Placeholder for ML model
            print("✓ Models trained successfully (placeholder)")
        except Exception as e:
            print(f"⚠️  Model training failed: {e}")
            self.current_model = None
    
    def _get_current_prices(self, data, date):
        """Get current prices for all tickers"""
        current_prices = {}
        for ticker in self.tickers:
            if ticker in data.columns and date in data.index:
                current_prices[ticker] = data.loc[date, ticker]
        return current_prices
    
    def _convert_signals_to_cpp(self, signals):
        """Convert Python signals to C++ Signal format"""
        cpp_signals = []
        for ticker, signal_strength in signals.items():
            signal = quant_cpp.Signal()
            signal.ticker = ticker
            signal.side = "BUY" if signal_strength > 0 else "SELL"
            signal.qty = abs(signal_strength) * self.max_position_size  # Scale by max position size
            signal.limit_price = 0  # Market order
            cpp_signals.append(signal)
        return cpp_signals
    
    def _generate_signals_realistic(self, data, current_date):
        """Generate trading signals using Python logic"""
        signals = {}
        debug_first_day = False
        # Only print for the very first call in the backtest
        if hasattr(self, '_debug_signals_printed'):
            debug_first_day = False
        else:
            debug_first_day = True
            self._debug_signals_printed = True
            print(f"\n[DEBUG] Signal component breakdown for {current_date.strftime('%Y-%m-%d')}")
        
        for ticker in self.tickers:
            close_col = f"{ticker}_Close"
            if close_col not in data.columns:
                continue
            
            price_series = data[close_col].dropna()
            if len(price_series) < 50:  # Need sufficient data
                continue
            
            # Calculate returns
            returns = price_series.pct_change().dropna()
            
            # Generate multiple signal types
            momentum_signal = self._calculate_momentum_signal(price_series, returns)
            mean_reversion_signal = self._calculate_mean_reversion_signal(price_series)
            
            # ML signal if model is available
            ml_signal = 0
            if self.current_model is not None:
                ml_signal = self._calculate_ml_signal(ticker, price_series, data)
            
            # Combine signals with weights
            combined_signal = (
                0.4 * momentum_signal +
                0.3 * mean_reversion_signal +
                0.3 * ml_signal
            )
            
            # Apply volatility adjustment
            volatility = returns.std()
            volatility_adjustment = self._calculate_volatility_adjustment(volatility)
            combined_signal *= volatility_adjustment
            
            # Scale signal strength
            scaled_signal = self._scale_signal_strength(combined_signal)
            
            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(scaled_signal, data)
            
            # DEBUG: Print all signal components for the first day
            if debug_first_day:
                print(f"  {ticker}: momentum={momentum_signal:.6f}, meanrev={mean_reversion_signal:.6f}, ml={ml_signal:.6f}, combined={combined_signal:.6f}, scaled={scaled_signal:.6f}, volatility={volatility:.6f}, quality={signal_quality:.6f}")
            
            # For debugging, include all signals
            signals[ticker] = scaled_signal
        
        return signals
    
    def _calculate_momentum_signal(self, price_series, returns):
        """Calculate momentum-based signal"""
        # Multiple timeframe momentum
        short_momentum = price_series.pct_change(5).iloc[-1]  # 5-day
        medium_momentum = price_series.pct_change(20).iloc[-1]  # 20-day
        long_momentum = price_series.pct_change(60).iloc[-1]  # 60-day
        
        # RSI
        rsi = self._calculate_rsi_series(price_series, 14).iloc[-1]
        rsi_signal = (rsi - 50) / 50  # Normalize to [-1, 1]
        
        # MACD
        macd_signal = self._calculate_macd_features(price_series)['macd_signal']
        
        # Combine momentum signals
        momentum_signal = (
            0.3 * short_momentum +
            0.4 * medium_momentum +
            0.2 * long_momentum +
            0.1 * rsi_signal
        )
        
        return momentum_signal
    
    def _calculate_mean_reversion_signal(self, price_series):
        """Calculate mean reversion signal"""
        # Bollinger Bands
        bb_features = self._calculate_bollinger_features(price_series)
        bb_signal = bb_features['bb_signal']
        
        # Z-score
        z_score = self._calculate_z_score(price_series, 20)
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(price_series)
        
        # Mean reversion signal
        mean_reversion_signal = (
            0.5 * bb_signal +
            0.3 * z_score +
            0.2 * (1 - trend_strength)  # Less trend = more mean reversion
        )
        
        return mean_reversion_signal
    
    def _calculate_ml_signal(self, ticker, price_series, historical_data):
        """Calculate ML-based signal"""
        try:
            if self.current_model is None:
                return 0
            
            # Create features for ML model
            features = self._create_ticker_features(price_series)
            
            # Make prediction
            prediction = self.current_model.predict([features])[0]
            
            # Convert to signal
            ml_signal = (prediction - 0.5) * 2  # Convert [0,1] to [-1,1]
            
            return ml_signal
            
        except Exception as e:
            print(f"ML signal calculation failed for {ticker}: {e}")
            return 0
    
    def _create_ticker_features(self, price_series):
        """Create feature vector for ML model"""
        returns = price_series.pct_change().dropna()
        
        features = [
            returns.mean(),  # Mean return
            returns.std(),   # Volatility
            returns.skew(),  # Skewness
            returns.kurtosis(),  # Kurtosis
            price_series.pct_change(5).iloc[-1],  # 5-day return
            price_series.pct_change(20).iloc[-1],  # 20-day return
            self._calculate_rsi_series(price_series, 14).iloc[-1],  # RSI
            self._calculate_bollinger_features(price_series)['bb_position'],  # BB position
            self._calculate_macd_features(price_series)['macd'],  # MACD
            self._calculate_trend_strength(price_series)  # Trend strength
        ]
        
        return features
    
    def _calculate_rsi_series(self, prices, period=14):
        """Calculate RSI series"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_features(self, prices):
        """Calculate Bollinger Bands features"""
        sma = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        current_price = prices.iloc[-1]
        bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        
        # Signal based on position
        if current_price > upper_band.iloc[-1]:
            bb_signal = -0.5  # Sell signal
        elif current_price < lower_band.iloc[-1]:
            bb_signal = 0.5   # Buy signal
        else:
            bb_signal = 0
        
        return {
            'bb_position': bb_position,
            'bb_signal': bb_signal,
            'upper_band': upper_band.iloc[-1],
            'lower_band': lower_band.iloc[-1]
        }
    
    def _calculate_macd_features(self, prices):
        """Calculate MACD features"""
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
        """Calculate Z-score for mean reversion"""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        z_score = (prices.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        return z_score
    
    def _calculate_trend_strength(self, prices):
        """Calculate trend strength"""
        def rolling_slope(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]
        
        # Calculate rolling slope
        trend_strength = prices.rolling(window=20).apply(rolling_slope).iloc[-1]
        return abs(trend_strength)
    
    def _calculate_volatility_adjustment(self, volatility):
        """Adjust signal based on volatility"""
        # Higher volatility = lower signal strength
        if volatility > 0.05:  # High volatility
            return 0.5
        elif volatility > 0.03:  # Medium volatility
            return 0.8
        else:  # Low volatility
            return 1.0
    
    def _scale_signal_strength(self, signal):
        """Scale signal strength to reasonable range"""
        # Apply sigmoid scaling
        scaled_signal = np.tanh(signal * 2)  # Scale and bound to [-1, 1]
        return scaled_signal
    
    def _calculate_signal_quality(self, signal, historical_data):
        """Calculate signal quality metric"""
        # Simple quality metric based on signal strength and consistency
        signal_strength = abs(signal)
        
        # Historical accuracy (simplified)
        # In a real system, you'd track actual signal accuracy
        base_quality = 0.7  # Assume 70% base accuracy
        
        # Adjust based on signal strength
        if signal_strength > 0.8:
            quality = base_quality * 1.2
        elif signal_strength > 0.5:
            quality = base_quality * 1.0
        else:
            quality = base_quality * 0.8
        
        return min(quality, 1.0)
    
    def _check_stop_losses(self, current_prices, date):
        """Check and execute stop losses"""
        for ticker, position in self.positions.items():
            if ticker not in current_prices:
                continue
                
            current_price = current_prices[ticker]
            entry_price = position.get('entry_price', 0)
            
            if entry_price > 0:
                # Calculate stop loss price
                if position['quantity'] > 0:  # Long position
                    stop_price = entry_price * (1 - self.stop_loss_rate)
                    if current_price <= stop_price:
                        # Execute stop loss
                        self._execute_stop_loss(ticker, current_price, date, "stop_loss")
                
                elif position['quantity'] < 0:  # Short position
                    stop_price = entry_price * (1 + self.stop_loss_rate)
                    if current_price >= stop_price:
                        # Execute stop loss
                        self._execute_stop_loss(ticker, current_price, date, "stop_loss")
    
    def _check_trailing_stops(self, current_prices, date):
        """Check and execute trailing stops"""
        for ticker, position in self.positions.items():
            if ticker not in current_prices:
                continue
                
            current_price = current_prices[ticker]
            entry_price = position.get('entry_price', 0)
            
            if entry_price > 0:
                # Update peak price for trailing stop
                if ticker not in self.peak_prices:
                    self.peak_prices[ticker] = entry_price
                else:
                    if position['quantity'] > 0:  # Long position
                        self.peak_prices[ticker] = max(self.peak_prices[ticker], current_price)
                    else:  # Short position
                        self.peak_prices[ticker] = min(self.peak_prices[ticker], current_price)
                
                # Check trailing stop
                peak_price = self.peak_prices[ticker]
                
                if position['quantity'] > 0:  # Long position
                    trailing_stop_price = peak_price * (1 - self.trailing_stop_rate)
                    if current_price <= trailing_stop_price:
                        # Execute trailing stop
                        self._execute_stop_loss(ticker, current_price, date, "trailing_stop")
                
                elif position['quantity'] < 0:  # Short position
                    trailing_stop_price = peak_price * (1 + self.trailing_stop_rate)
                    if current_price >= trailing_stop_price:
                        # Execute trailing stop
                        self._execute_stop_loss(ticker, current_price, date, "trailing_stop")
    
    def _execute_stop_loss(self, ticker, current_price, date, stop_type):
        """Execute stop loss or trailing stop"""
        position = self.positions[ticker]
        quantity = position['quantity']
        
        # Calculate P&L
        entry_price = position['entry_price']
        pnl = (current_price - entry_price) * quantity
        
        # Update cash and portfolio
        self.cash += (current_price * abs(quantity)) - (self.commission_rate * abs(quantity) * current_price)
        # Recalculate portfolio value based on remaining positions
        remaining_value = sum(pos['quantity'] * current_price if ticker == pos_ticker else 0 
                            for pos_ticker, pos in self.positions.items())
        self.portfolio_value = self.cash + remaining_value
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'ticker': ticker,
            'action': 'sell' if quantity > 0 else 'buy',
            'quantity': abs(quantity),
            'price': current_price,
            'pnl': pnl,
            'stop_type': stop_type
        })
        
        # Close position
        del self.positions[ticker]
        if ticker in self.peak_prices:
            del self.peak_prices[ticker]
        
        print(f"🛑 {stop_type.upper()} executed for {ticker}: P&L = ${pnl:.2f}")
    
    def _check_drawdown_limit(self):
        """Check if maximum drawdown limit is reached"""
        if len(self.portfolio_history) < 2:
            return False
        
        # Calculate current drawdown
        peak_value = max(entry['portfolio_value'] for entry in self.portfolio_history)
        current_value = self.portfolio_history[-1]['portfolio_value']
        current_drawdown = (peak_value - current_value) / peak_value
        
        return current_drawdown >= self.max_drawdown_limit
    
    def _calculate_performance_realistic(self):
        """Calculate comprehensive performance metrics"""
        if len(self.portfolio_history) < 2:
            return
        
        # Extract portfolio values
        portfolio_values = [entry['portfolio_value'] for entry in self.portfolio_history]
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        self.total_return = (portfolio_values[-1] / self.initial_capital) - 1
        annualized_return = self.total_return * (252 / len(returns)) if len(returns) > 0 else 0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        self.sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        self.max_drawdown = np.min(drawdown)
        
        # Store daily returns
        self.daily_returns = returns.tolist()
        
        print(f"📊 Performance Summary:")
        print(f"   Total Return: {self.total_return:.2%}")
        print(f"   Annualized Return: {annualized_return:.2%}")
        print(f"   Volatility: {volatility:.2%}")
        print(f"   Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {self.max_drawdown:.2%}")
    
    def _run_monte_carlo_simulations(self, backtest_results, n_simulations=1000):
        """Run Monte Carlo simulations for robustness testing"""
        if not backtest_results or 'daily_returns' not in backtest_results:
            return {}
        
        returns = np.array(backtest_results['daily_returns'])
        if len(returns) == 0:
            return {}
        
        print(f"🎲 Running {n_simulations} Monte Carlo simulations...")
        
        # Bootstrap simulation
        simulation_results = []
        for i in range(n_simulations):
            # Resample returns with replacement
            resampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate cumulative return
            cumulative_return = np.prod(1 + resampled_returns) - 1
            
            # Calculate Sharpe ratio
            annualized_return = cumulative_return * (252 / len(resampled_returns))
            volatility = np.std(resampled_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative_values = np.cumprod(1 + resampled_returns)
            peak = np.maximum.accumulate(cumulative_values)
            drawdown = (cumulative_values - peak) / peak
            max_drawdown = np.min(drawdown)
            
            simulation_results.append({
                'total_return': cumulative_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            })
        
        # Calculate confidence intervals
        returns_array = np.array([r['total_return'] for r in simulation_results])
        sharpe_array = np.array([r['sharpe_ratio'] for r in simulation_results])
        drawdown_array = np.array([r['max_drawdown'] for r in simulation_results])
        
        mc_results = {
            'n_simulations': n_simulations,
            'total_return': {
                'mean': np.mean(returns_array),
                'std': np.std(returns_array),
                'percentile_5': np.percentile(returns_array, 5),
                'percentile_95': np.percentile(returns_array, 95)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_array),
                'std': np.std(sharpe_array),
                'percentile_5': np.percentile(sharpe_array, 5),
                'percentile_95': np.percentile(sharpe_array, 95)
            },
            'max_drawdown': {
                'mean': np.mean(drawdown_array),
                'std': np.std(drawdown_array),
                'percentile_5': np.percentile(drawdown_array, 5),
                'percentile_95': np.percentile(drawdown_array, 95)
            }
        }
        
        print(f"✓ Monte Carlo simulations completed")
        return mc_results
    
    def _generate_realistic_report(self, backtest_results, mc_results):
        """Generate comprehensive backtest report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE BACKTEST REPORT")
        print("=" * 80)
        
        # Basic performance metrics
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Portfolio Value: ${backtest_results['final_portfolio_value']:,.2f}")
        print(f"   Total Return: {backtest_results['total_return']:.2%}")
        print(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"   Maximum Drawdown: {backtest_results['max_drawdown']:.2%}")
        
        # Risk metrics
        if self.portfolio_history:
            portfolio_values = [entry['portfolio_value'] for entry in self.portfolio_history]
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 and np.any(returns <= var_95) else 0
            
            print(f"\n⚠️  RISK METRICS:")
            print(f"   Annualized Volatility: {volatility:.2%}")
            print(f"   95% VaR: {var_95:.2%}")
            print(f"   95% CVaR: {cvar_95:.2%}")
        
        # Trade statistics
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            if 'pnl' in trades_df.columns:
                winning_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] < 0]
                
                win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
                avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                
                print(f"\n💰 TRADE STATISTICS:")
                print(f"   Total Trades: {len(trades_df)}")
                print(f"   Win Rate: {win_rate:.2%}")
                print(f"   Average Win: ${avg_win:.2f}")
                print(f"   Average Loss: ${avg_loss:.2f}")
                print(f"   Profit Factor: {profit_factor:.2f}")
        
        # Monte Carlo results
        if mc_results:
            print(f"\n🎲 MONTE CARLO SIMULATION RESULTS ({mc_results['n_simulations']} simulations):")
            print(f"   Total Return:")
            print(f"     Mean: {mc_results['total_return']['mean']:.2%}")
            print(f"     Std: {mc_results['total_return']['std']:.2%}")
            print(f"     5th Percentile: {mc_results['total_return']['percentile_5']:.2%}")
            print(f"     95th Percentile: {mc_results['total_return']['percentile_95']:.2%}")
            
            print(f"   Sharpe Ratio:")
            print(f"     Mean: {mc_results['sharpe_ratio']['mean']:.2f}")
            print(f"     Std: {mc_results['sharpe_ratio']['std']:.2f}")
            print(f"     5th Percentile: {mc_results['sharpe_ratio']['percentile_5']:.2f}")
            print(f"     95th Percentile: {mc_results['sharpe_ratio']['percentile_95']:.2f}")
        
        # Save results to files
        self._save_results_to_files(backtest_results, mc_results)
        
        print(f"\n💾 Results saved to files:")
        print(f"   - backtest_results_v2.json")
        print(f"   - portfolio_history_v2.csv")
        print(f"   - trade_history_v2.csv")
        
        print("\n" + "=" * 80)
    
    def _save_results_to_files(self, backtest_results, mc_results):
        """Save results to files for further analysis"""
        
        # Save backtest results
        results_data = {
            'backtest_results': backtest_results,
            'monte_carlo_results': mc_results,
            'parameters': {
                'initial_capital': self.initial_capital,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'tickers': self.tickers,
                'commission_rate': self.commission_rate,
                'slippage_rate': self.slippage_rate,
                'max_position_size': self.max_position_size,
                'max_portfolio_risk': self.max_portfolio_risk,
                'stop_loss_rate': self.stop_loss_rate,
                'trailing_stop_rate': self.trailing_stop_rate
            }
        }
        
        with open('backtest_results_v2.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save portfolio history
        if self.portfolio_history:
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_df.to_csv('portfolio_history_v2.csv', index=False)
        
        # Save trade history
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            trade_df.to_csv('trade_history_v2.csv', index=False)
    
    def _close_all_positions(self, final_date, current_prices):
        """Close all positions at the end of backtest"""
        for ticker, position in list(self.positions.items()):
            if ticker in current_prices:
                current_price = current_prices[ticker]
                quantity = position['quantity']
                entry_price = position['entry_price']
                
                # Calculate P&L
                pnl = (current_price - entry_price) * quantity
                
                # Update cash
                self.cash += (current_price * abs(quantity)) - (self.commission_rate * abs(quantity) * current_price)
                
                # Record trade
                self.trade_history.append({
                    'date': final_date,
                    'ticker': ticker,
                    'action': 'sell' if quantity > 0 else 'buy',
                    'quantity': abs(quantity),
                    'price': current_price,
                    'pnl': pnl,
                    'stop_type': 'close'
                })
                
                # Remove position
                del self.positions[ticker]
        
        # Update final portfolio value
        self.portfolio_value = self.cash
    
    def _run_rolling_validation(self, data, window_years=2, step_days=63):
        """Run rolling window validation"""
        print(f"🔁 Rolling validation: {window_years} year windows, {step_days} day steps")
        
        window_days = int(window_years * 252)
        results = []
        
        for start_idx in range(0, len(data) - window_days, step_days):
            end_idx = start_idx + window_days
            
            # Split into train/test
            train_end = start_idx + int(window_days * 0.7)
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:end_idx]
            
            # Run mini-backtest
            mini_results = self._run_walk_forward_backtest(train_data, test_data)
            if mini_results:
                results.append(mini_results)
        
        if results:
            avg_return = np.mean([r['total_return'] for r in results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            print(f"✓ Rolling validation: Avg Return = {avg_return:.2%}, Avg Sharpe = {avg_sharpe:.2f}")

def main():
    """Main function to run the backtest"""
    print("🚀 Starting Realistic Backtest System v2...")
    
    # Initialize system
    backtest_system = RealisticBacktestSystemV2(
        initial_capital=1000000,
        start_date="2020-01-01",
        end_date="2024-12-31",
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],  # Sample tickers
        use_rolling_validation=True,
        use_advanced_signals=True
    )
    
    # Run backtest
    results, mc_results = backtest_system.run_realistic_backtest()
    
    if results:
        print("\n✅ Backtest completed successfully!")
    else:
        print("\n❌ Backtest failed!")

if __name__ == "__main__":
    main()
