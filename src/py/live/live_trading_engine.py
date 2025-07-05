#!/usr/bin/env python3
"""
Live Trading Engine for Real-Time Statistical Arbitrage
Integrates Alpaca trading with the enhanced statistical arbitrage model
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import json
import logging
from zoneinfo import ZoneInfo
warnings.filterwarnings('ignore')

# Import our components
# If alpaca-trade-api is not found, ensure it is installed: pip install alpaca-trade-api
from src.py.live.alpaca_trading_interface import AlpacaTradingInterface
from src.py.core.data import EnhancedDataLoader, get_sp500_tickers
from src.py.backtest.engine import RealisticBacktestSystemV2
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)

# --- TRADING WINDOW CONFIG ---
# To change the trading window, adjust these times (24h format, NY time)
TRADING_START = "15:00"  # 3:00 PM NY time
TRADING_END = "16:00"    # 4:00 PM NY time (market close)

class LiveTradingEngine:
    """
    Live trading engine that runs the statistical arbitrage model in real-time
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, 
                 initial_capital: float = 100000.0, paper: bool = True, max_new_positions_per_cycle: int = 5,
                 max_open_positions: int = 20, max_daily_loss: float = 0.03, max_turnover: float = 0.3):
        """
        Initialize live trading engine
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            initial_capital: Initial capital for position sizing
            paper: Whether to use paper trading
            max_new_positions_per_cycle: Maximum number of new positions to open per trading cycle
            max_open_positions: Maximum number of open positions
            max_daily_loss: Maximum daily loss
            max_turnover: Maximum daily turnover
        """
        # Initialize Alpaca interface
        self.alpaca = AlpacaTradingInterface(api_key, secret_key, paper)
        
        # Immediately fetch current positions from Alpaca
        self.current_positions = self.alpaca.get_current_positions()
        
        # Initialize components
        self.tickers = get_sp500_tickers()
        self.data_loader = EnhancedDataLoader()
        self.backtest_system = RealisticBacktestSystemV2(
            initial_capital=initial_capital,
            start_date="2017-01-01",
            end_date="2024-12-31",
            tickers=self.tickers,
            commission_rate=0.0005,
            slippage_rate=0.0002,
            market_impact_rate=0.00005,
            max_position_size=0.025,
            max_portfolio_risk=0.02,
            stop_loss_rate=0.08,
            max_drawdown_limit=0.08,
            trailing_stop_rate=0.05,
            train_period_years=2.0,
            retrain_frequency_days=63,
            out_of_sample_start="2020-01-01",
            use_rolling_validation=False,
            use_advanced_signals=True
        )
        
        # Trading parameters
        self.initial_capital = initial_capital
        self.paper = paper
        
        # Trading schedule
        self.trading_start = TRADING_START
        self.trading_end = TRADING_END
        self.rebalance_interval = 60  # Rebalance every 60 minutes
        self.last_rebalance = None
        self.ny_tz = ZoneInfo("America/New_York")
        
        # Risk management
        self.max_position_size = 0.05  # 5% max position
        self.max_daily_turnover = 0.20  # 20% max daily turnover
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.trailing_stop_pct = 0.02  # 2% trailing stop
        
        # Performance tracking
        self.portfolio_history = []
        self.trade_history = []
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # State tracking
        self.is_running = False
        self.stop_losses = {}
        self.trailing_stops = {}
        
        # New positions per cycle
        self.max_new_positions_per_cycle = max_new_positions_per_cycle
        self.max_open_positions = max_open_positions
        self.max_daily_loss = max_daily_loss
        self.max_turnover = max_turnover
        self.daily_turnover = 0.0
        self.open_positions = set()
        
        logging.info(f"Live Trading Engine initialized ({'PAPER' if paper else 'LIVE'} trading)")
        logging.info(f"Initial Capital: ${initial_capital:,.2f}")
        logging.info(f"Tickers: {len(self.tickers)}")
    
    def is_trading_time(self) -> bool:
        """Check if it's trading time (always compare in US/Eastern)"""
        now = datetime.now(self.ny_tz)
        start = datetime.strptime(self.trading_start, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day, tzinfo=self.ny_tz)
        end = datetime.strptime(self.trading_end, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day, tzinfo=self.ny_tz)
        in_window = start <= now < end
        if not in_window:
            logging.info(f"Not trading: current NY time {now.strftime('%H:%M:%S')} is outside trading window {self.trading_start}-{self.trading_end}.")
        else:
            logging.info(f"Trading active: current NY time {now.strftime('%H:%M:%S')} is within trading window.")
        return in_window
    
    def get_market_data(self):
        """Fetch latest market data using Alpaca's new SDK (daily bars)"""
        try:
            symbols = self.tickers if hasattr(self, 'tickers') else []
            if not symbols:
                logging.error("No tickers specified for market data.")
                return None
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                limit=100
            )
            bars = self.alpaca.data_client.get_stock_bars(request)
            data = {}
            for symbol in symbols:
                barlist = bars.data.get(symbol, [])
                if barlist:
                    data[symbol] = [{
                        'time': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    } for bar in barlist]
            if not data:
                logging.error("No market data returned from Alpaca.")
                return None
            return data
        except Exception as e:
            logging.error(f"Error getting market data: {e}")
            return None
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading signals using the robust backtest logic"""
        try:
            if data.empty:
                return {}
            # Get current date (last available data point)
            current_date = data.index[-1]
            # Use the backtest system's signal generation
            signals = self.backtest_system._generate_signals_realistic(data, current_date)
            # Apply live risk management (position limits, etc.)
            filtered_signals = self.apply_live_risk_management(signals)
            # --- BEGIN ADDED LOGGING ---
            buy_count = sum(1 for v in filtered_signals.values() if v > 0.001)
            sell_count = sum(1 for v in filtered_signals.values() if v < -0.001)
            hold_count = sum(1 for v in filtered_signals.values() if abs(v) <= 0.001)
            logging.info(f"Signal distribution: {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD/FLAT out of {len(filtered_signals)} total signals")
            # --- END ADDED LOGGING ---
            # --- BEGIN CAP NEW POSITIONS LOGIC ---
            current_positions = self.alpaca.get_current_positions()
            new_signals = {k: v for k, v in filtered_signals.items() if k not in current_positions and abs(v) > 0.001}
            if len(new_signals) > self.max_new_positions_per_cycle:
                # Sort by absolute signal strength, keep only top N
                top_new = dict(sorted(new_signals.items(), key=lambda item: abs(item[1]), reverse=True)[:self.max_new_positions_per_cycle])
                # Keep all signals for tickers we already hold, plus the top N new
                filtered_signals = {k: v for k, v in filtered_signals.items() if k in current_positions or k in top_new}
                logging.info(f"Capped new positions to {self.max_new_positions_per_cycle} this cycle.")
            # --- END CAP NEW POSITIONS LOGIC ---
            logging.info(f"Generated signals for {len(filtered_signals)} tickers")
            return filtered_signals
        except Exception as e:
            logging.error(f"Error calculating signals: {e}")
            return {}
    
    def apply_live_risk_management(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Apply risk management for live trading"""
        try:
            current_positions = self.alpaca.get_current_positions()
            account_info = self.alpaca.get_account_info()
            if not account_info:
                return {}
            portfolio_value = account_info['portfolio_value']
            limited_signals = {}
            open_positions_count = sum(1 for qty in current_positions.values() if abs(qty) > 0)
            # Aggressive close logic and risk controls
            for ticker in set(list(signals.keys()) + list(current_positions.keys())):
                signal = signals.get(ticker, 0)
                current_position = current_positions.get(ticker, 0)
                position_value = abs(current_position) * portfolio_value
                # Max open positions
                if open_positions_count >= self.max_open_positions and ticker not in current_positions:
                    continue
                # Max turnover
                if self.daily_turnover > self.max_turnover * portfolio_value:
                    continue
                # Max daily loss
                if abs(self.daily_pnl) > self.max_daily_loss * portfolio_value:
                    logging.warning("Max daily loss reached. No new trades will be opened today.")
                    continue
                # Aggressive close: if signal weakens or reverses
                if abs(current_position) > 0:
                    if (np.sign(signal) != np.sign(current_position) and abs(signal) > 0.01) or abs(signal) < 0.01:
                        limited_signals[ticker] = -np.sign(current_position) * self.max_position_size
                        logging.info(f"[AGGRESSIVE CLOSE] {ticker}: Closing position of {current_position} shares due to weak/reverse signal {signal:.4f}")
                        continue
                # Usual close logic
                if abs(signal) < 0.001 and abs(current_position) > 0:
                    limited_signals[ticker] = -np.sign(current_position) * self.max_position_size
                    logging.info(f"[CLOSE] {ticker}: Closing position of {current_position} shares")
                elif abs(signal) > 0.001:
                    limited_signals[ticker] = signal
                    logging.info(f"[SIGNAL] {ticker}: Signal={signal:.4f}, Position={current_position}")
            return limited_signals
        except Exception as e:
            logging.error(f"Error applying risk management: {e}")
            return {}
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker"""
        try:
            prices = self.alpaca.get_latest_prices([ticker])
            return prices.get(ticker)
        except Exception as e:
            logging.error(f"Error getting price for {ticker}: {e}")
            return None
    
    def execute_trades(self, signals: Dict[str, float]) -> List[Dict]:
        """Execute trades based on signals"""
        try:
            if not signals:
                return []
            # Get current prices
            current_prices = self.alpaca.get_latest_prices(list(signals.keys()))
            if not current_prices:
                logging.error("No current prices available")
                return []
            # Randomize order execution
            import random
            items = list(signals.items())
            random.shuffle(items)
            randomized_signals = dict(items)
            # Latency measurement
            start_time = time.time()
            executed_trades = self.alpaca.execute_trades(randomized_signals, current_prices)
            latency = time.time() - start_time
            logging.info(f"Trade execution latency: {latency:.4f} seconds")
            # Update turnover
            for trade in executed_trades:
                self.daily_turnover += abs(trade['value'])
            # Update stop losses and trailing stops
            for trade in executed_trades:
                ticker = trade['ticker']
                side = trade['side']
                price = trade['price']
                if side == 'buy':
                    self.stop_losses[ticker] = price * (1 - self.stop_loss_pct)
                    self.trailing_stops[ticker] = price * (1 - self.trailing_stop_pct)
                elif side == 'sell':
                    self.stop_losses[ticker] = price * (1 + self.stop_loss_pct)
                    self.trailing_stops[ticker] = price * (1 + self.trailing_stop_pct)
            self.trade_history.extend(executed_trades)
            self.total_trades += len(executed_trades)
            logging.info(f"Executed {len(executed_trades)} trades")
            return executed_trades
        except Exception as e:
            logging.error(f"Error executing trades: {e}")
            return []
    
    def update_portfolio_tracking(self):
        """Update portfolio tracking and performance metrics"""
        try:
            # Get current positions and prices
            current_positions = self.alpaca.get_current_positions()
            current_prices = self.alpaca.get_latest_prices(list(current_positions.keys()))
            
            # Calculate portfolio value
            portfolio_value = self.alpaca.get_portfolio_value(current_prices)
            
            # Update portfolio history
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'positions': current_positions.copy(),
                'prices': current_prices.copy()
            })
            
            # Calculate daily P&L
            if len(self.portfolio_history) > 1:
                prev_value = self.portfolio_history[-2]['portfolio_value']
                self.daily_pnl = portfolio_value - prev_value
            
            logging.info(f"Portfolio Value: ${portfolio_value:,.2f}, Daily P&L: ${self.daily_pnl:,.2f}")
            
        except Exception as e:
            logging.error(f"Error updating portfolio tracking: {e}")
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            logging.info("Starting trading cycle...")
            # Check if it's trading time
            if not self.is_trading_time():
                logging.info("Not trading time, skipping cycle")
                return
            # Get market data
            data = self.get_market_data()
            if data is None:
                logging.error("No market data available")
                return
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                # Flatten the dict of lists into a DataFrame
                df_list = []
                for symbol, bars in data.items():
                    for bar in bars:
                        bar_copy = bar.copy()
                        bar_copy['symbol'] = symbol
                        df_list.append(bar_copy)
                if df_list:
                    data = pd.DataFrame(df_list)
                    if 'time' in data.columns:
                        data['time'] = pd.to_datetime(data['time'])
                        data = data.set_index('time')
                else:
                    data = pd.DataFrame()
            # Calculate signals
            signals = self.calculate_signals(data)
            if not signals:
                logging.info("No signals generated")
                return
            # Execute trades
            executed_trades = self.execute_trades(signals)
            # Update portfolio tracking
            self.update_portfolio_tracking()
            # Update last rebalance time
            self.last_rebalance = datetime.now(ZoneInfo("America/New_York"))
            logging.info(f"Trading cycle completed. Executed {len(executed_trades)} trades")
        except Exception as e:
            logging.error(f"Error in trading cycle: {e}")
    
    def start_trading(self, check_interval: int = 60):
        """
        Start live trading
        
        Args:
            check_interval: Seconds between trading cycle checks
        """
        try:
            logging.info("Starting live trading...")
            self.is_running = True
            
            # Initial portfolio setup
            self.update_portfolio_tracking()
            
            while self.is_running:
                try:
                    # Run trading cycle
                    self.run_trading_cycle()
                    
                    # Wait for next cycle
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logging.info("Trading stopped by user")
                    break
                except Exception as e:
                    logging.error(f"Error in trading loop: {e}")
                    time.sleep(check_interval)
            
            # Clean up
            self.stop_trading()
            
        except Exception as e:
            logging.error(f"Error starting trading: {e}")
    
    def stop_trading(self):
        """Stop live trading"""
        try:
            logging.info("Stopping live trading...")
            self.is_running = False
            
            # Cancel all pending orders
            self.alpaca.cancel_all_orders()
            
            # Final portfolio update
            self.update_portfolio_tracking()
            
            # Save trading results
            self.save_trading_results()
            
            logging.info("Live trading stopped")
            
        except Exception as e:
            logging.error(f"Error stopping trading: {e}")
    
    def save_trading_results(self):
        """Save trading results to files"""
        try:
            # Save portfolio history
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_df.to_csv('live_portfolio_history.csv', index=False)
            
            # Save trade history
            trade_df = pd.DataFrame(self.trade_history)
            trade_df.to_csv('live_trade_history.csv', index=False)
            
            # Save performance summary
            performance_summary = {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                'final_portfolio_value': self.portfolio_history[-1]['portfolio_value'] if self.portfolio_history else 0,
                'total_return': (self.portfolio_history[-1]['portfolio_value'] / self.initial_capital - 1) if self.portfolio_history else 0,
                'trading_start': self.portfolio_history[0]['timestamp'] if self.portfolio_history else None,
                'trading_end': self.portfolio_history[-1]['timestamp'] if self.portfolio_history else None
            }
            
            with open('live_performance_summary.json', 'w') as f:
                json.dump(performance_summary, f, indent=2, default=str)
            
            logging.info("Trading results saved")
            
        except Exception as e:
            logging.error(f"Error saving trading results: {e}")
    
    def get_trading_status(self) -> Dict:
        """Get current trading status"""
        try:
            alpaca_status = self.alpaca.get_trading_status()
            
            status = {
                'is_running': self.is_running,
                'is_trading_time': self.is_trading_time(),
                'last_rebalance': self.last_rebalance,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                'daily_pnl': self.daily_pnl,
                'current_positions_count': len(self.current_positions),
                **alpaca_status
            }
            
            return status
            
        except Exception as e:
            logging.error(f"Error getting trading status: {e}")
            return {} 