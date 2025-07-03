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
from alpaca_trading_interface import AlpacaTradingInterface
from core.enhanced_data import EnhancedDataLoader, get_sp500_tickers
from backtest.realistic_backtest_system_v2 import RealisticBacktestSystemV2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)

class LiveTradingEngine:
    """
    Live trading engine that runs the statistical arbitrage model in real-time
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, 
                 initial_capital: float = 100000.0, paper: bool = True):
        """
        Initialize live trading engine
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            initial_capital: Initial capital for position sizing
            paper: Whether to use paper trading
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
        self.trading_start = "09:30"
        self.trading_end = "15:45"  # Stop 15 minutes before close
        self.rebalance_interval = 60  # Rebalance every 60 minutes
        self.last_rebalance = None
        
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
        
        logging.info(f"Live Trading Engine initialized ({'PAPER' if paper else 'LIVE'} trading)")
        logging.info(f"Initial Capital: ${initial_capital:,.2f}")
        logging.info(f"Tickers: {len(self.tickers)}")
    
    def is_trading_time(self) -> bool:
        """Check if it's trading time (always compare in US/Eastern)"""
        now = datetime.now(ZoneInfo("America/New_York"))
        current_time = now.strftime("%H:%M")
        logging.info(f"[DEBUG] Now (NY): {now}, current_time: {current_time}, trading window: {self.trading_start}-{self.trading_end}")

        if not self.alpaca.is_market_open():
            logging.info("[DEBUG] Market is not open according to Alpaca.")
            return False

        if current_time < self.trading_start or current_time > self.trading_end:
            logging.info(f"[DEBUG] Current time {current_time} is outside trading window.")
            return False

        if self.last_rebalance is None:
            logging.info("[DEBUG] No last rebalance, will trade.")
            return True

        # Compare in seconds, not minutes
        time_since_rebalance = (now - self.last_rebalance).total_seconds()
        logging.info(f"[DEBUG] Time since last rebalance: {time_since_rebalance} sec, interval: {self.rebalance_interval * 60} sec")
        return time_since_rebalance >= self.rebalance_interval * 60
    
    def get_market_data(self) -> pd.DataFrame:
        """Get current market data for all tickers using the latest Alpaca SDK"""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            # Get historical data for the last 300 days
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=300)
            request_params = StockBarsRequest(
                symbol_or_symbols=self.tickers,
                timeframe=TimeFrame.Day,
                start=str(start_date),
                end=str(end_date)
            )
            bars = self.alpaca.data_client.get_stock_bars(request_params)
            if not bars or not hasattr(bars, 'data') or not bars.data:
                logging.error(f"No market data received (bars object: {bars})")
                return pd.DataFrame()
            # Convert to DataFrame
            all_records = []
            for symbol, barlist in bars.data.items():
                for bar in barlist:
                    record = bar.copy() if isinstance(bar, dict) else bar.__dict__.copy()
                    record['symbol'] = symbol
                    all_records.append(record)
            df = pd.DataFrame(all_records)
            if df.empty:
                logging.error("No market data available after DataFrame conversion")
                return pd.DataFrame()
            # Pivot to wide format: one column per ticker per field
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            df = df.pivot(index='date', columns='symbol')
            # Flatten multi-index columns and standardize to TICKER_Close, TICKER_Open, etc.
            df.columns = [f"{sym}_{field.capitalize()}" for field, sym in df.columns]
            df = df.sort_index()
            logging.info(f"Retrieved market data: {df.shape}")
            logging.info(f"DataFrame columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logging.error(f"Error getting market data: {e}")
            import traceback; traceback.print_exc()
            return pd.DataFrame()
    
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
            logging.info(f"Generated signals for {len(filtered_signals)} tickers")
            return filtered_signals
        except Exception as e:
            logging.error(f"Error calculating signals: {e}")
            return {}
    
    def apply_live_risk_management(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Apply risk management for live trading"""
        try:
            # Get current positions and account info
            current_positions = self.alpaca.get_current_positions()
            account_info = self.alpaca.get_account_info()
            
            if not account_info:
                return {}
            
            portfolio_value = account_info['portfolio_value']
            limited_signals = {}
            
            # Combine tickers from signals and current positions
            all_tickers = set(list(signals.keys()) + list(current_positions.keys()))
            for ticker in all_tickers:
                signal = signals.get(ticker, 0)
                # Apply position size limit
                if abs(signal) > self.max_position_size:
                    signal = np.sign(signal) * self.max_position_size
                
                current_position = current_positions.get(ticker, 0)
                position_value = abs(current_position) * portfolio_value
                
                # Apply daily turnover limit
                if position_value > portfolio_value * self.max_daily_turnover:
                    continue
                
                # Apply stop losses
                if ticker in self.stop_losses:
                    current_price = self.get_current_price(ticker)
                    if current_price:
                        if signal > 0 and current_price <= self.stop_losses[ticker]:  # Long position
                            signal = 0  # Close position
                        elif signal < 0 and current_price >= self.stop_losses[ticker]:  # Short position
                            signal = 0  # Close position
                
                # If signal is zero but we have a position, generate a close signal
                if abs(signal) < 0.001 and abs(current_position) > 0:
                    # Negative of current position to close it
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
            
            # Execute trades
            executed_trades = self.alpaca.execute_trades(signals, current_prices)
            
            # Update stop losses and trailing stops
            for trade in executed_trades:
                ticker = trade['ticker']
                side = trade['side']
                price = trade['price']
                
                if side == 'buy':
                    # Set stop loss for long position
                    self.stop_losses[ticker] = price * (1 - self.stop_loss_pct)
                    self.trailing_stops[ticker] = price * (1 - self.trailing_stop_pct)
                elif side == 'sell':
                    # Set stop loss for short position
                    self.stop_losses[ticker] = price * (1 + self.stop_loss_pct)
                    self.trailing_stops[ticker] = price * (1 + self.trailing_stop_pct)
            
            # Record trades
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
            if data.empty:
                logging.error("No market data available")
                return
            
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