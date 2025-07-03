#!/usr/bin/env python3
"""
Alpaca Trading Interface for Real-Time Paper Trading
Integrates with the statistical arbitrage model for live trading
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import logging

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment
    ALPACA_AVAILABLE = True
except ImportError:
    print("Warning: Alpaca SDK not installed. Install with: pip install alpaca-py")
    ALPACA_AVAILABLE = False

class AlpacaTradingInterface:
    """
    Alpaca trading interface for real-time paper trading
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, paper: bool = True):
        """
        Initialize Alpaca trading interface
        
        Args:
            api_key: Alpaca API key (if None, will try to get from environment)
            secret_key: Alpaca secret key (if None, will try to get from environment)
            paper: Whether to use paper trading (default: True)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca SDK not available. Install with: pip install alpaca-py")
        
        # Get API credentials
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # Trading parameters
        self.paper = paper
        self.min_order_size = 1  # Minimum shares per order
        self.max_order_size = 10000  # Maximum shares per order
        self.order_timeout = 30  # Seconds to wait for order execution
        
        # Portfolio tracking
        self.positions = {}
        self.orders = []
        self.trade_history = []
        
        # Market hours (EST)
        self.market_open = "09:30"
        self.market_close = "16:00"
        
        print(f"Alpaca Trading Interface initialized ({'PAPER' if paper else 'LIVE'} trading)")
        print(f"Account: {self.trading_client.get_account().account_number}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"Error checking market status: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'account_number': account.account_number,
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'daytrade_count': account.daytrade_count,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            current_positions = {}
            
            for position in positions:
                ticker = position.symbol
                shares = float(position.qty)
                if shares != 0:
                    current_positions[ticker] = shares
            
            self.positions = current_positions
            return current_positions
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {}
    
    def get_historical_data(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime = None, timeframe: TimeFrame = TimeFrame.Day) -> pd.DataFrame:
        """
        Get historical data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data (default: now)
            timeframe: Data timeframe (default: daily)
        
        Returns:
            DataFrame with OHLCV data for all symbols
        """
        if end_date is None:
            end_date = datetime.now()
        
        all_data = {}
        
        for symbol in symbols:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date,
                    adjustment=Adjustment.ALL
                )
                
                bars = self.data_client.get_stock_bars(request)
                
                if bars and len(bars) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame([{
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    } for bar in bars])
                    
                    df.set_index('timestamp', inplace=True)
                    
                    # Add to combined data
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        all_data[f"{symbol}_{col.capitalize()}"] = df[col]
                
            except Exception as e:
                print(f"Error getting data for {symbol}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.DataFrame(all_data)
        combined_df.sort_index(inplace=True)
        
        return combined_df
    
    def get_latest_prices(self, symbols: list) -> dict:
        """Get the latest close price for each symbol in the list using Alpaca SDK"""
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=None,  # Let Alpaca use default (most recent)
                end=None
            )
            bars = self.data_client.get_stock_bars(request_params)
            prices = {}
            for symbol in symbols:
                barlist = bars.data.get(symbol, [])
                if barlist:
                    last_bar = barlist[-1]
                    price = last_bar['close'] if isinstance(last_bar, dict) else getattr(last_bar, 'close', None)
                    if price is not None:
                        prices[symbol] = price
                    else:
                        logging.error(f"No close price found for {symbol} in last bar: {last_bar}")
                else:
                    logging.error(f"No bars found for {symbol}")
            return prices
        except Exception as e:
            logging.error(f"Error getting latest prices: {e}")
            return {}
    
    def place_market_order(self, symbol: str, qty: int, side: str) -> Optional[str]:
        """
        Place a market order
        
        Args:
            symbol: Stock symbol
            qty: Number of shares (positive for buy, negative for sell)
            side: 'buy' or 'sell'
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Validate order
            if abs(qty) < self.min_order_size:
                print(f"Order size {qty} below minimum {self.min_order_size}")
                return None
            
            if abs(qty) > self.max_order_size:
                print(f"Order size {qty} above maximum {self.max_order_size}")
                return None
            
            # Create order request
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=abs(qty),
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            # Place order
            order = self.trading_client.submit_order(order_data)
            
            # Wait for order to be processed
            start_time = time.time()
            while time.time() - start_time < self.order_timeout:
                order_status = self.trading_client.get_order_by_id(order.id)
                if order_status.status in ['filled', 'partially_filled', 'rejected', 'canceled']:
                    break
                time.sleep(1)
            
            # Record order
            self.orders.append({
                'id': order.id,
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'status': order_status.status if 'order_status' in locals() else 'unknown',
                'timestamp': datetime.now()
            })
            
            print(f"Order placed: {side} {abs(qty)} {symbol} - Status: {order_status.status if 'order_status' in locals() else 'unknown'}")
            return order.id
            
        except Exception as e:
            print(f"Error placing order for {symbol}: {e}")
            return None
    
    def execute_trades(self, target_positions: Dict[str, float], current_prices: Dict[str, float]) -> List[Dict]:
        """
        Execute trades to reach target positions
        
        Args:
            target_positions: Target position sizes (ticker -> position_size)
            current_prices: Current prices for position sizing
        
        Returns:
            List of executed trades
        """
        if not self.is_market_open():
            print("Market is closed. Cannot execute trades.")
            return []
        
        # Get current positions
        current_positions = self.get_current_positions()
        account_info = self.get_account_info()
        
        if not account_info:
            print("Could not get account information")
            return []
        
        portfolio_value = account_info['portfolio_value']
        executed_trades = []
        
        for ticker, target_position in target_positions.items():
            if ticker not in current_prices:
                print(f"No price data for {ticker}")
                continue
            
            current_price = current_prices[ticker]
            current_shares = current_positions.get(ticker, 0)
            
            # Calculate target shares
            target_value = target_position * portfolio_value
            target_shares = int(target_value / current_price)
            
            # Calculate shares to trade
            shares_to_trade = target_shares - current_shares
            
            if abs(shares_to_trade) < self.min_order_size:
                continue
            
            # Place order
            side = 'buy' if shares_to_trade > 0 else 'sell'
            order_id = self.place_market_order(ticker, abs(shares_to_trade), side)
            
            if order_id:
                executed_trades.append({
                    'ticker': ticker,
                    'shares': shares_to_trade,
                    'side': side,
                    'price': current_price,
                    'value': abs(shares_to_trade * current_price),
                    'order_id': order_id,
                    'timestamp': datetime.now()
                })
        
        return executed_trades
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        try:
            account_info = self.get_account_info()
            cash = account_info.get('cash', 0)
            
            positions_value = 0
            current_positions = self.get_current_positions()
            
            for ticker, shares in current_positions.items():
                if ticker in current_prices:
                    positions_value += shares * current_prices[ticker]
            
            return cash + positions_value
        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def cancel_all_orders(self):
        """Cancel all pending orders"""
        try:
            self.trading_client.cancel_all_orders()
            print("All orders cancelled")
        except Exception as e:
            print(f"Error cancelling orders: {e}")
    
    def get_trading_status(self) -> Dict:
        """Get current trading status"""
        try:
            account_info = self.get_account_info()
            current_positions = self.get_current_positions()
            market_open = self.is_market_open()
            
            return {
                'market_open': market_open,
                'portfolio_value': account_info.get('portfolio_value', 0),
                'cash': account_info.get('cash', 0),
                'buying_power': account_info.get('buying_power', 0),
                'positions_count': len(current_positions),
                'daytrade_count': account_info.get('daytrade_count', 0),
                'pattern_day_trader': account_info.get('pattern_day_trader', False)
            }
        except Exception as e:
            print(f"Error getting trading status: {e}")
            return {} 