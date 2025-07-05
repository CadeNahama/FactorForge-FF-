import os
import logging
from typing import Optional, List, Dict
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class AlpacaTradingInterface:
    """
    Basic interface for Alpaca paper/live trading.
    """
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, paper: bool = True):
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.paper = paper
        logging.info(f"AlpacaTradingInterface initialized in {'paper' if paper else 'live'} mode.")

    def is_market_open(self) -> bool:
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logging.error(f"Error checking market open: {e}")
            return False

    def place_market_order(self, symbol: str, qty: int, side: str) -> Optional[str]:
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_data)
            logging.info(f"Order placed: {side} {qty} {symbol}")
            return order.id
        except Exception as e:
            logging.error(f"Order failed: {e}")
            return None

    def get_account(self):
        try:
            return self.trading_client.get_account()
        except Exception as e:
            logging.error(f"Error getting account: {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            barset = self.trading_client.get_barset(symbol, '1Min', limit=1)
            bar = barset[symbol][0] if symbol in barset and barset[symbol] else None
            if bar:
                return bar.c
            else:
                logging.warning(f"No price data for {symbol}")
                return None
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {e}")
            return None

    def get_current_positions(self) -> Dict[str, float]:
        """Return current open positions as a dict {symbol: qty, ...}"""
        try:
            positions = self.trading_client.get_all_positions()
            return {pos.symbol: float(pos.qty) for pos in positions}
        except Exception as e:
            logging.error(f"Error getting current positions: {e}")
            return {}

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                limit=1
            )
            bars = self.data_client.get_stock_bars(request)
            prices = {}
            for symbol in symbols:
                barlist = bars.data.get(symbol, [])
                if barlist:
                    last_bar = barlist[-1]
                    prices[symbol] = last_bar.close
            return prices
        except Exception as e:
            logging.error(f"Error getting latest prices: {e}")
            return {}

    def get_portfolio_value(self, *args, **kwargs) -> Optional[float]:
        try:
            account = self.trading_client.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            logging.error(f"Error getting portfolio value: {e}")
            return None

    @property
    def api(self):
        raise AttributeError("'AlpacaTradingInterface' no longer has an 'api' attribute. Use 'trading_client' or 'data_client' instead.")

    def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            self.trading_client.cancel_orders()
            logging.info("All open orders cancelled.")
        except Exception as e:
            logging.error(f"Error cancelling all orders: {e}") 