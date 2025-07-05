#!/usr/bin/env python3
"""
Enhanced Data Loader for Quantitative Trading System
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from typing import List, Optional, Dict
import requests
import time

warnings.filterwarnings('ignore')

class EnhancedDataLoader:
    """Enhanced data loader with multiple data sources and error handling"""
    
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_data(self, tickers: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Download data for multiple tickers with enhanced error handling"""
        
        print(f"📊 Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_data = {}
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            try:
                print(f"  Downloading {ticker} ({i+1}/{len(tickers)})...")
                
                # Download data using yfinance with volume
                ticker_data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    progress=False,
                    auto_adjust=True,
                    actions=False  # Don't include dividends/splits in volume
                )
                
                if not ticker_data.empty:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        # Flatten MultiIndex columns
                        ticker_data.columns = [f"{ticker}_{col[0]}" for col in ticker_data.columns]
                    else:
                        # Rename columns to include ticker prefix
                        ticker_data.columns = [f"{ticker}_{col}" for col in ticker_data.columns]
                    
                    all_data[ticker] = ticker_data
                    print(f"    ✓ {ticker}: {len(ticker_data)} rows")
                else:
                    failed_tickers.append(ticker)
                    print(f"    ❌ {ticker}: No data available")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                failed_tickers.append(ticker)
                print(f"    ❌ {ticker}: Error - {str(e)}")
                continue
        
        if not all_data:
            print("❌ No data downloaded successfully")
            return None
        
        # Combine all data
        print(f"🔗 Combining data from {len(all_data)} successful downloads...")
        
        # Find common date range
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index)
        
        common_dates = sorted(list(all_dates))
        
        # Create combined DataFrame
        combined_data = pd.DataFrame(index=pd.DatetimeIndex(common_dates))
        
        for ticker, data in all_data.items():
            for col in data.columns:
                combined_data[col] = data[col]
        
        # Forward fill missing values (within reasonable limits)
        combined_data = combined_data.fillna(method='ffill', limit=5)
        
        print(f"✓ Combined data shape: {combined_data.shape}")
        print(f"✓ Date range: {combined_data.index[0]} to {combined_data.index[-1]}")
        
        if failed_tickers:
            print(f"⚠️  Failed tickers: {failed_tickers}")
        
        return combined_data
    
    def get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers"""
        # Return a subset of major S&P 500 stocks for testing
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 
            'AMD', 'INTC', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS',
            'PYPL', 'BAC', 'ADBE', 'CRM', 'NKE', 'CMCSA', 'XOM', 'ABT', 'KO',
            'PFE', 'TMO', 'AVGO', 'COST', 'DHR', 'ACN', 'VZ', 'MRK', 'PEP',
            'ABBV', 'TXN', 'LLY', 'CVX', 'WMT', 'MCD', 'BMY', 'UNP', 'LOW',
            'HON', 'RTX', 'T', 'QCOM', 'UPS', 'IBM', 'SPGI', 'CAT', 'GS',
            'AMGN', 'DE', 'GILD', 'MS', 'SCHW', 'AXP', 'PLD', 'ADI', 'GE'
        ]
    
    def get_etf_universe(self) -> List[str]:
        """Get ETF universe"""
        return [
            'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'TLT', 'GLD', 'SLV',
            'VTI', 'VEA', 'VWO', 'BND', 'VNQ', 'XLF', 'XLK', 'XLE', 'XLV',
            'XLI', 'XLP', 'XLU', 'XLB', 'XLY', 'XLC', 'XLRE', 'XOP', 'XBI'
        ]
    
    def get_market_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get market data for specific tickers"""
        data = self.download_data(tickers, start_date, end_date)
        if data is None:
            return {}
        
        # Split data by ticker
        market_data = {}
        for ticker in tickers:
            ticker_cols = [col for col in data.columns if col.startswith(f"{ticker}_")]
            if ticker_cols:
                market_data[ticker] = data[ticker_cols]
        
        return market_data
    
    def calculate_returns(self, data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Calculate returns for given tickers"""
        returns = pd.DataFrame(index=data.index)
        
        for ticker in tickers:
            close_col = f"{ticker}_Close"
            if close_col in data.columns:
                returns[ticker] = data[close_col].pct_change()
        
        return returns.dropna()
    
    def get_risk_free_rate(self, start_date: str, end_date: str) -> float:
        """Get risk-free rate (simplified - using 3-month Treasury yield)"""
        try:
            # Download 3-month Treasury yield as proxy for risk-free rate
            tbill_data = yf.download('^IRX', start=start_date, end=end_date, progress=False)
            if not tbill_data.empty:
                return tbill_data['Close'].iloc[-1] / 100  # Convert from percentage
        except:
            pass
        
        # Default risk-free rate
        return 0.02  # 2% annual rate

def get_sp500_tickers():
    """Fetch the latest S&P 500 tickers from Wikipedia."""
    import pandas as pd
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    # Alpaca uses BRK.B and BF.B, not BRK-B/BF-B
    tickers = [t.replace('.', '-') for t in tickers]
    tickers = [t.replace('BRK-B', 'BRK.B').replace('BF-B', 'BF.B') for t in tickers]
    return tickers
