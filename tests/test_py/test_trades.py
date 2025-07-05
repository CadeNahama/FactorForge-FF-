#!/usr/bin/env python3

import sys
sys.path.append('.')
from realistic_backtest_system import RealisticBacktestSystem
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_trades():
    # Create a shorter backtest period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Just 30 days

    # Get data for a single stock
    symbol = 'AAPL'
    print(f"Downloading data for {symbol}...")
    data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    print(f'Downloaded {len(data)} days of data for {symbol}')

    # Create backtest system
    backtest = RealisticBacktestSystem(
        initial_capital=100000,
        symbols=[symbol],
        data=data
    )

    # Run a quick backtest
    print('Running quick backtest...')
    try:
        results, trade_history = backtest.run_realistic_backtest()
        print(f'Backtest completed. Portfolio value: ${results["final_portfolio_value"]:.2f}')
        print(f'Number of trades: {len(trade_history) if trade_history is not None else 0}')
        
        if trade_history is not None and len(trade_history) > 0:
            print('Sample trades:')
            for i, trade in enumerate(trade_history[:5]):
                print(f'  {i+1}. {trade}')
        else:
            print('No trades were executed during the backtest period.')
            
    except Exception as e:
        print(f'Error during backtest: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trades() 