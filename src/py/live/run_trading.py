#!/usr/bin/env python3
"""
Live Trading Runner for Statistical Arbitrage
Runs the enhanced statistical arbitrage model with Alpaca paper trading
"""

import os
import sys
import time
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.py.live.live_trading_engine import LiveTradingEngine

print("DEBUG: ALPACA_API_KEY =", os.environ.get("ALPACA_API_KEY"))
print("DEBUG: ALPACA_SECRET_KEY =", os.environ.get("ALPACA_SECRET_KEY"))

def setup_environment():
    """Setup environment variables and check requirements"""
    print("=" * 80)
    print("LIVE TRADING SETUP - STATISTICAL ARBITRAGE")
    print("=" * 80)
    
    # Check for Alpaca credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Alpaca API credentials not found!")
        print("Please set the following environment variables:")
        print("export ALPACA_API_KEY='your_api_key_here'")
        print("export ALPACA_SECRET_KEY='your_secret_key_here'")
        print("\nGet your API keys from: https://app.alpaca.markets/")
        return False
    
    print("✅ Alpaca API credentials found")
    
    # Check for required packages
    try:
        import alpaca_trade_api
        print("✅ Alpaca SDK installed")
    except ImportError:
        print("❌ Alpaca SDK not installed!")
        print("Install with: pip install alpaca-trade-api")
        return False
    
    try:
        import pandas
        import numpy
        import yfinance
        print("✅ Required packages installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: pip install pandas numpy yfinance")
        return False
    
    return True

def print_trading_config(paper: bool, initial_capital: float, check_interval: int):
    """Print trading configuration"""
    print("\n" + "=" * 80)
    print("TRADING CONFIGURATION")
    print("=" * 80)
    print(f"Mode: {'PAPER TRADING' if paper else 'LIVE TRADING'}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Check Interval: {check_interval} seconds")
    print(f"Trading Hours: 9:30 AM - 3:45 PM EST")
    print(f"Rebalance Interval: 60 minutes")
    print("\nRisk Management:")
    print("✓ Max Position Size: 5%")
    print("✓ Max Daily Turnover: 20%")
    print("✓ Stop Loss: 3%")
    print("✓ Trailing Stop: 2%")
    print("\nStrategies:")
    print("✓ Pairs Trading (30%)")
    print("✓ Mean Reversion (35%)")
    print("✓ Momentum (35%)")
    print("=" * 80)

def main():
    """Main function to run live trading"""
    parser = argparse.ArgumentParser(description='Live Statistical Arbitrage Trading')
    parser.add_argument('--live', action='store_true', help='Use live trading (default: paper)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital (default: 100000)')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    parser.add_argument('--test', action='store_true', help='Run a single trading cycle for testing')
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        return
    
    # Configuration
    paper = not args.live
    initial_capital = args.capital
    check_interval = args.interval
    
    # Print configuration
    print_trading_config(paper, initial_capital, check_interval)
    
    # Confirm before starting
    if paper:
        print("\n⚠️  WARNING: This will execute REAL trades on your Alpaca account!")
        print("Make sure you understand the risks and have sufficient capital.")
    else:
        print("\n⚠️  WARNING: This will execute REAL trades on your Alpaca account!")
        print("Make sure you understand the risks and have sufficient capital.")
    
    confirm = input("\nDo you want to continue? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        print("Trading cancelled.")
        return
    
    try:
        # Initialize trading engine
        print("\nInitializing trading engine...")
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        engine = LiveTradingEngine(
            api_key=api_key,
            secret_key=secret_key,
            initial_capital=initial_capital,
            paper=paper
        )
        
        # Test mode
        if args.test:
            print("\nRunning test cycle...")
            engine.run_trading_cycle()
            print("Test cycle completed.")
            return
        
        # Start live trading
        print("\nStarting live trading...")
        print("Press Ctrl+C to stop trading")
        print("-" * 80)
        
        engine.start_trading(check_interval=check_interval)
        
    except KeyboardInterrupt:
        print("\n\nTrading stopped by user")
    except Exception as e:
        print(f"\nError in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nTrading session ended.")
        print("Check the following files for results:")
        print("- live_portfolio_history.csv")
        print("- live_trade_history.csv")
        print("- live_performance_summary.json")
        print("- live_trading.log")

if __name__ == "__main__":
    main() 