# Quantitative Trading System: FactorForge-FF

## Project Overview
FactorForge-FF is an institutional-grade quantitative trading platform designed for both robust backtesting and live trading. It combines Python orchestration with high-performance C++ execution for latency-critical components, enabling realistic simulation and live deployment of advanced trading strategies.

## Features
- **Realistic Backtesting**: Institutional-grade backtest engine with rolling validation, Monte Carlo simulation, and advanced risk controls.
- **Live Trading**: Seamless integration with Alpaca for real-time trading, including position sync and robust risk management.
- **C++ Integration**: Latency-critical execution and risk modules are implemented in C++ for speed, exposed to Python via bindings.
- **Advanced Risk Management**: Position sizing, stop loss, take profit, trailing stops, max drawdown, daily loss limits, and more.
- **Signal Generation**: Multi-factor signals (momentum, mean reversion, ML, volatility, volume, correlation, regime detection).
- **Comprehensive Reporting**: Automated performance, risk, and trade statistics with CSV/JSON export.
- **Extensible Design**: Modular architecture for easy strategy and component extension.

## Installation
### Prerequisites
- Python 3.10+
- C++17 compiler (for native modules)
- [Alpaca API account](https://alpaca.markets/)
- [Alpaca SDK](https://github.com/alpacahq/alpaca-py):
  ```sh
  pip install alpaca-py
  ```
- Other Python dependencies:
  ```sh
  pip install -r python/requirements.txt
  ```

### C++ Build (if modifying C++ code)
1. Navigate to `cpp/`:
   ```sh
   cd cpp
   mkdir build && cd build
   cmake ..
   make
   ```
2. Ensure the resulting `.so` file is in `python/core/` for Python to import.

### Alpaca API Setup
- Set your API keys as environment variables:
  ```sh
  export ALPACA_API_KEY='your_api_key'
  export ALPACA_SECRET_KEY='your_secret_key'
  ```

## Usage
### Backtesting
Run the realistic backtest system:
```sh
python3 python/backtest/realistic_backtest_system_v2.py
```
- Results are saved to `backtest_results_v2.json`, `portfolio_history_v2.csv`, and `trade_history_v2.csv`.

### Live Trading
Run the live trading engine (paper or live):
```sh
cd python/live
python3 run_live_trading.py --capital 100000 --interval 60
```
- Use `--live` for live trading (default is paper).
- Results are saved to `live_portfolio_history.csv`, `live_trade_history.csv`, and `live_performance_summary.json`.

## Configuration
- **Backtest parameters**: Set in `python/backtest/realistic_backtest_system_v2.py` or via class arguments.
- **Live trading parameters**: Set in `python/live/live_trading_engine.py` or via command-line args.
- **C++ risk/exec parameters**: See `cpp/` headers and Python bindings.

## File Structure
```
NEW-Quant-Algo/
├── cpp/                  # C++ core modules (execution, risk, bindings)
├── python/
│   ├── backtest/         # Backtesting systems
│   ├── core/             # Data loader, ML, C++ bindings
│   ├── live/             # Live trading engine, Alpaca interface
│   └── requirements.txt  # Python dependencies
├── data/                 # Data files (if any)
├── *history*.csv/json    # Output files
├── README.md             # This file
└── ...
```

## Troubleshooting
- **Alpaca SDK Import Errors**: Ensure `alpaca-py` is installed and Python version is compatible.
- **C++ Import Errors**: Rebuild the C++ module and check `.so` location.
- **API Key Issues**: Double-check environment variables.
- **Trading Skips Cycles**: Ensure rebalance interval logic is correct (now fixed to use seconds).
- **Data Issues**: Check data availability and format in `EnhancedDataLoader`.

## License
MIT License. See `LICENSE` file for details.

## Contact
For questions, issues, or contributions:
- GitHub: [CadeNahama/FactorForge-FF-](https://github.com/CadeNahama/FactorForge-FF-)
- Email: [your-email@example.com] 