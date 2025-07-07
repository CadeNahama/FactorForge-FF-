# Quantitative Trading Algorithm - Modular Backtesting System

A sophisticated quantitative trading system with modular architecture, featuring walk-forward backtesting, multi-factor alpha models, and realistic portfolio simulation.

## 🏗️ Architecture

### Modular Design
The system is built with a modular architecture that separates concerns and allows for easy component swapping:

```
src/py/
├── common/                    # Core modules
│   ├── alpha_models.py       # Signal generation
│   ├── risk_evaluator.py     # Risk management
│   ├── execution_handlers.py # Order execution
│   ├── report_generator.py   # Performance reporting
│   └── data_pipeline.py      # Data loading
├── backtest/                 # Backtesting engine
│   └── engine_modular_backtest.py
└── live/                     # Live trading (future)
```

### Core Components

#### 1. AlphaModel (AlphaModelRealistic)
- **Multi-factor signal generation** using momentum, mean reversion, and correlation
- **Momentum signals:** 5-day, 20-day, 60-day returns + RSI
- **Mean reversion:** Bollinger Bands, Z-score, trend strength
- **Correlation:** Rolling correlation with SPY for diversification
- **ML-ready:** Scaffolded for future machine learning integration

#### 2. RiskEvaluator
- **Position sizing** controls
- **Price validation** and liquidity checks
- **Expandable** for stop-loss, drawdown limits, etc.

#### 3. ExecutionHandler (BasicSimulatedExecutionHandler)
- **Simulated order execution** with mock fills
- **Position tracking** and cash management
- **Ready for real execution** integration

#### 4. ReportGenerator (BasicReportGenerator)
- **Trade logging** and portfolio tracking
- **Performance metrics** calculation
- **Summary reporting** with key statistics

#### 5. DataPipeline
- **Historical data loading** via EnhancedDataLoader
- **Multi-symbol support** with flexible date ranges
- **Data preprocessing** and validation

## 🚀 Features

### Walk-Forward Backtesting
- **Day-by-day simulation** using only historical data available at each point
- **Realistic portfolio tracking** with positions, cash, and P&L
- **Performance metrics** including Sharpe ratio, drawdown, and total return

### Multi-Factor Alpha Model
- **Momentum strategies** across multiple timeframes
- **Mean reversion** using statistical indicators
- **Correlation-based diversification** with market index
- **Volatility adjustment** for signal scaling

### Configuration-Driven
- **YAML configuration** for all parameters
- **Easy parameter tuning** without code changes
- **Flexible symbol lists** and date ranges

## 📊 Performance Metrics

The system calculates and reports:
- **Total Return** - Overall portfolio performance
- **Sharpe Ratio** - Risk-adjusted returns
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Trade Statistics** - Number of trades, buy/sell distribution

## 🛠️ Installation & Setup

### Prerequisites
```bash
  pip install -r python/requirements.txt
  ```

### Configuration
1. Copy `config/config.yaml` and modify parameters:
   ```yaml
   start_date: '2022-01-01'
   end_date: '2022-12-31'
   symbols:
     - AAPL
     - MSFT
     - SPY
   initial_capital: 1000000
   ```

### Running the Backtest
```bash
PYTHONPATH=src/py python src/py/backtest/engine_modular_backtest.py
```

## 📈 Example Output

```
🚀 Starting Walk-Forward Backtest Simulation
==================================================
📊 Loading market data...
✓ Data loaded: 251 days, 15 columns
🧠 Initializing alpha model...
🔄 Running walk-forward simulation...
  Processed 251/251 days...

📈 Generating performance report...
Total trades: 343
Buys: 124
Sells: 219
Tickers traded: MSFT, AAPL

📊 PERFORMANCE METRICS:
   Initial Capital: $1,000,000.00
   Final Portfolio Value: $810,432.53
   Total Return: -18.96%
   Sharpe Ratio: -2.09
   Maximum Drawdown: -21.73%
   Total Trading Days: 251
```

## 🔧 Customization

### Adding New Alpha Factors
1. Extend `AlphaModelRealistic` in `src/py/common/alpha_models.py`
2. Add new signal calculation methods
3. Update signal combination weights

### Implementing New Risk Rules
1. Extend `RiskEvaluator` in `src/py/common/risk_evaluator.py`
2. Add new risk checks in the `evaluate` method
3. Configure risk parameters in `config.yaml`

### Adding Real Execution
1. Implement new `ExecutionHandler` class
2. Add real broker API integration
3. Update the backtest engine to use real execution

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python tests/test_cpp/test_cpp_integration.py
```

## 📚 Learning Path

This project demonstrates:
1. **Modular software architecture** for trading systems
2. **Walk-forward backtesting** methodology
3. **Multi-factor alpha modeling**
4. **Risk management** implementation
5. **Performance evaluation** and reporting

## 🔮 Future Enhancements

### Planned Features
- **Machine Learning integration** for signal generation
- **Real-time data processing** for live trading
- **Advanced risk management** (VaR, stress testing)
- **Alternative data sources** (sentiment, news, weather)
- **Multi-asset support** (options, futures, crypto)

### Alternative Data Integration
- **Social media sentiment** analysis
- **News sentiment** processing
- **Weather data** impact modeling
- **Satellite imagery** analysis

## 📄 License

This project is for educational and research purposes. Use at your own risk.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## ⚠️ Disclaimer

This software is for educational purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss and is not suitable for all investors.

## 📞 Contact

For questions or contributions, please open an issue on GitHub. 