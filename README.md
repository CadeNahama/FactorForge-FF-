# Sentiment-Driven Trading Pipeline

This project is a clean, modern pipeline for collecting, processing, and analyzing alternative data (Reddit, News, Twitter) for quantitative trading using sentiment analysis and NLP features.

## How to Download and Work With This Repo

1. **Clone the repository:**
    ```bash
    git clone https://github.com/CadeNahama/FactorForge-FF-.git
    cd FactorForge-FF-
    ```
2. **Set up your Python environment:**
    ```bash
    python3 -m venv altdata-venv
    source altdata-venv/bin/activate
    pip install -r requirements.txt  # or install dependencies manually
    ```
3. **Configure API keys:**
    - Edit `config/api_keys.yaml` with your Reddit, NewsAPI, and Twitter credentials.
4. **Collect data:**
    ```bash
    python data_collection/reddit_collector.py
    python data_collection/news_collector.py
    # python data_collection/twitter_collector.py  # if desired
    ```
5. **Add rolling sentiment:**
    ```bash
    python src/sentiment/rolling_features.py --input data/raw/news_TSLA_20250707_114023.csv --output data/raw/news_TSLA_rolling.csv --window 5 --group_col news_source_encoding --time_col published_at
    python src/sentiment/rolling_features.py --input data/raw/reddit_TSLA_20250707_114018.csv --output data/raw/reddit_TSLA_rolling.csv --window 5 --group_col subreddit_source --time_col created_utc
    ```
6. **Use enriched CSVs for ML, alpha models, or trading signals.**

## Roadmap & Next Steps

### 1. Data Collection & Ingestion
- [x] Reddit, News, Twitter collectors (basic)
- [ ] Expand to more sources (e.g., StockTwits, Discord, YouTube, financial news APIs)
- [ ] Add scheduling/automation (cron jobs, Airflow, etc.)
- [ ] Real-time streaming support

### 2. Text Preprocessing & NLP
- [x] spaCy-based cleaning, tokenization, lemmatization
- [ ] Language detection and filtering (for multi-lingual support)
- [ ] Advanced entity recognition (tickers, companies, people)
- [ ] Custom stopword lists and domain-specific cleaning

### 3. Sentiment Analysis
- [x] VADER sentiment scoring
- [ ] Experiment with transformer-based sentiment models (FinBERT, RoBERTa, etc.)
- [ ] Ensemble or hybrid sentiment scoring
- [ ] Sentiment calibration for financial context

### 4. Feature Engineering
- [x] Text length, keyword counts, all-caps, punctuation
- [x] Time features (hour, day, weekend)
- [x] Source encoding
- [x] Rolling sentiment (signal smoothing)
- [ ] More advanced features: topic modeling, volatility proxies, event detection
- [ ] Feature selection/importance analysis

### 5. Alpha Model Integration
- [ ] Use features in ML or rule-based trading models (XGBoost, LightGBM, logistic regression, etc.)
- [ ] Combine sentiment with price/volume/technical features
- [ ] Model explainability and backtesting

### 6. Backtest & Live Trading
- [ ] Integrate with a backtest/live trading engine (Zipline, Backtrader, custom)
- [ ] Simulate order execution, slippage, and transaction costs
- [ ] Live signal deployment and monitoring

### 7. Deployment & Automation
- [ ] Schedule collectors and pipeline (cron, Airflow, Prefect, etc.)
- [ ] Dockerize for reproducibility
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Monitoring, alerting, and logging

### 8. Collaboration & Scaling
- [ ] Modularize for team contributions
- [ ] Add unit/integration tests
- [ ] Documentation and code examples
- [ ] Open source community engagement

### Vision: Where We're Going
- Build a robust, extensible platform for alternative data-driven trading
- Incorporate state-of-the-art NLP and ML for alpha generation
- Enable rapid research, backtesting, and live deployment of new signals
- Foster a collaborative, open-source quant research community

## Features
- Collects data from Reddit, News, and Twitter (optional)
- Cleans and preprocesses text (spaCy)
- Computes sentiment scores (VADER)
- Extracts actionable NLP features (tokens, text length, keyword counts, all-caps, punctuation, etc.)
- Computes rolling sentiment for trend detection
- Outputs enriched CSVs for ML/alpha models or direct trading signals

## Directory Structure
- `src/sentiment/` — NLP, sentiment, and feature engineering modules
- `data_collection/` — Data collectors for Reddit, News, Twitter
- `data/raw/` — Output CSVs
- `config/` — API keys and config
- `altdata-venv/` — Python virtual environment

## Requirements
- Python 3.9+
- See `requirements.txt` for dependencies (spaCy, vaderSentiment, pandas, praw, pyyaml, etc.)

## Next Steps
- Integrate with your alpha model or trading engine
- Experiment with new features and alternative data sources
- Automate and deploy for live trading

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