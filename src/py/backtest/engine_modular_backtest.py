from common.alpha_models import AlphaModelRealistic
from common.risk_evaluator import RiskEvaluator
from common.execution_handlers import BasicSimulatedExecutionHandler
from common.report_generator import BasicReportGenerator
from common.data_pipeline import DataPipeline
import yaml
import os
import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, config):
        # Initialize all modular components
        self.data_pipeline = DataPipeline()
        self.alpha_model = AlphaModelRealistic()
        self.risk_evaluator = RiskEvaluator()
        self.execution_handler = BasicSimulatedExecutionHandler()
        self.report_generator = BasicReportGenerator()
        self.config = config
        
        # Portfolio state
        self.initial_capital = config.get('initial_capital', 1000000)
        self.cash = self.initial_capital
        self.positions = {}  # {ticker: quantity}
        self.portfolio_history = []
        self.daily_returns = []

    def run(self):
        """Run walk-forward backtest simulation"""
        print("🚀 Starting Walk-Forward Backtest Simulation")
        print("=" * 50)
        
        # Load data
        print("📊 Loading market data...")
        data = self.data_pipeline.load_data(
            self.config['start_date'],
            self.config['end_date'],
            self.config['symbols'],
            self.config['bar_size']
        )
        
        if data is None or data.empty:
            print("❌ No data loaded, exiting.")
            return
        
        print(f"✓ Data loaded: {data.shape[0]} days, {data.shape[1]} columns")
        
        # Initialize alpha model with initial data
        print("🧠 Initializing alpha model...")
        self.alpha_model.fit(data)
        
        # Walk-forward simulation
        print("🔄 Running walk-forward simulation...")
        trading_days = data.index.sort_values()
        
        for i, current_date in enumerate(trading_days):
            # Get historical data up to current date (exclusive)
            historical_data = data.loc[trading_days[:i+1]]
            current_data = data.loc[current_date:current_date]
            
            # Generate signals using only historical data
            signals = self.alpha_model.predict(historical_data)
            
            # Get current prices
            current_prices = self._get_current_prices(current_data)
            
            # Risk evaluation
            adjusted_signals = self.risk_evaluator.evaluate(
                self.positions, signals, current_data
            )
            
            # Execute trades
            execution_results = self.execution_handler.execute(adjusted_signals)
            
            # Update portfolio state
            self._update_portfolio(execution_results, current_prices, current_date)
            
            # Log trades
            for trade in execution_results:
                self.report_generator.log_trade(trade)
            
            # Record portfolio state
            self._record_portfolio_state(current_date, current_prices)
            
            # Progress indicator
            if i % 50 == 0:
                print(f"  Processed {i+1}/{len(trading_days)} days...")
        
        # Generate final report
        print("\n📈 Generating performance report...")
        summary = self.report_generator.generate_summary()
        print(summary)
        
        # Calculate and display performance metrics
        self._calculate_performance_metrics()
        
        return {
            'portfolio_history': self.portfolio_history,
            'final_portfolio_value': self._calculate_portfolio_value(current_prices),
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown
        }

    def _get_current_prices(self, current_data):
        """Extract current prices for all symbols"""
        prices = {}
        for symbol in self.config['symbols']:
            close_col = f"{symbol}_Close"
            if close_col in current_data.columns:
                price = current_data[close_col].iloc[0]
                if pd.notna(price) and price > 0:
                    prices[symbol] = price
        return prices

    def _update_portfolio(self, execution_results, current_prices, current_date):
        """Update portfolio positions and cash based on executed trades"""
        for trade in execution_results:
            ticker = trade['ticker']
            action = trade['action']
            quantity = trade['quantity']
            price = trade['price']
            
            # Calculate trade value
            trade_value = quantity * price
            
            # Update positions
            if action == 'buy':
                if ticker in self.positions:
                    self.positions[ticker] += quantity
                else:
                    self.positions[ticker] = quantity
                self.cash -= trade_value
            else:  # sell
                if ticker in self.positions:
                    self.positions[ticker] -= quantity
                    if self.positions[ticker] == 0:
                        del self.positions[ticker]
                else:
                    self.positions[ticker] = -quantity
                self.cash += trade_value

    def _record_portfolio_state(self, current_date, current_prices):
        """Record current portfolio state for historical tracking"""
        portfolio_value = self._calculate_portfolio_value(current_prices)
        
        self.portfolio_history.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy()
        })
        
        # Calculate daily return
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)

    def _calculate_portfolio_value(self, current_prices):
        """Calculate total portfolio value (cash + positions)"""
        position_value = 0
        for ticker, quantity in self.positions.items():
            if ticker in current_prices:
                position_value += quantity * current_prices[ticker]
        return self.cash + position_value

    def _calculate_performance_metrics(self):
        """Calculate key performance metrics"""
        if len(self.portfolio_history) < 2:
            return
        
        # Total return
        initial_value = self.portfolio_history[0]['portfolio_value']
        final_value = self.portfolio_history[-1]['portfolio_value']
        self.total_return = (final_value - initial_value) / initial_value
        
        # Sharpe ratio
        if len(self.daily_returns) > 0:
            returns_array = np.array(self.daily_returns)
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            self.sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            self.sharpe_ratio = 0
        
        # Maximum drawdown
        portfolio_values = [entry['portfolio_value'] for entry in self.portfolio_history]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        self.max_drawdown = np.min(drawdown)
        
        # Print metrics
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Initial Capital: ${initial_value:,.2f}")
        print(f"   Final Portfolio Value: ${final_value:,.2f}")
        print(f"   Total Return: {self.total_return:.2%}")
        print(f"   Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"   Maximum Drawdown: {self.max_drawdown:.2%}")
        print(f"   Total Trading Days: {len(self.portfolio_history)}")


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config from YAML file
    config_path = os.path.join(os.path.dirname(__file__), '../../../config/config.yaml')
    config = load_config(config_path)
    engine = BacktestEngine(config)
    results = engine.run()

if __name__ == "__main__":
    main() 