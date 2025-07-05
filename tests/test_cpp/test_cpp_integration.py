#!/usr/bin/env python3
"""
Test script to verify C++ execution engine integration
"""

import sys
sys.path.append('.')

try:
    import quant_cpp
    from quant_cpp import ExecutionEngine, Signal, Fill, PortfolioState, RiskParams
    print("✅ C++ modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import C++ modules: {e}")
    sys.exit(1)

def test_cpp_integration():
    """Test the C++ execution engine with simple signals"""
    
    print("\n🧪 Testing C++ Execution Engine...")
    
    # Initialize C++ components
    execution_engine = ExecutionEngine()
    portfolio_state = PortfolioState()
    portfolio_state.cash = 100000.0
    portfolio_state.positions = {}
    portfolio_state.total_transaction_costs = 0.0
    
    risk_params = RiskParams()
    risk_params.max_position_size = 0.05
    risk_params.max_portfolio_risk = 0.02
    risk_params.commission_rate = 0.001
    risk_params.slippage_rate = 0.0005
    risk_params.market_impact_rate = 0.0001
    risk_params.initial_capital = 100000.0
    
    # Create test signals
    signals = []
    
    # Signal 1: Buy AAPL
    signal1 = Signal()
    signal1.ticker = "AAPL"
    signal1.side = "BUY"
    signal1.qty = 0.02  # 2% of portfolio
    signal1.limit_price = 150.0
    signals.append(signal1)
    
    # Signal 2: Sell MSFT
    signal2 = Signal()
    signal2.ticker = "MSFT"
    signal2.side = "SELL"
    signal2.qty = 0.01  # 1% of portfolio
    signal2.limit_price = 300.0
    signals.append(signal2)
    
    # Current prices
    current_prices = {
        "AAPL": 150.0,
        "MSFT": 300.0
    }
    
    print(f"📊 Initial portfolio state:")
    print(f"   Cash: ${portfolio_state.cash:,.2f}")
    print(f"   Positions: {portfolio_state.positions}")
    
    print(f"\n📈 Executing {len(signals)} signals...")
    for i, signal in enumerate(signals):
        print(f"   Signal {i+1}: {signal.ticker} {signal.side} {signal.qty:.3f} @ ${signal.limit_price:.2f}")
    
    # Execute signals
    fills = execution_engine.executeSignals(signals, current_prices, portfolio_state, risk_params)
    
    print(f"\n✅ Execution completed!")
    print(f"   Fills generated: {len(fills)}")
    
    for i, fill in enumerate(fills):
        print(f"   Fill {i+1}: {fill.ticker} {fill.side} {fill.qty:.3f} @ ${fill.price:.2f}")
        print(f"      Costs: commission=${fill.commission:.2f}, slippage=${fill.slippage:.2f}, total=${fill.total_costs:.2f}")
    
    print(f"\n📊 Final portfolio state:")
    print(f"   Cash: ${portfolio_state.cash:,.2f}")
    print(f"   Positions: {dict(portfolio_state.positions)}")
    print(f"   Total transaction costs: ${portfolio_state.total_transaction_costs:.2f}")
    
    # Test risk limits
    print(f"\n🔒 Testing risk limits...")
    
    # Try to exceed position limit
    large_signal = Signal()
    large_signal.ticker = "AAPL"
    large_signal.side = "BUY"
    large_signal.qty = 0.10  # 10% - exceeds 5% limit
    large_signal.limit_price = 150.0
    
    risk_check = execution_engine.checkRiskLimits([large_signal], portfolio_state, risk_params)
    print(f"   Risk check for large position: {'❌ REJECTED' if not risk_check else '✅ ACCEPTED'}")
    
    return True

if __name__ == "__main__":
    success = test_cpp_integration()
    if success:
        print("\n🎉 C++ integration test completed successfully!")
    else:
        print("\n❌ C++ integration test failed!")
        sys.exit(1) 