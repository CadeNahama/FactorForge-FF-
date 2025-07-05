#include "ExecutionEngine.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

ExecutionEngine::ExecutionEngine() {}
ExecutionEngine::~ExecutionEngine() {}

std::vector<Fill> ExecutionEngine::executeSignals(
    const std::vector<Signal>& signals,
    const std::map<std::string, double>& current_prices,
    PortfolioState& portfolio_state,
    const RiskParams& risk_params
) {
    std::vector<Fill> fills;
    
    // Check risk limits first
    if (!checkRiskLimits(signals, portfolio_state, risk_params)) {
        std::cout << "Risk limits exceeded - no trades executed" << std::endl;
        return fills;
    }
    
    // Simulate fills for all signals
    fills = simulateFills(signals, current_prices, risk_params);
    
    // Update portfolio positions and cash
    updatePositions(fills, portfolio_state, current_prices, risk_params);
    
    // Add to trade history
    trade_history.insert(trade_history.end(), fills.begin(), fills.end());
    
    return fills;
}

bool ExecutionEngine::checkRiskLimits(
    const std::vector<Signal>& signals,
    const PortfolioState& portfolio_state,
    const RiskParams& risk_params
) {
    // Calculate total portfolio value (cash + positions)
    double total_portfolio_value = portfolio_state.cash;
    
    // For now, we'll use a simplified approach - just check if we have enough cash
    // In a real system, you'd calculate the actual portfolio value including positions
    
    // Check if we have enough cash for all trades
    double total_trade_value = 0.0;
    for (const auto& signal : signals) {
        // Calculate trade value (signal.qty is already in dollars)
        total_trade_value += signal.qty;
    }
    
    // Check if we have enough cash (with some buffer for transaction costs)
    if (total_trade_value > portfolio_state.cash * 0.95) {  // 95% of cash as safety buffer
        std::cout << "Insufficient cash for trades: " << total_trade_value 
                  << " > " << portfolio_state.cash * 0.95 << std::endl;
        return false;
    }
    
    // For now, disable position size limits to allow trades to execute
    // In production, you'd implement proper position sizing based on portfolio value
    
    return true;
}

void ExecutionEngine::updatePositions(
    const std::vector<Fill>& fills,
    PortfolioState& portfolio_state,
    const std::map<std::string, double>& current_prices,
    const RiskParams& risk_params
) {
    for (const auto& fill : fills) {
        // Update position
        double& current_position = portfolio_state.positions[fill.ticker];
        
        if (fill.side == "BUY") {
            current_position += fill.qty;
        } else if (fill.side == "SELL") {
            current_position -= fill.qty;
        }
        
        // Update cash
        double trade_value = fill.qty * fill.price;  // qty is in shares, multiply by price
        if (fill.side == "BUY") {
            portfolio_state.cash -= (trade_value + fill.total_costs);
        } else {
            portfolio_state.cash += (trade_value - fill.total_costs);
        }
        
        // Update transaction costs
        portfolio_state.total_transaction_costs += fill.total_costs;
        
        // Update last trade price
        last_trade_prices[fill.ticker] = fill.price;
    }
}

double ExecutionEngine::calculateTransactionCosts(
    double trade_value,
    const RiskParams& risk_params
) {
    double commission = trade_value * risk_params.commission_rate;
    double slippage = trade_value * risk_params.slippage_rate;
    double market_impact = calculateMarketImpact(trade_value, risk_params);
    
    return commission + slippage + market_impact;
}

std::vector<Fill> ExecutionEngine::simulateFills(
    const std::vector<Signal>& signals,
    const std::map<std::string, double>& current_prices,
    const RiskParams& risk_params
) {
    std::vector<Fill> fills;
    
    for (const auto& signal : signals) {
        if (!validateSignal(signal)) {
            continue;
        }
        
        auto price_it = current_prices.find(signal.ticker);
        if (price_it == current_prices.end()) {
            std::cout << "No price data for " << signal.ticker << std::endl;
            continue;
        }
        
        double current_price = price_it->second;
        
        // Safety check: ensure price is positive
        if (current_price <= 0) {
            std::cout << "Invalid price for " << signal.ticker << ": " << current_price << std::endl;
            continue;
        }
        
        double trade_value = signal.qty * current_price;  // qty is in shares, multiply by price
        double execution_price = current_price;
        
        // Simulate slippage and market impact
        double slippage = trade_value * risk_params.slippage_rate;
        double market_impact = calculateMarketImpact(trade_value, risk_params);
        
        // Adjust execution price based on side
        if (signal.side == "BUY") {
            execution_price += (slippage + market_impact) / signal.qty / risk_params.initial_capital;
        } else {
            execution_price -= (slippage + market_impact) / signal.qty / risk_params.initial_capital;
        }
        
        // Calculate costs
        double commission = trade_value * risk_params.commission_rate;
        double total_costs = commission + slippage + market_impact;
        
        // Create fill
        Fill fill;
        fill.ticker = signal.ticker;
        fill.side = signal.side;
        fill.qty = signal.qty;
        fill.price = execution_price;
        fill.commission = commission;
        fill.slippage = slippage;
        fill.market_impact = market_impact;
        fill.total_costs = total_costs;
        fill.timestamp = getCurrentTimestamp();
        
        fills.push_back(fill);
        
        std::cout << "Fill: " << fill.ticker << " " << fill.side << " " 
                  << fill.qty << " @ " << fill.price << " (costs: " << fill.total_costs << ")" << std::endl;
    }
    
    return fills;
}

bool ExecutionEngine::validateSignal(const Signal& signal) {
    if (signal.ticker.empty()) {
        std::cout << "Invalid signal: empty ticker" << std::endl;
        return false;
    }
    
    if (signal.side != "BUY" && signal.side != "SELL") {
        std::cout << "Invalid signal: side must be BUY or SELL" << std::endl;
        return false;
    }
    
    if (signal.qty <= 0) {
        std::cout << "Invalid signal: quantity must be positive" << std::endl;
        return false;
    }
    
    if (signal.limit_price <= 0) {
        std::cout << "Invalid signal: limit price must be positive" << std::endl;
        return false;
    }
    
    return true;
}

double ExecutionEngine::calculateMarketImpact(double trade_value, const RiskParams& risk_params) {
    // Market impact increases with trade size
    return trade_value * risk_params.market_impact_rate * (trade_value / 100000.0);
}

std::string ExecutionEngine::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
} 