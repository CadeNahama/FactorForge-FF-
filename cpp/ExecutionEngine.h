#pragma once
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

// Signal structure from Python
struct Signal {
    std::string ticker;
    std::string side;  // "BUY" or "SELL"
    double qty;        // Target quantity (fraction of portfolio)
    double limit_price;
};

// Fill structure returned to Python
struct Fill {
    std::string ticker;
    std::string side;
    double qty;
    double price;
    double commission;
    double slippage;
    double market_impact;
    double total_costs;
    std::string timestamp;
};

// Portfolio state
struct PortfolioState {
    double cash;
    std::map<std::string, double> positions;  // ticker -> position (fraction of portfolio)
    double total_transaction_costs;
};

// Risk parameters
struct RiskParams {
    double max_position_size;
    double max_portfolio_risk;
    double commission_rate;
    double slippage_rate;
    double market_impact_rate;
    double initial_capital;
};

class ExecutionEngine {
public:
    ExecutionEngine();
    ~ExecutionEngine();

    // Main execution method - receives signals from Python and returns fills
    std::vector<Fill> executeSignals(
        const std::vector<Signal>& signals,
        const std::map<std::string, double>& current_prices,
        PortfolioState& portfolio_state,
        const RiskParams& risk_params
    );

    // Risk management methods
    bool checkRiskLimits(
        const std::vector<Signal>& signals,
        const PortfolioState& portfolio_state,
        const RiskParams& risk_params
    );

    // Position management
    void updatePositions(
        const std::vector<Fill>& fills,
        PortfolioState& portfolio_state,
        const std::map<std::string, double>& current_prices,
        const RiskParams& risk_params
    );

    // Utility methods
    double calculateTransactionCosts(
        double trade_value,
        const RiskParams& risk_params
    );

    std::vector<Fill> simulateFills(
        const std::vector<Signal>& signals,
        const std::map<std::string, double>& current_prices,
        const RiskParams& risk_params
    );

private:
    // Internal state
    std::map<std::string, double> last_trade_prices;
    std::vector<Fill> trade_history;
    
    // Helper methods
    bool validateSignal(const Signal& signal);
    double calculateMarketImpact(double trade_value, const RiskParams& risk_params);
    std::string getCurrentTimestamp();
}; 