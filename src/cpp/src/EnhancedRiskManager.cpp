#include "EnhancedRiskManager.h"
#include <cmath>
#include <numeric>

EnhancedRiskManager::EnhancedRiskManager(const Config& config)
    : config_(config) {}

EnhancedRiskManager::~EnhancedRiskManager() {}

// Calculate individual position risk metrics (stateless, const)
std::map<std::string, double> EnhancedRiskManager::calculatePositionRisk(
    const std::string& ticker, double price, double volatility, double positionSize) const {
    std::map<std::string, double> riskMetrics;
    // Dollar risk
    riskMetrics["dollar_risk"] = positionSize * price * volatility;
    // VaR (Value at Risk) - 95% confidence (z = -1.645)
    double var_quantile = -1.645;
    riskMetrics["var"] = positionSize * price * var_quantile * volatility;
    // Expected Shortfall (Conditional VaR)
    double pdf = 0.103; // N(-1.645) ~ 0.103
    riskMetrics["expected_shortfall"] = positionSize * price * volatility * pdf / 0.05;
    // Max loss
    riskMetrics["max_loss"] = positionSize * price;
    // Beta risk placeholder
    riskMetrics["beta_risk"] = riskMetrics["dollar_risk"];
    return riskMetrics;
}

// Calculate portfolio-level risk metrics (simple volatility, VaR, ES)
std::map<std::string, double> EnhancedRiskManager::calculatePortfolioRisk(
    const std::map<std::string, double>& positions,
    const std::map<std::string, double>& prices,
    const std::map<std::string, std::vector<double>>& returnsData) const {
    std::map<std::string, double> portfolioRisk;
    if (positions.empty()) return portfolioRisk;
    // Calculate portfolio value and weights
    double totalValue = 0.0;
    for (const auto& kv : positions) {
        totalValue += std::abs(kv.second * prices.at(kv.first));
    }
    std::map<std::string, double> weights;
    for (const auto& kv : positions) {
        weights[kv.first] = (kv.second * prices.at(kv.first)) / totalValue;
    }
    // Calculate portfolio variance (diagonal only, for simplicity)
    double portfolioVar = 0.0;
    for (const auto& kv : weights) {
        const auto& ticker = kv.first;
        double w = kv.second;
        const auto& rets = returnsData.at(ticker);
        double mean = std::accumulate(rets.begin(), rets.end(), 0.0) / rets.size();
        double var = 0.0;
        for (double r : rets) var += (r - mean) * (r - mean);
        var /= rets.size();
        portfolioVar += w * w * var;
    }
    double portfolioVol = std::sqrt(portfolioVar);
    // Portfolio risk metrics
    portfolioRisk["total_risk"] = portfolioVol;
    portfolioRisk["volatility"] = portfolioVol;
    portfolioRisk["var"] = -1.645 * portfolioVol;
    portfolioRisk["expected_shortfall"] = 0.103 * portfolioVol / 0.05;
    return portfolioRisk;
}

// Placeholder stubs (const)
std::map<std::string, double> EnhancedRiskManager::optimizePositionSizes(
    const std::map<std::string, double>&, const std::map<std::string, double>&,
    const std::map<std::string, std::vector<double>>&, double) const { return {}; }

std::map<std::string, double> EnhancedRiskManager::applyRiskLimits(
    const std::map<std::string, double>&, const std::map<std::string, double>&,
    const std::map<std::string, std::vector<double>>&) const { return {}; }

std::string EnhancedRiskManager::generateRiskReport(
    const std::map<std::string, double>&, const std::map<std::string, double>&,
    const std::map<std::string, std::vector<double>>&) const { return ""; }

// Symmetric interface: clamp signals/positions to risk config
std::map<std::string, double> EnhancedRiskManager::clampSignals(
    const std::map<std::string, double>& signals,
    const std::map<std::string, double>& current_positions,
    double portfolio_value) const {
    std::map<std::string, double> output;
    int open_positions = 0;
    for (const auto& kv : current_positions) {
        if (std::abs(kv.second) > 0.0) open_positions += 1;
    }
    for (const auto& kv : signals) {
        const std::string& ticker = kv.first;
        double signal = kv.second;
        // Clamp position sizing to max allowed per config
        if (open_positions >= 20 && current_positions.find(ticker) == current_positions.end())
            continue; // max open positions mimic
        double value = std::abs(signal) * portfolio_value;
        if (value > config_.max_position_size * portfolio_value)
            signal = (signal > 0 ? 1.0 : -1.0) * config_.max_position_size;
        output[ticker] = signal;
    }
    return output;
}

// Smart pointer factory
std::unique_ptr<EnhancedRiskManager> EnhancedRiskManager::Create(const Config& config) {
    return std::unique_ptr<EnhancedRiskManager>(new EnhancedRiskManager(config));
}