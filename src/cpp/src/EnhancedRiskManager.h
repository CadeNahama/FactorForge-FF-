#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>

// Modern, RAII-safe, modular EnhancedRiskManager interface
class EnhancedRiskManager {
public:
    // Configuration struct for risk parameters
    struct Config {
        double max_portfolio_risk = 0.13;
        double max_position_size = 0.05;
        // Extendable: add future config members here
    };

    explicit EnhancedRiskManager(const Config& config);
    ~EnhancedRiskManager();

    // Stateless, const-correct methods
    std::map<std::string, double> calculatePositionRisk(
        const std::string& ticker, double price, double volatility, double positionSize) const;

    std::map<std::string, double> calculatePortfolioRisk(
        const std::map<std::string, double>& positions,
        const std::map<std::string, double>& prices,
        const std::map<std::string, std::vector<double>>& returnsData) const;

    std::map<std::string, double> optimizePositionSizes(
        const std::map<std::string, double>& signals,
        const std::map<std::string, double>& prices,
        const std::map<std::string, std::vector<double>>& returnsData,
        double targetReturn) const;
    
    std::map<std::string, double> applyRiskLimits(
        const std::map<std::string, double>& positions,
        const std::map<std::string, double>& prices,
        const std::map<std::string, std::vector<double>>& returnsData) const;

    std::string generateRiskReport(
        const std::map<std::string, double>& positions,
        const std::map<std::string, double>& prices,
        const std::map<std::string, std::vector<double>>& returnsData) const;

    // Symmetric to Python: clamps signals and positions according to risk config
    std::map<std::string, double> clampSignals(
        const std::map<std::string, double>& signals,
        const std::map<std::string, double>& current_positions,
        double portfolio_value) const;

    // Factory for smart pointer safety
    static std::unique_ptr<EnhancedRiskManager> Create(const Config& config);

    // Accessors for config
    double getMaxPortfolioRisk() const { return config_.max_portfolio_risk; }
    double getMaxPositionSize() const { return config_.max_position_size; }

private:
    Config config_;
};