#pragma once
#include <string>
#include <vector>
#include <map>

class EnhancedRiskManager {
public:
    EnhancedRiskManager();
    ~EnhancedRiskManager();

    double max_portfolio_risk;
    double max_position_size;

    std::map<std::string, double> calculatePositionRisk(const std::string& ticker, double price, double volatility, double positionSize);
    std::map<std::string, double> calculatePortfolioRisk(const std::map<std::string, double>& positions, const std::map<std::string, double>& prices, const std::map<std::string, std::vector<double>>& returnsData);
    std::map<std::string, double> optimizePositionSizes(const std::map<std::string, double>& signals, const std::map<std::string, double>& prices, const std::map<std::string, std::vector<double>>& returnsData, double targetReturn);
    std::map<std::string, double> applyRiskLimits(const std::map<std::string, double>& positions, const std::map<std::string, double>& prices, const std::map<std::string, std::vector<double>>& returnsData);
    std::string generateRiskReport(const std::map<std::string, double>& positions, const std::map<std::string, double>& prices, const std::map<std::string, std::vector<double>>& returnsData);
}; 