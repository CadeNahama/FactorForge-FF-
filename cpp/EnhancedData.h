#pragma once
#include <string>
#include <vector>
#include <map>

class EnhancedData {
public:
    EnhancedData();
    ~EnhancedData();

    std::map<std::string, std::vector<double>> downloadData(const std::vector<std::string>& tickers, const std::string& startDate, const std::string& endDate);
    std::vector<std::string> getSP500Tickers();
    std::vector<std::string> getETFUniverse();
    std::map<std::string, std::vector<double>> getMarketData(const std::vector<std::string>& tickers, const std::string& startDate, const std::string& endDate);
    std::map<std::string, std::vector<double>> calculateReturns(const std::map<std::string, std::vector<double>>& data, const std::vector<std::string>& tickers);
    double getRiskFreeRate(const std::string& startDate, const std::string& endDate);
}; 