#pragma once
#include <string>
#include <map>

class RiskManager {
public:
    RiskManager(double maxPositionSize, double maxExposure);
    ~RiskManager();

    bool checkOrderRisk(const std::string& symbol, double quantity, double price, const std::map<std::string, double>& currentPositions);

private:
    double maxPositionSize;
    double maxExposure;
}; 