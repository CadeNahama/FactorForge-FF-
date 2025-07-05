#include "RiskManager.h"
#include <iostream>

RiskManager::RiskManager(double maxPositionSize_, double maxExposure_)
    : maxPositionSize(maxPositionSize_), maxExposure(maxExposure_) {}
RiskManager::~RiskManager() {}

bool RiskManager::checkOrderRisk(const std::string& symbol, double quantity, double price, const std::map<std::string, double>& currentPositions) {
    double newPosition = currentPositions.count(symbol) ? currentPositions.at(symbol) + quantity : quantity;
    if (std::abs(newPosition) > maxPositionSize) {
        std::cout << "Risk check failed: position size limit exceeded for " << symbol << std::endl;
        return false;
    }
    double totalExposure = 0.0;
    for (const auto& kv : currentPositions) {
        totalExposure += std::abs(kv.second * price);
    }
    totalExposure += std::abs(quantity * price);
    if (totalExposure > maxExposure) {
        std::cout << "Risk check failed: total exposure limit exceeded" << std::endl;
        return false;
    }
    return true;
} 