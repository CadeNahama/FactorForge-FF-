#pragma once
#include <string>
#include <vector>
#include <map>

class LiveTradingEngine {
public:
    LiveTradingEngine(double initialCapital, bool paper);
    ~LiveTradingEngine();

    void startTrading();
    void stopTrading();
    void runTradingCycle();
    std::map<std::string, double> getMarketData();
    std::map<std::string, double> calculateSignals(const std::map<std::string, double>& data);
    void executeTrades(const std::map<std::string, double>& signals);
    void applyRiskManagement();
    void saveTradingResults();
    std::map<std::string, double> getTradingStatus();

private:
    double initialCapital;
    bool paper;
    // Add other members as needed
}; 