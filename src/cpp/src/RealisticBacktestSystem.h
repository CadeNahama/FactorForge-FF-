#pragma once
#include <string>
#include <vector>
#include <map>

class RealisticBacktestSystem {
public:
    RealisticBacktestSystem(double initialCapital);
    ~RealisticBacktestSystem();

    void runBacktest();
    std::map<std::string, double> generateSignals(const std::map<std::string, std::vector<double>>& data, const std::string& currentDate);
}; 