#pragma once
#include <string>
#include <vector>
#include <map>

class AdvancedFeatures {
public:
    AdvancedFeatures();
    ~AdvancedFeatures();

    std::map<std::string, double> extractTechnicalFeatures(const std::vector<double>& prices, const std::vector<double>& volumes);
    std::map<std::string, double> extractAlternativeDataFeatures(const std::string& ticker);
    std::map<std::string, double> extractMacroeconomicFeatures();
    std::vector<std::string> selectOptimalFeatures(const std::map<std::string, std::vector<double>>& featureMatrix, const std::vector<double>& target, int topN);
}; 