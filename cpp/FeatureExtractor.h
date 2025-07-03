#pragma once
#include <vector>

class FeatureExtractor {
public:
    FeatureExtractor();
    ~FeatureExtractor();

    std::vector<double> extractFeatures(const std::vector<double>& prices);
}; 