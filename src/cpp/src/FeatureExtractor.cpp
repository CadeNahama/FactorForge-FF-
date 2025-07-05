#include "FeatureExtractor.h"
#include <numeric>

FeatureExtractor::FeatureExtractor() {}
FeatureExtractor::~FeatureExtractor() {}

std::vector<double> FeatureExtractor::extractFeatures(const std::vector<double>& prices) {
    std::vector<double> features;
    if (prices.empty()) return features;
    // Example: simple moving average
    double sum = std::accumulate(prices.begin(), prices.end(), 0.0);
    features.push_back(sum / prices.size());
    // Add more features as needed
    return features;
} 