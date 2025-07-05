#include "SignalGenerator.h"
#include <vector>

SignalGenerator::SignalGenerator() {}
SignalGenerator::~SignalGenerator() {}

// Generate signal: 1 = buy, -1 = sell, 0 = hold
int SignalGenerator::generateSignal(const std::vector<double>& features) {
    if (features.empty()) return 0;
    // Assume features[2] is momentum (from AdvancedFeatures)
    double threshold = 0.5; // Example threshold
    double momentum = features.size() > 2 ? features[2] : 0.0;
    if (momentum > threshold) return 1;
    if (momentum < -threshold) return -1;
    return 0;
    // TODO: Expand to use more features and ML models
} 