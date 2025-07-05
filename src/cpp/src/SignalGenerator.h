#pragma once
#include <vector>

class SignalGenerator {
public:
    SignalGenerator();
    ~SignalGenerator();

    int generateSignal(const std::vector<double>& features); // -1 = sell, 0 = hold, 1 = buy
}; 