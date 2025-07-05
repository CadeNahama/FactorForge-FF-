#include "OrderManager.h"
#include "RiskManager.h"
#include "FeatureExtractor.h"
#include "SignalGenerator.h"
#include <iostream>
#include <vector>
#include <map>

int main() {
    // Example price data
    std::vector<double> prices = {100.0, 101.0, 102.0, 103.0, 104.0};

    // Feature extraction
    FeatureExtractor fe;
    std::vector<double> features = fe.extractFeatures(prices);

    // Signal generation
    SignalGenerator sg;
    int signal = sg.generateSignal(features);
    std::cout << "Signal: " << signal << std::endl;

    // Risk management
    RiskManager rm(100, 10000); // max 100 shares per symbol, $10,000 exposure
    std::map<std::string, double> positions = {{"AAPL", 50}};
    double price = 104.0;
    double qty = 10.0 * signal; // buy/sell 10 shares if signal
    if (signal != 0 && rm.checkOrderRisk("AAPL", qty, price, positions)) {
        // Order management
        OrderManager om;
        Order order = {"AAPL", qty, price, signal > 0 ? "buy" : "sell", "market", ""};
        std::string orderId = om.sendOrder(order);
        std::cout << "Order ID: " << orderId << std::endl;
    } else {
        std::cout << "No trade or risk check failed." << std::endl;
    }
    return 0;
} 