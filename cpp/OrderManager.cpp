#include "OrderManager.h"
#include <iostream>
#include <chrono>
#include <ctime>

OrderManager::OrderManager() : orderCounter(0) {}
OrderManager::~OrderManager() {}

std::string OrderManager::sendOrder(const Order& order) {
    std::string orderId = "ORD" + std::to_string(++orderCounter);
    Order newOrder = order;
    newOrder.status = "new";
    // Add timestamp (for simulation)
    // In real HFT, connect here to broker/exchange API
    orders[orderId] = newOrder;
    std::cout << "Order sent: " << orderId << " (" << order.symbol << ", qty: " << order.quantity << ", side: " << order.side << ")" << std::endl;
    // Simulate immediate fill for market orders
    if (order.type == "market") {
        orders[orderId].status = "filled";
        std::cout << "Order filled: " << orderId << std::endl;
    }
    return orderId;
}

bool OrderManager::cancelOrder(const std::string& orderId) {
    auto it = orders.find(orderId);
    if (it != orders.end() && it->second.status == "new") {
        it->second.status = "cancelled";
        std::cout << "Order cancelled: " << orderId << std::endl;
        return true;
    }
    std::cout << "Cancel failed: Order not found or already processed." << std::endl;
    return false;
}

Order OrderManager::getOrderStatus(const std::string& orderId) {
    if (orders.count(orderId)) {
        return orders[orderId];
    }
    std::cout << "Order status: Not found for " << orderId << std::endl;
    return Order{};
}

std::vector<Order> OrderManager::getOpenOrders() const {
    std::vector<Order> openOrders;
    for (const auto& kv : orders) {
        if (kv.second.status == "new") {
            openOrders.push_back(kv.second);
        }
    }
    return openOrders;
} 