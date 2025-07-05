#pragma once
#include <string>
#include <vector>
#include <map>

struct Order {
    std::string symbol;
    double quantity;
    double price;
    std::string side; // "buy" or "sell"
    std::string type; // "market" or "limit"
    std::string status; // "new", "filled", etc.
};

class OrderManager {
public:
    OrderManager();
    ~OrderManager();

    std::string sendOrder(const Order& order);
    bool cancelOrder(const std::string& orderId);
    Order getOrderStatus(const std::string& orderId);
    std::vector<Order> getOpenOrders() const;

private:
    std::map<std::string, Order> orders;
    int orderCounter;
}; 