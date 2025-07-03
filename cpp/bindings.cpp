#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "OrderManager.h"
#include "EnhancedRiskManager.h"
#include "AdvancedFeatures.h"
#include "SignalGenerator.h"
#include "ExecutionEngine.h"

namespace py = pybind11;

PYBIND11_MODULE(quant_cpp, m) {
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("symbol", &Order::symbol)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("price", &Order::price)
        .def_readwrite("side", &Order::side)
        .def_readwrite("type", &Order::type)
        .def_readwrite("status", &Order::status);

    py::class_<OrderManager>(m, "OrderManager")
        .def(py::init<>())
        .def("sendOrder", &OrderManager::sendOrder)
        .def("cancelOrder", &OrderManager::cancelOrder)
        .def("getOrderStatus", &OrderManager::getOrderStatus)
        .def("getOpenOrders", &OrderManager::getOpenOrders);

    py::class_<EnhancedRiskManager>(m, "EnhancedRiskManager")
        .def(py::init<>())
        .def("calculatePositionRisk", &EnhancedRiskManager::calculatePositionRisk)
        .def("calculatePortfolioRisk", &EnhancedRiskManager::calculatePortfolioRisk)
        .def("applyRiskLimits", &EnhancedRiskManager::applyRiskLimits)
        .def_readwrite("max_portfolio_risk", &EnhancedRiskManager::max_portfolio_risk)
        .def_readwrite("max_position_size", &EnhancedRiskManager::max_position_size);

    py::class_<AdvancedFeatures>(m, "AdvancedFeatures")
        .def(py::init<>())
        .def("extractTechnicalFeatures", &AdvancedFeatures::extractTechnicalFeatures);

    py::class_<SignalGenerator>(m, "SignalGenerator")
        .def(py::init<>())
        .def("generateSignal", &SignalGenerator::generateSignal);

    // New ExecutionEngine bindings
    py::class_<Signal>(m, "Signal")
        .def(py::init<>())
        .def_readwrite("ticker", &Signal::ticker)
        .def_readwrite("side", &Signal::side)
        .def_readwrite("qty", &Signal::qty)
        .def_readwrite("limit_price", &Signal::limit_price);

    py::class_<Fill>(m, "Fill")
        .def(py::init<>())
        .def_readwrite("ticker", &Fill::ticker)
        .def_readwrite("side", &Fill::side)
        .def_readwrite("qty", &Fill::qty)
        .def_readwrite("price", &Fill::price)
        .def_readwrite("commission", &Fill::commission)
        .def_readwrite("slippage", &Fill::slippage)
        .def_readwrite("market_impact", &Fill::market_impact)
        .def_readwrite("total_costs", &Fill::total_costs)
        .def_readwrite("timestamp", &Fill::timestamp);

    py::class_<PortfolioState>(m, "PortfolioState")
        .def(py::init<>())
        .def_readwrite("cash", &PortfolioState::cash)
        .def_readwrite("positions", &PortfolioState::positions)
        .def_readwrite("total_transaction_costs", &PortfolioState::total_transaction_costs);

    py::class_<RiskParams>(m, "RiskParams")
        .def(py::init<>())
        .def_readwrite("max_position_size", &RiskParams::max_position_size)
        .def_readwrite("max_portfolio_risk", &RiskParams::max_portfolio_risk)
        .def_readwrite("commission_rate", &RiskParams::commission_rate)
        .def_readwrite("slippage_rate", &RiskParams::slippage_rate)
        .def_readwrite("market_impact_rate", &RiskParams::market_impact_rate)
        .def_readwrite("initial_capital", &RiskParams::initial_capital);

    py::class_<ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<>())
        .def("executeSignals", &ExecutionEngine::executeSignals)
        .def("checkRiskLimits", &ExecutionEngine::checkRiskLimits)
        .def("updatePositions", &ExecutionEngine::updatePositions)
        .def("calculateTransactionCosts", &ExecutionEngine::calculateTransactionCosts)
        .def("simulateFills", &ExecutionEngine::simulateFills);
} 