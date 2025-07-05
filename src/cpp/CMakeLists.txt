cmake_minimum_required(VERSION 3.14)
project(quant_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(COMMAND python3 -m pybind11 --cmakedir OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

file(GLOB SOURCES
    "*.cpp"
    )

pybind11_add_module(quant_cpp bindings.cpp OrderManager.cpp EnhancedRiskManager.cpp AdvancedFeatures.cpp SignalGenerator.cpp ExecutionEngine.cpp) 