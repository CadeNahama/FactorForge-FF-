#pragma once
#include <vector>
#include <string>
#include <map>

class EnsembleMLSystem {
public:
    EnsembleMLSystem();
    ~EnsembleMLSystem();

    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    std::vector<double> predict(const std::vector<std::vector<double>>& X);
    void evaluate(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    void saveModel(const std::string& filepath);
    void loadModel(const std::string& filepath);
}; 