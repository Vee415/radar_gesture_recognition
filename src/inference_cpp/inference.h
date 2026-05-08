#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <onnxruntime_cxx_api.h>

struct LatencyStats {
    double mean_ms;
    double std_ms;
    double p95_ms;
    double p99_ms;
    double min_ms;
    double max_ms;
};

class GestureInference {
public:
    explicit GestureInference(const std::string& model_path);
    std::pair<int, float> predict(const std::vector<float>& input);
    LatencyStats benchmark(const std::vector<float>& input, int n_iterations = 100);

private:
    Ort::Env env_;
    Ort::Session session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_name_strings_;
    std::vector<std::string> output_name_strings_;
};