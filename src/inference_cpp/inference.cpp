#include "inference.h"

#ifdef _WIN32
#include <windows.h>
#endif

GestureInference::GestureInference(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "gesture"),
      session_(nullptr) {

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // On Windows, ORT requires wide string for model path
#ifdef _WIN32
    int size = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
    std::wstring wmodel_path(size - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &wmodel_path[0], size);
    session_ = Ort::Session(env_, wmodel_path.c_str(), session_options);
#else
    session_ = Ort::Session(env_, model_path.c_str(), session_options);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_inputs = session_.GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator);
        input_name_strings_.push_back(name.get());
    }

    size_t num_outputs = session_.GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator);
        output_name_strings_.push_back(name.get());
    }

    for (auto& s : input_name_strings_) input_names_.push_back(s.c_str());
    for (auto& s : output_name_strings_) output_names_.push_back(s.c_str());
}

std::pair<int, float> GestureInference::predict(const std::vector<float>& input) {
    std::vector<int64_t> shape = {1, 4, 32, 32};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input.data()),
        input.size(), shape.data(), shape.size());

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), static_cast<int>(output_names_.size()));

    float* logits = outputs[0].GetTensorMutableData<float>();
    auto& type_shape = outputs[0].GetTensorTypeAndShapeInfo();
    size_t n_classes = type_shape.GetShape()[1];

    int predicted_class = 0;
    float max_logit = logits[0];
    for (size_t i = 1; i < n_classes; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            predicted_class = static_cast<int>(i);
        }
    }

    return {predicted_class, max_logit};
}

LatencyStats GestureInference::benchmark(const std::vector<float>& input, int n_iterations) {
    // Warmup
    for (int i = 0; i < 5; ++i) {
        predict(input);
    }

    std::vector<double> latencies;
    latencies.reserve(n_iterations);

    for (int i = 0; i < n_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        predict(input);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
    }

    std::sort(latencies.begin(), latencies.end());

    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / n_iterations;
    double variance = 0.0;
    for (auto l : latencies) variance += (l - mean) * (l - mean);
    variance /= n_iterations;

    int p95_idx = static_cast<int>(0.95 * n_iterations) - 1;
    int p99_idx = static_cast<int>(0.99 * n_iterations) - 1;

    return {
        mean,
        std::sqrt(variance),
        latencies[p95_idx],
        latencies[p99_idx],
        latencies.front(),
        latencies.back()
    };
}