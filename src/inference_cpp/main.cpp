#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "inference.h"
#include "preprocessor.h"

void write_benchmark_report(
    const std::string& model_path,
    const LatencyStats& stats,
    int predicted_class,
    float confidence,
    int n_iterations,
    const std::string& output_dir) {

    std::string model_type = (model_path.find("quant") != std::string::npos) ? "INT8" : "FP32";

    std::ofstream report(output_dir + "/latency_benchmark.md");
    report << "# Latency Benchmark\n\n";
    report << "## Model\n";
    report << "- Path: `" << model_path << "`\n";
    report << "- Precision: " << model_type << "\n";
    report << "- Input: (1, 4, 32, 32)\n\n";
    report << "## Results\n\n";
    report << "| Metric | Value |\n";
    report << "|--------|-------|\n";
    report << "| Mean latency | " << stats.mean_ms << " ms |\n";
    report << "| Std deviation | " << stats.std_ms << " ms |\n";
    report << "| P95 latency | " << stats.p95_ms << " ms |\n";
    report << "| P99 latency | " << stats.p99_ms << " ms |\n";
    report << "| Min latency | " << stats.min_ms << " ms |\n";
    report << "| Max latency | " << stats.max_ms << " ms |\n";
    report << "| Iterations | " << n_iterations << " |\n\n";
    report << "## Sample Prediction\n";
    report << "- Predicted class: " << predicted_class << "\n";
    report << "- Confidence (logit): " << confidence << "\n";
    report.close();

    std::cout << "Benchmark report saved to " << output_dir << "/latency_benchmark.md\n";
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/gesture_model.onnx";
    int n_iterations = 100;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) n_iterations = std::atoi(argv[2]);

    std::cout << "Loading model: " << model_path << std::endl;
    GestureInference inf(model_path);

    // Create random test input (1, 4, 32, 32)
    std::vector<float> input(4 * 32 * 32);
    for (auto& v : input) v = static_cast<float>(rand()) / RAND_MAX;

    // Single prediction
    auto [pred_class, confidence] = inf.predict(input);
    std::cout << "Prediction: class=" << pred_class
              << " confidence=" << confidence << std::endl;

    // Benchmark
    std::cout << "\nRunning benchmark (" << n_iterations << " iterations)..." << std::endl;
    LatencyStats stats = inf.benchmark(input, n_iterations);

    std::cout << "\n=== Latency Results ===\n";
    std::cout << "Mean:  " << stats.mean_ms << " ms\n";
    std::cout << "Std:   " << stats.std_ms << " ms\n";
    std::cout << "P95:   " << stats.p95_ms << " ms\n";
    std::cout << "P99:   " << stats.p99_ms << " ms\n";
    std::cout << "Min:   " << stats.min_ms << " ms\n";
    std::cout << "Max:   " << stats.max_ms << " ms\n";

    // Write report
    write_benchmark_report(model_path, stats, pred_class, confidence, n_iterations, "reports");

    return 0;
}