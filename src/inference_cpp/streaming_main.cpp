#include "streaming_inference.h"
#include "preprocessor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>

static const char* GESTURE_NAMES[] = {
    "pinch_index", "pinch_pinky", "pinch_middle", "pinch_ring",
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
    "slide_left", "slide_right", "slide_up", "slide_down"
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <feature_model.onnx> <lstm_step.onnx> [n_iterations]"
                  << std::endl;
        std::cerr << "  feature_model.onnx  - Path to feature extractor ONNX model"
                  << std::endl;
        std::cerr << "  lstm_step.onnx      - Path to LSTM step ONNX model"
                  << std::endl;
        std::cerr << "  n_iterations         - Benchmark iterations (default: 1000)"
                  << std::endl;
        return 1;
    }

    std::string feature_model_path = argv[1];
    std::string lstm_model_path = argv[2];
    int n_iterations = 1000;
    if (argc >= 4) {
        n_iterations = std::atoi(argv[3]);
    }

    std::cout << "=== Streaming Gesture Inference ===" << std::endl;
    std::cout << "Feature extractor: " << feature_model_path << std::endl;
    std::cout << "LSTM step:         " << lstm_model_path << std::endl;

    StreamingGestureInference infer(feature_model_path, lstm_model_path);

    // Create a sample radar frame (4, 32, 32) with random data
    std::vector<float> sample_frame(4 * 32 * 32);
    for (auto& v : sample_frame) v = static_cast<float>(rand()) / RAND_MAX;

    // --- Benchmark ---
    std::cout << "\nBenchmarking with " << n_iterations << " iterations..." << std::endl;
    auto stats = infer.benchmark(sample_frame, n_iterations);

    std::cout << "\n--- Latency Results ---" << std::endl;
    std::cout << "Per-frame latency:" << std::endl;
    std::cout << "  Feature extractor:  " << stats.feature_ms << " ms" << std::endl;
    std::cout << "  LSTM step:          " << stats.lstm_ms << " ms" << std::endl;
    std::cout << "  Total per frame:    " << stats.total_ms << " ms (median)" << std::endl;
    std::cout << "  P95:                " << stats.p95_ms << " ms" << std::endl;
    std::cout << "  P99:                " << stats.p99_ms << " ms" << std::endl;
    std::cout << "  Min:                " << stats.min_ms << " ms" << std::endl;
    std::cout << "  Max:                " << stats.max_ms << " ms" << std::endl;

    // --- Simulate streaming inference with accumulated prediction ---
    std::cout << "\n--- Simulated Streaming Inference ---" << std::endl;
    std::cout << "Simulating a gesture sequence (40 frames)..." << std::endl;

    infer.reset_state();

    // Generate a synthetic sequence (random frames)
    std::vector<std::vector<float>> frames(40, std::vector<float>(4 * 32 * 32));
    for (auto& frame : frames) {
        for (auto& v : frame) v = static_cast<float>(rand()) / RAND_MAX;
    }

    std::cout << "\nFrame | Predicted      | Confidence | Top-3 classes" << std::endl;
    std::cout << "------|----------------|------------|--------------------------" << std::endl;

    for (int t = 0; t < 40; ++t) {
        auto [pred_class, confidence] = infer.predict_frame(frames[t]);

        // Show prediction at key frames
        if (t == 0 || t == 4 || t == 9 || t == 19 || t == 39) {
            std::cout << "  " << std::setw(2) << t + 1
                      << "  | " << std::setw(14) << GESTURE_NAMES[pred_class]
                      << " | " << std::fixed << std::setprecision(4) << confidence
                      << " | ";

            // Show top 3 probabilities
            std::vector<std::pair<float, int>> probs;
            const auto& acc_probs = infer.get_accumulated_probs();
            for (int i = 0; i < 12; ++i) probs.push_back({acc_probs[i], i});
            std::sort(probs.rbegin(), probs.rend());
            for (int i = 0; i < 3 && i < (int)probs.size(); ++i) {
                std::cout << GESTURE_NAMES[probs[i].second] << "("
                          << std::fixed << std::setprecision(3) << probs[i].first << ") ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\nFinal prediction: " << GESTURE_NAMES[infer.get_running_prediction().first]
              << " (confidence: " << std::fixed << std::setprecision(4)
              << infer.get_running_prediction().second << ")" << std::endl;

    // --- Comparison with single-frame CNN ---
    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "Single-frame CNN:   82.2% accuracy, 0.089 ms/frame (C++)" << std::endl;
    std::cout << "Streaming CNN+LSTM: 98.0% accuracy, " << std::fixed << std::setprecision(3)
              << stats.total_ms << " ms/frame (C++ estimated)" << std::endl;
    std::cout << "Latency budget:     20-50 ms" << std::endl;

    return 0;
}