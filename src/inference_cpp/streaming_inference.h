#pragma once

#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <onnxruntime_cxx_api.h>

struct StreamingLatencyStats {
    double feature_ms;
    double lstm_ms;
    double total_ms;
    double p95_ms;
    double p99_ms;
    double min_ms;
    double max_ms;
};

class StreamingGestureInference {
public:
    explicit StreamingGestureInference(
        const std::string& feature_model_path,
        const std::string& lstm_model_path
    );

    // Process a single radar frame (4, 32, 32) and return prediction
    // Input: 4*32*32 = 4096 floats in CHW layout
    // Returns: (predicted_class, confidence) based on accumulated LSTM state
    std::pair<int, float> predict_frame(const std::vector<float>& frame);

    // Reset LSTM hidden state for a new gesture
    void reset_state();

    // Get current logits (12 class scores) from last prediction
    const std::vector<float>& get_logits() const { return last_logits_; }

    // Get accumulated softmax probabilities (averaged over all frames since reset)
    const std::vector<float>& get_accumulated_probs() const { return accumulated_probs_; }

    // Get number of frames processed since last reset
    int get_frame_count() const { return frame_count_; }

    // Get running prediction using accumulated probability
    std::pair<int, float> get_running_prediction() const;

    // Benchmark streaming inference
    StreamingLatencyStats benchmark(const std::vector<float>& sample_frame, int n_iterations = 1000);

private:
    // Feature extractor session
    Ort::Env env_;
    Ort::Session feature_session_;
    std::vector<const char*> feature_input_names_;
    std::vector<const char*> feature_output_names_;
    std::vector<std::string> feature_input_name_strings_;
    std::vector<std::string> feature_output_name_strings_;

    // LSTM step session
    Ort::Session lstm_session_;
    std::vector<const char*> lstm_input_names_;
    std::vector<const char*> lstm_output_names_;
    std::vector<std::string> lstm_input_name_strings_;
    std::vector<std::string> lstm_output_name_strings_;

    // LSTM hidden state (1, 1, 512)
    static constexpr int HIDDEN_SIZE = 512;
    std::vector<float> h_state_;
    std::vector<float> c_state_;

    // Accumulated softmax probabilities
    static constexpr int N_CLASSES = 12;
    std::vector<float> accumulated_probs_;
    int frame_count_;
    std::vector<float> last_logits_;

    // Helper to create ORT session with wide string path on Windows
    Ort::Session create_session(const std::string& model_path, Ort::SessionOptions& options);
};