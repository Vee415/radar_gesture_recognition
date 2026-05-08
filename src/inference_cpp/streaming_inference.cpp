#include "streaming_inference.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <numeric>

StreamingGestureInference::StreamingGestureInference(
    const std::string& feature_model_path,
    const std::string& lstm_model_path
) : env_(ORT_LOGGING_LEVEL_WARNING, "streaming"),
    feature_session_(nullptr),
    lstm_session_(nullptr) {

    // Feature extractor session
    Ort::SessionOptions feature_opts;
    feature_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    feature_session_ = create_session(feature_model_path, feature_opts);

    // LSTM step session
    Ort::SessionOptions lstm_opts;
    lstm_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    lstm_session_ = create_session(lstm_model_path, lstm_opts);

    // Extract input/output names for feature extractor
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_feat_inputs = feature_session_.GetInputCount();
    for (size_t i = 0; i < num_feat_inputs; ++i) {
        auto name = feature_session_.GetInputNameAllocated(i, allocator);
        feature_input_name_strings_.push_back(name.get());
    }
    for (size_t i = 0; i < feature_session_.GetOutputCount(); ++i) {
        auto name = feature_session_.GetOutputNameAllocated(i, allocator);
        feature_output_name_strings_.push_back(name.get());
    }
    for (auto& s : feature_input_name_strings_) feature_input_names_.push_back(s.c_str());
    for (auto& s : feature_output_name_strings_) feature_output_names_.push_back(s.c_str());

    // Extract input/output names for LSTM step
    size_t num_lstm_inputs = lstm_session_.GetInputCount();
    for (size_t i = 0; i < num_lstm_inputs; ++i) {
        auto name = lstm_session_.GetInputNameAllocated(i, allocator);
        lstm_input_name_strings_.push_back(name.get());
    }
    for (size_t i = 0; i < lstm_session_.GetOutputCount(); ++i) {
        auto name = lstm_session_.GetOutputNameAllocated(i, allocator);
        lstm_output_name_strings_.push_back(name.get());
    }
    for (auto& s : lstm_input_name_strings_) lstm_input_names_.push_back(s.c_str());
    for (auto& s : lstm_output_name_strings_) lstm_output_names_.push_back(s.c_str());

    // Initialize hidden state
    reset_state();
}

Ort::Session StreamingGestureInference::create_session(
    const std::string& model_path,
    Ort::SessionOptions& options) {

#ifdef _WIN32
    int size = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
    std::wstring wmodel_path(size - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &wmodel_path[0], size);
    return Ort::Session(env_, wmodel_path.c_str(), options);
#else
    return Ort::Session(env_, model_path.c_str(), options);
#endif
}

void StreamingGestureInference::reset_state() {
    h_state_.assign(HIDDEN_SIZE, 0.0f);
    c_state_.assign(HIDDEN_SIZE, 0.0f);
    accumulated_probs_.assign(N_CLASSES, 0.0f);
    frame_count_ = 0;
    last_logits_.assign(N_CLASSES, 0.0f);
}

std::pair<int, float> StreamingGestureInference::predict_frame(
    const std::vector<float>& frame) {

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Step 1: Feature extraction (1, 4, 32, 32) -> (1, 256)
    std::vector<int64_t> feature_input_shape = {1, 4, 32, 32};
    auto feature_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(frame.data()),
        frame.size(), feature_input_shape.data(), feature_input_shape.size());

    auto feature_output = feature_session_.Run(
        Ort::RunOptions{nullptr},
        feature_input_names_.data(), &feature_tensor, 1,
        feature_output_names_.data(), static_cast<int>(feature_output_names_.size()));

    // Get feature vector and reshape to (1, 1, 256)
    float* feature_data = feature_output[0].GetTensorMutableData<float>();
    std::vector<int64_t> lstm_input_shape = {1, 1, 256};

    // Step 2: LSTM step with hidden state
    std::vector<int64_t> hidden_shape = {1, 1, HIDDEN_SIZE};

    auto feature_tensor_lstm = Ort::Value::CreateTensor<float>(
        memory_info, feature_data, 256, lstm_input_shape.data(), lstm_input_shape.size());

    auto h_tensor = Ort::Value::CreateTensor<float>(
        memory_info, h_state_.data(), HIDDEN_SIZE,
        hidden_shape.data(), hidden_shape.size());

    auto c_tensor = Ort::Value::CreateTensor<float>(
        memory_info, c_state_.data(), HIDDEN_SIZE,
        hidden_shape.data(), hidden_shape.size());

    std::vector<Ort::Value> lstm_inputs;
    lstm_inputs.push_back(std::move(feature_tensor_lstm));
    lstm_inputs.push_back(std::move(h_tensor));
    lstm_inputs.push_back(std::move(c_tensor));

    std::vector<const char*> lstm_input_names_arr = {
        lstm_input_names_[0], lstm_input_names_[1], lstm_input_names_[2]
    };

    auto lstm_output = lstm_session_.Run(
        Ort::RunOptions{nullptr},
        lstm_input_names_arr.data(), lstm_inputs.data(), 3,
        lstm_output_names_.data(), static_cast<int>(lstm_output_names_.size()));

    // Extract outputs
    float* logits_data = lstm_output[0].GetTensorMutableData<float>();
    float* h_new = lstm_output[1].GetTensorMutableData<float>();
    float* c_new = lstm_output[2].GetTensorMutableData<float>();

    // Update hidden state
    std::memcpy(h_state_.data(), h_new, HIDDEN_SIZE * sizeof(float));
    std::memcpy(c_state_.data(), c_new, HIDDEN_SIZE * sizeof(float));

    // Store logits
    last_logits_.assign(logits_data, logits_data + N_CLASSES);

    // Compute softmax for this frame
    float max_logit = *std::max_element(last_logits_.begin(), last_logits_.end());
    std::vector<float> probs(N_CLASSES);
    float sum = 0.0f;
    for (int i = 0; i < N_CLASSES; ++i) {
        probs[i] = std::exp(last_logits_[i] - max_logit);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;

    // Accumulate probabilities (running average)
    frame_count_++;
    for (int i = 0; i < N_CLASSES; ++i) {
        accumulated_probs_[i] = accumulated_probs_[i] * (1.0f - 1.0f / frame_count_) + probs[i] / frame_count_;
    }

    return get_running_prediction();
}

std::pair<int, float> StreamingGestureInference::get_running_prediction() const {
    int predicted_class = 0;
    float max_prob = accumulated_probs_[0];
    for (int i = 1; i < N_CLASSES; ++i) {
        if (accumulated_probs_[i] > max_prob) {
            max_prob = accumulated_probs_[i];
            predicted_class = i;
        }
    }
    return {predicted_class, max_prob};
}

StreamingLatencyStats StreamingGestureInference::benchmark(
    const std::vector<float>& sample_frame, int n_iterations) {

    // Warmup
    reset_state();
    for (int i = 0; i < 10; ++i) {
        predict_frame(sample_frame);
    }

    // Benchmark individual components
    std::vector<double> feature_latencies;
    std::vector<double> lstm_latencies;
    std::vector<double> total_latencies;
    feature_latencies.reserve(n_iterations);
    lstm_latencies.reserve(n_iterations);
    total_latencies.reserve(n_iterations);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> feature_input_shape = {1, 4, 32, 32};
    std::vector<int64_t> lstm_input_shape = {1, 1, 256};
    std::vector<int64_t> hidden_shape = {1, 1, HIDDEN_SIZE};

    for (int i = 0; i < n_iterations; ++i) {
        reset_state();

        // Feature extraction timing
        auto start_total = std::chrono::high_resolution_clock::now();

        auto feature_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(sample_frame.data()),
            sample_frame.size(), feature_input_shape.data(), feature_input_shape.size());

        auto start_feat = std::chrono::high_resolution_clock::now();
        auto feature_output = feature_session_.Run(
            Ort::RunOptions{nullptr},
            feature_input_names_.data(), &feature_tensor, 1,
            feature_output_names_.data(), static_cast<int>(feature_output_names_.size()));
        auto end_feat = std::chrono::high_resolution_clock::now();

        float* feature_data = feature_output[0].GetTensorMutableData<float>();

        // LSTM step timing
        auto feature_tensor_lstm = Ort::Value::CreateTensor<float>(
            memory_info, feature_data, 256, lstm_input_shape.data(), lstm_input_shape.size());

        auto h_tensor = Ort::Value::CreateTensor<float>(
            memory_info, h_state_.data(), HIDDEN_SIZE,
            hidden_shape.data(), hidden_shape.size());

        auto c_tensor = Ort::Value::CreateTensor<float>(
            memory_info, c_state_.data(), HIDDEN_SIZE,
            hidden_shape.data(), hidden_shape.size());

        std::vector<Ort::Value> lstm_inputs;
        lstm_inputs.push_back(std::move(feature_tensor_lstm));
        lstm_inputs.push_back(std::move(h_tensor));
        lstm_inputs.push_back(std::move(c_tensor));

        std::vector<const char*> lstm_input_names_arr = {
            lstm_input_names_[0], lstm_input_names_[1], lstm_input_names_[2]
        };

        auto start_lstm = std::chrono::high_resolution_clock::now();
        auto lstm_output = lstm_session_.Run(
            Ort::RunOptions{nullptr},
            lstm_input_names_arr.data(), lstm_inputs.data(), 3,
            lstm_output_names_.data(), static_cast<int>(lstm_output_names_.size()));
        auto end_lstm = std::chrono::high_resolution_clock::now();

        auto end_total = std::chrono::high_resolution_clock::now();

        // Update hidden state
        float* h_new = lstm_output[1].GetTensorMutableData<float>();
        float* c_new = lstm_output[2].GetTensorMutableData<float>();
        std::memcpy(h_state_.data(), h_new, HIDDEN_SIZE * sizeof(float));
        std::memcpy(c_state_.data(), c_new, HIDDEN_SIZE * sizeof(float));

        feature_latencies.push_back(std::chrono::duration<double, std::milli>(end_feat - start_feat).count());
        lstm_latencies.push_back(std::chrono::duration<double, std::milli>(end_lstm - start_lstm).count());
        total_latencies.push_back(std::chrono::duration<double, std::milli>(end_total - start_total).count());
    }

    std::sort(total_latencies.begin(), total_latencies.end());

    return {
        /*feature_ms=*/std::accumulate(feature_latencies.begin(), feature_latencies.end(), 0.0) / n_iterations,
        /*lstm_ms=*/std::accumulate(lstm_latencies.begin(), lstm_latencies.end(), 0.0) / n_iterations,
        /*total_ms=*/total_latencies[total_latencies.size() / 2],
        /*p95_ms=*/total_latencies[static_cast<int>(0.95 * n_iterations) - 1],
        /*p99_ms=*/total_latencies[static_cast<int>(0.99 * n_iterations) - 1],
        /*min_ms=*/total_latencies[0],
        /*max_ms=*/total_latencies[total_latencies.size() - 1]
    };
}