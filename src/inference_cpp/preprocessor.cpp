#include "preprocessor.h"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

std::vector<float> load_npy_test_data(const std::string& path) {
    // Simple .npy loader for float32 arrays
    // Reads the numpy file header and raw data
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read magic string
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Not a valid .npy file: " + path);
    }

    // Read version
    uint8_t major = 0, minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hlen = 0;
        file.read(reinterpret_cast<char*>(&hlen), 2);
        header_len = hlen;
    } else if (major == 2) {
        uint32_t hlen = 0;
        file.read(reinterpret_cast<char*>(&hlen), 4);
        header_len = hlen;
    }

    // Skip header
    file.seekg(10 + header_len, std::ios::beg);

    // Read remaining data as float32
    file.seekg(0, std::ios::end);
    auto end_pos = file.tellg();
    file.seekg(10 + header_len, std::ios::beg);
    auto data_size = end_pos - file.tellg();

    std::vector<float> data(data_size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), data_size);

    return data;
}

std::vector<float> preprocess_sample(const std::vector<float>& raw) {
    // Normalize to [0, 1] per channel (4 channels, 32x32)
    // raw shape: (4, 32, 32) = 4096 elements
    const int channels = 4;
    const int spatial = 32 * 32;

    std::vector<float> output(raw.size());

    for (int c = 0; c < channels; ++c) {
        float min_val = *std::min_element(
            raw.begin() + c * spatial,
            raw.begin() + (c + 1) * spatial);
        float max_val = *std::max_element(
            raw.begin() + c * spatial,
            raw.begin() + (c + 1) * spatial);
        float range = max_val - min_val;

        for (int i = 0; i < spatial; ++i) {
            float val = raw[c * spatial + i];
            output[c * spatial + i] = (range > 1e-8f) ? (val - min_val) / range : 0.0f;
        }
    }

    return output;
}