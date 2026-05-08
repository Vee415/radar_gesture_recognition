#pragma once

#include <string>
#include <vector>

std::vector<float> load_npy_test_data(const std::string& path);
std::vector<float> preprocess_sample(const std::vector<float>& raw);