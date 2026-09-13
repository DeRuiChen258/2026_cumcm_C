#pragma once
#include <string>
#include <utility>
#include <vector>

#include "model/panel.h"

namespace io {

void write_npy_f64(const std::string& path, const double* data, const std::vector<size_t>& shape);
void write_csv(const std::string& path, const std::vector<std::vector<std::string>>& rows);
void write_text(const std::string& path, const std::string& content);
std::string fmt(double v, int precision = 10);
std::string json_escape(const std::string& s);

std::string manifest_json(const model::Panel& panel, double seconds, long peak_rss_kb,
                          const std::vector<std::pair<std::string, std::string>>& inputs,
                          const std::vector<std::pair<std::string, std::string>>& outputs);
std::string slots_json(const model::Panel& panel);
std::string dates_json(const model::Panel& panel);

}  // namespace io
