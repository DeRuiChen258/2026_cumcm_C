#include "io/writers.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace io {
namespace {

std::string pad_header(const std::string& dict) {
  // 总长度 = 8(魔数+版本) + 2(头长度) + 头，头要补齐到 64 的整数倍
  size_t len = dict.size() + 1;  // trailing '\n'
  const size_t total = 10 + len;
  const size_t pad = (64 - (total % 64)) % 64;
  std::string out = dict;
  out.append(pad, ' ');
  out.push_back('\n');
  return out;
}

}  // namespace

void write_npy_f64(const std::string& path, const double* data, const std::vector<size_t>& shape) {
  std::string shape_str;
  if (shape.size() == 1) {
    shape_str = "(" + std::to_string(shape[0]) + ",)";
  } else {
    shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      shape_str += std::to_string(shape[i]);
      if (i + 1 < shape.size()) shape_str += ", ";
    }
    shape_str += ")";
  }
  size_t count = 1;
  for (size_t s : shape) count *= s;
  const std::string dict = "{'descr': '<f8', 'fortran_order': False, 'shape': " + shape_str + ", }";
  const std::string header = pad_header(dict);

  std::FILE* fp = std::fopen(path.c_str(), "wb");
  if (fp == nullptr) throw std::runtime_error("cannot write " + path);
  const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  std::fwrite(magic, 1, 6, fp);
  const unsigned char version[2] = {1, 0};
  std::fwrite(version, 1, 2, fp);
  const uint16_t hlen = static_cast<uint16_t>(header.size());
  unsigned char lb[2] = {static_cast<unsigned char>(hlen & 0xFF),
                         static_cast<unsigned char>((hlen >> 8) & 0xFF)};
  std::fwrite(lb, 1, 2, fp);
  std::fwrite(header.data(), 1, header.size(), fp);
  std::fwrite(data, sizeof(double), count, fp);
  std::fclose(fp);
}

void write_text(const std::string& path, const std::string& content) {
  std::FILE* fp = std::fopen(path.c_str(), "wb");
  if (fp == nullptr) throw std::runtime_error("cannot write " + path);
  std::fwrite(content.data(), 1, content.size(), fp);
  std::fclose(fp);
}

void write_csv(const std::string& path, const std::vector<std::vector<std::string>>& rows) {
  std::string out;
  for (const auto& row : rows) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i) out += ",";
      const std::string& cell = row[i];
      const bool needs_quote = cell.find_first_of(",\"\n\r") != std::string::npos;
      if (!needs_quote) {
        out += cell;
      } else {
        out += '"';
        for (char c : cell) {
          if (c == '"') out += '"';
          out += c;
        }
        out += '"';
      }
    }
    out += "\n";  // LF only, UTF-8 without BOM
  }
  write_text(path, out);
}

std::string fmt(double v, int precision) {
  if (std::isnan(v)) return "nan";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
  return std::string(buf);
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

std::string slots_json(const model::Panel& panel) {
  std::string out = "{\n  \"convention\": \"right-endpoint timestamps of 10-minute slots\",\n  \"slots\": [";
  for (size_t i = 0; i < panel.slots.size(); ++i) {
    if (i) out += ", ";
    out += "\"" + json_escape(panel.slots[i]) + "\"";
  }
  out += "]\n}\n";
  return out;
}

std::string dates_json(const model::Panel& panel) {
  std::string out = "{\n  \"days\": " + std::to_string(panel.dates.size()) + ",\n  \"dates\": [";
  for (size_t i = 0; i < panel.dates.size(); ++i) {
    if (i) out += ", ";
    out += "\"" + panel.dates[i] + "\"";
  }
  out += "]\n}\n";
  return out;
}

std::string manifest_json(const model::Panel& panel, double seconds, long peak_rss_kb,
                          const std::vector<std::pair<std::string, std::string>>& inputs,
                          const std::vector<std::pair<std::string, std::string>>& outputs) {
  std::string out = "{\n";
  out += "  \"producer\": \"code/cleaning_cpp (C++17, zlib+tinyxml2)\",\n";
  out += "  \"contract\": \"prompt v1: C++ performs all value-level cleaning; Python is read-only\",\n";
  out += "  \"rows\": " + std::to_string(panel.pv_actual.rows()) + ",\n";
  out += "  \"cols\": " + std::to_string(panel.pv_actual.cols()) + ",\n";
  out += "  \"forecast_shape\": [365, 4, 24],\n";
  out += "  \"day1_shape\": [144, 3],\n";
  out += "  \"dtype\": \"<f8\",\n";
  out += "  \"missing_policy\": \"NaN (never 0)\",\n";
  out += "  \"time_convention\": \"slot k (1..144) <-> right endpoint 10k minutes <-> attachment column k+1\",\n";
  out += "  \"audit_records\": " + std::to_string(panel.audit.size()) + ",\n";
  out += "  \"quarantine_records\": " + std::to_string(panel.quarantine.size()) + ",\n";
  out += "  \"sigma_floor_kw\": " + fmt(panel.sigma_floor) + ",\n";
  out += "  \"wall_seconds\": " + fmt(seconds, 6) + ",\n";
  out += "  \"peak_rss_kb\": " + std::to_string(peak_rss_kb) + ",\n";
  out += "  \"inputs_sha256\": {\n";
  for (size_t i = 0; i < inputs.size(); ++i) {
    out += "    \"" + json_escape(inputs[i].first) + "\": \"" + inputs[i].second + "\"" +
           (i + 1 < inputs.size() ? "," : "") + "\n";
  }
  out += "  },\n  \"outputs_sha256\": {\n";
  for (size_t i = 0; i < outputs.size(); ++i) {
    out += "    \"" + json_escape(outputs[i].first) + "\": \"" + outputs[i].second + "\"" +
           (i + 1 < outputs.size() ? "," : "") + "\n";
  }
  out += "  }\n}\n";
  return out;
}

}  // namespace io
