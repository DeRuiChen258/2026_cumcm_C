#pragma once
#include <cstdint>
#include <string>

namespace util {

// 一个精简版 SHA-256（按 FIPS 180-4 写），用来算 manifest 里的校验和。
class Sha256 {
 public:
  Sha256() { reset(); }
  void reset();
  void update(const uint8_t* data, size_t len);
  std::string hexdigest();
  static std::string file(const std::string& path);
  static std::string bytes(const std::string& data);
  static std::string hex_of(const std::string& raw);

 private:
  void transform(const uint8_t* block);
  uint32_t h_[8];
  uint8_t buf_[64];
  size_t buf_len_;
  uint64_t total_len_;
};

}  // namespace util
