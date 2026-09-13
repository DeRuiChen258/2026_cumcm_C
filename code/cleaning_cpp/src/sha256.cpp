#include "util/sha256.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace util {
namespace {

const uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u,
};

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

void Sha256::reset() {
  static const uint32_t init[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::memcpy(h_, init, sizeof(h_));
  buf_len_ = 0;
  total_len_ = 0;
}

void Sha256::transform(const uint8_t* block) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
           (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
           static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
  uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = hh + s1 + ch + K[i] + w[i];
    const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    hh = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }
  h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
  h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
}

void Sha256::update(const uint8_t* data, size_t len) {
  total_len_ += len;
  while (len > 0) {
    const size_t take = (len < (64 - buf_len_)) ? len : (64 - buf_len_);
    std::memcpy(buf_ + buf_len_, data, take);
    buf_len_ += take;
    data += take;
    len -= take;
    if (buf_len_ == 64) {
      transform(buf_);
      buf_len_ = 0;
    }
  }
}

std::string Sha256::hexdigest() {
  const uint64_t bits = total_len_ * 8ULL;
  uint8_t pad[72];
  pad[0] = 0x80;
  size_t pad_len = (buf_len_ < 56) ? (56 - buf_len_) : (120 - buf_len_);
  std::memset(pad + 1, 0, pad_len - 1);
  for (int i = 0; i < 8; ++i) pad[pad_len + i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
  update(pad, pad_len + 8);
  // 补完位之后缓冲区正好空，可以直接输出
  char out[65];
  for (int i = 0; i < 8; ++i) {
    std::snprintf(out + i * 8, 9, "%08x", h_[i]);
  }
  out[64] = '\0';
  return std::string(out);
}

std::string Sha256::hex_of(const std::string& raw) {
  Sha256 s;
  s.update(reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
  return s.hexdigest();
}

std::string Sha256::bytes(const std::string& data) { return hex_of(data); }

std::string Sha256::file(const std::string& path) {
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) return std::string();
  Sha256 s;
  std::vector<uint8_t> buf(1 << 16);
  size_t n = 0;
  while ((n = std::fread(buf.data(), 1, buf.size(), fp)) > 0) s.update(buf.data(), n);
  std::fclose(fp);
  return s.hexdigest();
}

}  // namespace util
