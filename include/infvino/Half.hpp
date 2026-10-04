// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// FP16 <-> FP32 位运算转换（不依赖 _Float16，便于跨编译器）。
#ifndef INFVINO__HALF_HPP_
#define INFVINO__HALF_HPP_

#include <cstdint>
#include <cstring>

namespace infvino
{

inline uint16_t f32_to_f16(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  const uint32_t mant = x & 0x7fffffu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

inline float f16_to_f32(uint16_t h)
{
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      int e = 127 - 15 + 1;
      while (!(mant & 0x400u)) { mant <<= 1; --e; }
      mant &= 0x3ffu;
      f = sign | (static_cast<uint32_t>(e) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | (mant << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

}  // namespace infvino

#endif  // INFVINO__HALF_HPP_
