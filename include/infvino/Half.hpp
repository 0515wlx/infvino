// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// FP16 <-> FP32 位运算转换（不依赖 _Float16，便于跨编译器）。
#ifndef INFVINO__HALF_HPP_
#define INFVINO__HALF_HPP_

#include <cstddef>
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

// ---------------------------------------------------------------------------
// 批量转换（热路径：整网输入 f32->f16、输出 f16->f32）。
// 实现与上面的标量版**逐位一致**（同样的截断/denormal flush 语义），但在支持的
// x86 上用 AVX2 无分支实现；否则回退标量循环。
// 动机（docs/budget-analysis-3models.md §9）：ClBackend 每次推理对 1.23M 输入 +
// 0.47M 输出做转换，实测在项目编译选项下约 1.59 ms/帧（yolo），是净耗时里一块
// 未被优化的固定开销。批量版 ~0.5 ms。
// ---------------------------------------------------------------------------
void f32_to_f16_bulk(const float * in, uint16_t * out, size_t n);
void f16_to_f32_bulk(const uint16_t * in, float * out, size_t n);

}  // namespace infvino

#endif  // INFVINO__HALF_HPP_
