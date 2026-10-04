// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Half 批量转换实现：AVX2 无分支版（与 Half.hpp 标量语义逐位一致）+ 标量回退。
#include "infvino/Half.hpp"

#include <cstdlib>
#include <string>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace infvino
{

#if defined(__x86_64__) || defined(__i386__)
namespace
{
// 与 f32_to_f16 标量版**逐位一致**：同样的截断（mant>>13）、exp<=0 直接返回 sign
// （flush denormal）、exp>=31 返回 inf。无分支写成 SIMD 后可被 GCC 向量化。
__attribute__((target("avx2")))
void f32_to_f16_avx2(const float * in, uint16_t * out, size_t n)
{
  size_t i = 0;
  const __m256i one = _mm256_set1_epi32(1);
  for (; i + 8 <= n; i += 8)
  {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
    const __m256i sign =
      _mm256_and_si256(_mm256_srli_epi32(x, 16), _mm256_set1_epi32(0x8000));
    const __m256i e = _mm256_sub_epi32(
      _mm256_and_si256(_mm256_srli_epi32(x, 23), _mm256_set1_epi32(0xff)),
      _mm256_set1_epi32(112));  // 127 - 15
    const __m256i mant = _mm256_and_si256(x, _mm256_set1_epi32(0x7fffff));
    const __m256i normal = _mm256_or_si256(
      sign, _mm256_or_si256(_mm256_slli_epi32(e, 10), _mm256_srli_epi32(mant, 13)));
    const __m256i le0  = _mm256_cmpgt_epi32(one, e);       // e <= 0
    const __m256i ge31 = _mm256_cmpgt_epi32(e, _mm256_set1_epi32(30));  // e >= 31
    __m256i r = _mm256_blendv_epi8(normal, sign, le0);
    r = _mm256_blendv_epi8(r, _mm256_or_si256(sign, _mm256_set1_epi32(0x7c00)), ge31);
    const __m128i lo = _mm256_castsi256_si128(r);
    const __m128i hi = _mm256_extracti128_si256(r, 1);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), _mm_packus_epi32(lo, hi));
  }
  for (; i < n; ++i) out[i] = f32_to_f16(in[i]);
}

// 与 f16_to_f32 标量版逐位一致。denormal（exp==0 && mant!=0）走标量路径，正常值
// 走 SIMD 快路径；模型输出基本无 denormal。
__attribute__((target("avx2")))
void f16_to_f32_avx2(const uint16_t * in, float * out, size_t n)
{
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
  {
    const __m256i x =
      _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i)));
    const __m256i sign =
      _mm256_slli_epi32(_mm256_and_si256(x, _mm256_set1_epi32(0x8000)), 16);
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(x, 10), _mm256_set1_epi32(0x1f));
    const __m256i mant = _mm256_and_si256(x, _mm256_set1_epi32(0x3ff));
    const __m256i normal = _mm256_or_si256(
      sign,
      _mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(112)), 23),
                      _mm256_slli_epi32(mant, 13)));
    const __m256i isz = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
    const __m256i mz  = _mm256_cmpeq_epi32(mant, _mm256_setzero_si256());
    const __m256i zz  = _mm256_and_si256(isz, mz);              // exp==0 && mant==0 -> sign
    const __m256i inf = _mm256_cmpeq_epi32(e, _mm256_set1_epi32(31));
    __m256i r = _mm256_blendv_epi8(normal, sign, zz);
    r = _mm256_blendv_epi8(r, _mm256_or_si256(sign, _mm256_set1_epi32(0x7f800000)), inf);
    const __m256i denorm = _mm256_andnot_si256(mz, isz);
    if (_mm256_movemask_epi8(denorm))
      for (int j = 0; j < 8; ++j) out[i + static_cast<size_t>(j)] = f16_to_f32(in[i + static_cast<size_t>(j)]);
    else
      _mm256_storeu_ps(out + i, _mm256_castsi256_ps(r));
  }
  for (; i < n; ++i) out[i] = f16_to_f32(in[i]);
}

bool hasAvx2()
{
  static const bool v = __builtin_cpu_supports("avx2");
  static const bool off = []() {
    const char * e = std::getenv("INFVINO_NO_SIMD_HALF");
    return e && *e && std::string(e) != "0";
  }();
  return v && !off;
}
}  // namespace
#endif

void f32_to_f16_bulk(const float * in, uint16_t * out, size_t n)
{
#if defined(__x86_64__) || defined(__i386__)
  if (hasAvx2()) { f32_to_f16_avx2(in, out, n); return; }
#endif
  for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16(in[i]);
}

void f16_to_f32_bulk(const uint16_t * in, float * out, size_t n)
{
#if defined(__x86_64__) || defined(__i386__)
  if (hasAvx2()) { f16_to_f32_avx2(in, out, n); return; }
#endif
  for (size_t i = 0; i < n; ++i) out[i] = f16_to_f32(in[i]);
}

}  // namespace infvino
