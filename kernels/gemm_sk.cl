// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// gemm_sk_f16 —— **小 GEMM 专用通路**：lane 沿 K 归约（split-K within a sub-group），
// 面向「输出很小、K 很大」的 1x1 conv / fc 层。
//
// 动机（实测，见 docs/kernel.md / R32）：
//   mobilenetv3-small 与 yolo 头部有大量 M=Cout 小、N=HW 小、K=Cin 大的层，例如
//   `96x49x576`、`40x196x240`、`24x784x72`、`Cout1_N6400_Cin64`。现有 `gemm_f16`
//   按 BM×BN 切输出，这些形状只能得到 1–4 个 work-group（严重网格饥饿），实测仅
//   ~0.2 ops/EU/cyc；缩小 tile/BK 反而因 staging/barrier 变差。
//
// 数据通路：一个 work-group = 一个 sub-group（SK_SG=16 lane），处理 TM×TN 个输出。
//   * lane 沿 K 分块：k = lane + SK_SG*t —— A[m][k] 相邻 lane 连续（coalesced）；
//   * 每个 lane 累加 TM×TN 个部分和（fp32），最后 `sub_group_reduce_add` 树归约；
//   * 网格 = ceil(N/TN) × ceil(M/TM) 个 sub-group —— 小输出也能拿到几百个 WG；
//   * K 循环按 SK_UK 展开（多路独立链，隐藏 half→float 延迟）。
//
// 数值：fp32 累加 + 树归约，与 `gemm_f16`（fp16 累加）**不是逐位一致**，但更精确；
//   fp16 容差判据（model_check mean_rel<2e-2）下等价。
//
// 编译期参数（-D）：SK_TM / SK_TN / SK_SG / SK_UK / ACT / RES
//   ACT: 0=none 1=SiLU 2=Relu 3=Hardswish 4=Hardsigmoid 5=Sigmoid（与 gemm.cl 同码）
//   RES: 1 = epilogue 在激活后加残差 Res[M][N]
//
// License: original infvino code (project license).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifndef SK_TM
#define SK_TM 8
#endif
#ifndef SK_TN
#define SK_TN 4
#endif
#ifndef SK_SG
#define SK_SG 16
#endif
#ifndef SK_UK
#define SK_UK 4
#endif
#ifndef ACT
#define ACT 0
#endif
#ifndef RES
#define RES 0
#endif
// R51 D5: MUL_SCALE=1 -> activation is scaled per input channel (k) on load, fusing a
// one-operand channel-broadcast Mul (SE `x * scale[c]`) into the conv prologue so the
// separate elementwise pass (and its materialised output) disappears. Rounds the
// product to half before mad -> numerically identical to conv(scaled-half-tensor).
#ifndef MUL_SCALE
#define MUL_SCALE 0
#endif

inline half sk_activate(half v)
{
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));
#elif ACT == 2
  return (half)fmax((float)v, 0.0f);
#elif ACT == 3
  float f = (float)v;
  return (half)(f * fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);
#elif ACT == 4
  float f = (float)v;
  return (half)(fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);
#elif ACT == 5
  return (half)(1.0f / (1.0f + exp(-(float)v)));
#else
  return v;
#endif
}

__attribute__((intel_reqd_sub_group_size(SK_SG)))
__attribute__((reqd_work_group_size(SK_SG, 1, 1)))
__kernel void gemm_sk_f16(
  __global const half *restrict A,     // [M][K]  (weight [Cout][Cin], row-major)
  __global const half *restrict B,     // [K][N]  (activation [Cin][HW], row-major)
  __global half *restrict C,           // [M][N]
  const int M, const int N, const int K,
  __global const half *restrict Bias,  // [M] or null
  __global const half *restrict Res    // [M][N] or null (RES=1)
#if MUL_SCALE
  , __global const half *restrict Scale // [K] (MUL_SCALE=1)
#endif
)
{
  const int lane = get_local_id(0);
  const int n0 = get_group_id(0) * SK_TN;
  const int m0 = get_group_id(1) * SK_TM;

  float acc[SK_TM][SK_TN];
#pragma unroll
  for (int i = 0; i < SK_TM; ++i)
#pragma unroll
    for (int j = 0; j < SK_TN; ++j) acc[i][j] = 0.0f;

  int k = lane;
  // 主循环：每次覆盖 SK_UK 个 k（每个 lane），提供 SK_UK 条独立累加链。
#pragma unroll 1
  for (; k + (SK_UK - 1) * SK_SG < K; k += SK_SG * SK_UK) {
#pragma unroll
    for (int u = 0; u < SK_UK; ++u) {
      const int kk = k + u * SK_SG;
      half a[SK_TM], b[SK_TN];
#pragma unroll
      for (int i = 0; i < SK_TM; ++i) {
        const int m = m0 + i;
        a[i] = (m < M) ? A[(size_t)m * K + kk] : (half)0;
      }
#pragma unroll
      for (int j = 0; j < SK_TN; ++j) {
        const int n = n0 + j;
        b[j] = (n < N) ? B[(size_t)kk * N + n] : (half)0;
#if MUL_SCALE
        if (Scale != 0) b[j] = (half)((float)b[j] * (float)Scale[kk]);
#endif
      }
#pragma unroll
      for (int i = 0; i < SK_TM; ++i)
#pragma unroll
        for (int j = 0; j < SK_TN; ++j) acc[i][j] += (float)a[i] * (float)b[j];
    }
  }
  // 尾部
  for (; k < K; k += SK_SG) {
    half a[SK_TM], b[SK_TN];
#pragma unroll
    for (int i = 0; i < SK_TM; ++i) {
      const int m = m0 + i;
      a[i] = (m < M) ? A[(size_t)m * K + k] : (half)0;
    }
#pragma unroll
    for (int j = 0; j < SK_TN; ++j) {
      const int n = n0 + j;
      b[j] = (n < N) ? B[(size_t)k * N + n] : (half)0;
#if MUL_SCALE
      if (Scale != 0) b[j] = (half)((float)b[j] * (float)Scale[k]);
#endif
    }
#pragma unroll
    for (int i = 0; i < SK_TM; ++i)
#pragma unroll
      for (int j = 0; j < SK_TN; ++j) acc[i][j] += (float)a[i] * (float)b[j];
  }

  // sub-group 归约：**所有 lane 必须在一致控制流里调用 collective**（先归约到 s[]，
  // 再只让 lane 0 写回），否则部分 lane 跳过会导致错误的树归约（首版即栽在这里）。
  float s[SK_TM][SK_TN];
#pragma unroll
  for (int i = 0; i < SK_TM; ++i)
#pragma unroll
    for (int j = 0; j < SK_TN; ++j) s[i][j] = sub_group_reduce_add(acc[i][j]);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < SK_TM; ++i) {
      const int m = m0 + i;
      if (m >= M) continue;
      const half bv = (Bias != 0) ? Bias[m] : (half)0;
#pragma unroll
      for (int j = 0; j < SK_TN; ++j) {
        const int n = n0 + j;
        if (n >= N) continue;
        half v = (half)(s[i][j] + (float)bv);
        v = sk_activate(v);
#if RES
        if (Res != 0) v = v + Res[(size_t)m * N + n];
#endif
        C[(size_t)m * N + n] = v;
      }
    }
  }
}
