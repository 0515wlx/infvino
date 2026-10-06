// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// conv1x1_f16 —— 1x1 卷积（pointwise）专用 kernel，融合 bias + 激活 (+ 可选残差)。
//
// 动机（见 docs/kernel.md R6 / benchmark.md）：mobilenetv3-small 里 1x1 pointwise
// 占整网 kernel 时间的 ~2/3，此前全部路由到通用 `gemm_f16`，且每层还要额外拆出
// `bias_add` + `ew_unary` 两次 kernel launch。形状 M=Cout、N=H*W、K=Cin 里有大量
// N 很小（49/196）、M 中等（96..576）的层，GEMM 的 BM=128/BN=64 会严重网格饥饿
// （例：96x49x576 只有 1 个 work-group）。
//
// 数据通路：权重在 plan 生成期重排为 [Cin][Cout]（行主序，Cout 连续）。
//   * lane 方向 = 空间（dim0），相邻 lane 对应相邻空间位置 -> X/Y 完全 coalesced；
//   * 每 work-item 计算 TM 个输出通道 x TN 个空间位置；沿 Cin 归约；
//   * 每个 ci 只需 1 次 TN 宽的 X 读 + 1 次 TM 宽的 W 读（连续 half8/half2），
//     再做 TMxTN 个独立 mad -> 高 ILP、极低 load:FMA，且寄存器占用小
//     （acc = TM*TN half），可支撑海量 work-item 隐藏 L3/全局延迟；
//   * 全部操作数从 GPU-L3 读取：这些层 X+W 总共几十 KB–1 MB，天然驻留 L3。
//
// 网格：dim0 = ceil(HW / TN)，dim1 = ceil(Cout / TM)。
//
// 编译期参数（-D）：
//   TM  : 每 work-item 的输出通道数（建议 8/16，配合连续权重宽载）
//   TN  : 每 work-item 的空间位置数
//   ACT : 0=none 1=SiLU 2=Hardswish 3=Relu 4=Hardsigmoid（与 ops.cl ew_unary 同码）
//   RES : 1 = epilogue 先加残差输入 Res[Cout][HW]
//   SG  : 强制子组宽度（0 = 由 IGC 决定）
//
// License: original infvino code (project license). NOT derived from OpenVINO;
// see kernels/conv_ov.cl + THIRD_PARTY_NOTICES.md for the Apache-2.0 OV port.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 4
#endif
#ifndef ACT
#define ACT 0
#endif
#ifndef RES
#define RES 0
#endif
#ifndef SG
#define SG 0
#endif
#ifndef UNROLL
#define UNROLL 4
#endif

inline half c1x1_activate(half v) {
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));                          // SiLU
#elif ACT == 2
  return (half)fmax((float)v, 0.0f);                            // Relu
#elif ACT == 3
  float f = (float)v;
  return (half)(f * fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);  // Hardswish
#elif ACT == 4
  float f = (float)v;
  return (half)(fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);      // Hardsigmoid
#elif ACT == 5
  return (half)(1.0f / (1.0f + exp(-(float)v)));                // Sigmoid
#else
  return v;                                                     // ACT == 0: none
#endif
}

#if SG
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
__kernel void conv1x1_f16(
  __global const half *restrict Wt,    // [Cin][Cout] (repacked at plan build)
  __global const half *restrict X,     // [Cin][HW]
  __global const half *restrict Bias,  // [Cout] or null
  __global const half *restrict Res,   // [Cout][HW] or null (RES=1)
  __global half *restrict Y,           // [Cout][HW]
  const int Cin, const int Cout, const int HW) {
  const int n0 = get_global_id(0) * TN;
  const int m0 = get_global_id(1) * TM;

  half acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (half)0;

#pragma unroll UNROLL
  for (int ci = 0; ci < Cin; ++ci) {
    half xr[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + j;
      xr[j] = (n < HW) ? X[ci * HW + n] : (half)0;
    }
    // Consecutive output channels -> one wide (contiguous) weight load per lane.
    __global const half *wrow = Wt + (size_t)ci * Cout + m0;
    half wr[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i)
      wr[i] = (m0 + i < Cout) ? wrow[i] : (half)0;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = mad(wr[i], xr[j], acc[i][j]);
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int m = m0 + i;
    if (m >= Cout) continue;
    const half b = (Bias != 0) ? Bias[m] : (half)0;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + j;
      if (n >= HW) continue;
      half v = c1x1_activate(acc[i][j] + b);
#if RES
      v = v + Res[(size_t)m * HW + n];   // residual is applied AFTER activation
#endif
      Y[(size_t)m * HW + n] = v;
    }
  }
}

// ---------------------------------------------------------------------------
// conv1x1_gemv_f16 —— HW==1（N=1）的 1x1 conv / fc 专用：把每个输出通道的
// Cin 归约拆到 16 个 lane（split-K），sub_group_reduce_add 归约。
//
// 这是 mobilenetv3-small 的最大单点：SE 的 fc1/fc2 与 classifier 全是 N=1 的
// 矩阵-向量积，而通用 GEMM 在 N=1 时 BN=64 有 63/64 的 lane 被浪费、且网格只有
// ceil(M/BM) 个 work-group（例 1000x1x1024 只有 8 个），实测占整网 gemm 的 ~56%。
//
// 映射：一个 work-group = 一个 sub-group（16 lane）= 一个输出通道 m。
//   迭代 k = lane, lane+16, ...：W[m][k] 相邻 lane 连续（coalesced），X[k] 也连续；
//   每 lane 累加后 sub_group_reduce_add。权重保持自然 [Cout][Cin] 布局，无需转置。
//   网格 dim0 = Cout * 16（lws=16）→ work-group 数 = Cout（=1000 时 ~1000 个
//   sub-group，足以喂满 80 EU 的 7 threads/EU）。
//
// 编译期参数：ACT（同 conv1x1_f16）、SG（默认 16）、GEMV_TM（每子组输出通道数）。
//
// R48 §4bis「conv1x1 N=1 无多输出」：原实现一个子组只算一个输出通道，X 向量被每个
// 输出通道的子组重复读 Cout 次（X 读流量 = Cout*Cin）。GEMV_TM=T 时一个子组同时算 T 个
// 输出通道：X[k] 只读一次、供 T 个权重行复用（X 读流量 /T），并把 T 条独立归约链交错以
// 提升 ILP。T=1 与旧行为**逐位一致**（每个输出通道的累加顺序、reduction 不变）。
// 网格：dim0 = ceil(Cout/T)*GEMV_SG；kernel 由 group_id*T 得起始通道。
// ---------------------------------------------------------------------------
#ifndef GEMV_SG
#define GEMV_SG 16
#endif
#ifndef GEMV_TM
#define GEMV_TM 1
#endif

__attribute__((intel_reqd_sub_group_size(GEMV_SG)))
__attribute__((reqd_work_group_size(GEMV_SG, 1, 1)))
__kernel void conv1x1_gemv_f16(
  __global const half *restrict W,     // [Cout][Cin]  (natural layout)
  __global const half *restrict X,     // [Cin]
  __global const half *restrict Bias,  // [Cout] or null
  __global const half *restrict Res,   // [Cout] or null (RES=1)
  __global half *restrict Y,           // [Cout]
  const int Cin, const int Cout) {
  const int lane = get_local_id(0);
  const int m0 = get_group_id(0) * GEMV_TM;
  if (m0 >= Cout) return;

  float acc[GEMV_TM];
#pragma unroll
  for (int t = 0; t < GEMV_TM; ++t) acc[t] = 0.0f;

  for (int k = lane; k < Cin; k += GEMV_SG) {
    const float x = (float)X[k];
#pragma unroll
    for (int t = 0; t < GEMV_TM; ++t) {
      const int m = m0 + t;
      if (m < Cout) acc[t] += (float)W[(size_t)m * Cin + k] * x;
    }
  }

#pragma unroll
  for (int t = 0; t < GEMV_TM; ++t) {
    const int m = m0 + t;
    if (m >= Cout) continue;
    float s = sub_group_reduce_add(acc[t]);
    if (lane == 0) {
      half v = (half)s;
      if (Bias != 0) v = v + Bias[m];
      v = c1x1_activate(v);
#if RES
      v = v + Res[m];
#endif
      Y[m] = v;
    }
  }
}
