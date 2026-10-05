// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// ---------------------------------------------------------------------------
// Self-contained adaptation of OpenVINO's
//   src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_f16_1x1.cl
// (kernel `convolution_gpu_bfyx_f16_1x1`, selector
//  `ConvolutionKernel_b_fs_yx_fsv16_1x1`, the kernel OV's kernel selector picks
//  for 1x1 f16 pointwise layers on this TGL iGPU).
//
// Why this exists (per-op OV profiling, 2026-10): on mobilenetv3-small OV runs
// *all* pointwise (1x1) convs in ~0.42 ms while infvino's NCHW GEMM/GEMV path
// takes ~2.0 ms (≈4.8x).  OV's 1x1 kernel requires the blocked `b_fs_yx_fsv16`
// input/output layout, which infvino never had for the 1x1 family.
//
// Data path kept faithful:
//   * sub-group lanes map to OUTPUT CHANNELS (16 per fsv16 block, 1 lane == 1
//     output channel); a lane accumulates a vector of X_BLOCK consecutive output
//     x positions, so the per-input-channel shuffle is amortised over X_BLOCK
//     packed FMAs;
//   * input is read in `b_fs_yx_fsv16` ([C/16][H][W][16]) with
//     `intel_sub_group_block_read_usN` (lane l gets channel l, columns x..x+N-1);
//   * weights in `os_is_yx_isv16_osv16` = [Cout/16][Cin/16][isv16][osv16], read
//     with two `block_read_us8` (lane gets its own output channel's 16 input
//     channels);
//   * output written either `b_fs_yx_fsv16` (OUT_FSV16=1, keeps the blocked
//     chain alive, matching OV) or plain bfyx (OUT_FSV16=0, drop-in for infvino).
//
// Simplified vs upstream: stride=1/pad=0 only (a 1x1 conv), optional
// SLM_DIV split-K reduction, runtime leftovers, no JIT OOB macro layer.
//
// See THIRD_PARTY_NOTICES.md / third_party/openvino/LICENSE.
// Upstream: https://github.com/openvinotoolkit/openvino/.../convolution_gpu_bfyx_f16_1x1.cl
// ---------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#ifndef SG              // sub-group size == fsv16 block
#define SG 16
#endif
#ifndef X_BLOCK         // consecutive output columns per lane (OV's X_BLOCK_SIZE)
#define X_BLOCK 4
#endif
#ifndef SLM_DIV         // split-K across sub-groups inside the work-group
#define SLM_DIV 1
#endif
// R48 D1: 输出空间 tiling —— 每个 work-item 处理 Y_BLOCK 个连续输出行，权重（寄存器里
// 的 wei[16]）跨行复用，把权重读指令摊薄到 Y_BLOCK 个输出行上。原版每行每 K 步都重读
// 权重，是 blocked 1x1 的发射/带宽主项（Cout/X_BLOCK 倍于输入读）。数值上与 Y_BLOCK=1
// **逐位一致**（每个输出的 K 累加顺序不变）。
#ifndef Y_BLOCK
#define Y_BLOCK 1
#endif
#ifndef ACT             // 1x1 act codes (same as kernels/conv1x1.cl):
#define ACT 0           //  1=SiLU 2=ReLU 3=HardSwish 4=HardSigmoid 5=Sigmoid 0=none
#endif
#ifndef OUT_FSV16
#define OUT_FSV16 0
#endif
// R48 D4: 消费者侧融合 reorder —— 输入直接读 **NCHW** [Cin][H][W]，lane 按自己的输入通道
// gc=k*16+sglid 逐元素读取，省掉独立的 `reorder_bfyx_to_fsv16` pass 及其 launch。
// 数值逐位一致（K 累加顺序、每输出计算不变）。lane 间地址跨 H*W（非合并读），故只在
// **Cin 较小**（reorder 以 launch 为主、输入大概率 L2 命中）时值得——由 autotune 按签名选。
#ifndef IN_NCHW
#define IN_NCHW 0
#endif
#ifndef RES             // 1 = 加残差（在激活之后，语义同 gemm_f16/conv1x1.cl）
#define RES 0
#endif
#ifndef FIT_CIN         // Cin % 16 == 0 -> no input-channel leftover predicate
#define FIT_CIN 0
#endif
#ifndef FIT_COUT        // Cout % 16 == 0 -> no output-channel leftover predicate
#define FIT_COUT 0
#endif
#ifndef FIT_WH          // W % X_BLOCK == 0 -> no column-wrap predicate
#define FIT_WH 0
#endif
// R51 D5: MUL_SCALE=1 -> activation scaled per input channel (lane's gc) on load,
// fusing a channel-broadcast Mul (SE `x*scale[c]`) into the conv prologue (removes the
// separate elementwise pass). Product rounded to half before mad -> bitwise-identical.
#ifndef MUL_SCALE
#define MUL_SCALE 0
#endif

#define FS 16           // FEATURE_SLICE_SIZE

// Vector accumulator / packed shuffle (OV's GET_SRC).
#if X_BLOCK == 8
#define VEC_T half8
#define RWV_T ushort8
#define AS_RW(x) as_ushort8(x)
#define AS_V(x) as_half8(x)
#define BLOCK_READ(p) intel_sub_group_block_read_us8(p)
#elif X_BLOCK == 4
#define VEC_T half4
#define RWV_T ushort4
#define AS_RW(x) as_ushort4(x)
#define AS_V(x) as_half4(x)
#define BLOCK_READ(p) intel_sub_group_block_read_us4(p)
#elif X_BLOCK == 2
#define VEC_T half2
#define RWV_T ushort2
#define AS_RW(x) as_ushort2(x)
#define AS_V(x) as_half2(x)
#define BLOCK_READ(p) intel_sub_group_block_read_us2(p)
#else
#error "conv1x1_blk: X_BLOCK must be 2, 4 or 8"
#endif

#define GET_SRC(data, id) AS_V(intel_sub_group_shuffle(AS_RW(data), (uint)(id)))

inline half c1x1blk_activate(half v) {
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));                          // SiLU
#elif ACT == 2
  return (half)fmax((float)v, 0.0f);                            // ReLU
#elif ACT == 3
  float f = (float)v;
  return (half)(f * fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);  // HardSwish
#elif ACT == 4
  float f = (float)v;
  return (half)(fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);      // HardSigmoid
#elif ACT == 5
  return (half)(1.0f / (1.0f + exp(-(float)v)));                // Sigmoid
#else
  return v;                                                     // none
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(1, SG * SLM_DIV, 1)))
__kernel void conv1x1_blk(
  __global const half *restrict input,    // b_fs_yx_fsv16 [Cin/16][H][W][16]
  __global const half *restrict weights,  // os_is_yx_isv16_osv16 [Cout/16][Cin/16][16][16]
  __global const half *restrict bias,     // [Cout] or null
  __global half *restrict output,         // fsv16 or bfyx (OUT_FSV16)
  __global const half *restrict residual, // [Cout][H][W] NCHW or null (RES=1)
  const int Cin, const int H, const int W,
  const int Cout
#if MUL_SCALE
  , __global const half *restrict Scale    // [Cin] (MUL_SCALE=1)
#endif
)
{
  const int sglid = get_sub_group_local_id();
  const int lid1  = get_local_id(1);
  const int feature_per_wg = SG;
  const int feature_sub_block = lid1 / feature_per_wg;
  const int feature_block = get_group_id(1);     // output-channel / 16 block

  const int X_BLOCKS = (W + X_BLOCK - 1) / X_BLOCK;
  const int xy = get_global_id(0);
  const int x0 = (xy % X_BLOCKS) * X_BLOCK;
  const int y0 = (xy / X_BLOCKS) * Y_BLOCK;   // R48 D1: 输出行 tiling 的起始行

  const int ic_blocks = (Cin + FS - 1) / FS;
  const int input_y_pitch  = FS * W;
  const int input_fs_pitch = FS * W * H;
  const int input_xoff     = x0 * FS;

  const int filter_os_pitch = FS * FS * ic_blocks;
  const int filter_offset   = feature_block * filter_os_pitch;

#if X_BLOCK > 1
  typedef VEC_T vec_t;
#else
  typedef half vec_t;
#endif

  // R48 D1: Y_BLOCK 个独立累加器（每个输出行一个）。
  vec_t dst[Y_BLOCK];
#pragma unroll
  for (int r = 0; r < Y_BLOCK; ++r) dst[r] = (vec_t)0;

#if SLM_DIV > 1
  __local vec_t partial_summ[Y_BLOCK * SG * SLM_DIV];
#endif

#if SLM_DIV > 1
  for (int k = feature_sub_block * ic_blocks / SLM_DIV;
       k < (feature_sub_block + 1) * ic_blocks / SLM_DIV; ++k)
#else
  for (int k = 0; k < ic_blocks; ++k)
#endif
  {
    const int gc = k * FS + sglid;
#if FIT_CIN
    const bool in_left = false;
#else
    const bool in_left = (gc >= Cin);
#endif

    // R48 D1: 权重每 K 步只读一次，跨 Y_BLOCK 个输出行复用（原版每行重读）。
    const int woff = filter_offset + k * FS * FS;
    ushort8 w0 = intel_sub_group_block_read_us8((__global const ushort *)weights + woff);
    ushort8 w1 = intel_sub_group_block_read_us8((__global const ushort *)weights + woff + 8 * FS);
    half wei[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) wei[j] = as_half(w0[j]);
#pragma unroll
    for (int j = 0; j < 8; ++j) wei[8 + j] = as_half(w1[j]);

#pragma unroll
    for (int r = 0; r < Y_BLOCK; ++r) {
      const int y = y0 + r;
      VEC_T src = (VEC_T)0;
      if (y < H) {
#if IN_NCHW
        // R48 D4: 直接读 NCHW（lane 自己的输入通道 gc）。无独立 reorder pass。
#pragma unroll
        for (int i = 0; i < X_BLOCK; ++i) {
          const int xx = x0 + i;
          src[i] = (in_left || xx >= W) ? (half)0
                   : input[(size_t)gc * H * W + (size_t)y * W + xx];
        }
#else
        const int base = y * input_y_pitch + input_xoff + k * input_fs_pitch;
#if FIT_WH
        // Aligned: one packed block read (lane l gets channel l, columns x0..x0+X-1).
        if (!in_left) src = AS_V(BLOCK_READ((__global const ushort *)input + base));
#else
        if (!in_left && x0 + X_BLOCK <= W) {
          src = AS_V(BLOCK_READ((__global const ushort *)input + base));
        } else {
#pragma unroll
          for (int i = 0; i < X_BLOCK; ++i) {
            const int xx = x0 + i;
            src[i] = (in_left || xx >= W) ? (half)0 : input[base + i * FS + sglid];
          }
        }
#endif
#endif
      }
#if MUL_SCALE
      if (Scale != 0 && !in_left) {
#pragma unroll
        for (int i = 0; i < X_BLOCK; ++i)
          src[i] = (half)((float)src[i] * (float)Scale[gc]);
      }
#endif
#pragma unroll
      for (int id = 0; id < 16; ++id)
        dst[r] = mad(wei[id], GET_SRC(src, id), dst[r]);
    }
  }

  const int oc = feature_block * FS + sglid;
#if FIT_COUT
  const bool oob = false;
#else
  const bool oob = (oc >= Cout);
#endif

#if SLM_DIV > 1
#pragma unroll
  for (int r = 0; r < Y_BLOCK; ++r) partial_summ[r * SG * SLM_DIV + lid1] = dst[r];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (feature_sub_block == 0) {
#pragma unroll
    for (int r = 0; r < Y_BLOCK; ++r)
#pragma unroll
      for (int i = 1; i < SLM_DIV; ++i) dst[r] += partial_summ[r * SG * SLM_DIV + sglid + i * SG];
  } else {
    return;  // only sub-block 0 writes the output
  }
#endif

#pragma unroll
  for (int r = 0; r < Y_BLOCK; ++r) {
    const int y = y0 + r;
    if (y >= H) continue;
    if (bias != 0 && !oob) dst[r] += (vec_t)(bias[oc]);  // broadcast bias
#pragma unroll
    for (int i = 0; i < X_BLOCK; ++i) {
      const int ox = x0 + i;
      if (ox >= W || oob) continue;
      half v = c1x1blk_activate(dst[r][i]);
#if RES
      if (residual != 0) v = v + residual[((size_t)oc * H + y) * W + ox];
#endif
#if OUT_FSV16
      output[(((size_t)feature_block * H + y) * W + ox) * FS + sglid] = v;
#else
      output[((size_t)oc * H + y) * W + ox] = v;
#endif
    }
  }
}

// bfyx [C][H][W] -> b_fs_yx_fsv16 [C/16][H][W][16].  Shared with conv_blk.cl's
// reorder but kept local so this file is self-contained (same signature).
__kernel void reorder_bfyx_to_fsv16_1x1(
  __global const half *restrict in,
  __global half *restrict out,
  const int C, const int H, const int W) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int c = get_global_id(2);
  if (x >= W || y >= H || c >= C) return;
  out[(((size_t)(c / 16) * H + y) * W + x) * 16 + (c % 16)] =
      in[((size_t)c * H + y) * W + x];
}
