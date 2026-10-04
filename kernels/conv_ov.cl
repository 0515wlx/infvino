// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// ---------------------------------------------------------------------------
// This file is a self-contained adaptation of OpenVINO's
//   src/plugins/intel_gpu/src/kernel_selector/cl_kernels/
//       convolution_gpu_bfyx_os_iyx_osv32.cl
// (kernel name `convolution_gpu_bfyx_os_iyx_osv32`), which is
//   Copyright (C) 2018-2026 Intel Corporation, licensed under Apache-2.0.
//
// infvino vendors the *algorithm and data path* of that kernel here so it can be
// used without pulling in OpenVINO as a build/runtime dependency. The OpenCL
// source has been inlined into a single self-contained file (OV's `include/*.cl`
// helper headers and JIT macro layer are replaced by local definitions), but the
// data path is kept faithful:
//
//   * sub-group lanes map to OUTPUT CHANNELS (OSV_SIZE=32, 2 channels per lane);
//   * the input block for one input channel is loaded once into a per-lane
//     register array `in[]` and distributed with `sub_group_broadcast`;
//   * weights are read with `intel_sub_group_block_read_us2` from an OSV-swizzled
//     layout (see the host-side `swizzleOsvWeights` helper);
//   * NO __local memory and NO barrier;
//   * bias + activation in the output phase.
//
// See THIRD_PARTY_NOTICES.md and third_party/openvino/LICENSE for the full
// attribution and the Apache-2.0 license text.
//
// Upstream reference (fetched 2026): 
//   https://github.com/openvinotoolkit/openvino/blob/master/
//   src/plugins/intel_gpu/src/kernel_selector/cl_kernels/
//   convolution_gpu_bfyx_os_iyx_osv32.cl
// ---------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#ifndef OBW
#ifdef TX
#define OBW TX
#else
#define OBW 8            // OUTPUT_BLOCK_WIDTH
#endif
#endif
#ifndef OBH
#ifdef TY
#define OBH TY
#else
#define OBH 2            // OUTPUT_BLOCK_HEIGHT
#endif
#endif
#ifndef STRIDE
#define STRIDE 1
#endif
#ifndef PAD
#define PAD 1
#endif
#ifndef DIL
#define DIL 1
#endif
#ifndef ACT
#define ACT 0            // 0=none 1=SiLU 2=Hardswish (infvino conv3x3 act codes)
#endif
#ifndef RES
#define RES 0            // 1 = add residual[Cout][Hout][Wout] after activation
#endif
#ifndef SG
#define SG 16            // SUB_GROUP_SIZE
#endif
#ifndef OSV
#define OSV 32           // OSV_SIZE
#endif
// Round 28 (P2): compile-time shape specialization. When the output is exactly
// tile-aligned / channel-aligned the per-output predicates are dead code and IGC
// can drop them (no runtime branch — the earlier R24 attempt used a *runtime*
// interior test and lost). Set by the autotuner as extra candidates.
#ifndef FIT_WH
#define FIT_WH 0         // Wout % OBW == 0 && Hout % OBH == 0
#endif
#ifndef FIT_COUT
#define FIT_COUT 0       // Cout % (2*SG) == 0
#endif
// R37: split the input-channel loop across SLM_DIV sub-groups in one work-group
// (upstream SLM_DIV_FACTOR-style), then reduce in SLM. More in-flight work per WG
// to hide global-read latency on occupancy-limited shapes. 1 = disabled.
#ifndef SLM_DIV
#define SLM_DIV 1
#endif
// R41 diagnostic (kernel_bench only; production never sets it): isolate the two
// global feeds from the FMA pipeline. 0=off, 1=replace the input tile load with a
// constant, 2=replace the weight block read with a constant. Numerically wrong by
// design — perf attribution only.
#ifndef PROBE
#define PROBE 0
#endif

// OV's get_bfyx_req_input_block_dims(): round the required input width up to a
// whole sub-group (read_chunk_size == SUB_GROUP_SIZE == 16), min one chunk.
#define IN_REQ_W (((OBW - 1) * STRIDE) + ((3 - 1) * DIL) + 1)
#define IN_REQ_H (((OBH - 1) * STRIDE) + ((3 - 1) * DIL) + 1)
#define IN_BLOCK_WIDTH ((((IN_REQ_W) + SG - 1) / SG) * SG)
#define IN_BLOCK_ARRAY_SIZE (((IN_REQ_H) * (IN_BLOCK_WIDTH) + SG - 1) / SG)

// R41: load one input channel's block (OV's scatter mapping) into DST[*]. Factored
// out so the prefetch (PF) path can issue it a full `kd` ahead of its use.
#define OV_LOAD_BLOCK(DST, KDV)                                                \
  _Pragma("unroll")                                                            \
  for (int q_ = 0; q_ < IN_BLOCK_ARRAY_SIZE; ++q_) {                           \
    const int pos_ = q_ * SG;                                                  \
    const int row_ = pos_ / IN_BLOCK_WIDTH;                                    \
    const int col_ = lid + (pos_ % IN_BLOCK_WIDTH);                            \
    const int yy_ = base_y + row_;                                             \
    const int xx_ = base_x + col_;                                             \
    half v_ = (half)0;                                                         \
    if (yy_ >= 0 && yy_ < H && xx_ >= 0 && xx_ < W)                            \
      v_ = input[((size_t)(KDV) * H + yy_) * W + xx_];                         \
    DST[q_] = v_;                                                              \
  }

inline half ov_activate(half v) {
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));
#elif ACT == 3
  float f = (float)v;
  return (half)(f * fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);
#else
  return v;
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
// R39: the SLM_DIV sub-groups that split the input-channel loop must be laid out
// along the *fastest-changing* local dimension after x, so that the SG lanes of a
// sub-group share the same `sub`. With the old (1, SLM_DIV, SG) geometry the
// sub-group was formed across the y/z dims (y fastest), so `get_local_id(1)` varied
// *within* a sub-group: different lanes accumulated different kd ranges and the
// reduction then read uninitialized `partial[]` slots -> NaN/inf. All SLM_DIV
// sub-groups now live in local dim 2 (local dims 1 = 1), which for SLM_DIV==1 is
// bit-identical to the pre-R37 geometry.
__attribute__((reqd_work_group_size(1, 1, SG * SLM_DIV)))
__kernel void conv3x3_ov(
  __global const half *restrict input,    // [Cin][H][W] (bfyx, contiguous)
  __global const half *restrict weights,  // OSV-swizzled [ceil(Cout/32)][Cin][3][3][32]
  __global const half *restrict bias,     // [Cout] or null
  __global const half *restrict residual, // [Cout][Hout][Wout] or null (RES=1)
  __global half *restrict output,         // [Cout][Hout][Wout]
  const int Cin, const int H, const int W,
  const int Cout, const int Hout, const int Wout) {
  const int oc  = get_global_id(0) * OBW;   // output column block
  const int orr = get_group_id(1) * OBH;    // output row block
  // OV: fm = get_global_id(2) (includes the sub-group lane); the feature-map
  // group is fm / SUB_GROUP_SIZE, and each lane owns channel fmg*OSV + lid.
  // R39: local dim 2 holds the SG lanes *and* the SLM_DIV sub-groups, so the
  // output-channel sub-group is the work-group id along dim 2 (local dim 2 size is
  // SG*SLM_DIV) and the lanes of a sub-group share the same `sub`.
  const int fmg = get_group_id(2);          // output-channel sub-group (lanes live in local z)
  const int lid = get_sub_group_local_id();
  const int sub = get_sub_group_id();        // R37/R39: sub-block over input channels
  const int feature_idx = fmg * OSV + lid;   // this lane's first output channel

  half in[IN_BLOCK_ARRAY_SIZE];
#if PF
  half in2[IN_BLOCK_ARRAY_SIZE];   // R41 prefetch buffer (diagnostic)
#endif
  half2 out[OBW * OBH];
#pragma unroll
  for (int i = 0; i < OBW * OBH; ++i) out[i] = (half2)(0, 0);

  const int base_x = oc * STRIDE - PAD;
  const int base_y = orr * STRIDE - PAD;
  const int wbase = fmg * Cin * 9 * OSV;   // in halfs

#if SLM_DIV > 1
  const int kd_begin = sub * Cin / SLM_DIV;
  const int kd_end = (sub + 1) * Cin / SLM_DIV;
#else
  const int kd_begin = 0;
  const int kd_end = Cin;
#endif
#if PF
  OV_LOAD_BLOCK(in, kd_begin)   // prime the software pipeline
#endif
  for (int kd = kd_begin; kd < kd_end; ++kd) {
#if PF
    // `in` already holds kd's block; issue kd+1's load now so its ~150-cyc global
    // latency overlaps this iteration's mads.
    if (kd + 1 < kd_end) OV_LOAD_BLOCK(in2, kd + 1)
#else
    // ---- load the input block for this input channel (OV's scatter mapping) ----
#pragma unroll
    for (int q = 0; q < IN_BLOCK_ARRAY_SIZE; ++q) {
      const int pos = q * SG;
      const int row = pos / IN_BLOCK_WIDTH;
      const int col = lid + (pos % IN_BLOCK_WIDTH);
      const int yy = base_y + row;
      const int xx = base_x + col;
      half v = (half)0;
#if PROBE == 1
      v = (half)0.25;   // R41: no input global load
#elif PROBE == 3
      v = (half)0.25;   // R41: no input load (combined probe)
#else
      if (yy >= 0 && yy < H && xx >= 0 && xx < W)
        v = input[((size_t)kd * H + yy) * W + xx];
#endif
      in[q] = v;
    }
#endif

#pragma unroll
    for (int kr = 0; kr < 3; ++kr) {
#pragma unroll
      for (int kc = 0; kc < 3; ++kc) {
        // One block read gives every lane its own two output-channel weights.
        const int woff = wbase + (kd * 9 + kr * 3 + kc) * OSV;
#if PROBE == 2
        const half2 w = (half2)((half)0.5, (half)0.5);   // R41: no weight global load
#elif PROBE == 3
        const half2 w = (half2)((half)0.5, (half)0.5);   // R41: no weight load (combined)
#else
        const half2 w = as_half2(intel_sub_group_block_read_us2(
          (__global const ushort *)weights + woff));
#endif
#pragma unroll
        for (int br = 0; br < OBH; ++br) {
#pragma unroll
          for (int bc = 0; bc < OBW; ++bc) {
            const int y_pos = br * STRIDE + kr * DIL;
            const int x_pos = bc * STRIDE + kc * DIL;
            const int e = y_pos * IN_BLOCK_WIDTH + x_pos;
            const half val = sub_group_broadcast(in[e / SG], e % SG);
            out[br * OBW + bc] = mad(w, (half2)(val, val), out[br * OBW + bc]);
          }
        }
      }
    }
#if PF
    // rotate the prefetch buffer for the next iteration
#pragma unroll
    for (int q = 0; q < IN_BLOCK_ARRAY_SIZE; ++q) in[q] = in2[q];
#endif
  }

#if SLM_DIV > 1
  // Reduce the per-sub-group partial sums; sub-block 0 owns the final output.
  __local half2 partial[SLM_DIV * SG * (OBW * OBH)];
#pragma unroll
  for (int i = 0; i < OBW * OBH; ++i) partial[(sub * SG + lid) * (OBW * OBH) + i] = out[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sub == 0) {
#pragma unroll
    for (int s = 1; s < SLM_DIV; ++s)
#pragma unroll
      for (int i = 0; i < OBW * OBH; ++i)
        out[i] += partial[(s * SG + lid) * (OBW * OBH) + i];
  }
  if (sub != 0) return;
#endif

  // ---- output phase: 2 channels per lane (fid = 0,1), bias + activation ----
#pragma unroll
  for (int fid = 0; fid < 2; ++fid) {
    const int ch = fmg * OSV + fid * SG + lid;
#if !FIT_COUT
    if (ch >= Cout) continue;
#endif
    const half b = (bias != 0) ? bias[ch] : (half)0;
#pragma unroll
    for (int r = 0; r < OBH; ++r) {
      const int oy = orr + r;
#if !FIT_WH
      if (oy >= Hout) continue;
#endif
#pragma unroll
      for (int c = 0; c < OBW; ++c) {
        const int ox = oc + c;
#if !FIT_WH
        if (ox >= Wout) continue;
#endif
        const half2 a = out[r * OBW + c];
        const half v = (fid == 0) ? a.s0 : a.s1;
        const size_t oidx = ((size_t)ch * Hout + oy) * Wout + ox;
        half dst = ov_activate(v + b);
#if RES
        dst = dst + residual[oidx];   // residual after activation
#endif
#if PROBE == 4
        if (dst == (half)12345.0f) output[oidx] = dst;   // R41: no store traffic
#else
        output[oidx] = dst;
#endif
      }
    }
  }
}
