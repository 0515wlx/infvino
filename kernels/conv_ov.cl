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

// OV's get_bfyx_req_input_block_dims(): round the required input width up to a
// whole sub-group (read_chunk_size == SUB_GROUP_SIZE == 16), min one chunk.
#define IN_REQ_W (((OBW - 1) * STRIDE) + ((3 - 1) * DIL) + 1)
#define IN_REQ_H (((OBH - 1) * STRIDE) + ((3 - 1) * DIL) + 1)
#define IN_BLOCK_WIDTH ((((IN_REQ_W) + SG - 1) / SG) * SG)
#define IN_BLOCK_ARRAY_SIZE (((IN_REQ_H) * (IN_BLOCK_WIDTH) + SG - 1) / SG)

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
__attribute__((reqd_work_group_size(1, 1, SG)))
__kernel void conv3x3_ov(
  __global const half *restrict input,    // [Cin][H][W] (bfyx, contiguous)
  __global const half *restrict weights,  // OSV-swizzled [ceil(Cout/32)][Cin][3][3][32]
  __global const half *restrict bias,     // [Cout] or null
  __global const half *restrict residual, // [Cout][Hout][Wout] or null (RES=1)
  __global half *restrict output,         // [Cout][Hout][Wout]
  const int Cin, const int H, const int W,
  const int Cout, const int Hout, const int Wout) {
  const int oc  = get_global_id(0) * OBW;   // output column block
  const int orr = get_global_id(1) * OBH;   // output row block
  // OV: fm = get_global_id(2) (includes the sub-group lane); the feature-map
  // group is fm / SUB_GROUP_SIZE, and each lane owns channel fmg*OSV + lid.
  const int fmg = get_global_id(2) / SG;
  const int lid = get_sub_group_local_id();
  const int feature_idx = fmg * OSV + lid;   // this lane's first output channel

  half in[IN_BLOCK_ARRAY_SIZE];
  half2 out[OBW * OBH];
#pragma unroll
  for (int i = 0; i < OBW * OBH; ++i) out[i] = (half2)(0, 0);

  const int base_x = oc * STRIDE - PAD;
  const int base_y = orr * STRIDE - PAD;
  const int wbase = fmg * Cin * 9 * OSV;   // in halfs

  for (int kd = 0; kd < Cin; ++kd) {
    // ---- load the input block for this input channel (OV's scatter mapping) ----
#pragma unroll
    for (int q = 0; q < IN_BLOCK_ARRAY_SIZE; ++q) {
      const int pos = q * SG;
      const int row = pos / IN_BLOCK_WIDTH;
      const int col = lid + (pos % IN_BLOCK_WIDTH);
      const int yy = base_y + row;
      const int xx = base_x + col;
      half v = (half)0;
      if (yy >= 0 && yy < H && xx >= 0 && xx < W)
        v = input[((size_t)kd * H + yy) * W + xx];
      in[q] = v;
    }

#pragma unroll
    for (int kr = 0; kr < 3; ++kr) {
#pragma unroll
      for (int kc = 0; kc < 3; ++kc) {
        // One block read gives every lane its own two output-channel weights.
        const int woff = wbase + (kd * 9 + kr * 3 + kc) * OSV;
        const half2 w = as_half2(intel_sub_group_block_read_us2(
          (__global const ushort *)weights + woff));
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
  }

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
        output[oidx] = dst;
      }
    }
  }
}
