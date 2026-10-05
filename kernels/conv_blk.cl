// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// ---------------------------------------------------------------------------
// Self-contained adaptation of OpenVINO's
//   src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_f16.cl
// (kernel `convolution_gpu_bfyx_f16`, selector
//  `ConvolutionKernel_b_fs_yx_fsv16`, which is what OV's kernel selector picks
//  for 3x3 f16 layers on this TGL iGPU).
//
// Data path kept faithful:
//   * sub-group lanes map to OUTPUT CHANNELS (16 channels per work-group,
//     1 lane == 1 output channel);
//   * each lane accumulates a *vector* of OUTPUT_X_BLOCK_SIZE consecutive output
//     x positions (`dst`), so one `mad` covers OBW positions -> the per-input-
//     channel shuffle is amortised over OBW/2 packed FMAs;
//   * input is read in the blocked `b_fs_yx_fsv16` layout ([C/16][H][W][16]):
//     lane `l` loads input channel (icb*16 + l) for the whole input line, then
//     `sub_group_shuffle` distributes channel `id`'s line to all lanes;
//   * weights in `os_is_yx_isv16_osv16` = [Cout/16][Cin/16][3][3][isv16][osv16],
//     read with `intel_sub_group_block_read_us8` (lane gets its own channel);
//   * output written back in plain bfyx [Cout][Hout][Wout] (OV's OUTPUT_FORMAT_BFYX
//     post-reorder-fused path) so it is drop-in for infvino.
//
// Simplified vs upstream: no grouping, no SLM_DIV_FACTOR, no
// MULTIPLE_GROUPS_INPUT_PRELOAD, no OOB-guard macros. Leftover input/output
// channels are handled at runtime.
//
// See THIRD_PARTY_NOTICES.md / third_party/openvino/LICENSE.
// Upstream: https://github.com/openvinotoolkit/openvino/.../convolution_gpu_bfyx_f16.cl
// ---------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#ifndef OBW            // OUTPUT_X_BLOCK_SIZE (OV uses 2/4/8)
#define OBW 8
#endif
#ifndef STRIDE
#define STRIDE 1
#endif
#ifndef PAD
#define PAD 1
#endif
#ifndef ACT
#define ACT 0             // 0=none 1=SiLU 2=Hardswish (infvino codes)
#endif
#ifndef SG
#define SG 16
#endif
// Round 28 (P2): compile-time shape specialization (see conv_ov.cl). Removing the
// leftover predicates when the shape is tile/channel aligned lets IGC drop the
// branch entirely (numerically identical).
#ifndef FIT_WH
#define FIT_WH 0          // Wout % OBW == 0 && Hout % 1 == 0
#endif
#ifndef FIT_COUT
#define FIT_COUT 0        // Cout % 16 == 0
#endif
#ifndef FIT_CIN
#define FIT_CIN 0         // Cin % 16 == 0
#endif
// R36 (P1-layout): persist the blocked layout on the output side. When the consumer
// is another `conv3x3_blk` (and every consumer is), we write `b_fs_yx_fsv16` directly
// instead of bfyx, so the consumer can read it with zero reorder. Requires Cout % 16 == 0
// (the planner only sets it then), so the store index stays in bounds.
#ifndef OUT_FSV16
#define OUT_FSV16 0
#endif
// R37: split the input-channel loop across SLM_DIV sub-groups in one
// work-group (upstream `SLM_DIV_FACTOR`), then reduce in SLM. More in-flight work
// per WG to hide the global-read latency that dominates the occupancy-limited
// shapes. 1 = disabled.
#ifndef SLM_DIV
#define SLM_DIV 1
#endif

#define FEATURE_SLICE_SIZE 16
// INPUT_LINE_SIZE = stride*(OBW-1) + (3-1)*dil + 1 = stride*(OBW-1) + 3
#define INPUT_LINE_SIZE (STRIDE * (OBW - 1) + 3)
#define INPUT_BLOCK_SIZE (((INPUT_LINE_SIZE * 3) + SG - 1) / SG)

// Vector accumulator type + the packed sub-group shuffle (OV's GET_SRC): shuffle
// OBW halves as one packed ushort vector so IGC lowers it to a wide move and the
// subsequent `mad` runs packed (half2/half4/half8).
#if OBW == 8
#define VEC_T half8
#define BLK_AS_VEC(x) as_half8(x)
#define BLK_AS_USVEC(x) as_ushort8(x)
#elif OBW == 4
#define VEC_T half4
#define BLK_AS_VEC(x) as_half4(x)
#define BLK_AS_USVEC(x) as_ushort4(x)
#elif OBW == 2
#define VEC_T half2
#define BLK_AS_VEC(x) as_half2(x)
#define BLK_AS_USVEC(x) as_ushort2(x)
#else
#error "conv_blk: OBW must be 2, 4 or 8"
#endif
#define BLK_GET_SRC(data, id)                                                  \
  BLK_AS_VEC(intel_sub_group_shuffle(BLK_AS_USVEC(data), (uint)(id)))

inline half blk_activate(half v) {
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));                        // SiLU
#elif ACT == 3
  float f = (float)v;
  return (half)(f * fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);  // Hardswish
#else
  return v;
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(1, SG * SLM_DIV, 1)))
__kernel void conv3x3_blk(
  __global const half *restrict input,    // [Cin/16][H][W][16]
  __global const half *restrict weights,  // [Cout/16][Cin/16][3][3][16 isv][16 osv]
  __global const half *restrict bias,     // [Cout] or null
  __global half *restrict output,         // [Cout][Hout][Wout] (bfyx)
  const int Cin, const int H, const int W,
  const int Cout, const int Hout, const int Wout) {
  const int lid = get_sub_group_local_id();
  const int lid1 = get_local_id(1);
  const int feature_sub_block = lid1 / SG;
  const int f_block = get_group_id(1);
  const int xy = get_global_id(0);
  const int X_BLOCKS = (Wout + OBW - 1) / OBW;
  const int x = (xy % X_BLOCKS) * OBW;
  const int y = xy / X_BLOCKS;

  const int input_x = x * STRIDE - PAD;
  const int input_y = y * STRIDE - PAD;

  const int input_x_pitch = 16;
  const int input_y_pitch = 16 * W;
  const int input_fs_pitch = 16 * W * H;
  const int input_offset = (input_y * W + input_x) * 16;

  const int filter_isv_pitch = 16;
  const int filter_x_pitch = 16 * 16;
  const int filter_y_pitch = filter_x_pitch * 3;
  const int filter_is_pitch = filter_y_pitch * 3;
  const int ic_blocks = (Cin + 15) / 16;
  const int filter_os_pitch = filter_is_pitch * ic_blocks;
  const int filter_offset = f_block * filter_os_pitch;

  VEC_T dst = (VEC_T)0;

#if SLM_DIV > 1
  const int icb_begin = feature_sub_block * ic_blocks / SLM_DIV;
  const int icb_end = (feature_sub_block + 1) * ic_blocks / SLM_DIV;
#else
  const int icb_begin = 0;
  const int icb_end = ic_blocks;
#endif
  for (int icb = icb_begin; icb < icb_end; ++icb) {
    const int gc = icb * 16 + lid;                 // this lane's input channel
#if FIT_CIN
    const bool in_left = false;
#else
    const bool in_left = (gc >= Cin);              // input-channel leftover
#endif
    const int fs_off = input_offset + icb * input_fs_pitch;
#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
      const int iy = input_y + kh;
      const bool row_ok = (iy >= 0) && (iy < H);
      const int line_base = fs_off + kh * input_y_pitch;
      half line_cache[INPUT_LINE_SIZE];
      // Fast path: the whole input line is in bounds and this is not an
      // input-channel leftover -> load it with coalesced 8-wide block reads
      // (lane l takes input channel l, columns xb..xb+7), mirroring OV.
      const bool line_ok = row_ok && !in_left && (input_x >= 0) &&
                           (input_x + INPUT_LINE_SIZE <= W);
      if (line_ok) {
        int xb = 0;
#pragma unroll
        for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
          ushort8 vv = intel_sub_group_block_read_us8(
              (__global const ushort *)input + line_base + xb * input_x_pitch);
#pragma unroll
          for (int j = 0; j < 8; ++j) line_cache[xb + j] = as_half(vv[j]);
        }
#pragma unroll
        for (; xb < INPUT_LINE_SIZE; ++xb)
          line_cache[xb] = input[line_base + xb * input_x_pitch + lid];
      } else {
#pragma unroll
        for (int xb = 0; xb < INPUT_LINE_SIZE; ++xb) {
          const int ix = input_x + xb;
          half v = (half)0;
          if (!in_left && row_ok && ix >= 0 && ix < W)
            v = input[line_base + xb * input_x_pitch + lid];
          line_cache[xb] = v;
        }
      }
#pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        const int woff = filter_offset + icb * filter_is_pitch +
                         kh * filter_y_pitch + kw * filter_x_pitch;
        // lane `lid` gets the 16 input-channel weights of its own output channel
        ushort8 w0 = intel_sub_group_block_read_us8((__global const ushort *)weights + woff);
        ushort8 w1 = intel_sub_group_block_read_us8((__global const ushort *)weights + woff + 8 * filter_isv_pitch);
        half wei[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) wei[j] = as_half(w0[j]);
#pragma unroll
        for (int j = 0; j < 8; ++j) wei[8 + j] = as_half(w1[j]);

        VEC_T src = (VEC_T)0;
#pragma unroll
        for (int i = 0; i < OBW; ++i) src[i] = line_cache[kw + STRIDE * i];

#pragma unroll
        for (int id = 0; id < 16; ++id) {
          const VEC_T sv = BLK_GET_SRC(src, id);
          dst = mad(wei[id], sv, dst);
        }
      }
    }
  }

#if SLM_DIV > 1
  // Reduce the per-sub-group partial sums. Sub-block 0 owns the final output.
  __local VEC_T partial_summ[SG * SLM_DIV];
  partial_summ[lid1] = dst;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (feature_sub_block == 0) {
#pragma unroll
    for (int i = 1; i < SLM_DIV; ++i) dst += partial_summ[(lid1 % SG) + i * SG];
  }
  if (feature_sub_block != 0) return;
#endif

  const int oc = f_block * 16 + lid;
#if FIT_COUT
  const bool out_left = false;
#else
  const bool out_left = (oc >= Cout);
#endif
  half b = (bias != 0 && !out_left) ? bias[oc] : (half)0;
  VEC_T res;
#pragma unroll
  for (int i = 0; i < OBW; ++i) res[i] = blk_activate(dst[i] + b);
#if OUT_FSV16
#pragma unroll
  for (int i = 0; i < OBW; ++i) {
    const int ox = x + i;
#if FIT_WH
    if (!out_left)
#else
    if (!out_left && ox < Wout && y < Hout)
#endif
    {
      // b_fs_yx_fsv16: [Cout/16][Hout][Wout][16]; oc == f_block*16 + lid.
      output[(((size_t)f_block * Hout + y) * Wout + ox) * 16 + lid] = res[i];
    }
  }
#else
  // R37: coalesced vector store. The lane's OBW outputs are
  // contiguous in bfyx; one vstoreN replaces OBW scalar stores when the base is
  // aligned. (OV's VSTORE path.)
  const size_t obase = ((size_t)oc * Hout + y) * Wout + x;
#if FIT_WH
  const bool vec_ok = !out_left && (x + OBW <= Wout) && ((obase % OBW) == 0);
#else
  const bool vec_ok = !out_left && y < Hout && (x + OBW <= Wout) && ((obase % OBW) == 0);
#endif
  if (vec_ok) {
#if OBW == 8
    vstore8(res, 0, output + obase);
#elif OBW == 4
    vstore4(res, 0, output + obase);
#else
    vstore2(res, 0, output + obase);
#endif
  } else {
#pragma unroll
    for (int i = 0; i < OBW; ++i) {
      const int ox = x + i;
#if FIT_WH
      if (!out_left)
#else
      if (!out_left && ox < Wout && y < Hout)
#endif
        output[obase + i] = res[i];
    }
  }
#endif
}

// bfyx [C][H][W] -> b_fs_yx_fsv16 [C/16][H][W][16] (host-side reorder before
// the blocked conv; OV keeps tensors blocked network-wide, infvino converts
// per-layer so the rest of the pipeline stays plain bfyx).
//
// R52 note: a per-channel-block variant (one work-item writes a contiguous 16-lane
// block) was tried to coalesce the writes for Cin%16!=0 — it helps large-spatial
// tiny-Cin (Cin=3 640x640: 8.5→22.8 GB/s) but **regressed** the network's actual
// reorder shapes (small spatial W<16 with 40..288 channels: mb reorder 0.079→0.26
// ms/frame), because the strided per-channel reads no longer map to a full x
// sub-group. Net negative → kept the per-element mapping (coalesced reads). See
// docs/round53. Reorder is near roofline (50-70 GB/s) for every shape the models
// actually convert.
__kernel void reorder_bfyx_to_fsv16(
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
