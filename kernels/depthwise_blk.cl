// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// ---------------------------------------------------------------------------
// Self-contained adaptation of OpenVINO's
//   src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_f16_depthwise.cl
// (kernel `convolution_gpu_bfyx_f16_depthwise`): sub-group lanes = 16 output
// channels (one `b_fs_yx_fsv16` block per work-group), X_BLOCK consecutive output
// columns per lane read with `intel_sub_group_block_read_usN`, weights in the
// blocked `[C/16][K][K][16]` layout, output either `b_fs_yx_fsv16` (OUT_FSV16=1,
// keeps the persistent blocked chain) or plain bfyx.
//
// Purpose (Phase 2): OV runs mobilenet depthwise in 0.154 ms vs infvino 0.39 ms,
// and — more importantly — the blocked layout lets a 1x1 -> depthwise -> 1x1 chain
// stay in `b_fs_yx_fsv16` end-to-end, removing the per-layer `reorder(blk)` that
// the blocked 1x1 otherwise pays.
//
// Simplified vs upstream: stride=1 fast path + general scalar fallback for
// stride>1 / boundaries, runtime leftovers, no JIT macro layer.
//
// R50: added `Y_BLOCK` (output rows per work-item).  The classic kernel loads
// `K` input rows to produce ONE output row; `Y_BLOCK=T` produces T consecutive
// rows while loading only `(T-1)*S + K` rows, so each input row is loaded once
// and reused across the `K` output rows it contributes to (sliding window over
// ir).  This is the depthwise analogue of `conv1x1_blk`'s Y_BLOCK (R48 D1) and
// targets the instruction/feed bound measured by the ISA histogram (add/mov/shl/
// send dominate over mad).  Accumulation order per output is unchanged
// (kh = 0..K-1) => bit-identical to Y_BLOCK=1.  Default Y_BLOCK=1 keeps the
// original behaviour.
//
// See THIRD_PARTY_NOTICES.md / third_party/openvino/LICENSE.
// ---------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#ifndef SG
#define SG 16
#endif
#ifndef X_BLOCK
#define X_BLOCK 8
#endif
#ifndef DWK             // kernel size (K x K)
#define DWK 3
#endif
#ifndef STRIDE
#define STRIDE 1
#endif
#ifndef PAD
#define PAD 1
#endif
#ifndef ACT             // depthwise 规范激活码（与 conv_general.cl 的 dw_activate 一致）：
#define ACT 0           //  1=SiLU 2=ReLU 3=HardSwish 4=HardSigmoid 0=none
#endif
#ifndef OUT_FSV16
#define OUT_FSV16 0
#endif
#ifndef Y_BLOCK         // R50: output rows per work-item (1 = original)
#define Y_BLOCK 1
#endif

#if X_BLOCK == 8
#define VEC_T half8
#define AS_V(x) as_half8(x)
#define BLOCK_READ(p) intel_sub_group_block_read_us8(p)
#elif X_BLOCK == 4
#define VEC_T half4
#define AS_V(x) as_half4(x)
#define BLOCK_READ(p) intel_sub_group_block_read_us4(p)
#elif X_BLOCK == 2
#define VEC_T half2
#define AS_V(x) as_half2(x)
#define BLOCK_READ(p) intel_sub_group_block_read_us2(p)
#else
#error "depthwise_blk: X_BLOCK must be 2, 4 or 8"
#endif

// Input columns spanned by one X_BLOCK output block: (X_BLOCK-1)*S + K.
#define DW_SPAN ((X_BLOCK - 1) * STRIDE + DWK)
// Input rows spanned by one Y_BLOCK output block: (Y_BLOCK-1)*S + K.
#define DW_IRCOUNT ((Y_BLOCK - 1) * STRIDE + DWK)
// Packed 8-wide block read used only for line loading (independent of X_BLOCK).
#define DW_BLK8(p) as_half8(intel_sub_group_block_read_us8(p))

inline half dwblk_activate(half v) {
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));                          // SiLU
#elif ACT == 3
  float f = (float)v;
  return (half)(f * fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);  // HardSwish
#elif ACT == 2
  return (half)fmax((float)v, 0.0f);                           // ReLU
#elif ACT == 4
  float f = (float)v;
  return (half)(fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);      // HardSigmoid
#else
  return v;
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(1, SG, 1)))
__kernel void depthwise_blk(
  __global const half *restrict input,    // b_fs_yx_fsv16 [C/16][H][W][16]
  __global const half *restrict weights,  // [C/16][K][K][16]
  __global const half *restrict bias,     // [C] or null
  __global half *restrict output,         // fsv16 or bfyx
  const int C, const int H, const int W,
  const int Ho, const int Wo) {
  const int lane = get_sub_group_local_id();
  const int f_block = get_group_id(1);
  const int c = f_block * SG + lane;
  const bool cok = (c < C);

  const int X_BLOCKS = (Wo + X_BLOCK - 1) / X_BLOCK;
  const int xy = get_global_id(0);
  const int x0 = (xy % X_BLOCKS) * X_BLOCK;
  const int y0 = (xy / X_BLOCKS) * Y_BLOCK;   // first output row of this WI

  const int input_y_pitch = SG * W;
  const int input_fs_pitch = SG * W * H;
  const int base_cb = f_block * input_fs_pitch;
  const int colbase = x0 * STRIDE - PAD;

  // R50: weights hoisted once per WI (reused across Y_BLOCK rows and X_BLOCK cols).
  half wt[DWK * DWK];
#pragma unroll
  for (int i = 0; i < DWK * DWK; ++i) wt[i] = weights[(f_block * DWK * DWK + i) * SG + lane];

  half acc[Y_BLOCK * X_BLOCK];
#pragma unroll
  for (int i = 0; i < Y_BLOCK * X_BLOCK; ++i) acc[i] = (half)0;

  // Sliding window over the (Y_BLOCK-1)*S + K input rows: each row is loaded once
  // and applied to every output row t for which kh = ir - t*S lies in [0, K).
#pragma unroll
  for (int ir = 0; ir < DW_IRCOUNT; ++ir) {
    const int iy = y0 * STRIDE - PAD + ir;
    const bool rowok = (iy >= 0 && iy < H);

    half line[DW_SPAN];
    if (rowok) {
      const int row_base = base_cb + iy * input_y_pitch;
      // Load the whole input span for this row ONCE (one register line), then reuse
      // it across all (t,kw) taps — the key difference vs re-reading per tap.
      if (cok && colbase >= 0 && colbase + DW_SPAN <= W) {
        int j = 0;
#pragma unroll
        for (; j + 8 <= DW_SPAN; j += 8) {
          half8 v = DW_BLK8((__global const ushort *)input + row_base + (colbase + j) * SG);
#pragma unroll
          for (int i2 = 0; i2 < 8; ++i2) line[j + i2] = v[i2];
        }
#pragma unroll
        for (; j < DW_SPAN; ++j) line[j] = input[row_base + (colbase + j) * SG + lane];
      } else {
#pragma unroll
        for (int j = 0; j < DW_SPAN; ++j) {
          const int col = colbase + j;
          line[j] = (cok && col >= 0 && col < W) ? input[row_base + col * SG + lane] : (half)0;
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < DW_SPAN; ++j) line[j] = (half)0;
    }

#pragma unroll
    for (int t = 0; t < Y_BLOCK; ++t) {
      const int kh = ir - t * STRIDE;
      if (kh < 0 || kh >= DWK) continue;   // compile-time after full unroll
#pragma unroll
      for (int kw = 0; kw < DWK; ++kw) {
#pragma unroll
        for (int j = 0; j < X_BLOCK; ++j)
          acc[t * X_BLOCK + j] =
              mad(line[j * STRIDE + kw], wt[kh * DWK + kw], acc[t * X_BLOCK + j]);
      }
    }
  }

  const half b = (bias != 0 && cok) ? bias[c] : (half)0;
#pragma unroll
  for (int t = 0; t < Y_BLOCK; ++t) {
    const int yy = y0 + t;
    if (yy >= Ho) continue;
#pragma unroll
    for (int j = 0; j < X_BLOCK; ++j) {
      const int xx = x0 + j;
      if (!cok || xx >= Wo) continue;
      const half v = dwblk_activate((half)(acc[t * X_BLOCK + j] + b));
#if OUT_FSV16
      output[(((size_t)f_block * Ho + yy) * Wo + xx) * SG + lane] = v;
#else
      output[((size_t)c * Ho + yy) * Wo + xx] = v;
#endif
    }
  }
}
