// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Round 17: output-channel-vectorized direct 3x3 conv (groups=1, fp16).
// Inspired by OpenVINO's `convolution_gpu_bfyx_os_iyx_osv32` (os = output
// stationary, iyx = input y-x, osv32 = output channels per SIMD vector).
//
// Make the SUB-GROUP LANES the output channels. Each lane owns VECO consecutive
// output channels and TM output columns; the (sub-group-uniform) input strip is
// shared across lanes, and each weight vector is reused across TM columns:
//   -> weight-load : mad = 1 : TM   (conv3x3_f16 with TM=1 is 1:1)
//   -> accumulators = TM * VECO/2 half2 per lane (the channel block is spread
//      across lanes, so it does NOT grow with CB like conv3x3_f16 does).
//
// Work-group covers TX output columns x TY rows x CB output channels.
//   local(0) = RX * SG   (rx = lx / SG : spatial-x tile, lane = lx % SG : channel)
//   local(1) = RY * CG   (row = ly % RY : output row,     cg   = ly / RY : channel group)
//   gz       = output channel block
// with RX = TX/TM, RY = TY, CG = CB/(SG*VECO).
//
// Knobs (-D): TX TY TM VECO CB CINC STRIDE PAD ACT UNROLL_CI SG
//   VECO: channels per lane (1 = scalar, 2/4 = half2-packed pairs)
//   ACT : 0=none 1=SiLU 2=Hardswish
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef TX
#define TX 16
#endif
#ifndef TY
#define TY 8
#endif
#ifndef TM
#define TM 4
#endif
#ifndef VECO
#define VECO 2
#endif
#ifndef CB
#define CB 32
#endif
#ifndef CINC
#define CINC 16
#endif
#ifndef STRIDE
#define STRIDE 1
#endif
#ifndef PAD
#define PAD 1
#endif
#ifndef ACT
#define ACT 0
#endif
#ifndef UNROLL_CI
#define UNROLL_CI 3
#endif
#ifndef SG
#error "conv3x3_osv requires -DSG (sub-group width = lanes)"
#endif

#define KH 3
#define KW 3
#define KHKW (KH * KW)
#define IN_ROWS (TY * STRIDE - STRIDE + KH)
#define IN_COLS (TX * STRIDE - STRIDE + KW)
#define RX (TX / TM)
#define RY (TY)
#define CG (CB / (SG * VECO))
#define NTHREADS (RX * SG * RY * CG)
#define STRIPN ((TM - 1) * STRIDE + KW)
#define NVEC (VECO / 2)

inline half activate_osv(half v) {
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));                        // SiLU
#elif ACT == 2
  float f = (float)v;
  float t = clamp(f + 3.0f, 0.0f, 6.0f) / 6.0f;               // Hardswish
  return (half)(f * t);
#else
  return v;
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(RX * SG, RY * CG, 1)))
__kernel void conv3x3_osv(
  __global const half *restrict X,     // [Cin][H][W]
  __global const half *restrict Wt,    // [Cout][Cin][3][3]
  __global const half *restrict Bias,  // [Cout] or null
  __global half *restrict Y,           // [Cout][Hout][Wout]
  const int Cin, const int H, const int W,
  const int Cout, const int Hout, const int Wout) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int tid = ly * (RX * SG) + lx;
  const int gx = get_group_id(0);
  const int gy = get_group_id(1);
  const int gz = get_group_id(2);

  __local half Xs[CINC][IN_ROWS][IN_COLS];
  __local half Ws[CINC][KHKW][CB];   // channel-contiguous

  const int x0 = gx * TX * STRIDE - PAD;
  const int y0 = gy * TY * STRIDE - PAD;
  const int out_c0 = gz * CB;

  const int lane = lx % SG;          // channel lane within the group
  const int rx = lx / SG;            // spatial-x tile
  const int row = ly % RY;           // output row
  const int cg = ly / RY;            // channel group
  // first output channel owned by this lane
  const int oc_l0 = cg * SG * VECO + lane * VECO;

  half2 acc[TM][NVEC];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int v = 0; v < NVEC; ++v) acc[i][v] = (half2)(0, 0);

  const int cchunks = (Cin + CINC - 1) / CINC;
  for (int cc = 0; cc < cchunks; ++cc) {
    const int cbase = cc * CINC;
    // ---- stage input halo (shared by all CB channels) ----
    for (int idx = tid; idx < CINC * IN_ROWS * IN_COLS; idx += NTHREADS) {
      int ci = idx / (IN_ROWS * IN_COLS);
      int rem = idx % (IN_ROWS * IN_COLS);
      int r = rem / IN_COLS, c = rem % IN_COLS;
      int gc = cbase + ci;
      int yy = y0 + r, xx = x0 + c;
      half v = (half)0;
      if (gc < Cin && yy >= 0 && yy < H && xx >= 0 && xx < W)
        v = X[((size_t)gc * H + yy) * W + xx];
      Xs[ci][r][c] = v;
    }
    // ---- stage weights [ci][kk][t] ----
    for (int idx = tid; idx < CB * CINC * KHKW; idx += NTHREADS) {
      int t = idx % CB;
      int rem = idx / CB;
      int ci = rem / KHKW;
      int kk = rem % KHKW;
      int gout = out_c0 + t;
      int gc = cbase + ci;
      half v = (half)0;
      if (gout < Cout && gc < Cin) v = Wt[((size_t)gout * Cin + gc) * KHKW + kk];
      Ws[ci][kk][t] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int lrow = row * STRIDE;
    const int lcol0 = rx * TM * STRIDE;
#pragma unroll UNROLL_CI
    for (int ci = 0; ci < CINC; ++ci) {
#pragma unroll
      for (int kh = 0; kh < KH; ++kh) {
        // sub-group-uniform strip (all lanes read the same address)
        half strip[STRIPN];
#pragma unroll
        for (int i = 0; i < STRIPN; ++i)
          strip[i] = Xs[ci][lrow + kh][lcol0 + i];
#pragma unroll
        for (int kw = 0; kw < KW; ++kw) {
          // per-lane weight vector, reused across TM columns
          half2 wv[NVEC];
#pragma unroll
          for (int v = 0; v < NVEC; ++v)
            wv[v] = *(__local half2 *)&Ws[ci][kh * KW + kw][oc_l0 + 2 * v];
#pragma unroll
          for (int i = 0; i < TM; ++i) {
            const half x = strip[i * STRIDE + kw];
            const half2 xx = (half2)(x, x);
#pragma unroll
            for (int v = 0; v < NVEC; ++v) acc[i][v] = mad(xx, wv[v], acc[i][v]);
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // ---- store ----
  const int oy = gy * TY + row;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int ox = gx * TX + rx * TM + i;
    if (oy < Hout && ox < Wout) {
#pragma unroll
      for (int v = 0; v < NVEC; ++v) {
        const int oc = out_c0 + oc_l0 + 2 * v;
        const half2 a = acc[i][v];
        half b0 = Bias ? Bias[oc] : (half)0;
        half b1 = Bias ? Bias[oc + 1] : (half)0;
        if (oc < Cout)
          Y[((size_t)oc * Hout + oy) * Wout + ox] = activate_osv(a.s0 + b0);
        if (oc + 1 < Cout)
          Y[((size_t)(oc + 1) * Hout + oy) * Wout + ox] = activate_osv(a.s1 + b1);
      }
    }
  }
}
