// Native direct conv (3x3), groups=1, fp16 in/out — avoids im2col materialization.
//
// Roofline (Round 2): materialized im2col of the dominant 3x3 64->64 80x80 layer
// has AI ~30 FLOP/byte < ridge ~72 -> bandwidth bound on this iGPU (only ~23 GB/s).
// Keeping the input halo tile on-chip keeps AI ~270 -> compute bound.
//
// Work-group:
//   gx = output column tile (TX cols)   gid(0)
//   gy = output row tile    (TY rows)   gid(1)
//   gz = output channel tile(CB chans)  gid(2)
// Work-item (lx,ly): TM output columns x CB output channels (whole channel block),
// accumulating over input-channel chunks (CINC).  Input halo staged once per Cin
// chunk and reused for all CB channels (no per-channel reload).
//
// VECC=1 packs two output channels into a half2 accumulator (channel-contiguous
// weights) -> half the accumulator registers + packed fp16 mad; this is what lets
// CB grow (less input re-read) without spilling.
//
// Knobs (-D): TX TY TM CB CINC STRIDE PAD ACT VECC UNROLL_CI
//   ACT: 0=none 1=SiLU 2=Hardswish
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef TX
#define TX 64
#endif
#ifndef TY
#define TY 8
#endif
#ifndef TM
#define TM 4
#endif
#ifndef CB
#define CB 16
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
#ifndef VECC
#define VECC 1
#endif
#ifndef UNROLL_CI
#define UNROLL_CI 1
#endif
// Round 15: force the sub-group (SIMD) width, like the GEMM (SG=16 avoids the
// IGC SIMD8 cliff). 0 = let IGC decide.
#ifndef SG
#define SG 0
#endif

#define KH 3
#define KW 3
#define KHKW (KH * KW)
#define IN_ROWS (TY * STRIDE - STRIDE + KH)
#define IN_COLS (TX * STRIDE - STRIDE + KW)
#define LX (TX / TM)
#define LY (TY)
#define NTHREADS (LX * LY)

inline half activate_h(half v) {
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

#if SG
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
__attribute__((reqd_work_group_size(LX, LY, 1)))
__kernel void conv3x3_f16(
  __global const half *restrict X,     // [Cin][H][W]
  __global const half *restrict Wt,    // [Cout][Cin][3][3]
  __global const half *restrict Bias,  // [Cout] or null
  __global half *restrict Y,           // [Cout][Hout][Wout]
  const int Cin, const int H, const int W,
  const int Cout, const int Hout, const int Wout) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int tid = ly * LX + lx;
  const int gx = get_group_id(0);
  const int gy = get_group_id(1);
  const int gz = get_group_id(2);

  __local half Xs[CINC][IN_ROWS][IN_COLS];
  __local half Ws[CINC][KHKW][CB];   // channel-contiguous

  const int x0 = gx * TX * STRIDE - PAD;
  const int y0 = gy * TY * STRIDE - PAD;
  const int out_c0 = gz * CB;

#if VECC
  half2 acc[TM][CB / 2];
#define ACCN (CB / 2)
#else
  half acc[TM][CB];
#define ACCN (CB)
#endif
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int t = 0; t < ACCN; ++t) acc[i][t] = (half)(0);

  const int cchunks = (Cin + CINC - 1) / CINC;
  for (int cc = 0; cc < cchunks; ++cc) {
    const int cbase = cc * CINC;
    // ---- stage input halo tile (once per Cin chunk) ----
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
    // ---- stage weights for this channel block ----
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

    const int lrow = ly * STRIDE;
    const int lcol0 = lx * TM * STRIDE;
#define STRIPN ((TM - 1) * STRIDE + KW)
#pragma unroll UNROLL_CI
    for (int ci = 0; ci < CINC; ++ci) {
#pragma unroll
      for (int kh = 0; kh < KH; ++kh) {
        // Contiguous input strip for this row, reused across all kw.
        half strip[STRIPN];
#pragma unroll
        for (int i = 0; i < STRIPN; ++i) strip[i] = Xs[ci][lrow + kh][lcol0 + i];
#if VECC
#pragma unroll
        for (int kw = 0; kw < KW; ++kw) {
#pragma unroll
          for (int t2 = 0; t2 < CB / 2; ++t2) {
            const half2 w2 = *(__local half2 *)&Ws[ci][kh * KW + kw][2 * t2];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
              const half x = strip[i * STRIDE + kw];
              acc[i][t2] = mad((half2)(x, x), w2, acc[i][t2]);
            }
          }
        }
#else
#pragma unroll
        for (int kw = 0; kw < KW; ++kw) {
#pragma unroll
          for (int t = 0; t < CB; ++t) {
            const half w = Ws[ci][kh * KW + kw][t];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
              const half x = strip[i * STRIDE + kw];
              acc[i][t] = mad(x, w, acc[i][t]);
            }
          }
        }
#endif
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // ---- store ----
  const int oy = gy * TY + ly;
  for (int t = 0; t < CB; ++t) {
    const int gout = out_c0 + t;
    if (gout < Cout) {
      half b = Bias ? Bias[gout] : (half)0;
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int ox = gx * TX + lx * TM + i;
        if (oy < Hout && ox < Wout) {
#if VECC
          half v = ((t & 1) ? acc[i][t / 2].s1 : acc[i][t / 2].s0) + b;
#else
          half v = acc[i][t] + b;
#endif
          Y[((size_t)gout * Hout + oy) * Wout + ox] = activate_h(v);
        }
      }
    }
  }
}
