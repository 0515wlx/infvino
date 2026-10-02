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
// Round 17: SLM weight-load vector width (halfs): 2 (half2, original), 4, 8.
// The weight tile Ws[ci][kk][CB] is channel-contiguous, so one wide load can feed
// several half2 mads (CB must be a multiple of WVEC).
#ifndef WVEC
#define WVEC 2
#endif
// Round 17 diagnostic (R14 style): PROBE bit0 = replace SLM weight load with a
// runtime register constant (keeps the mad structure); bit1 = same for the input
// strip. Isolates operand/L1/SLM feed cost from the mad issue itself.
#ifndef PROBE
#define PROBE 0
#endif
// Round 15: force the sub-group (SIMD) width, like the GEMM (SG=16 avoids the
// IGC SIMD8 cliff). 0 = let IGC decide.
#ifndef SG
#define SG 0
#endif
// Round 18: coalesced weight staging. The stock loop makes `t` (output channel)
// the fastest index, so consecutive lanes read Wt[gout][gc][kk] with stride
// Cin*KHKW -> every lane a different cache line (fully uncoalesced global read,
// re-done for every spatial work-group). WCOAL=1 instead makes (ci,kk) fastest
// so a sub-group reads one contiguous Cin*KHKW block per output channel.
#ifndef WCOAL
#define WCOAL 0
#endif
// Round 18: XGN=1 drops the input halo SLM staging entirely and reads the
// per-thread strip straight from global memory (the strip is reused across all
// CB output channels, so the global read is coalesced across the sub-group and
// amortised). Removes the Xs divides/modulos + SLM writes at the cost of more
// (L3-resident) global loads.
#ifndef XGN
#define XGN 0
#endif
// Round 19: WGL=1 tests the user's "use the (software) L2 to relieve SLM"
// hypothesis at its strongest point: keep the input halo in SLM, but drop the
// weight SLM tile `Ws` entirely and read the weights straight from the 3.75 MB
// GPU-L3 (330 GB/s). Wt must be pre-repacked to [Cin][KHKW][Cout] so the CB
// weights for one (ci,kk) are contiguous (coalesced/broadcast block read).
// Note: compute-API "L2" *is* the GPU L3 on Xe-LP; there is no faster data L2.
#ifndef WGL
#define WGL 0
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
#if !(PROBE & 4)
    // ---- stage input halo tile (once per Cin chunk) ----
#if !XGN
    for (int idx = tid; idx < CINC * IN_ROWS * IN_COLS; idx += NTHREADS) {
      int ci = idx / (IN_ROWS * IN_COLS);
      int rem = idx % (IN_ROWS * IN_COLS);
      int r = rem / IN_COLS, c = rem % IN_COLS;
      int gc = cbase + ci;
      int yy = y0 + r, xx = x0 + c;
      half v = (half)0;
#if (PROBE & 16)
      const half pc_x = (half)((idx & 3) + 1);   // R18: staging sans global read
      if (gc < Cin && yy >= 0 && yy < H && xx >= 0 && xx < W) v = pc_x;
#else
      if (gc < Cin && yy >= 0 && yy < H && xx >= 0 && xx < W)
        v = X[((size_t)gc * H + yy) * W + xx];
#endif
      Xs[ci][r][c] = v;
    }
#endif
    // ---- stage weights for this channel block ----
#if !WGL
#if WCOAL
    for (int idx = tid; idx < CB * CINC * KHKW; idx += NTHREADS) {
      int t = idx / (CINC * KHKW);   // output channel (slowest) -> coalesced reads
      int r = idx % (CINC * KHKW);   // (ci,kk) fastest
      int ci = r / KHKW;
      int kk = r % KHKW;
      int gout = out_c0 + t;
      int gc = cbase + ci;
      half v = (half)0;
#if (PROBE & 16)
      const half pc_w = (half)((idx & 3) + 1);   // R18: staging sans global read
      if (gout < Cout && gc < Cin) v = pc_w;
#else
      if (gout < Cout && gc < Cin) v = Wt[((size_t)gout * Cin + gc) * KHKW + kk];
#endif
      Ws[ci][kk][t] = v;
    }
#else
    for (int idx = tid; idx < CB * CINC * KHKW; idx += NTHREADS) {
      int t = idx % CB;
      int rem = idx / CB;
      int ci = rem / KHKW;
      int kk = rem % KHKW;
      int gout = out_c0 + t;
      int gc = cbase + ci;
      half v = (half)0;
#if (PROBE & 16)
      const half pc_w = (half)((idx & 3) + 1);   // R18: staging sans global read
      if (gout < Cout && gc < Cin) v = pc_w;
#else
      if (gout < Cout && gc < Cin) v = Wt[((size_t)gout * Cin + gc) * KHKW + kk];
#endif
      Ws[ci][kk][t] = v;
    }
#endif
#endif
#endif
#if !(PROBE & 8)
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    const int lrow = ly * STRIDE;
    const int lcol0 = lx * TM * STRIDE;
#define STRIPN ((TM - 1) * STRIDE + KW)
#pragma unroll UNROLL_CI
    for (int ci = 0; ci < CINC; ++ci) {
#pragma unroll
      for (int kh = 0; kh < KH; ++kh) {
        // Contiguous input strip for this row, reused across all kw.
        half strip[STRIPN];
#if (PROBE & 2)
        const half pc_x = (half)((H & 7) | 1);
#pragma unroll
        for (int i = 0; i < STRIPN; ++i) strip[i] = pc_x;
#elif XGN
        const int gci = cbase + ci;
        const int gyy = y0 + lrow + kh;
#pragma unroll
        for (int i = 0; i < STRIPN; ++i) {
          const int gxx = x0 + lcol0 + i;
          half s = (half)0;
          if (gci < Cin && gyy >= 0 && gyy < H && gxx >= 0 && gxx < W)
            s = X[((size_t)gci * H + gyy) * W + gxx];
          strip[i] = s;
        }
#else
#pragma unroll
        for (int i = 0; i < STRIPN; ++i) strip[i] = Xs[ci][lrow + kh][lcol0 + i];
#endif
#if VECC
#pragma unroll
        for (int kw = 0; kw < KW; ++kw) {
#if WGL
#pragma unroll
          for (int t2 = 0; t2 < CB / 2; ++t2) {
            // Weight tile lives in GPU L3, repacked [Cin][KHKW][Cout].
            const half2 w2 = *( (__global const half2 *)&Wt[
                (((size_t)(cbase + ci) * KHKW) + (kh * KW + kw)) * Cout + out_c0 + 2 * t2] );
#pragma unroll
            for (int i = 0; i < TM; ++i) {
              const half x = strip[i * STRIDE + kw];
              acc[i][t2] = mad((half2)(x, x), w2, acc[i][t2]);
            }
          }
#elif WVEC == 8
#pragma unroll
          for (int t8 = 0; t8 < CB / 8; ++t8) {
            const half8 w8 = *(__local half8 *)&Ws[ci][kh * KW + kw][8 * t8];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
              const half x = strip[i * STRIDE + kw];
#pragma unroll
              for (int q = 0; q < 4; ++q)
                acc[i][4 * t8 + q] =
                  mad((half2)(x, x), (half2)(w8[2 * q], w8[2 * q + 1]), acc[i][4 * t8 + q]);
            }
          }
#elif WVEC == 4
#pragma unroll
          for (int t4 = 0; t4 < CB / 4; ++t4) {
            const half4 w4 = *(__local half4 *)&Ws[ci][kh * KW + kw][4 * t4];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
              const half x = strip[i * STRIDE + kw];
#pragma unroll
              for (int q = 0; q < 2; ++q)
                acc[i][2 * t4 + q] =
                  mad((half2)(x, x), (half2)(w4[2 * q], w4[2 * q + 1]), acc[i][2 * t4 + q]);
            }
          }
#else
#pragma unroll
          for (int t2 = 0; t2 < CB / 2; ++t2) {
#if (PROBE & 1)
            const half pc_w = (half)((W & 7) | 1);
            const half2 w2 = (half2)(pc_w, pc_w);
#else
            const half2 w2 = *(__local half2 *)&Ws[ci][kh * KW + kw][2 * t2];
#endif
#pragma unroll
            for (int i = 0; i < TM; ++i) {
              const half x = strip[i * STRIDE + kw];
              acc[i][t2] = mad((half2)(x, x), w2, acc[i][t2]);
            }
          }
#endif
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
#if !(PROBE & 8)
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
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

// ---------------------------------------------------------------------------
// Round 16: register-tiled direct conv. Each thread computes TM output columns
// x TN output channels, so a loaded weight vector is reused across TM columns
// (the old kernel had TM=1 -> 1:1 weight-load:FMA). Threads inside a work-group
// split the CB channels into CB/TN groups (RTLY = TY * CB/TN).
//   Work-item (lx,ly): ly -> row = ly/(CB/TN), channel group cg = ly%(CB/TN).
// Knobs: TX TY TM TN CB CINC STRIDE PAD ACT UNROLL_CI SG
// ---------------------------------------------------------------------------
#ifndef TN
#define TN 8
#endif
#define RTCGPW (CB / TN)
#define RTLX (TX / TM)
#define RTLY (TY * RTCGPW)
#define RTT (RTLX * RTLY)
#define RTSTRIP ((TM - 1) * STRIDE + KW)

#if SG
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
__attribute__((reqd_work_group_size(RTLX, RTLY, 1)))
__kernel void conv3x3_rt(
  __global const half *restrict X,
  __global const half *restrict Wt,
  __global const half *restrict Bias,
  __global half *restrict Y,
  const int Cin, const int H, const int W,
  const int Cout, const int Hout, const int Wout) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int tid = ly * RTLX + lx;
  const int gx = get_group_id(0);
  const int gy = get_group_id(1);
  const int gz = get_group_id(2);

  __local half Xs[CINC][IN_ROWS][IN_COLS];
  __local half Ws[CINC][KHKW][CB];

  const int x0 = gx * TX * STRIDE - PAD;
  const int y0 = gy * TY * STRIDE - PAD;
  const int out_c0 = gz * CB;
  const int row = ly / RTCGPW;
  const int cg = ly % RTCGPW;

  half acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int t = 0; t < TN; ++t) acc[i][t] = (half)0;

  const int cchunks = (Cin + CINC - 1) / CINC;
  for (int cc = 0; cc < cchunks; ++cc) {
    const int cbase = cc * CINC;
    for (int idx = tid; idx < CINC * IN_ROWS * IN_COLS; idx += RTT) {
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
    for (int idx = tid; idx < CB * CINC * KHKW; idx += RTT) {
      int t = idx % CB;
      int rem = idx / CB;
      int ci = rem / KHKW;
      int kk = rem % KHKW;
      int gout = out_c0 + t;
      int gc = cbase + ci;
      half v = (half)0;
#if (PROBE & 16)
      const half pc_w = (half)((idx & 3) + 1);   // R18: staging sans global read
      if (gout < Cout && gc < Cin) v = pc_w;
#else
      if (gout < Cout && gc < Cin) v = Wt[((size_t)gout * Cin + gc) * KHKW + kk];
#endif
      Ws[ci][kk][t] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll UNROLL_CI
    for (int ci = 0; ci < CINC; ++ci) {
#pragma unroll
      for (int kh = 0; kh < KH; ++kh) {
        half strip[RTSTRIP];
#pragma unroll
        for (int i = 0; i < RTSTRIP; ++i)
          strip[i] = Xs[ci][row * STRIDE + kh][lx * TM * STRIDE + i];
#pragma unroll
        for (int kw = 0; kw < KW; ++kw) {
          half w[TN];
#pragma unroll
          for (int t = 0; t < TN; ++t) w[t] = Ws[ci][kh * KW + kw][cg * TN + t];
#pragma unroll
          for (int i = 0; i < TM; ++i) {
            const half x = strip[i * STRIDE + kw];
#pragma unroll
            for (int t = 0; t < TN; ++t) acc[i][t] = mad(x, w[t], acc[i][t]);
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const int oy = gy * TY + row;
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    const int gout = out_c0 + cg * TN + t;
    if (gout < Cout) {
      half b = Bias ? Bias[gout] : (half)0;
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int ox = gx * TX + lx * TM + i;
        if (oy < Hout && ox < Wout)
          Y[((size_t)gout * Hout + oy) * Wout + ox] = activate_h(acc[i][t] + b);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Round 18: double-buffered CINC software pipeline for the native 3x3 direct
// conv.  R17.5 pinned the dominant loss to the global->SLM staging phase
// (9.9 -> 16.4 ops/EU/cyc when staging is skipped): with a single SLM buffer
// the per-chunk barrier serialises stage and compute, so the global-load
// latency is fully exposed.  Here the next CINC chunk is staged into the
// alternate SLM buffer *before* computing the current one, so the load latency
// overlaps the mads.  The chunk loop is 2x unrolled with *literal* buffer
// indices so IGC can resolve the SLM aliasing (the R9/R12 GEMM lesson: a
// runtime buffer index makes IGC drop the loop body).
//
// SLM footprint doubles (2 buffers), so pair DBUF with CINC=8 to stay in the
// ~12-16 KB/WG occupancy sweet spot (or use CINC=16 and accept fewer resident
// work-groups).  Same knobs as conv3x3_f16 (half2/vector accumulators only).
// ---------------------------------------------------------------------------
#define CV_NBUF 2

#define CV_STR(x) #x
#define CV_XSTR(x) CV_STR(x)

#define CV_STAGE(B, CBASE)                                                        \
  do {                                                                            \
    for (int idx = tid; idx < CINC * IN_ROWS * IN_COLS; idx += NTHREADS) {         \
      int ci = idx / (IN_ROWS * IN_COLS);                                          \
      int rem = idx % (IN_ROWS * IN_COLS);                                         \
      int r = rem / IN_COLS, c = rem % IN_COLS;                                    \
      int gc = (CBASE) + ci;                                                       \
      int yy = y0 + r, xx = x0 + c;                                                \
      half v = (half)0;                                                            \
      if (gc < Cin && yy >= 0 && yy < H && xx >= 0 && xx < W)                       \
        v = X[((size_t)gc * H + yy) * W + xx];                                     \
      Xs[B][ci][r][c] = v;                                                         \
    }                                                                              \
    for (int idx = tid; idx < CB * CINC * KHKW; idx += NTHREADS) {                  \
      int t = idx % CB;                                                            \
      int rem = idx / CB;                                                          \
      int ci = rem / KHKW;                                                         \
      int kk = rem % KHKW;                                                         \
      int gout = out_c0 + t;                                                       \
      int gc = (CBASE) + ci;                                                       \
      half v = (half)0;                                                            \
      if (gout < Cout && gc < Cin) v = Wt[((size_t)gout * Cin + gc) * KHKW + kk];  \
      Ws[B][ci][kk][t] = v;                                                        \
    }                                                                              \
  } while (0)

#define CV_COMPUTE(B)                                                             \
  do {                                                                            \
    const int lrow = ly * STRIDE;                                                  \
    const int lcol0 = lx * TM * STRIDE;                                            \
    _Pragma(CV_XSTR(unroll UNROLL_CI))                                             \
    for (int ci = 0; ci < CINC; ++ci) {                                            \
      _Pragma(CV_XSTR(unroll))                                                     \
      for (int kh = 0; kh < KH; ++kh) {                                            \
        half strip[STRIPN];                                                        \
        _Pragma(CV_XSTR(unroll))                                                   \
        for (int i = 0; i < STRIPN; ++i)                                           \
          strip[i] = Xs[B][ci][lrow + kh][lcol0 + i];                              \
        _Pragma(CV_XSTR(unroll))                                                   \
        for (int kw = 0; kw < KW; ++kw) {                                          \
          _Pragma(CV_XSTR(unroll))                                                 \
          for (int t2 = 0; t2 < CB / 2; ++t2) {                                    \
            const half2 w2 = *(__local half2 *)&Ws[B][ci][kh * KW + kw][2 * t2];   \
            _Pragma(CV_XSTR(unroll))                                               \
            for (int i = 0; i < TM; ++i) {                                         \
              const half x = strip[i * STRIDE + kw];                               \
              acc[i][t2] = mad((half2)(x, x), w2, acc[i][t2]);                     \
            }                                                                      \
          }                                                                        \
        }                                                                          \
      }                                                                            \
    }                                                                              \
  } while (0)

#if SG
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
__attribute__((reqd_work_group_size(LX, LY, 1)))
__kernel void conv3x3_db(
  __global const half *restrict X,
  __global const half *restrict Wt,
  __global const half *restrict Bias,
  __global half *restrict Y,
  const int Cin, const int H, const int W,
  const int Cout, const int Hout, const int Wout) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int tid = ly * LX + lx;
  const int gx = get_group_id(0);
  const int gy = get_group_id(1);
  const int gz = get_group_id(2);

  __local half Xs[CV_NBUF][CINC][IN_ROWS][IN_COLS];
  __local half Ws[CV_NBUF][CINC][KHKW][CB];

  const int x0 = gx * TX * STRIDE - PAD;
  const int y0 = gy * TY * STRIDE - PAD;
  const int out_c0 = gz * CB;

  half2 acc[TM][CB / 2];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int t = 0; t < CB / 2; ++t) acc[i][t] = (half)0;

  const int cchunks = (Cin + CINC - 1) / CINC;

  // prologue: stage chunk 0 into buffer 0
  CV_STAGE(0, 0);
  barrier(CLK_LOCAL_MEM_FENCE);

  int cc = 0;
  for (; cc + 1 < cchunks; cc += 2) {
    CV_STAGE(1, (cc + 1) * CINC);   // prefetch next chunk (latency overlaps compute)
    CV_COMPUTE(0);                   // compute current chunk from buffer 0
    barrier(CLK_LOCAL_MEM_FENCE);
    CV_STAGE(0, (cc + 2) * CINC);    // prefetch chunk cc+2 (harmless/no-op at the end)
    CV_COMPUTE(1);                   // compute chunk cc+1 from buffer 1
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (cc < cchunks) {
    CV_COMPUTE(0);                   // odd trailing chunk left in buffer 0
  }

  const int oy = gy * TY + ly;
  for (int t = 0; t < CB; ++t) {
    const int gout = out_c0 + t;
    if (gout < Cout) {
      half b = Bias ? Bias[gout] : (half)0;
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int ox = gx * TX + lx * TM + i;
        if (oy < Hout && ox < Wout) {
          half v = ((t & 1) ? acc[i][t / 2].s1 : acc[i][t / 2].s0) + b;
          Y[((size_t)gout * Hout + oy) * Wout + ox] = activate_h(v);
        }
      }
    }
  }
}
