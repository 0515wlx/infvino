// Specialized first-layer direct conv: tiny input-channel count (CIN <= 4).
//
// Motivation: the network stem (e.g. yolov8n/11 `320x320 s2 p1 Cin3 -> Cout16`) is
// the single worst `ratio` layer in the whole net (measured ops/EU/cyc ~3 vs the
// middle standard ~16). The OV osv32 port pads Cin 3 -> 16 (13/16 lanes wasted) and
// the native SLM kernel's staging/barrier dominates a 3-channel chunk.
//
// Data path (CIN<=4):
//   * lanes (sub-group) map to consecutive output columns x; each work-item
//     computes ALL Cout output channels for ONE output pixel;
//   * weights are host-repacked to [Cin*9][Cout] (channel-contiguous) so the inner
//     loop reads a half2 of two output channels per tap; the read is the *same*
//     address for every lane in the sub-group -> hardware broadcast, L1-resident
//     (432 halfs = 864 B). No SLM, no barrier, no per-work-group staging;
//   * the store loop over channels writes consecutive x per channel across the
//     sub-group -> coalesced.
//
// Numerical semantics are identical to native/OV 3x3 conv (same fp16 mad
// accumulation order over (ci,kh,kw)); only the work decomposition changes.
//
// Compile-time: CIN COUT STRIDE PAD ACT WGS.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef CIN
#define CIN 3
#endif
#ifndef COUT
#define COUT 16
#endif
#ifndef STRIDE
#define STRIDE 2
#endif
#ifndef PAD
#define PAD 1
#endif
#ifndef ACT
#define ACT 1
#endif
#ifndef SG
#define SG 16
#endif
#ifndef WGS
#define WGS 128
#endif

#define KHW (CIN * 9)

inline half act_c3(half v)
{
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));               // SiLU
#elif ACT == 2
  float f = (float)v;
  float t = clamp(f + 3.0f, 0.0f, 6.0f) / 6.0f;      // Hardswish
  return (half)(f * t);
#elif ACT == 3
  return (half)max((float)v, 0.0f);                  // ReLU
#elif ACT == 4
  float f = (float)v;
  return (half)(clamp(f + 3.0f, 0.0f, 6.0f) / 6.0f); // Hardsigmoid
#else
  return v;
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void conv3x3_cin3(
  __global const half *restrict X,     // [CIN][H][W]
  __global const half *restrict Wt,    // [CIN*9][COUT]  (host-repacked, channel-contiguous)
  __global const half *restrict Bias,  // [COUT] or null
  __global half *restrict Y,           // [COUT][Hout][Wout]
  const int H, const int W, const int Hout, const int Wout)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= Wout || y >= Hout) return;

  half2 acc[COUT / 2];
#pragma unroll
  for (int t = 0; t < COUT / 2; ++t) acc[t] = (half2)(0, 0);

  const int iy0 = y * STRIDE - PAD;
  const int ix0 = x * STRIDE - PAD;
#pragma unroll
  for (int ci = 0; ci < CIN; ++ci)
  {
#pragma unroll
    for (int kh = 0; kh < 3; ++kh)
    {
      const int iy = iy0 + kh;
#pragma unroll
      for (int kw = 0; kw < 3; ++kw)
      {
        const int ix = ix0 + kw;
        half v = (half)0;
        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
          v = X[((size_t)ci * H + iy) * W + ix];
        const half2 v2 = (half2)(v, v);
        const int    k  = (ci * 3 + kh) * 3 + kw;
#pragma unroll
        for (int t = 0; t < COUT / 2; ++t)
          acc[t] = mad(v2, *(__global const half2 *)&Wt[k * COUT + 2 * t], acc[t]);
      }
    }
  }

#pragma unroll
  for (int t = 0; t < COUT / 2; ++t)
  {
    const int  c0 = 2 * t;
    const half b0 = Bias ? Bias[c0] : (half)0;
    const half b1 = Bias ? Bias[c0 + 1] : (half)0;
    Y[((size_t)c0 * Hout + y) * Wout + x]       = act_c3(acc[t].s0 + b0);
    Y[((size_t)(c0 + 1) * Hout + y) * Wout + x] = act_c3(acc[t].s1 + b1);
  }
}
