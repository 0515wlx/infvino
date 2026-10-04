// General conv (arbitrary KxK, stride, pad, groups) — correctness-first path used
// for the mobilenet convs (5x5 / 3x3 stride-2 / depthwise).  One work-item computes
// one output pixel for one output channel; for groups>1 the input-channel loop is
// restricted to the group's channels.  fp16 in/out, fp32 accumulate.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void conv_general(__global const half *restrict X,   // [Cin][H][W]
                           __global const half *restrict Wt,  // [Cout][Cin/g][K][K]
                           __global const half *restrict Bias,// [Cout] or null
                           __global half *restrict Y,         // [Cout][Hout][Wout]
                           const int Cin, const int H, const int W,
                           const int Cout, const int Hout, const int Wout,
                           const int K, const int S, const int P, const int G,
                           const int act) {
  const int idx = get_global_id(0);
  const int total = Cout * Hout * Wout;
  if (idx >= total) return;
  const int oc = idx / (Hout * Wout);
  const int rem = idx % (Hout * Wout);
  const int oy = rem / Wout, ox = rem % Wout;

  const int cin_g = Cin / G;         // input channels per group
  const int cout_g = Cout / G;       // output channels per group
  const int g = oc / cout_g;
  const int ic0 = g * cin_g;

  float acc = Bias ? (float)Bias[oc] : 0.0f;
  for (int ic = 0; ic < cin_g; ++ic) {
    const int gc = ic0 + ic;
    __global const half *xplane = X + (size_t)gc * H * W;
    __global const half *wplane = Wt + ((size_t)oc * cin_g + ic) * K * K;
    for (int ky = 0; ky < K; ++ky) {
      const int yy = oy * S - P + ky;
      if (yy < 0 || yy >= H) continue;
      for (int kx = 0; kx < K; ++kx) {
        const int xx = ox * S - P + kx;
        if (xx < 0 || xx >= W) continue;
        acc += (float)xplane[(size_t)yy * W + xx] * (float)wplane[ky * K + kx];
      }
    }
  }
  float r = acc;
  if (act == 1) r = acc / (1.0f + exp(-acc));                        // SiLU
  else if (act == 2) r = acc * fmin(fmax(acc + 3.0f, 0.0f), 6.0f) / 6.0f;  // Hardswish
  else if (act == 3) r = fmax(acc, 0.0f);                            // ReLU
  else if (act == 4) r = fmin(fmax(acc + 3.0f, 0.0f), 6.0f) / 6.0f;  // Hardsigmoid
  Y[idx] = (half)r;
}

// ---------------------------------------------------------------------------
// Round 16: native depthwise conv (groups == Cin == Cout). One work-item = one
// output element (perfectly coalesced 1D grid) with fp16 accumulate and
// COMPILE-TIME K/S/P/ACT so the tap loops fully unroll and exactly one activation
// path is emitted (the old conv_general took K/S/ACT as runtime args and emitted
// every activation variant). Measured ~equal to conv_general on mobilenet and
// ~8% faster on the yolo11 depthwise ops.
// ---------------------------------------------------------------------------
#ifndef DW_K
#define DW_K 3
#endif
#ifndef DW_S
#define DW_S 1
#endif
#ifndef DW_P
#define DW_P 1
#endif
#ifndef DW_ACT
#define DW_ACT 0
#endif

static inline half dw_activate(half v) {
#if DW_ACT == 1
  float f = (float)v; return (half)(f / (1.0f + exp(-f)));
#elif DW_ACT == 3
  float f = (float)v; return (half)(f * (fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f));
#elif DW_ACT == 2
  return (half)fmax((float)v, 0.0f);
#elif DW_ACT == 4
  float f = (float)v; return (half)(fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);
#else
  return v;
#endif
}

__kernel void depthwise_f16(
  __global const half *restrict X,     // [C][H][W]
  __global const half *restrict Wt,    // [C][K][K]
  __global const half *restrict Bias,  // [C] or null
  __global half *restrict Y,           // [C][Ho][Wo]
  const int C, const int H, const int W, const int Ho, const int Wo) {
  const int idx = get_global_id(0);
  const int total = C * Ho * Wo;
  if (idx >= total) return;
  const int ox = idx % Wo;
  const int t = idx / Wo;
  const int oy = t % Ho;
  const int c = t / Ho;

  const __global half *xplane = X + (size_t)c * H * W;
  const __global half *wplane = Wt + (size_t)c * DW_K * DW_K;
  half acc = (half)0;
#pragma unroll
  for (int kh = 0; kh < DW_K; ++kh) {
    const int yy = oy * DW_S - DW_P + kh;
    if (yy < 0 || yy >= H) continue;
    const __global half *xrow = xplane + (size_t)yy * W;
#pragma unroll
    for (int kw = 0; kw < DW_K; ++kw) {
      const int xx = ox * DW_S - DW_P + kw;
      if (xx < 0 || xx >= W) continue;
      acc = mad(xrow[xx], wplane[kh * DW_K + kw], acc);
    }
  }
  const half b = Bias ? Bias[c] : (half)0;
  Y[idx] = dw_activate((half)(acc + b));
}

// ---------------------------------------------------------------------------
// R29: register-blocked depthwise (sliding window).  The scalar kernel above
// issues K*K independent global loads + one mad each per output, so it is
// latency-bound on the global loads (~0.3 ops/EU/cyc).  Here one work-item
// computes DW_TW consecutive x outputs for one channel: the input strip
// ((DW_TW-1)*DW_S + DW_K values) is loaded once per row and reused across taps
// and outputs, and the K*K weights are loaded once into registers.  fp16
// accumulation in the same (kh,kw) order => bit-identical to depthwise_f16.
//   knobs: DW_K, DW_S, DW_P, DW_ACT, DW_TW
#ifndef DW_TW
#define DW_TW 4
#endif
#define DW_STRLEN ((DW_TW - 1) * DW_S + DW_K)

__kernel void depthwise_v(
  __global const half *restrict X,     // [C][H][W]
  __global const half *restrict Wt,    // [C][K][K]
  __global const half *restrict Bias,  // [C] or null
  __global half *restrict Y,           // [C][Ho][Wo]
  const int C, const int H, const int W, const int Ho, const int Wo) {
  const int x0 = get_global_id(0) * DW_TW;
  const int oy = get_global_id(1);
  const int c  = get_global_id(2);
  if (oy >= Ho || c >= C) return;

  half w[DW_K * DW_K];
#pragma unroll
  for (int i = 0; i < DW_K * DW_K; ++i) w[i] = Wt[(size_t)c * DW_K * DW_K + i];
  const __global half *xplane = X + (size_t)c * H * W;

  half acc[DW_TW];
#pragma unroll
  for (int t = 0; t < DW_TW; ++t) acc[t] = (half)0;

#pragma unroll
  for (int kh = 0; kh < DW_K; ++kh) {
    const int yy = oy * DW_S - DW_P + kh;
    if (yy < 0 || yy >= H) continue;
    const __global half *xrow = xplane + (size_t)yy * W;
    half strip[DW_STRLEN];
#pragma unroll
    for (int j = 0; j < DW_STRLEN; ++j) {
      const int xx = x0 * DW_S - DW_P + j;
      strip[j] = (xx >= 0 && xx < W) ? xrow[xx] : (half)0;
    }
#pragma unroll
    for (int t = 0; t < DW_TW; ++t)
#pragma unroll
      for (int kw = 0; kw < DW_K; ++kw)
        acc[t] = mad(strip[t * DW_S + kw], w[kh * DW_K + kw], acc[t]);
  }

  const half b = Bias ? Bias[c] : (half)0;
#pragma unroll
  for (int t = 0; t < DW_TW; ++t) {
    const int ox = x0 + t;
    if (ox < Wo) Y[((size_t)c * Ho + oy) * Wo + ox] = dw_activate((half)(acc[t] + b));
  }
}

// ---------------------------------------------------------------------------
// R31: padded depthwise.  ISA reverse-engineering showed the scalar/vector kernels
// are dominated by addressing + per-element boundary predicates (cmp/shl/add), not
// FMA.  Here the input is materialised once per frame into a zero border
// Xp[C][Hp][Wpad] by `depthwise_pad` (borders zero-filled at allocation time, only
// the interior is rewritten each frame), so `depthwise_vp` has **no boundary
// checks at all**.  It accumulates in the same (kh,kw) order as `depthwise_v` and
// out-of-range taps read the zero border => bit-identical output.
//
// Layout: Xp[c][P+oy][P+ox] = X[c][oy][ox]; Hp = H + 2P; Wpad covers the deepest
// tap any work-group reads (see PlanModel::dwPadInput).
__kernel void depthwise_pad(
  __global const half *restrict X,   // [C][H][W]
  __global half *restrict Xp,        // [C][Hp][Wpad], borders pre-zeroed
  const int C, const int H, const int W, const int Hp, const int Wpad, const int P) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int c = get_global_id(2);
  if (x >= W || y >= H || c >= C) return;
  Xp[((size_t)c * Hp + (y + P)) * Wpad + (x + P)] =
      X[((size_t)c * H + y) * W + x];
}

__kernel void depthwise_vp(
  __global const half *restrict Xp,    // [C][Hp][Wpad] (padded, zero border)
  __global const half *restrict Wt,    // [C][K][K]
  __global const half *restrict Bias,  // [C] or null
  __global half *restrict Y,           // [C][Ho][Wo]
  const int C, const int Hp, const int Wpad, const int Ho, const int Wo) {
  const int x0 = get_global_id(0) * DW_TW;
  const int oy = get_global_id(1);
  const int c  = get_global_id(2);
  if (oy >= Ho || c >= C) return;

  half w[DW_K * DW_K];
#pragma unroll
  for (int i = 0; i < DW_K * DW_K; ++i) w[i] = Wt[(size_t)c * DW_K * DW_K + i];
  const __global half *xplane = Xp + (size_t)c * Hp * Wpad;

  half acc[DW_TW];
#pragma unroll
  for (int t = 0; t < DW_TW; ++t) acc[t] = (half)0;

#pragma unroll
  for (int kh = 0; kh < DW_K; ++kh) {
    // padded row index = oy*S - P + kh + P = oy*S + kh (always in [0, Hp)).
    const __global half *xrow = xplane + (size_t)(oy * DW_S + kh) * Wpad;
    half strip[DW_STRLEN];
#pragma unroll
    for (int j = 0; j < DW_STRLEN; ++j) strip[j] = xrow[x0 * DW_S + j];
#pragma unroll
    for (int t = 0; t < DW_TW; ++t)
#pragma unroll
      for (int kw = 0; kw < DW_K; ++kw)
        acc[t] = mad(strip[t * DW_S + kw], w[kh * DW_K + kw], acc[t]);
  }

  const half b = Bias ? Bias[c] : (half)0;
#pragma unroll
  for (int t = 0; t < DW_TW; ++t) {
    const int ox = x0 + t;
    if (ox < Wo) Y[((size_t)c * Ho + oy) * Wo + ox] = dw_activate((half)(acc[t] + b));
  }
}
