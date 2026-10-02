// Generic elementwise / layout / pooling / softmax kernels (fp16, NCHW) for the
// graph runner.  All numeric parameters are computed by the ONNX→plan converter
// and passed as attrs, so these kernels stay simple.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ---- elementwise binary with broadcasting (ONNX-style, align right) ----
// dims buffer: [rank out.., rank a-stride.., rank b-stride..]
//   a/b strides are in ELEMENTS for the operand's own contiguous layout
//   (0 for broadcast dims).
__kernel void ew_binary_bcast(__global const half *restrict a, __global const half *restrict b,
                              __global half *restrict y, const int n, const int op,
                              const int rank, __global const int *restrict dims) {
  const int i = get_global_id(0);
  if (i >= n) return;
  int rem = i, aidx = 0, bidx = 0;
  for (int r = rank - 1; r >= 0; --r) {
    const int orr = dims[r];
    const int coord = rem % orr;
    rem /= orr;
    aidx += coord * dims[rank + r];
    bidx += coord * dims[2 * rank + r];
  }
  const float av = (float)a[aidx];
  const float bv = (float)b[bidx];
  float r = av;
  if (op == 0) r = av + bv;
  else if (op == 1) r = av - bv;
  else if (op == 2) r = av * bv;
  else r = av / bv;
  y[i] = (half)r;
}

// ---- elementwise binary: 0=add 1=sub 2=mul 3=div ; b_scalar broadcasts b[0] ----
__kernel void ew_binary(__global const half *restrict a, __global const half *restrict b,
                        __global half *restrict y, const int n, const int op,
                        const int b_scalar) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const float av = (float)a[i];
  const float bv = (float)(b_scalar ? b[0] : b[i]);
  float r = av;
  if (op == 0) r = av + bv;
  else if (op == 1) r = av - bv;
  else if (op == 2) r = av * bv;
  else r = av / bv;
  y[i] = (half)r;
}

// ---- elementwise unary: 0=sigmoid 1=silu 2=relu 3=hardswish 4=hardsigmoid ----
__kernel void ew_unary(__global const half *restrict x, __global half *restrict y,
                       const int n, const int op) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const float v = (float)x[i];
  float r = v;
  if (op == 0) r = 1.0f / (1.0f + exp(-v));
  else if (op == 1) r = v / (1.0f + exp(-v));
  else if (op == 2) r = fmax(v, 0.0f);
  else if (op == 3) r = v * fmin(fmax(v + 3.0f, 0.0f), 6.0f) / 6.0f;
  else if (op == 4) r = fmin(fmax(v + 3.0f, 0.0f), 6.0f) / 6.0f;
  y[i] = (half)r;
}

// ---- copy `cnt` channels starting at c0 of x[HW*C] to y at dst_off ----
__kernel void copy_c(__global const half *restrict x, __global half *restrict y,
                     const int HW, const int c0, const int cnt, const int dst_off) {
  const int i = get_global_id(0);
  if (i >= cnt * HW) return;
  const int c = i / HW, r = i % HW;
  y[dst_off + i] = x[(c0 + c) * HW + r];
}

// ---- slice `len` elements along an axis (outer,axdim,inner) ----
__kernel void slice_axis(__global const half *restrict x, __global half *restrict y,
                         const int outer, const int axdim, const int inner,
                         const int start, const int len) {
  const int i = get_global_id(0);
  const int total = outer * len * inner;
  if (i >= total) return;
  const int o = i / (len * inner);
  const int rem = i % (len * inner);
  y[i] = x[(o * axdim + start) * inner + rem];
}

// ---- concat up to 4 inputs along an axis (outer, inner) with axis dims ca..cd ----
// R23: 3-D grid (r, ax, o) removes the per-element integer div/mod of the original
// 1-D version; gid0 (inner) stays contiguous so reads/writes remain coalesced.
__kernel void concat4(__global const half *restrict a, const int ca,
                      __global const half *restrict b, const int cb,
                      __global const half *restrict c, const int cc,
                      __global const half *restrict d, const int cd,
                      __global half *restrict y, const int outer, const int inner) {
  const int r  = get_global_id(0);
  const int ax = get_global_id(1);
  const int o  = get_global_id(2);
  if (r >= inner || o >= outer) return;
  const int sum = ca + cb + cc + cd;
  if (ax >= sum) return;
  half v;
  if (ax < ca) v = a[(o * ca + ax) * inner + r];
  else if (ax < ca + cb) v = b[(o * cb + (ax - ca)) * inner + r];
  else if (ax < ca + cb + cc) v = c[(o * cc + (ax - ca - cb)) * inner + r];
  else v = d[(o * cd + (ax - ca - cb - cc)) * inner + r];
  y[(o * sum + ax) * inner + r] = v;
}

// ---- max pool (K,K) stride S pad P ----
__kernel void maxpool(__global const half *restrict x, __global half *restrict y,
                      const int C, const int H, const int W, const int Hout,
                      const int Wout, const int K, const int S, const int P) {
  const int i = get_global_id(0);
  const int total = C * Hout * Wout;
  if (i >= total) return;
  const int c = i / (Hout * Wout);
  const int rem = i % (Hout * Wout);
  const int oh = rem / Wout, ow = rem % Wout;
  float m = -3.4e38f;
  for (int kh = 0; kh < K; ++kh)
    for (int kw = 0; kw < K; ++kw) {
      const int yy = oh * S - P + kh, xx = ow * S - P + kw;
      if (yy >= 0 && yy < H && xx >= 0 && xx < W)
        m = fmax(m, (float)x[(c * H + yy) * W + xx]);
    }
  y[i] = (half)m;
}

// ---- nearest-neighbour resize by integer scale S (out = H*S, W*S) ----
__kernel void resize_nn(__global const half *restrict x, __global half *restrict y,
                        const int C, const int H, const int W, const int S) {
  const int Hout = H * S, Wout = W * S;
  const int i = get_global_id(0);
  const int total = C * Hout * Wout;
  if (i >= total) return;
  const int c = i / (Hout * Wout);
  const int rem = i % (Hout * Wout);
  const int oh = rem / Wout, ow = rem % Wout;
  y[i] = x[(c * H + oh / S) * W + ow / S];
}

// ---- softmax over one axis (outer, axdim, inner) ----
__kernel void softmax_axis(__global const half *restrict x, __global half *restrict y,
                           const int outer, const int axdim, const int inner) {
  const int i = get_global_id(0);
  const int total = outer * inner;
  if (i >= total) return;
  const int o = i / inner, r = i % inner;
  float m = -3.4e38f;
  for (int a = 0; a < axdim; ++a) m = fmax(m, (float)x[(o * axdim + a) * inner + r]);
  float s = 0.0f;
  for (int a = 0; a < axdim; ++a) s += exp((float)x[(o * axdim + a) * inner + r] - m);
  for (int a = 0; a < axdim; ++a)
    y[(o * axdim + a) * inner + r] =
      (half)(exp((float)x[(o * axdim + a) * inner + r] - m) / s);
}

// ---- transpose [1,D1,D2,I] with perm p (0 fixed): 3 modes ----
// mode 0: [1,D1,D2,I]->[1,D2,D1,I] (p=[0,2,1,3])
// mode 1: [1,D1,D2,I]->[1,D1,I,D2] (p=[0,1,3,2])
__kernel void permute_0213(__global const half *restrict x, __global half *restrict y,
                           const int D1, const int D2, const int I, const int mode) {
  const int i = get_global_id(0);
  const int total = D1 * D2 * I;
  if (i >= total) return;
  if (mode == 0) {
    const int d1 = i / (D2 * I);
    const int rem = i % (D2 * I);
    const int d2 = rem / I, r = rem % I;
    y[((d2 * D1 + d1) * I) + r] = x[i];
  } else {
    // out[d1][r][d2] = x[d1][d2][r]
    const int d1 = i / (D2 * I);
    const int rem = i % (D2 * I);
    const int d2 = rem / I, r = rem % I;
    y[(d1 * I + r) * D2 + d2] = x[i];
  }
}

// ---- global average pool over spatial: [C,H,W] -> [C] ----
__kernel void gap(__global const half *restrict x, __global half *restrict y,
                  const int C, const int HW) {
  const int c = get_global_id(0);
  if (c >= C) return;
  float s = 0.0f;
  for (int i = 0; i < HW; ++i) s += (float)x[c * HW + i];
  y[c] = (half)(s / (float)HW);
}

// ---- parallel global average pool: one work-group per channel, tree reduction.
// Round 22: the serial `gap` above launches only C work-items (e.g. C=16 for the
// 56x56->1x1 stem GAP) and is latency-bound (mobilenet measured 0.86 ms across 10
// GAPs). This version spreads each channel over GAP_WGS lanes and reduces in SLM.
#ifndef GAP_WGS
#define GAP_WGS 128
#endif
__kernel void gap_r(__global const half *restrict x, __global half *restrict y,
                    const int C, const int HW) {
  const int c = get_group_id(0);
  const int lid = get_local_id(0);
  if (c >= C) return;
  __local float s[GAP_WGS];
  float acc = 0.0f;
  for (int i = lid; i < HW; i += GAP_WGS) acc += (float)x[c * HW + i];
  s[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = GAP_WGS / 2; off > 0; off >>= 1) {
    if (lid < off) s[lid] += s[lid + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y[c] = (half)(s[0] / (float)HW);
}

// ---- broadcast add of a per-channel bias: y[c*HW+i] = x[c*HW+i] + b[c] ----
__kernel void bias_add(__global const half *restrict x, __global const half *restrict b,
                       __global half *restrict y, const int HW) {
  const int i = get_global_id(0);
  const int c = i / HW;
  y[i] = (half)((float)x[i] + (float)b[c]);
}

// ---- batched matmul over [B0,B1,M,K] x [B0,B1,K,N] -> [B0,B1,M,N] ----
__kernel void bmm(__global const half *restrict A, __global const half *restrict B2,
                  __global half *restrict Y,
                  const int B0, const int B1, const int M, const int K, const int N) {
  const int idx = get_global_id(0);
  const int bb = B0 * B1;
  const int total = bb * M * N;
  if (idx >= total) return;
  const int b = idx / (M * N);
  const int rem = idx % (M * N);
  const int m = rem / N, n = rem % N;
  float acc = 0.0f;
  __global const half *ap = A + ((size_t)b * M * K) + (size_t)m * K;
  __global const half *bp = B2 + ((size_t)b * K * N) + n;
  for (int k = 0; k < K; ++k) acc += (float)ap[k] * (float)bp[(size_t)k * N];
  Y[idx] = (half)acc;
}

// ===========================================================================
// Round 28: autotunable variants for the small (launch/bandwidth-bound) ops.
//
// These are numerically identical to the scalar versions above (same per-element
// expression, same reduction order) — they only change how many elements one
// work-item handles / how the grid indexes the output, so the autotuner can pick
// the fastest variant per shape without any numeric risk.
// ===========================================================================

// ---- elementwise binary specialized for the "one operand broadcast per channel"
// pattern (e.g. SE Mul: [C,1,1] * [C,HW]). 2-D grid (spatial, channel) removes the
// rank loop/div/mod; `a_channel` says which input is the per-channel one. ----
__kernel void ew_binary_ch(__global const half *restrict a, __global const half *restrict b,
                           __global half *restrict y, const int HW, const int C,
                           const int op, const int a_channel) {
  const int r = get_global_id(0);
  const int c = get_global_id(1);
  if (r >= HW || c >= C) return;
  const half av = a_channel ? a[c] : a[c * HW + r];
  const half bv = a_channel ? b[c * HW + r] : b[c];
  const int i = c * HW + r;
  float af = (float)av, bf = (float)bv, res = af;
  if (op == 0) res = af + bf;
  else if (op == 1) res = af - bf;
  else if (op == 2) res = af * bf;
  else res = af / bf;
  y[i] = (half)res;
}

// ---- vectorized elementwise binary: EW_VEC contiguous elements per work-item ----
#ifndef EW_VEC
#define EW_VEC 4
#endif
__kernel void ew_binary_v(__global const half *restrict a, __global const half *restrict b,
                          __global half *restrict y, const int n, const int op,
                          const int b_scalar) {
  const int base = get_global_id(0) * EW_VEC;
#pragma unroll
  for (int j = 0; j < EW_VEC; ++j) {
    const int i = base + j;
    if (i >= n) return;
    const float av = (float)a[i];
    const float bv = (float)(b_scalar ? b[0] : b[i]);
    float r = av;
    if (op == 0) r = av + bv;
    else if (op == 1) r = av - bv;
    else if (op == 2) r = av * bv;
    else r = av / bv;
    y[i] = (half)r;
  }
}

// ---- vectorized elementwise unary ----
__kernel void ew_unary_v(__global const half *restrict x, __global half *restrict y,
                         const int n, const int op) {
  const int base = get_global_id(0) * EW_VEC;
#pragma unroll
  for (int j = 0; j < EW_VEC; ++j) {
    const int i = base + j;
    if (i >= n) return;
    const float v = (float)x[i];
    float r = v;
    if (op == 0) r = 1.0f / (1.0f + exp(-v));
    else if (op == 1) r = v / (1.0f + exp(-v));
    else if (op == 2) r = fmax(v, 0.0f);
    else if (op == 3) r = v * fmin(fmax(v + 3.0f, 0.0f), 6.0f) / 6.0f;
    else if (op == 4) r = fmin(fmax(v + 3.0f, 0.0f), 6.0f) / 6.0f;
    y[i] = (half)r;
  }
}

// ---- vectorized concat4: EW_VEC contiguous inner elements per work-item ----
__kernel void concat4_v(__global const half *restrict a, const int ca,
                        __global const half *restrict b, const int cb,
                        __global const half *restrict c, const int cc,
                        __global const half *restrict d, const int cd,
                        __global half *restrict y, const int outer, const int inner) {
  const int r0 = get_global_id(0) * EW_VEC;
  const int ax = get_global_id(1);
  const int o  = get_global_id(2);
  if (o >= outer) return;
  const int sum = ca + cb + cc + cd;
  if (ax >= sum) return;
  __global const half *src;
  int off = 0;
  if (ax < ca) { src = a; off = (o * ca + ax) * inner; }
  else if (ax < ca + cb) { src = b; off = (o * cb + (ax - ca)) * inner; }
  else if (ax < ca + cb + cc) { src = c; off = (o * cc + (ax - ca - cb)) * inner; }
  else { src = d; off = (o * cd + (ax - ca - cb - cc)) * inner; }
  const int dst = (o * sum + ax) * inner;
#pragma unroll
  for (int j = 0; j < EW_VEC; ++j) {
    const int r = r0 + j;
    if (r >= inner) return;
    y[dst + r] = src[off + r];
  }
}

// ---- copy_c on a 2-D grid (row, channel): no per-element div/mod ----
__kernel void copy_c2(__global const half *restrict x, __global half *restrict y,
                      const int HW, const int c0, const int cnt, const int dst_off) {
  const int r = get_global_id(0);
  const int c = get_global_id(1);
  if (r >= HW || c >= cnt) return;
  y[dst_off + c * HW + r] = x[(c0 + c) * HW + r];
}

// ---- slice_axis on a 3-D grid (inner, len, outer): no per-element div/mod ----
__kernel void slice_axis3(__global const half *restrict x, __global half *restrict y,
                          const int outer, const int axdim, const int inner,
                          const int start, const int len) {
  const int r = get_global_id(0);
  const int a = get_global_id(1);
  const int o = get_global_id(2);
  if (r >= inner || a >= len || o >= outer) return;
  y[(o * len + a) * inner + r] = x[(o * axdim + start + a) * inner + r];
}

// ---- maxpool on a 3-D grid (Wout, Hout, C): no per-element div/mod ----
__kernel void maxpool3(__global const half *restrict x, __global half *restrict y,
                       const int C, const int H, const int W, const int Hout,
                       const int Wout, const int K, const int S, const int P) {
  const int ow = get_global_id(0);
  const int oh = get_global_id(1);
  const int c  = get_global_id(2);
  if (ow >= Wout || oh >= Hout || c >= C) return;
  float m = -3.4e38f;
  for (int kh = 0; kh < K; ++kh)
    for (int kw = 0; kw < K; ++kw) {
      const int yy = oh * S - P + kh, xx = ow * S - P + kw;
      if (yy >= 0 && yy < H && xx >= 0 && xx < W)
        m = fmax(m, (float)x[(c * H + yy) * W + xx]);
    }
  y[(c * Hout + oh) * Wout + ow] = (half)m;
}

// ---- resize_nn on a 3-D grid (Wout, Hout, C): no per-element div/mod ----
__kernel void resize_nn3(__global const half *restrict x, __global half *restrict y,
                         const int C, const int H, const int W, const int S) {
  const int Hout = H * S, Wout = W * S;
  const int ow = get_global_id(0);
  const int oh = get_global_id(1);
  const int c  = get_global_id(2);
  if (ow >= Wout || oh >= Hout || c >= C) return;
  y[(c * Hout + oh) * Wout + ow] = x[(c * H + oh / S) * W + ow / S];
}

// ---- permute_0213 on a 3-D grid: no per-element div/mod ----
//   mode 0: [1,D1,D2,I] -> [1,D2,D1,I]; grid (I, D2, D1)
//   mode 1: [1,D1,D2,I] -> [1,D1,I,D2]; grid (D2, I, D1)
__kernel void permute_0213_3d(__global const half *restrict x, __global half *restrict y,
                              const int D1, const int D2, const int I, const int mode) {
  const int a = get_global_id(0);
  const int b = get_global_id(1);
  const int d1 = get_global_id(2);
  if (d1 >= D1) return;
  if (mode == 0) {
    const int r = a, d2 = b;
    if (r >= I || d2 >= D2) return;
    y[(d2 * D1 + d1) * I + r] = x[(d1 * D2 + d2) * I + r];
  } else {
    const int d2 = a, r = b;
    if (d2 >= D2 || r >= I) return;
    y[(d1 * I + r) * D2 + d2] = x[(d1 * D2 + d2) * I + r];
  }
}

// ---- bmm on a 3-D grid (n, m, b): no per-element div/mod, same K order ----
__kernel void bmm2(__global const half *restrict A, __global const half *restrict B2,
                   __global half *restrict Y,
                   const int B0, const int B1, const int M, const int K, const int N) {
  const int n = get_global_id(0);
  const int m = get_global_id(1);
  const int b = get_global_id(2);
  if (n >= N || m >= M || b >= B0 * B1) return;
  __global const half *ap = A + ((size_t)b * M * K) + (size_t)m * K;
  __global const half *bp = B2 + ((size_t)b * K * N) + n;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += (float)ap[k] * (float)bp[(size_t)k * N];
  Y[((size_t)b * M + m) * N + n] = (half)acc;
}

// ---- softmax over last axis of [outer, axdim] flattened (inner=1) ----
