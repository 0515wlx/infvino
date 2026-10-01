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
