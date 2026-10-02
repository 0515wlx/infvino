// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Round 17b: SLM-free, sub-group-broadcast direct 3x3 conv, modelled on
// OpenVINO's `convolution_gpu_bfyx_os_iyx_osv32`.
//
// Data path (the important part): there is NO __local memory and NO barrier.
//   * Each sub-group (= SG lanes) owns SG*VECO output channels for an OBW x OBH
//     output block.
//   * The input block for one input channel is loaded ONCE into a per-lane
//     register array `in[]` (each lane holds elements lane, lane+SG, ...), then
//     `sub_group_broadcast` distributes the needed scalar to all lanes.
//   * Each lane reads its own weight (coalesced across lanes) and reuses it
//     across the whole OBW*OBH output block.
//
//   weight-load : mad = 1 : (OBW*OBH)      (conv3x3_f16 TM=1 is 1:1)
//
// Knobs: SG VECO OBW OBH STRIDE PAD ACT
//   VECO: output channels per lane (2 -> half2 accumulator)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifndef SG
#define SG 16
#endif
#ifndef VECO
#define VECO 2
#endif
// OBW/OBH are passed as TX/TY by Conv3x3Cfg::options().
#ifndef OBW
#ifdef TX
#define OBW TX
#else
#define OBW 4
#endif
#endif
#ifndef OBH
#ifdef TY
#define OBH TY
#else
#define OBH 4
#endif
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

#define NVEC (VECO / 2)
#define IN_ROWS (OBH * STRIDE - STRIDE + 3)
#define IN_COLS (OBW * STRIDE - STRIDE + 3)
#define NIN (IN_ROWS * IN_COLS)
#define NREG ((NIN + SG - 1) / SG)
#define NPIX (OBH * OBW)

inline half act_sg(half v) {
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));
#elif ACT == 2
  float f = (float)v;
  float t = clamp(f + 3.0f, 0.0f, 6.0f) / 6.0f;
  return (half)(f * t);
#else
  return v;
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(SG, 1, 1)))
__kernel void conv3x3_sg(
  __global const half *restrict X,     // [Cin][H][W]
  __global const half *restrict Wt,    // [Cout][Cin][3][3]
  __global const half *restrict Bias,  // [Cout] or null
  __global half *restrict Y,           // [Cout][Hout][Wout]
  const int Cin, const int H, const int W,
  const int Cout, const int Hout, const int Wout) {
  const int lane = get_local_id(0);
  const int gx = get_group_id(0), gy = get_group_id(1), gz = get_group_id(2);
  const int ox0 = gx * OBW;
  const int oy0 = gy * OBH;
  const int x0 = ox0 - PAD;
  const int y0 = oy0 - PAD;
  const int ocbase = gz * SG * VECO;   // lane owns channels ocbase + v*SG + lane

  half2 acc[NVEC][NPIX];
#pragma unroll
  for (int v = 0; v < NVEC; ++v)
#pragma unroll
    for (int p = 0; p < NPIX; ++p) acc[v][p] = (half2)(0, 0);

  for (int ci = 0; ci < Cin; ++ci) {
    // ---- load the input block once into per-lane registers ----
    half in[NREG];
#pragma unroll
    for (int r = 0; r < NREG; ++r) {
      const int e = r * SG + lane;
      half v = (half)0;
      if (e < NIN) {
        const int rr = e / IN_COLS, cc = e % IN_COLS;
        const int yy = y0 + rr * STRIDE, xx = x0 + cc * STRIDE;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
          v = X[((size_t)ci * H + yy) * W + xx];
      }
      in[r] = v;
    }
#pragma unroll
    for (int kr = 0; kr < 3; ++kr) {
#pragma unroll
      for (int kc = 0; kc < 3; ++kc) {
        half wv[VECO];
#pragma unroll
        for (int v = 0; v < VECO; ++v) {
          const int oc = ocbase + v * SG + lane;
          wv[v] = (oc < Cout) ? Wt[((size_t)oc * Cin + ci) * 9 + kr * 3 + kc] : (half)0;
        }
#pragma unroll
        for (int br = 0; br < OBH; ++br) {
#pragma unroll
          for (int bc = 0; bc < OBW; ++bc) {
            const int e = (br * STRIDE + kr) * IN_COLS + (bc * STRIDE + kc);
            const half val = sub_group_broadcast(in[e / SG], e % SG);
            const half2 xx = (half2)(val, val);
#pragma unroll
            for (int v = 0; v < NVEC; ++v)
              acc[v][br * OBW + bc] =
                mad(xx, (half2)(wv[2 * v], wv[2 * v + 1]), acc[v][br * OBW + bc]);
          }
        }
      }
    }
  }

  // ---- bias + activation + store ----
#pragma unroll
  for (int v = 0; v < VECO; ++v) {
    const int oc = ocbase + v * SG + lane;
    if (oc < Cout) {
      const half b = Bias ? Bias[oc] : (half)0;
#pragma unroll
      for (int br = 0; br < OBH; ++br) {
        const int oy = oy0 + br;
        if (oy < Hout) {
#pragma unroll
          for (int bc = 0; bc < OBW; ++bc) {
            const int ox = ox0 + bc;
            if (ox < Wout) {
              const half2 a = acc[v / 2][br * OBW + bc];
              half r = (v & 1) ? a.s1 : a.s0;
              Y[((size_t)oc * Hout + oy) * Wout + ox] = act_sg(r + b);
            }
          }
        }
      }
    }
  }
}
