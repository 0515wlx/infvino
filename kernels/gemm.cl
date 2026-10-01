// FP16 tiled GEMM: C[M,N] = A[M,K] * B[K,N]  (row-major, fp16 in/out).
//
// Targets Intel Xe (Gen12).  Compile-time knobs are passed as -D options so the
// bench can sweep without editing code:
//   BM x BN : block tile        BK : k-tile
//   TM x TN : per-work-item register tile
//   VEC2    : 1 => pack two k-elements into a half2 mad (packed fp16 FMA)
//   PAD     : local-memory padding to break bank conflicts
//
// B is staged transposed (BsT[col][k]) so that the k dimension is contiguous and
// a half2 load yields the (k, k+1) pair required by VEC2.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 8
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 4
#endif
#ifndef VEC2
#define VEC2 1
#endif
#ifndef PAD
#define PAD 0
#endif

#define LX (BN / TN)
#define LY (BM / TM)

__attribute__((reqd_work_group_size(LX, LY, 1)))
__kernel void gemm_f16(__global const half *restrict A,
                       __global const half *restrict B,
                       __global half *restrict C,
                       const int M, const int N, const int K) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int tid = ly * LX + lx;

  const int blockRow = get_group_id(1) * BM;
  const int blockCol = get_group_id(0) * BN;

  __local half As[BM][BK + PAD];
  __local half BsT[BN][BK + PAD];

#if VEC2
  half2 acc[TM][TN];
#else
  half acc[TM][TN];
#endif
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (half)(0);

  const int kTiles = (K + BK - 1) / BK;
  for (int kt = 0; kt < kTiles; ++kt) {
    const int k0 = kt * BK;
#pragma unroll
    for (int i = 0; i < (BM * BK + LX * LY - 1) / (LX * LY); ++i) {
      int idx = tid + i * (LX * LY);
      if (idx < BM * BK) {
        int r = idx / BK, c = idx % BK;
        int gr = blockRow + r, gc = k0 + c;
        As[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : (half)0;
      }
    }
#pragma unroll
    for (int i = 0; i < (BK * BN + LX * LY - 1) / (LX * LY); ++i) {
      int idx = tid + i * (LX * LY);
      if (idx < BK * BN) {
        int c = idx % BN, r = idx / BN;  // transposed staging
        int gr = k0 + r, gc = blockCol + c;
        BsT[c][r] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : (half)0;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if VEC2
#pragma unroll
    for (int kk = 0; kk < BK; kk += 2) {
      half2 a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        a[i] = *(__local half2 *)&As[ly * TM + i][kk];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        b[j] = *(__local half2 *)&BsT[lx * TN + j][kk];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = mad(a[i], b[j], acc[i][j]);
    }
#else
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      half a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[ly * TM + i][kk];
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = BsT[lx * TN + j][kk];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = mad(a[i], b[j], acc[i][j]);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int gr = blockRow + ly * TM + i;
      int gc = blockCol + lx * TN + j;
      if (gr < M && gc < N) {
#if VEC2
        C[(size_t)gr * N + gc] = acc[i][j].s0 + acc[i][j].s1;
#else
        C[(size_t)gr * N + gc] = acc[i][j];
#endif
      }
    }
}
