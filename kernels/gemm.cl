// FP16 tiled GEMM: C[M,N] = A[M,K] * B[K,N]  (row-major, fp16 in/out).
//
// Compile-time knobs (-D):
//   BM x BN : block tile        BK : k-tile
//   TM x TN : per-work-item register tile
//   VEC2    : 1 => pack two k-elements into a half2 mad
//   PAD     : local-memory padding
//   DBUF    : 1 => double-buffer k-tile staging. The k loop is unrolled by two
//             so the two SLM buffers have *compile-time* indices (a dynamic
//             buffer index defeats the address folding and costs ~2x).
//
// B is staged transposed (BsT[col][k]) so k is contiguous for the VEC2 pair.
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
#ifndef DBUF
#define DBUF 0
#endif

#define LX (BN / TN)
#define LY (BM / TM)
#define NTHR (LX * LY)
#define BKP (BK + PAD)

#if DBUF
#define NBUF 2
#else
#define NBUF 1
#endif

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

#if DBUF
  __local half As0[BM][BKP], As1[BM][BKP];
  __local half Bs0[BN][BKP], Bs1[BN][BKP];
#define ASPTR(B) ((B) ? As1 : As0)
#define BSPTR(B) ((B) ? Bs1 : Bs0)
#else
  __local half As0[BM][BKP], Bs0[BN][BKP];
#define ASPTR(B) (As0)
#define BSPTR(B) (Bs0)
#endif

#define STAGE_TILE(BUF, K0)                                                        \
  do {                                                                             \
    const int k0_ = (K0);                                                          \
    _Pragma("unroll") for (int i = 0; i < (BM * BK + NTHR - 1) / NTHR; ++i) {      \
      int idx = tid + i * NTHR;                                                    \
      if (idx < BM * BK) {                                                         \
        int r = idx / BK, c = idx % BK;                                            \
        int gr = blockRow + r, gc = k0_ + c;                                       \
        ASPTR(BUF)[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : (half)0;  \
      }                                                                            \
    }                                                                              \
    _Pragma("unroll") for (int i = 0; i < (BK * BN + NTHR - 1) / NTHR; ++i) {      \
      int idx = tid + i * NTHR;                                                    \
      if (idx < BK * BN) {                                                         \
        int c = idx % BN, r = idx / BN;                                            \
        int gr = k0_ + r, gc = blockCol + c;                                       \
        BSPTR(BUF)[c][r] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : (half)0;  \
      }                                                                            \
    }                                                                              \
  } while (0)

#if VEC2
#define ACC_T half2
#else
#define ACC_T half
#endif

#if VEC2
#define COMPUTE_TILE(BUF)                                                          \
  do {                                                                             \
    _Pragma("unroll") for (int kk = 0; kk < BK; kk += 2) {                         \
      half2 a[TM], b[TN];                                                          \
      _Pragma("unroll") for (int i = 0; i < TM; ++i)                               \
        a[i] = *(__local half2 *)&ASPTR(BUF)[ly * TM + i][kk];                     \
      _Pragma("unroll") for (int j = 0; j < TN; ++j)                               \
        b[j] = *(__local half2 *)&BSPTR(BUF)[lx * TN + j][kk];                     \
      _Pragma("unroll") for (int i = 0; i < TM; ++i)                               \
        _Pragma("unroll") for (int j = 0; j < TN; ++j)                             \
          acc[i][j] = mad(a[i], b[j], acc[i][j]);                                  \
    }                                                                              \
  } while (0)
#else
#define COMPUTE_TILE(BUF)                                                          \
  do {                                                                             \
    _Pragma("unroll") for (int kk = 0; kk < BK; ++kk) {                            \
      half a[TM], b[TN];                                                           \
      _Pragma("unroll") for (int i = 0; i < TM; ++i)                               \
        a[i] = ASPTR(BUF)[ly * TM + i][kk];                                        \
      _Pragma("unroll") for (int j = 0; j < TN; ++j)                               \
        b[j] = BSPTR(BUF)[lx * TN + j][kk];                                        \
      _Pragma("unroll") for (int i = 0; i < TM; ++i)                               \
        _Pragma("unroll") for (int j = 0; j < TN; ++j)                             \
          acc[i][j] = mad(a[i], b[j], acc[i][j]);                                  \
    }                                                                              \
  } while (0)
#endif

  ACC_T acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (half)(0);

  const int kTiles = (K + BK - 1) / BK;
#if DBUF
  STAGE_TILE(0, 0);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int kt = 0; kt < kTiles; kt += 2) {
    STAGE_TILE(1, (kt + 1) * BK);   // harmless zero-fill when past K
    COMPUTE_TILE(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    STAGE_TILE(0, (kt + 2) * BK);
    COMPUTE_TILE(1);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#else
  for (int kt = 0; kt < kTiles; ++kt) {
    STAGE_TILE(0, kt * BK);
    barrier(CLK_LOCAL_MEM_FENCE);
    COMPUTE_TILE(0);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#endif

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
#undef STAGE_TILE
#undef COMPUTE_TILE
#undef ASPTR
#undef BSPTR
}
