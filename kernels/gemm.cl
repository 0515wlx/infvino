// FP16 tiled GEMM: C[M,N] = A[M,K] * B[K,N]  (row-major, fp16 in/out).
//
// Round 8 design:
//   * B is staged ROW-MAJOR (Bs[BK][BN]) -> the transpose (and its BK-scaled SLM
//     bank conflicts) is gone; the previous transposed-B staging was what made
//     BK>=16 collapse.
//   * Both tiles are staged with VECTOR loads/stores (half4 by default) so the
//     staging instruction count drops ~4x.
//   * The inner loop reads b[] with a vector load too (N is contiguous).
//   * ASYNC=1 stages the tiles with async_work_group_copy (no register pressure).
//
// Knobs (-D): BM BN BK TM TN PAD VEC ASYNC SKIP_STAGE SKIP_COMPUTE
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef BM
#define BM 128
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
#ifndef PAD
#define PAD 0
#endif
#ifndef VEC
#define VEC 4
#endif
#ifndef ASYNC
#define ASYNC 0
#endif
#ifndef SKIP_STAGE
#define SKIP_STAGE 0
#endif
#ifndef SKIP_COMPUTE
#define SKIP_COMPUTE 0
#endif

#define LX (BN / TN)
#define LY (BM / TM)
#define NTHR (LX * LY)
#define AR (BK + PAD)
#define BR (BN + PAD)

// Staging vector chunks per row.
#define AC_PR (BK / VEC)
#define BC_PR (BN / VEC)

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

  __local half As[BM][AR];
  __local half Bs[BK][BR];


// ---- vectorized synchronous staging ----
#define STAGE_VEC(BUF_UNUSED, K0)                                                   \
  do {                                                                              \
    const int k0_ = (K0);                                                           \
    if ((K % VEC) == 0) {                                                           \
      _Pragma("unroll") for (int i = 0; i < (BM * AC_PR + NTHR - 1) / NTHR; ++i) {  \
        int idx = tid + i * NTHR;                                                   \
        if (idx < BM * AC_PR) {                                                     \
          int r = idx / AC_PR, c = (idx % AC_PR) * VEC;                             \
          int gr = blockRow + r, gc = k0_ + c;                                      \
          if (gr < M && gc + VEC <= K) {                                            \
            *(__local half4 *)&As[r][c] = *(__global const half4 *)&A[gr * K + gc];  \
          } else {                                                                  \
            _Pragma("unroll") for (int v = 0; v < VEC; ++v)                         \
              As[r][c + v] = (gr < M && gc + v < K) ? A[gr * K + gc + v] : (half)0; \
          }                                                                         \
        }                                                                           \
      }                                                                             \
    } else {                                                                        \
      _Pragma("unroll") for (int i = 0; i < (BM * BK + NTHR - 1) / NTHR; ++i) {     \
        int idx = tid + i * NTHR;                                                   \
        if (idx < BM * BK) {                                                        \
          int r = idx / BK, c = idx % BK;                                           \
          int gr = blockRow + r, gc = k0_ + c;                                      \
          As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0;                 \
        }                                                                           \
      }                                                                             \
    }                                                                               \
    if ((N % VEC) == 0) {                                                           \
      _Pragma("unroll") for (int i = 0; i < (BK * BC_PR + NTHR - 1) / NTHR; ++i) {  \
        int idx = tid + i * NTHR;                                                   \
        if (idx < BK * BC_PR) {                                                     \
          int r = idx / BC_PR, c = (idx % BC_PR) * VEC;                             \
          int gr = k0_ + r, gc = blockCol + c;                                      \
          if (gr < K && gc + VEC <= N) {                                            \
            *(__local half4 *)&Bs[r][c] = *(__global const half4 *)&B[gr * N + gc];  \
          } else {                                                                  \
            _Pragma("unroll") for (int v = 0; v < VEC; ++v)                         \
              Bs[r][c + v] = (gr < K && gc + v < N) ? B[gr * N + gc + v] : (half)0; \
          }                                                                         \
        }                                                                           \
      }                                                                             \
    } else {                                                                        \
      _Pragma("unroll") for (int i = 0; i < (BK * BN + NTHR - 1) / NTHR; ++i) {     \
        int idx = tid + i * NTHR;                                                   \
        if (idx < BK * BN) {                                                        \
          int r = idx / BN, c = idx % BN;                                           \
          int gr = k0_ + r, gc = blockCol + c;                                      \
          Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0;                 \
        }                                                                           \
      }                                                                             \
    }                                                                               \
  } while (0)

// ---- async (group DMA) staging of the contiguous B tile; A stays vector-sync ----
#define STAGE_ASYNC(K0)                                                             \
  do {                                                                              \
    const int k0_ = (K0);                                                           \
    _Pragma("unroll") for (int i = 0; i < (BM * AC_PR + NTHR - 1) / NTHR; ++i) {    \
      int idx = tid + i * NTHR;                                                     \
      if (idx < BM * AC_PR) {                                                       \
        int r = idx / AC_PR, c = (idx % AC_PR) * VEC;                              \
        int gr = blockRow + r, gc = k0_ + c;                                        \
        if (gr < M && gc + VEC <= K) {                                              \
          *(__local half4 *)&As[r][c] = *(__global const half4 *)&A[gr * K + gc];   \
        } else {                                                                    \
          _Pragma("unroll") for (int v = 0; v < VEC; ++v)                           \
            As[r][c + v] = (gr < M && gc + v < K) ? A[gr * K + gc + v] : (half)0;   \
        }                                                                           \
      }                                                                             \
    }                                                                               \
    if (((N & 1) == 0) && (blockCol + BN <= N) && (k0_ + BK <= K)) {                \
      event_t ev[BK];                                                               \
      _Pragma("unroll") for (int r = 0; r < BK; ++r)                                \
        ev[r] = async_work_group_copy((__local uint *)&Bs[r][0],                    \
                                      (__global const uint *)&B[(k0_ + r) * N + blockCol], \
                                      BN / 2, 0);                                   \
      wait_group_events(BK, ev);                                                    \
    } else {                                                                        \
      _Pragma("unroll") for (int i = 0; i < (BK * BN + NTHR - 1) / NTHR; ++i) {     \
        int idx = tid + i * NTHR;                                                   \
        if (idx < BK * BN) {                                                        \
          int r = idx / BN, c = idx % BN;                                           \
          int gr = k0_ + r, gc = blockCol + c;                                      \
          Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0;                 \
        }                                                                           \
      }                                                                             \
    }                                                                               \
  } while (0)

  half acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (half)(0);

#define COMPUTE_TILE()                                                              \
  do {                                                                              \
    _Pragma("unroll") for (int kk = 0; kk < BK; ++kk) {                            \
      half a[TM];                                                                   \
      half b[TN];                                                                   \
      _Pragma("unroll") for (int i = 0; i < TM; ++i) a[i] = As[ly * TM + i][kk];    \
      _Pragma("unroll") for (int v = 0; v < TN / VEC; ++v)                          \
        *((half4 *)&b[v * VEC]) = *(__local half4 *)&Bs[kk][lx * TN + v * VEC];     \
      _Pragma("unroll") for (int i = 0; i < TM; ++i)                               \
        _Pragma("unroll") for (int j = 0; j < TN; ++j)                             \
          acc[i][j] = mad(a[i], b[j], acc[i][j]);                                   \
    }                                                                               \
  } while (0)

  const int kTiles = (K + BK - 1) / BK;
#if SKIP_STAGE
  STAGE_VEC(0, 0);
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
  for (int kt = 0; kt < kTiles; ++kt) {
#if !SKIP_STAGE
#if ASYNC
    STAGE_ASYNC(kt * BK);
#else
    STAGE_VEC(0, kt * BK);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if SKIP_COMPUTE
    acc[0][0] += As[ly * TM][kt % AR] + Bs[lx * TN][kt % BR];
#else
    COMPUTE_TILE();
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int gr = blockRow + ly * TM + i;
      int gc = blockCol + lx * TN + j;
      if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
    }
#undef STAGE_VEC
#undef STAGE_ASYNC
#undef COMPUTE_TILE
}
