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
// Round 9 design:
//   * DBUF=1 double-buffers BOTH tiles and software-pipelines the k-loop: tile
//     kt+1 is staged into the other buffer *before* computing tile kt, so the
//     global->SLM latency overlaps the mad loop. The k-loop is explicitly
//     unrolled by two so the buffer indices stay compile-time constants (with a
//     runtime `(kt&1)` index IGC deletes the whole compute body -> 0 mad).
//   * DBUF doubles SLM/WG, so BK had to drop 32->16 to stay inside the ~16 KB
//     SLM budget; that trade is a net win (6.75 -> 11.8 @ 4096x512x512).
//   * SKIP_STAGE / SKIP_COMPUTE are bottleneck probes (diagnostic only).
//
// Round 12: the dominant lever turned out to be a codegen cliff, not SLM.
//   * IGC silently drops the *full* kernel to SIMD8 mads under the staging
//     register pressure; forcing SIMD16 with `-DSG=16`
//     (intel_reqd_sub_group_size) recovers the FPU width. SG is the requested
//     sub-group size (0 = let IGC decide; 16 is the Xe-LP sweet spot, 32 spills).
//   * Best config: keep the pipeline you like -- with SG=16 both DBUF=1 BK=16
//     and DBUF=0 BK=32 hit 13.7 ops/EU/cyc @4096x512x512 (vs 11.9 at SG=0):
//     one SLM buffer + double the k-tile is enough once the FPU width is fixed.
//   * PF=1 (prefetch the next tile into registers) is now unnecessary and, at
//     SG=0, forces SIMD8 -> 7.45. GN=1 (no __local, A/B from global) puts the
//     global latency on the mad chain -> 1.89. Both kept as negatives (default 0).
//   * SKIP_BARRIER=1 is a diagnostic (drop the k-loop barrier).
//
// Knobs (-D): BM BN BK TM TN PAD VEC ASYNC DBUF AT PF GN SG SKIP_STAGE SKIP_COMPUTE SKIP_BARRIER
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 16
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
#ifndef DBUF
#define DBUF 1
#endif
#ifndef PF
#define PF 0
#endif
#ifndef GN
#define GN 0
#endif
#if PF && GN
#error "PF and GN are mutually exclusive"
#endif
#ifndef AT
#define AT 0
#endif
#ifndef ASP
#define ASP (BM + PAD)
#endif
// The transposed-A layout (AT=1) indexes rows with stride ASP, so it is only
// valid without extra padding; the row-major path owns PAD.
#if AT && PAD
#error "AT=1 does not support PAD (both are negatives; see docs/kernel.md R11)"
#endif
#ifndef SKIP_STAGE
#define SKIP_STAGE 0
#endif
#ifndef SKIP_COMPUTE
#define SKIP_COMPUTE 0
#endif
#ifndef SKIP_BARRIER
#define SKIP_BARRIER 0
#endif
// Round 14: keep the exact mad structure but replace the per-kk SLM operand
// loads with register constants (diagnostic: isolates the feed from the
// occupancy/issue structure).
#ifndef NOLOAD
#define NOLOAD 0
#endif
// Force the compiler to keep SIMD16 (diagnostic probe for the SIMD8 cliff).
#ifndef SG
#define SG 0
#endif

// Fused epilogue (Round 22): when EPI=1 the kernel takes an extra `Bias` argument
// and applies `act(acc + bias)` at store time, so a 1x1 conv / fc no longer needs
// separate `bias_add` + `ew_unary` launches. ACT codes:
// 0=none 1=SiLU 2=Relu 3=Hardswish 4=Hardsigmoid 5=Sigmoid.
#ifndef EPI
#define EPI 0
#endif
#ifndef ACT
#define ACT 0
#endif
#ifndef RES
#define RES 0
#endif

inline half gemm_activate(half v) {
#if ACT == 1
  float f = (float)v;
  return (half)(f / (1.0f + exp(-f)));
#elif ACT == 2
  return (half)fmax((float)v, 0.0f);
#elif ACT == 3
  float f = (float)v;
  return (half)(f * fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);
#elif ACT == 4
  float f = (float)v;
  return (half)(fmin(fmax(f + 3.0f, 0.0f), 6.0f) / 6.0f);
#elif ACT == 5
  return (half)(1.0f / (1.0f + exp(-(float)v)));
#else
  return v;
#endif
}

#define LX (BN / TN)
#define LY (BM / TM)
#define NTHR (LX * LY)
#define AR (BK + PAD)
#define BR (BN + PAD)

// Staging vector chunks per row.
#define AC_PR (BK / VEC)
#define BC_PR (BN / VEC)

// Round 12: register-prefetch pipeline (PF=1). The global->SLM staging of the
// next k-tile is issued into registers *before* the current tile's mad loop, so
// the global latency hides behind compute. That needs only ONE SLM buffer
// (2*BK*(BM+BN) halfs) instead of the double-buffer's two, which lets BK double
// while staying inside the ~12 KB/WG occupancy wall -- halving the barrier count
// (the dominant cost, see docs/kernel.md R12).
#if PF
#define ACHUNKS ((BM * (BK / VEC)) / NTHR)
#define BCHUNKS ((BK * (BN / VEC)) / NTHR)
#if ((BM * (BK / VEC)) % NTHR) || ((BK * (BN / VEC)) % NTHR)
#error "PF requires BM*(BK/VEC) and BK*(BN/VEC) to be exact multiples of NTHR"
#endif
#if AT
#error "PF=1 is only implemented for the row-major A path"
#endif
#endif

#if SG
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
__attribute__((reqd_work_group_size(LX, LY, 1)))
__kernel void gemm_f16(__global const half *restrict A,
                       __global const half *restrict B,
                       __global half *restrict C,
                       const int M, const int N, const int K
#if EPI
                       , __global const half *restrict Bias
                       , __global const half *restrict Res
#endif
                       ) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int tid = ly * LX + lx;
  const int blockRow = get_group_id(1) * BM;
  const int blockCol = get_group_id(0) * BN;

  // SLM tiles (Round 9):
  //   DBUF=1 -> 2 A-buffers + 2 B-buffers (double-buffered software pipeline).
  //   DBUF=0 -> 1 A-buffer + 1 B-buffer (classic single-buffer).
  // DBUF=1 doubles SLM/WG, which is why the default BK is 16 (see Round 9).
#if !GN
#if AT
  __local half As_[PF ? 1 : (DBUF ? 2 : 1)][BK][ASP];
#else
  __local half As_[PF ? 1 : (DBUF ? 2 : 1)][BM][AR];
#endif
  __local half Bs_[PF ? 1 : (DBUF ? 2 : 1)][BK][BR];
#endif


// ---- vectorized synchronous staging of the A tile into buffer `pa` ----
#define STAGE_A(pa, K0)                                                             \
  do {                                                                              \
    const int k0_ = (K0);                                                           \
    if (AT) {                                                                       \
      __local half (*AsT)[ASP] = (__local half (*)[ASP])As_ + (pa) * BK;           \
      for (int idx = tid; idx < BM * BK; idx += NTHR) {                            \
        int r = idx / BK, c = idx % BK;                                             \
        int gr = blockRow + r, gc = k0_ + c;                                        \
        AsT[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0;                  \
      }                                                                             \
    } else if ((K % VEC) == 0) {                                                    \
      __local half (*As)[AR] = (__local half (*)[AR])As_ + (pa) * BM;              \
      if ((blockRow + BM <= M) && (k0_ + BK <= K)) {                               \
        _Pragma("unroll") for (int i = 0; i < (BM * AC_PR + NTHR - 1) / NTHR; ++i) { \
          int idx = tid + i * NTHR;                                                 \
          if (idx < BM * AC_PR) {                                                   \
            int r = idx / AC_PR, c = (idx % AC_PR) * VEC;                           \
            *(__local half4 *)&As[r][c] =                                           \
              *(__global const half4 *)&A[(blockRow + r) * K + k0_ + c];            \
          }                                                                         \
        }                                                                           \
      } else {                                                                      \
        _Pragma("unroll") for (int i = 0; i < (BM * AC_PR + NTHR - 1) / NTHR; ++i) { \
          int idx = tid + i * NTHR;                                                 \
          if (idx < BM * AC_PR) {                                                   \
            int r = idx / AC_PR, c = (idx % AC_PR) * VEC;                           \
            int gr = blockRow + r, gc = k0_ + c;                                    \
            if (gr < M && gc + VEC <= K) {                                          \
              *(__local half4 *)&As[r][c] = *(__global const half4 *)&A[gr * K + gc]; \
            } else {                                                                \
              _Pragma("unroll") for (int v = 0; v < VEC; ++v)                       \
                As[r][c + v] = (gr < M && gc + v < K) ? A[gr * K + gc + v] : (half)0; \
            }                                                                       \
          }                                                                         \
        }                                                                           \
      }                                                                             \
    } else {                                                                        \
      __local half (*As)[AR] = (__local half (*)[AR])As_ + (pa) * BM;              \
      _Pragma("unroll") for (int i = 0; i < (BM * BK + NTHR - 1) / NTHR; ++i) {     \
        int idx = tid + i * NTHR;                                                   \
        if (idx < BM * BK) {                                                        \
          int r = idx / BK, c = idx % BK;                                           \
          int gr = blockRow + r, gc = k0_ + c;                                      \
          As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0;                 \
        }                                                                           \
      }                                                                             \
    }                                                                               \
  } while (0)

// ---- vectorized synchronous staging of the B tile into buffer `pb` ----
#define STAGE_B(pb, K0)                                                             \
  do {                                                                              \
    __local half (*Bs)[BR] = (__local half (*)[BR])Bs_ + (pb) * BK;                \
    const int k0_ = (K0);                                                           \
    if ((N % VEC) == 0) {                                                           \
      if ((blockCol + BN <= N) && (k0_ + BK <= K)) {                                \
        _Pragma("unroll") for (int i = 0; i < (BK * BC_PR + NTHR - 1) / NTHR; ++i) { \
          int idx = tid + i * NTHR;                                                 \
          if (idx < BK * BC_PR) {                                                   \
            int r = idx / BC_PR, c = (idx % BC_PR) * VEC;                           \
            *(__local half4 *)&Bs[r][c] =                                           \
              *(__global const half4 *)&B[(k0_ + r) * N + blockCol + c];            \
          }                                                                         \
        }                                                                           \
      } else {                                                                      \
        _Pragma("unroll") for (int i = 0; i < (BK * BC_PR + NTHR - 1) / NTHR; ++i) { \
          int idx = tid + i * NTHR;                                                 \
          if (idx < BK * BC_PR) {                                                   \
            int r = idx / BC_PR, c = (idx % BC_PR) * VEC;                           \
            int gr = k0_ + r, gc = blockCol + c;                                    \
            if (gr < K && gc + VEC <= N) {                                          \
              *(__local half4 *)&Bs[r][c] = *(__global const half4 *)&B[gr * N + gc]; \
            } else {                                                                \
              _Pragma("unroll") for (int v = 0; v < VEC; ++v)                       \
                Bs[r][c + v] = (gr < K && gc + v < N) ? B[gr * N + gc + v] : (half)0; \
            }                                                                       \
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

// ---- combined synchronous staging (single buffer) ----
#define STAGE_VEC(pb, K0)                                                           \
  do {                                                                              \
    STAGE_A(pb, K0);                                                               \
    STAGE_B(pb, K0);                                                               \
  } while (0)

// ---- async (group DMA) staging of the contiguous B tile; A stays vector-sync ----
#define STAGE_ASYNC(pb, K0)                                                         \
  do {                                                                              \
    __local half (*As)[AR] = (__local half (*)[AR])As_ + (pb) * BM;                 \
    __local half (*Bs)[BR] = (__local half (*)[BR])Bs_ + (pb) * BK;                \
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

#if PF
// ---- Round 12 register prefetch -------------------------------------------
// Issue the global loads for tile `K0` into per-thread registers (pa/pb) so the
// latency overlaps the following COMPUTE_TILE. STORE_PREFETCH then commits those
// registers to the single SLM buffer after the compute barrier (no latency, just
// the SLM store). The chunk->(row,col) mapping is identical to STAGE_A/STAGE_B.
#define PREFETCH(K0)                                                             \
  do {                                                                            \
    const int k0_ = (K0);                                                         \
    if ((K % VEC) == 0) {                                                         \
      _Pragma("unroll") for (int i = 0; i < ACHUNKS; ++i) {                       \
        int idx = tid + i * NTHR;                                                 \
        int r = idx / AC_PR, c = (idx % AC_PR) * VEC;                             \
        int gr = blockRow + r, gc = k0_ + c;                                      \
        if (gr < M && gc + VEC <= K) {                                            \
          pa[i] = *(__global const half4 *)&A[gr * K + gc];                       \
        } else {                                                                  \
          _Pragma("unroll") for (int v = 0; v < VEC; ++v)                         \
            pa[i][v] = (gr < M && gc + v < K) ? A[gr * K + gc + v] : (half)0;     \
        }                                                                         \
      }                                                                           \
    } else {                                                                      \
      _Pragma("unroll") for (int i = 0; i < ACHUNKS; ++i) {                       \
        int idx = tid + i * NTHR;                                                 \
        int r = idx / (BK / VEC), c = (idx % (BK / VEC)) * VEC;                   \
        int gr = blockRow + r, gc = k0_ + c;                                      \
        _Pragma("unroll") for (int v = 0; v < VEC; ++v)                           \
          pa[i][v] = (gr < M && gc + v < K) ? A[gr * K + gc + v] : (half)0;       \
      }                                                                           \
    }                                                                             \
    if ((N % VEC) == 0) {                                                         \
      _Pragma("unroll") for (int i = 0; i < BCHUNKS; ++i) {                       \
        int idx = tid + i * NTHR;                                                 \
        int r = idx / BC_PR, c = (idx % BC_PR) * VEC;                             \
        int gr = k0_ + r, gc = blockCol + c;                                      \
        if (gr < K && gc + VEC <= N) {                                            \
          pb[i] = *(__global const half4 *)&B[gr * N + gc];                       \
        } else {                                                                  \
          _Pragma("unroll") for (int v = 0; v < VEC; ++v)                         \
            pb[i][v] = (gr < K && gc + v < N) ? B[gr * N + gc + v] : (half)0;     \
        }                                                                         \
      }                                                                           \
    } else {                                                                      \
      _Pragma("unroll") for (int i = 0; i < BCHUNKS; ++i) {                       \
        int idx = tid + i * NTHR;                                                 \
        int r = idx / (BN / VEC), c = (idx % (BN / VEC)) * VEC;                   \
        int gr = k0_ + r, gc = blockCol + c;                                      \
        _Pragma("unroll") for (int v = 0; v < VEC; ++v)                           \
          pb[i][v] = (gr < K && gc + v < N) ? B[gr * N + gc + v] : (half)0;       \
      }                                                                           \
    }                                                                             \
  } while (0)

#define STORE_PREFETCH()                                                         \
  do {                                                                            \
    __local half (*As)[AR] = (__local half (*)[AR])As_;                           \
    __local half (*Bs)[BR] = (__local half (*)[BR])Bs_;                           \
    _Pragma("unroll") for (int i = 0; i < ACHUNKS; ++i) {                         \
      int idx = tid + i * NTHR;                                                   \
      int r = idx / AC_PR, c = (idx % AC_PR) * VEC;                               \
      *(__local half4 *)&As[r][c] = pa[i];                                        \
    }                                                                             \
    _Pragma("unroll") for (int i = 0; i < BCHUNKS; ++i) {                         \
      int idx = tid + i * NTHR;                                                   \
      int r = idx / BC_PR, c = (idx % BC_PR) * VEC;                               \
      *(__local half4 *)&Bs[r][c] = pb[i];                                        \
    }                                                                             \
  } while (0)
#endif  // PF

  half acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (half)(0);

#if AT
// A is stored transposed AsT[BK][ASP]; the TM values for a thread are contiguous,
// so they come back as one or more vector loads instead of TM scalar loads.
#define LOAD_A(aptr, kk, pa)                                                            \
  do {                                                                              \
    __local half (*AsT)[ASP] = (__local half (*)[ASP])As_ + (pa) * BK;             \
    _Pragma("unroll") for (int v = 0; v < TM / 8; ++v)                              \
      *(half8 *)(&aptr[v * 8]) = *(__local half8 *)&AsT[kk][ly * TM + v * 8];       \
    _Pragma("unroll") for (int v = (TM / 8) * 8; v < TM; ++v)                       \
      aptr[v] = AsT[kk][ly * TM + v];                                               \
  } while (0)
#else
#define LOAD_A(aptr, kk, pa)                                                            \
  do {                                                                              \
    __local half (*As)[AR] = (__local half (*)[AR])As_ + (pa) * BM;                 \
    _Pragma("unroll") for (int i = 0; i < TM; ++i) aptr[i] = As[ly * TM + i][kk];   \
  } while (0)
#endif

// Round 14: explicit kk-level software pipeline -- prefetch the next kk's A/B
// operands while the current kk's mads run. Tries to hide the SLM load latency
// that NOLOAD shows costs ~35%.
#ifndef PIPE
#define PIPE 0
#endif

#if PIPE
#define LOAD_B(bptr, kk, pb)                                                        \
  do {                                                                              \
    __local half (*Bs)[BR] = (__local half (*)[BR])Bs_ + (pb) * BK;                \
    _Pragma("unroll") for (int v = 0; v < TN / VEC; ++v)                            \
      *((half4 *)&bptr[v * VEC]) = *(__local half4 *)&Bs[kk][lx * TN + v * VEC];    \
  } while (0)

#define COMPUTE_TILE(pa, pb)                                                        \
  do {                                                                              \
    half a[TM], b[TN], an[TM], bn[TN];                                              \
    LOAD_A(a, 0, pa);                                                               \
    LOAD_B(b, 0, pb);                                                               \
    _Pragma("unroll") for (int kk = 0; kk < BK; ++kk) {                            \
      if (kk + 1 < BK) { LOAD_A(an, kk + 1, pa); LOAD_B(bn, kk + 1, pb); }          \
      _Pragma("unroll") for (int i = 0; i < TM; ++i)                               \
        _Pragma("unroll") for (int j = 0; j < TN; ++j)                             \
          acc[i][j] = mad(a[i], b[j], acc[i][j]);                                   \
      _Pragma("unroll") for (int i = 0; i < TM; ++i) a[i] = an[i];                 \
      _Pragma("unroll") for (int j = 0; j < TN; ++j) b[j] = bn[j];                 \
    }                                                                               \
  } while (0)
#else
#define COMPUTE_TILE(pa, pb)                                                        \
  do {                                                                              \
    __local half (*Bs)[BR] = (__local half (*)[BR])Bs_ + (pb) * BK;                \
    _Pragma("unroll") for (int kk = 0; kk < BK; ++kk) {                            \
      half a[TM];                                                                   \
      half b[TN];                                                                   \
      _Pragma("unroll") for (int i = 0; i < TM; ++i) a[i] = (half)(lx + i);         \
      _Pragma("unroll") for (int j = 0; j < TN; ++j) b[j] = (half)(ly + j);         \
      if (!(NOLOAD)) {                                                              \
        LOAD_A(a, kk, pa);                                                          \
        _Pragma("unroll") for (int v = 0; v < TN / VEC; ++v)                        \
          *((half4 *)&b[v * VEC]) = *(__local half4 *)&Bs[kk][lx * TN + v * VEC];   \
      }                                                                             \
      _Pragma("unroll") for (int i = 0; i < TM; ++i)                               \
        _Pragma("unroll") for (int j = 0; j < TN; ++j)                             \
          acc[i][j] = mad(a[i], b[j], acc[i][j]);                                   \
    }                                                                               \
  } while (0)
#endif

  const int kTiles = (K + BK - 1) / BK;
#if SKIP_STAGE && !GN
  STAGE_VEC(0, 0);
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GN
  // ---- Round 12 no-SLM GEMM: read A/B straight from global (L1/L2) ----
  // No __local, hence no work-group barriers and no SLM capacity/occupancy
  // coupling at all. The WG-resident A/B tiles are re-read every kk but the
  // in-WG reuse (A broadcast over lx, B broadcast over ly) is served by the L1
  // cache. This is the direct test of whether the SLM wall is real.
  for (int kt = 0; kt < kTiles; ++kt) {
    _Pragma("unroll") for (int kk = 0; kk < BK; ++kk) {
      const int k = kt * BK + kk;
      half a[TM], b[TN];
      _Pragma("unroll") for (int i = 0; i < TM; ++i) {
        const int gr = blockRow + ly * TM + i;
        a[i] = (gr < M && k < K) ? A[gr * K + k] : (half)0;
      }
      _Pragma("unroll") for (int j = 0; j < TN; ++j) {
        const int gc = blockCol + lx * TN + j;
        b[j] = (k < K && gc < N) ? B[k * N + gc] : (half)0;
      }
      _Pragma("unroll") for (int i = 0; i < TM; ++i)
        _Pragma("unroll") for (int j = 0; j < TN; ++j)
          acc[i][j] = mad(a[i], b[j], acc[i][j]);
    }
  }
#elif PF
  // ---- Round 12 single-buffer pipeline with register prefetch ----
  // No second SLM buffer: tile kt+1 is fetched into registers (pa/pb) *before*
  // the tile-kt mad loop, so the global latency hides behind compute; the
  // registers are committed to the single SLM buffer after the compute barrier.
  // This keeps SLM at 2*BK*(BM+BN) halfs, so BK can double vs DBUF=1 and the
  // barrier count (2/k-tile) halves.
  {
    half4 pa[ACHUNKS];
    half4 pb[BCHUNKS];
    STAGE_VEC(0, 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kt = 0; kt < kTiles; ++kt) {
      if (kt + 1 < kTiles) PREFETCH((kt + 1) * BK);
      COMPUTE_TILE(0, 0);
      barrier(CLK_LOCAL_MEM_FENCE);
      if (kt + 1 < kTiles) STORE_PREFETCH();
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
#elif DBUF == 1
  // ---- full double-buffered software pipeline ----
  // Stage tile (kt+1) into the *other* buffer before computing tile kt, so the
  // global->SLM latency overlaps the mad loop. The k-tile loop is explicitly
  // unrolled by two so that the buffer indices are compile-time constants; with
  // a runtime `(kt&1)` index IGC drops the whole compute body (verified: 0 mad).
  STAGE_VEC(0, 0);
  barrier(CLK_LOCAL_MEM_FENCE);
  int kt = 0;
  for (; kt + 1 < kTiles; kt += 2) {
#if !SKIP_STAGE
    STAGE_VEC(1, (kt + 1) * BK);
#endif
    COMPUTE_TILE(0, 0);
    barrier(CLK_LOCAL_MEM_FENCE);
#if !SKIP_STAGE
    if (kt + 2 < kTiles) STAGE_VEC(0, (kt + 2) * BK);
#endif
    COMPUTE_TILE(1, 1);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (kt < kTiles) {  // odd tail: last tile already staged, just compute it
    COMPUTE_TILE(0, 0);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#else
  for (int kt = 0; kt < kTiles; ++kt) {
#if !SKIP_STAGE
#if ASYNC
    STAGE_ASYNC(0, kt * BK);
#else
    STAGE_VEC(0, kt * BK);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if SKIP_COMPUTE
    acc[0][0] += ((__local half *)As_)[ly * TM * AR] + ((__local half *)Bs_)[lx * TN * BR];
#else
    COMPUTE_TILE(0, 0);
#endif
#if !SKIP_BARRIER
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
  }
#endif

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int gr = blockRow + ly * TM + i;
      int gc = blockCol + lx * TN + j;
      if (gr < M && gc < N) {
#if EPI
        half v = acc[i][j] + (Bias ? Bias[gr] : (half)0);
        v = gemm_activate(v);
#if RES
        if (Res) v = v + Res[(size_t)gr * N + gc];
#endif
        C[gr * N + gc] = v;
#else
        C[gr * N + gc] = acc[i][j];
#endif
      }
    }
#undef STAGE_VEC
#undef STAGE_A
#undef STAGE_B
#undef STAGE_ASYNC
#undef COMPUTE_TILE
#undef LOAD_A
#if PF
#undef PREFETCH
#undef STORE_PREFETCH
#endif
#if PIPE
#undef LOAD_B
#endif
}
