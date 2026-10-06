// Micro-benchmarks for the Xe-LP roofline (Round 10): ALU latency/throughput and
// the cache-hierarchy size/bandwidth curve. Standalone; not used by inference.
//
// All kernels take explicit work sizes so a single launch is short (< 100 ms),
// per docs/benchmark_protocol.md.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ---------------------------------------------------------------------------
// A. FP16 / FP32 FMA latency & throughput.
//   DEPTH independent accumulator chains, ITERS iterations, WIDTH lanes/elem.
//   DEPTH=1 -> the loop is a single dependency chain  -> measures LATENCY.
//   DEPTH large -> enough ILP to saturate the FPU      -> measures THROUGHPUT.
// Each work-item runs the same chain; NTH work-items keep the EU busy.
// ---------------------------------------------------------------------------
#ifndef DEPTH
#define DEPTH 1
#endif
#ifndef ITERS
#define ITERS 4096
#endif
// Round 13: force sub-group width (0 = IGC decides). Applied to the FMA probes.
#ifndef SG
#define SG 0
#endif
#if SG
#define SGATTR __attribute__((intel_reqd_sub_group_size(SG)))
#else
#define SGATTR
#endif

__attribute__((reqd_work_group_size(64, 1, 1))) SGATTR
__kernel void fma_f32_lat(__global float *out, const float a, const float b) {
  float acc[DEPTH];
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) acc[d] = (float)(get_global_id(0) + d);
  for (int i = 0; i < ITERS; ++i) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) acc[d] = mad(acc[d], a, b);
  }
  float s = 0.f;
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) s += acc[d];
  if (s == -12345.678f) out[0] = s;  // never true; keeps the chain alive
}

__attribute__((reqd_work_group_size(64, 1, 1))) SGATTR
__kernel void fma_h1_lat(__global half *out, const half a, const half b) {
  half acc[DEPTH];
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) acc[d] = (half)(get_global_id(0) + d);
  for (int i = 0; i < ITERS; ++i) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) acc[d] = mad(acc[d], a, b);
  }
  half s = 0;
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) s += acc[d];
  if (s == (half)-12345.0f) out[0] = s;
}

// half2 / half4 / half8: DEPTH chains, each chain is a vector FMA.
__attribute__((reqd_work_group_size(64, 1, 1))) SGATTR
__kernel void fma_h2_lat(__global half *out, const half a, const half b) {
  half2 acc[DEPTH];
  const half2 av = (half2)(a, a), bv = (half2)(b, b);
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) acc[d] = (half2)((half)(get_global_id(0) + d), (half)0);
  for (int i = 0; i < ITERS; ++i) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) acc[d] = mad(acc[d], av, bv);
  }
  half2 s = (half2)(0, 0);
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) s += acc[d];
  if (s.s0 == (half)-12345.0f) out[0] = s.s0;
}

__attribute__((reqd_work_group_size(64, 1, 1))) SGATTR
__kernel void fma_h4_lat(__global half *out, const half a, const half b) {
  half4 acc[DEPTH];
  const half4 av = (half4)(a, a, a, a), bv = (half4)(b, b, b, b);
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) acc[d] = (half4)((half)(get_global_id(0) + d), 0, 0, 0);
  for (int i = 0; i < ITERS; ++i) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) acc[d] = mad(acc[d], av, bv);
  }
  half4 s = (half4)(0, 0, 0, 0);
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) s += acc[d];
  if (s.s0 == (half)-12345.0f) out[0] = s.s0;
}

__attribute__((reqd_work_group_size(64, 1, 1))) SGATTR
__kernel void fma_h8_lat(__global half *out, const half a, const half b) {
  half8 acc[DEPTH];
  const half8 av = (half8)(a, a, a, a, a, a, a, a);
  const half8 bv = (half8)(b, b, b, b, b, b, b, b);
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) acc[d] = (half8)((half)(get_global_id(0) + d), 0, 0, 0, 0, 0, 0, 0);
  for (int i = 0; i < ITERS; ++i) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) acc[d] = mad(acc[d], av, bv);
  }
  half8 s = (half8)(0, 0, 0, 0, 0, 0, 0, 0);
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) s += acc[d];
  if (s.s0 == (half)-12345.0f) out[0] = s.s0;
}

// RF read-port probe: both multiply operands are the SAME register (one GRF
// read instead of two). If FP16 FMA throughput is register-bandwidth limited,
// this should beat the two-distinct-operand fma_h8_lat.
__attribute__((reqd_work_group_size(64, 1, 1))) SGATTR
__kernel void fma_h8_1op(__global half *out, const half a, const half b) {
  half8 acc[DEPTH];
  const half8 av = (half8)(a, a, a, a, a, a, a, a);
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) acc[d] = (half8)((half)(get_global_id(0) + d), 0, 0, 0, 0, 0, 0, 0);
  for (int i = 0; i < ITERS; ++i) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) acc[d] = mad(acc[d], av, av);
  }
  half8 s = (half8)(0, 0, 0, 0, 0, 0, 0, 0);
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) s += acc[d];
  if (s.s0 == (half)-12345.0f) out[0] = s.s0;
}

// ---------------------------------------------------------------------------
// Round 27: occupancy / register-pressure probe.
//
// A *single* cyclic dependency chain (ILP == 1) that touches DEPTH registers:
//     r[(d+1) & (DEPTH-1)] = mad(r[d], a, b)     (DEPTH a power of two)
// Each outer iteration is one chain of DEPTH serial mads, so per work-item the
// time is ITERS*DEPTH*L_cyc (L = FP16 FMA latency) as long as the EU can host
// the work-items.  Aggregate throughput at saturation:
//     ops/EU/cyc = 32 * T_res / L      (T_res = resident sub-groups per EU)
// so the plateau directly gives T_res.  Raising DEPTH raises register pressure
// (each r[d] is one GRF at SIMD16) *without* changing ILP, so T_res(DEPTH)
// answers: is the 4 KB/thread register file per-thread (flat T_res) or shared
// (T_res falls as DEPTH rises)?
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(64, 1, 1))) SGATTR
__kernel void fma_cyc(__global half *out, const half a, const half b) {
  half r[DEPTH];
#pragma unroll
  for (int d = 0; d < DEPTH; ++d)
    r[d] = (half)((get_global_id(0) + d) & 7) * (half)0.125f + (half)1.0f;
  for (int i = 0; i < ITERS; ++i) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) r[(d + 1) & (DEPTH - 1)] = mad(r[d], a, b);
  }
  half s = (half)0;
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) s += r[d];
  if (s == (half)-12345.0f) out[0] = s;
}

// A dependent integer add chain (cheap ALU, same loop shape) to gauge the
// per-iteration loop overhead; subtract from depth=1 to get true ALU latency.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void add_lat(__global int *out, const int a) {
  int acc = get_global_id(0);
  for (int i = 0; i < ITERS; ++i) acc = acc + a;
  if (acc == 0x7fffffff) out[0] = acc;
}

// R47: work-group-size sweep for the occupancy probe. Same code as fma_cyc but
// work-group size is a macro, so we can find whether the EU threshold scales
// with WG size (raw threads) or is a fixed WG count (scheduler slots).
#ifndef WGS
#define WGS 64
#endif
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void fma_wgs(__global half *out, const half a, const half b) {
  half r[DEPTH];
#pragma unroll
  for (int d = 0; d < DEPTH; ++d)
    r[d] = (half)((get_global_id(0) + d) & 7) * (half)0.125f + (half)1.0f;
  for (int i = 0; i < ITERS; ++i) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) r[(d + 1) & (DEPTH - 1)] = mad(r[d], a, b);
  }
  half s = (half)0;
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) s += r[d];
  if (s == (half)-12345.0f) out[0] = s;
}

// ---------------------------------------------------------------------------
// B. SLM (L1) FMA feeding: mad reading one operand from __local, one from a
// register. Measures how fast the local-memory pipe can feed the FPU.
// ---------------------------------------------------------------------------__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void mad_slm(__global half *out, const uint iters) {
  __local half buf[4096];
  const int tid = get_local_id(0);
  for (int i = tid; i < 4096; i += 64) buf[i] = (half)(i & 7) * (half)0.125f;
  barrier(CLK_LOCAL_MEM_FENCE);
  half acc = (half)get_global_id(0);
  for (uint i = 0; i < iters; ++i) {
    half b = buf[(tid + i) & 4095];
    acc = mad(acc, b, (half)0.5f);
  }
  if (acc == (half)-12345.0f) out[0] = acc;
}

// ---------------------------------------------------------------------------
// C. Memory-latency pointer chase (reveals cache-hierarchy SIZES).
//    tbl holds a permutation so idx = tbl[idx] walks the whole working set with
//    a new access each step. Duration/count = per-access latency.
// ---------------------------------------------------------------------------
__kernel void mem_chase(__global const uint *restrict tbl, __global uint *sink,
                        const uint n, const uint iters) {
  uint idx = get_global_id(0) % n;
  for (uint i = 0; i < iters; ++i) idx = tbl[idx];
  sink[get_global_id(0)] = idx;
}

// ---------------------------------------------------------------------------
// D. Streaming scan over a working set of `n` halfs, reading every `step`-th
//    element. step=1 touches all lines (max traffic); larger step skips lines.
//    Used with a size sweep to get the bandwidth-vs-footprint curve.
// ---------------------------------------------------------------------------
__kernel void mem_scan(__global const half *restrict in, __global half *sink,
                       const uint n, const uint step, const uint iters) {
  const uint gid = get_global_id(0);
  const uint stride = (uint)get_global_size(0);
  half acc = (half)0;
  uint i = gid;
  for (uint it = 0; it < iters; ++it) {
    acc += in[i];
    i += stride * step;
    if (i >= n) i -= n;
  }
  if (acc == (half)-12345.0f) sink[0] = acc;
}

// ---------------------------------------------------------------------------
// E. SLM capacity probe: touch `kb` KB of __local and read it back. Reveals
//    the largest single-workgroup local allocation the device supports.
// ---------------------------------------------------------------------------
#ifndef SLM_KB
#define SLM_KB 16
#endif
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void slm_touch(__global uint *out, const uint iters) {
  __local uint buf[SLM_KB * 256];  // SLM_KB * 1024 bytes
  const int tid = get_local_id(0);
  for (uint i = tid; i < SLM_KB * 256; i += 64) buf[i] = i;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint acc = 0;
  for (uint i = 0; i < iters; ++i) acc += buf[(tid + i) % (SLM_KB * 256)];
  out[get_global_id(0)] = acc;
}

// ---------------------------------------------------------------------------
// F. Repeated streaming read of a footprint (internal loop) so a launch can be
//    long enough to be bandwidth-bound even for small footprints (L1/LLC).
//    Each pass reads every `step`-th element; traffic per pass = lines touched.
// ---------------------------------------------------------------------------
__kernel void scan_rep(__global const uint *restrict in, __global uint *sink,
                       const uint n, const uint step, const uint passes) {
  const uint gid = get_global_id(0);
  const uint stride = (uint)get_global_size(0);
  uint acc = 0;
  for (uint p = 0; p < passes; ++p) {
    for (uint i = gid; i < n; i += stride * step) acc += in[i];
  }
  if (acc == 0xdeadbeefu) sink[0] = acc;
}

// R47 calibration: per-work-item "footprint" load test. Each WI reads WS*uint
// (WS*4 bytes) starting at its own base (gid*WS), WS times, many passes. The
// RESIDENT working set = (concurrent WIs) * WS*4 — swept via the grid size. Used
// to calibrate perWG_bytes: effective bandwidth vs concurrent occupancy footprint.
// Layout is cache-line-strided (p*16 apart) so each WI keeps WS distinct lines hot.
__kernel void footprint_scan(__global const uint *restrict in, __global uint *sink,
                             const uint n, const uint passes, const uint ws) {
  const uint gid = get_global_id(0);
  const uint base = gid * ws;
  if (base + ws > n) return;
  uint acc = 0;
  for (uint p = 0; p < passes; ++p) {
    for (uint k = 0; k < ws; ++k) acc += in[base + ((k + p) % ws) * 16u];
  }
  if (acc == 0xdeadbeefu) sink[0] = acc;
}


// ---------------------------------------------------------------------------
// R55 L3 coupling calibration: **解耦驻留足迹与并发度**。
//   每个 work-item 反复触碰一块**私有、互不重叠**的窗口（`fp` 条 cache line，
//   每条 64 B），重复 `passes` 遍。取一个驻留波次时：
//       驻留足迹  R = gws · fp · 64 B
//       并发度    C = gws 线程
//       触碰次数  T = gws · fp · passes
//   在**固定 R** 下扫 (gws, fp) 即可把「容量/ miss」（由 R 决定）与
//   「并行度 / MLP」（由 C 决定）分开。passes=1 是纯流式（无复用，R 不影响命中）；
//   passes 大时 R 决定逐遍命中还是逐遍 Miss。
//   布局：第 gid 个 WI 的窗口 = [gid·fp·16, gid·fp·16 + (fp-1)·16]（uint 下标，
//   stride 16 uint = 64 B），故与相邻 WI 的窗口不相交。
// ---------------------------------------------------------------------------
__kernel void l3_probe(__global const uint *restrict in, __global uint *sink,
                       const uint fp, const uint passes) {
  const uint gid = get_global_id(0);
  __global const uint *base = in + (size_t)gid * (size_t)fp * 16u;
  uint a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (uint p = 0; p < passes; ++p) {
    uint k = 0;
    for (; k + 3u < fp; k += 4u) {
      a0 += base[(size_t)(k + 0u) * 16u];
      a1 += base[(size_t)(k + 1u) * 16u];
      a2 += base[(size_t)(k + 2u) * 16u];
      a3 += base[(size_t)(k + 3u) * 16u];
    }
    for (; k < fp; ++k) a0 += base[(size_t)k * 16u];
  }
  if ((a0 + a1 + a2 + a3) == 0xdeadbeefu) sink[0] = a0;
}

// ---------------------------------------------------------------------------
// R55 双租户污染探针：协作重读一个**共享热点**（`hot_lines` 条 line，重复 `hiters`
// 遍）。它紧跟在一次 aggressor 流（另一个 kernel，单独入队）之后运行；这一遍是
// 命中 L3 还是打到 DRAM，就揭示 aggressor 的在飞足迹有没有把热点逐出。
// 「B 的输入（热点）是否被 A 的占用（aggressor）冲掉」——即跨算子 L3 污染的直接测量。
// ---------------------------------------------------------------------------
__kernel void l3_hot_read(__global const uint *restrict hot, __global uint *sink,
                          const uint hot_lines, const uint hiters) {
  const uint gid = get_global_id(0);
  const uint gsz = get_global_size(0);
  uint acc = 0;
  for (uint p = 0; p < hiters; ++p)
    for (uint k = gid; k < hot_lines; k += gsz) acc += hot[(size_t)k * 16u];
  if (acc == 0xdeadbeefu) sink[0] = acc;
}

// ---------------------------------------------------------------------------
// H. Work-group barrier cost (Round 12). Each iteration does a real SLM store
//    + barrier + SLM load (the same shape as the GEMM k-loop's synchronisation),
//    so this measures the barrier+SLM-round-trip latency per iteration.
//    WG size is set with -DBWG.
// ---------------------------------------------------------------------------
#ifndef BWG
#define BWG 256
#endif
// BMODE 0: SLM store + barrier(LOCAL_FENCE) + SLM load  (GEMM-like)
// BMODE 1: SLM store + SLM load, NO barrier            (SLM round-trip only)
// BMODE 2: dependent ALU + barrier(LOCAL_FENCE)        (barrier only)
// BMODE 3: SLM store + barrier(0) + SLM load           (sync-only, no mem fence)
// BMODE 4: dependent ALU + barrier(0)
#ifndef BMODE
#define BMODE 0
#endif
__attribute__((reqd_work_group_size(BWG, 1, 1)))
__kernel void barrier_cost(__global uint *out, const uint iters) {
  __local uint lbuf[BWG];
  const int tid = get_local_id(0);
  uint acc = get_global_id(0) + 1u;
#if BMODE == 0
  for (uint i = 0; i < iters; ++i) {
    lbuf[tid] = acc; barrier(CLK_LOCAL_MEM_FENCE); acc += lbuf[(tid + 1) & (BWG - 1)];
  }
#elif BMODE == 1
  for (uint i = 0; i < iters; ++i) {
    lbuf[tid] = acc; acc += lbuf[(tid + 1) & (BWG - 1)];
  }
#elif BMODE == 2
  for (uint i = 0; i < iters; ++i) {
    acc = acc * 1664525u + 1013904223u; barrier(CLK_LOCAL_MEM_FENCE);
  }
#elif BMODE == 3
  for (uint i = 0; i < iters; ++i) {
    lbuf[tid] = acc; barrier(0); acc += lbuf[(tid + 1) & (BWG - 1)];
  }
#else
  for (uint i = 0; i < iters; ++i) {
    acc = acc * 1664525u + 1013904223u; barrier(0);
  }
#endif
  if (acc == 0xdeadbeefu) out[0] = acc;
}

// ---------------------------------------------------------------------------
// G. SLM read bandwidth: replay a co-resident __local buffer (the on-die L1
//    scratchpad) to get its aggregate bandwidth. SLM_KB controls the buffer.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void slm_bw(__global uint *out, const uint iters) {
  __local uint buf[SLM_KB * 256];
  const int tid = get_local_id(0);
  for (uint i = tid; i < SLM_KB * 256; i += 64) buf[i] = i + 1u;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  const uint m = SLM_KB * 256;
  for (uint i = 0; i < iters; ++i) {
    uint b = (i * 64 + tid) % m;
    a0 += buf[b];
    a1 += buf[(b + 64) % m];
    a2 += buf[(b + 128) % m];
    a3 += buf[(b + 192) % m];
  }
  out[get_global_id(0)] = a0 + a1 + a2 + a3;
}


// SLM read bandwidth with 16-byte vector loads (uint4), matching how the GEMM
// inner loop reads the B tile. Reveals the vector-fed SLM ceiling.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void slm_bw_v4(__global uint *out, const uint iters) {
  __local uint4 buf[SLM_KB * 64];
  const int tid = get_local_id(0);
  for (uint i = tid; i < SLM_KB * 64; i += 64) buf[i] = (uint4)(i, i + 1, i + 2, i + 3);
  barrier(CLK_LOCAL_MEM_FENCE);
  uint a = 0;
  const uint m = SLM_KB * 64;
  for (uint i = 0; i < iters; ++i) {
    uint4 v = buf[(i * 64 + tid) % m];
    a += v.s0 + v.s1 + v.s2 + v.s3;
  }
  out[get_global_id(0)] = a;
}

// --- Bank-conflict controlled probes ---------------------------------------
// MODE 0: scalar uint, consecutive (thread t reads word t)   -> conflict-free
// MODE 1: scalar uint, stride 32 words (all threads same bank) -> 32-way conflict
// MODE 2: scalar uint, all threads read the same word (broadcast) -> no traffic
// MODE 3: uint4, thread t reads uint4 t (16B/thread, 4 words/thread)
// MODE 4: uint4, thread t reads uint4 (t*32) (stride across banks)
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void slm_conf(__global uint *out, const uint iters) {
  __local uint buf[SLM_KB * 256];
  const int tid = get_local_id(0);
  for (uint i = tid; i < SLM_KB * 256; i += 64) buf[i] = i + 1u;
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint m = SLM_KB * 256;
  uint acc = 0;
#if MODE == 0
  for (uint i = 0; i < iters; ++i) acc += buf[(i * 64 + tid) & (m - 1)];
#elif MODE == 1
  for (uint i = 0; i < iters; ++i) acc += buf[((i * 32) + (tid & 31)) & (m - 1)];
#elif MODE == 2
  for (uint i = 0; i < iters; ++i) acc += buf[tid & 31];
#elif MODE == 3
  __local uint4 *b4 = (__local uint4 *)buf;
  for (uint i = 0; i < iters; ++i) {
    uint4 v = b4[(i * 64 + tid) & ((m / 4) - 1)];
    acc += v.s0 + v.s1 + v.s2 + v.s3;
  }
#elif MODE == 4
  __local uint4 *b4 = (__local uint4 *)buf;
  for (uint i = 0; i < iters; ++i) {
    uint4 v = b4[(i * 64 + ((tid * 8) & ((m / 4) - 1))) & ((m / 4) - 1)];
    acc += v.s0 + v.s1 + v.s2 + v.s3;
  }
#endif
  out[get_global_id(0)] = acc;
}

