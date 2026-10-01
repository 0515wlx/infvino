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

__attribute__((reqd_work_group_size(64, 1, 1)))
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

__attribute__((reqd_work_group_size(64, 1, 1)))
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
__attribute__((reqd_work_group_size(64, 1, 1)))
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

__attribute__((reqd_work_group_size(64, 1, 1)))
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

__attribute__((reqd_work_group_size(64, 1, 1)))
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

// A dependent integer add chain (cheap ALU, same loop shape) to gauge the
// per-iteration loop overhead; subtract from depth=1 to get true ALU latency.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void add_lat(__global int *out, const int a) {
  int acc = get_global_id(0);
  for (int i = 0; i < ITERS; ++i) acc = acc + a;
  if (acc == 0x7fffffff) out[0] = acc;
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
