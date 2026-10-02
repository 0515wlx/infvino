// R30c: read-vs-write asymmetry (DRAM).  Write probe stores; read probe reads.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// write-only: y[i] = const  (traffic = 4 B/elem write)
__kernel void wr_only(__global float *restrict y, const uint n, const float c) {
  const uint i = get_global_id(0);
  if (i < n) y[i] = c;
}

// read-only with L2/L3-bypass-friendly dependent chain (traffic = 4 B/elem read)
__kernel void rd_only(__global const float *restrict x, __global float *restrict sink,
                      const uint n) {
  const uint i = get_global_id(0);
  float acc = 0.0f;
  if (i < n) acc = x[i];
  if (acc == -12345.0f) sink[0] = acc;   // never true
}

// read-heavy / write-light: sum C consecutive inputs into one output (contraction).
// traffic = C*4 read + 4 write -> R:W = C:1.
__kernel void rd_sum(__global const float *restrict x, __global float *restrict y,
                     const uint nout, const uint C) {
  const uint o = get_global_id(0);
  if (o >= nout) return;
  float s = 0.0f;
  for (uint j = 0; j < C; ++j) s += x[o * C + j];
  y[o] = s;
}
