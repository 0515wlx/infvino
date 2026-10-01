// Memory bandwidth probes for roofline analysis (Round 2 decision).
//   copy : out[i] = in[i]              -> read + write
//   read : reduction sum of in[i]      -> read only
__kernel void copy_u32(__global const uint *restrict in, __global uint *restrict out,
                       const uint n) {
  const uint i = get_global_id(0);
  if (i < n) out[i] = in[i];
}

__kernel void readonly_u32(__global const uint *restrict in, __global uint *restrict sink,
                           const uint n) {
  const uint i = get_global_id(0);
  uint acc = 0;
  if (i < n) acc = in[i];
  if (acc == 0xdeadbeefu) sink[0] = acc;  // never true; keeps the read alive
}
