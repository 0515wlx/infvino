# Third-party notices

infvino is distributed under the project's own license. It **vendors** (embeds) a
small amount of third-party OpenCL source code, listed below. No third-party
library is linked or required at build or run time — only source code was adapted.

---

## OpenVINO (Intel Corporation)

- **Upstream project**: https://github.com/openvinotoolkit/openvino
- **Component used**: Intel GPU plugin kernel selector —
  `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_os_iyx_osv32.cl`
  (kernel `convolution_gpu_bfyx_os_iyx_osv32`).
- **License**: Apache License 2.0. Full text: [`third_party/openvino/LICENSE`](third_party/openvino/LICENSE).
- **Files in infvino derived from it**:
  - `kernels/conv_ov.cl` — a self-contained adaptation of the data path of the
    upstream kernel (sub-group lanes = output channels, register input block +
    `sub_group_broadcast`, `intel_sub_group_block_read_us2` weights, no SLM).
    The upstream `#include` helper headers and JIT macro layer were replaced by
    local definitions; the algorithm and memory access pattern are preserved.
  - The host-side OSV weight swizzle in `src/PlanModel.cpp` (`PlanModel::ovWeight`)
    and `src/tools/kernel_bench.cpp` / `src/tools/kernel_numtest.cpp` reproduce the
    `os_iyx_osv32` weight layout described by OpenVINO's
    `GET_FILTER_OS_IYX_OSV_INDEX` (see
    `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/include/batch_headers/fetch_weights.cl`).
  - `kernels/conv_blk.cl` — a self-contained adaptation of the data path of the
    upstream `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_f16.cl`
    (kernel `convolution_gpu_bfyx_f16`, selector `ConvolutionKernel_b_fs_yx_fsv16`):
    sub-group lanes = 16 output channels, blocked `b_fs_yx_fsv16` input,
    `os_is_yx_isv16_osv16` weights via `intel_sub_group_block_read_us8`, per-input
    vector `mad` over `OUTPUT_X_BLOCK_SIZE` columns, plain `bfyx` output. Upstream
    `#include` helper headers and JIT macro layer replaced by local definitions;
    also contains an original `reorder_bfyx_to_fsv16` helper.
  - The host-side blocked weight repack (`PlanModel::blkWeight`) and input reorder
    (`PlanModel::blkInput`) reproduce OpenVINO's `os_is_yx_isv16_osv16` /
    `b_fs_yx_fsv16` layouts.
  - `kernels/conv1x1_blk.cl` — a self-contained adaptation of the data path of the
    upstream `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_f16_1x1.cl`
    (kernel `convolution_gpu_bfyx_f16_1x1`, selector
    `ConvolutionKernel_b_fs_yx_fsv16_1x1`): sub-group lanes = 16 output channels,
    blocked `b_fs_yx_fsv16` input/output, `os_is_yx_isv16_osv16` (1x1) weights via
    `intel_sub_group_block_read_us8`, per-input vector `mad` over `X_BLOCK`
    columns, optional `SLM_DIV_FACTOR` split-K. Upstream `#include` helper headers
    and JIT macro layer replaced by local definitions; the `OUT_FSV16`
    persistent-layout output, the `RES` residual epilogue and the infvino
    activation codes are original. Host-side weight repack/verify in
    `src/tools/kernel_bench.cpp` (`benchConv1x1Blk`).
  - `kernels/depthwise_blk.cl` — a self-contained adaptation of the data path of the
    upstream `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_f16_depthwise.cl`
    (kernel `convolution_gpu_bfyx_f16_depthwise`): sub-group lanes = 16 output
    channels, blocked `b_fs_yx_fsv16` input/output, `[C/16][K][K][16]` weights via
    `intel_sub_group_block_read_us8`, per-row register line reused across the K
    taps, optional split-K. Host-side weight repack in `PlanModel::blkDwWeight`,
    bench in `benchDepthwiseBlk`.
- **Copyright notice retained**: `Copyright (C) 2018-2026 Intel Corporation`.
- **Modifications**: reduced to a single self-contained `.cl` file; added an
  infvino-specific fused activation/residual epilogue and a `RES` toggle; the
  surrounding dispatch/weight-reorder is original infvino code.

### Apache-2.0 compliance checklist

- [x] The upstream license text is included verbatim
      (`third_party/openvino/LICENSE`).
- [x] Each derived file carries the upstream copyright line and an
      `SPDX-License-Identifier: Apache-2.0` tag, plus a comment explaining what
      was changed (see the header of `kernels/conv_ov.cl`).
- [x] No upstream `NOTICE` file content is required to be propagated for this
      component (the OpenVINO source tree does not ship a NOTICE file that applies
      to these kernel files; attribution is provided here and in the file headers).
- [x] No OpenVINO code is linked as a library, so there is no binary redistribution
      obligation beyond the source attribution above.
