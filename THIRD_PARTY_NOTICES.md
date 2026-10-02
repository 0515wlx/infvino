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
