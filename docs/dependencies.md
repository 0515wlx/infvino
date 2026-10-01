# 依赖与版本清单

> infvino 是独立项目，**不依赖 OpenVINO / ROS**。任何版本变更都要更新本文件。

## 1. 运行时依赖（C++ 库）

| 依赖 | 说明 | 来源 |
|---|---|---|
| C++ | C++17 | 本工程 `CMakeLists.txt` |
| CMake | ≥ 3.16 | 构建 |
| OpenCV | 4.x（`core` / `imgproc` / `dnn`） | `libopencv-dev` |
| OpenCL | 1.2+（loader + 头文件） | `ocl-icd-opencl-dev`、`opencl-headers` |
| yaml-cpp | 系统版 | `libyaml-cpp-dev` |

- CMake：`find_package(OpenCV REQUIRED)`、`find_package(OpenCL REQUIRED)`、`find_package(yaml-cpp REQUIRED)`，
  无需硬编码路径。
- `.cl` kernel 源码默认从源码树 `kernels/` 运行时加载（编译期注入 `INFVINO_KERNEL_DIR`）；
  部署可用 `-DINFVINO_KERNEL_DIR_OVERRIDE=/path/to/kernels` 覆盖。

## 2. Intel iGPU 驱动栈（OpenCL 运行必需）

宿主需内核 `i915` + `/dev/dri/renderD*`；容器需 `--device=/dev/dri:/dev/dri --privileged`。
驱动组件（与团队 develop 镜像一致的版本）：

| 组件 | 版本 |
|---|---|
| Intel Graphics Compiler (IGC) | `1.0.17537.20` |
| Level-Zero GPU | `1.3.30872.22` |
| OpenCL ICD | `24.35.30872.22` |
| libigdgmm12 | `22.5.0` |

参考安装步骤见团队 `docker/Dockerfile-develop`（infvino 不复制该文件，仅依赖其产出的镜像）。

## 3. 开发/测试镜像

脚本默认使用 `infvino-dev:latest`（在 develop 镜像上补装 `ocl-icd-opencl-dev` +
`opencl-headers`），其中已含 OpenCV / yaml-cpp / cmake / OpenCL 头文件。
可用 `--image <your-image>` 指向自备镜像，只要满足第 1 节依赖即可。

## 4. Python 工具依赖（仅开发机，不入运行时）

| 工具 | 依赖 |
|---|---|
| `scripts/export_models.py` | `ultralytics`、`torch/torchvision`、`onnx` |
| `scripts/onnx2plan.py` | `onnx`、`numpy` |
| `scripts/numerical_check.py` / `model_check.py` / `engine_check.py` | `onnxruntime`、`numpy`、`onnx`（`engine_check` 另需 `opencv-python`、`pyyaml`） |
| `scripts/kernel_check.py` / `analyze_conv.py` | `numpy`、`onnx` |

## 5. 环境自检

```bash
# 宿主
clinfo -l                       # 期望出现 Intel(R) OpenCL Graphics / Iris Xe
ls /dev/dri                     # renderD128

# 容器内
cmake --version && g++ --version
ls /usr/include/CL/cl.h
pkg-config --modversion opencv4 # 若装了 pkg-config
```

## 6. 评审清单

- [ ] 是否仍无 OpenVINO / ROS 依赖？
- [ ] OpenCL 头文件/loader 是否满足？
- [ ] 新增算子是否更新了 `onnx2plan.py` 与文档？
- [ ] 本文件是否与实际 `CMakeLists.txt` 一致？
