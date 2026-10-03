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

宿主需内核 `i915` + `/dev/dri/renderD*`；容器只需挂载 render 节点
（`--device=/dev/dri/renderD128`，**不需要 `--privileged`**，见 `benchmark_protocol.md`）。

下表为编写文档时本机验证过的驱动组件版本，仅作参考；镜像构建时交由 Intel 官方
apt 源解析，不在仓库里硬编码：

| 组件 | 版本 |
|---|---|
| Intel Graphics Compiler (IGC) | `1.0.17537.20` |
| Level-Zero GPU | `1.3.30872.22` |
| OpenCL ICD | `24.35.30872.22` |
| libigdgmm12 | `22.5.0` |

参考安装步骤见仓库自带的 [`docker/Dockerfile`](../docker/Dockerfile)（infvino 自己维护，不依赖任何团队镜像）。

## 3. 开发/测试镜像

测试脚本默认使用 `infvino-dev:latest`，直接由本仓库的
[`docker/Dockerfile`](../docker/Dockerfile) 构建：

```bash
docker build -f docker/Dockerfile -t infvino-dev:latest .
```

镜像内含 OpenCV / OpenCL 头文件与 loader / yaml-cpp / cmake / g++ 以及 Intel GPU
计算运行时（OpenCL + Level-Zero，版本由 Intel 官方 apt 源解析，不硬编码）。
可用 `--image <your-image>` 指向自备镜像，只要满足第 1 节依赖即可。

## 4. Python 工具依赖（仅开发机，不入运行时）

依赖清单固定在仓库根目录的 requirements 文件中（版本已锁定）：

| 文件 | 用途 |
|---|---|
| [`requirements-dev.txt`](../requirements-dev.txt) | 数值检验 / 计划生成 / 分析工具（numpy、onnx、onnxruntime、opencv-python、PyYAML、packaging） |
| [`requirements-export.txt`](../requirements-export.txt) | 仅 `export_models.py` 需要的重型依赖（ultralytics、torch、torchvision） |
| [`requirements-ov.txt`](../requirements-ov.txt) | OpenVINO 对照基线（版本由脚本动态解析，见下） |

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt          # 日常开发
pip install -r requirements-export.txt       # 仅导出模型时需要
```

## 4b. OpenVINO 对照基线（可选）

OpenVINO **不是** infvino 的依赖，仅用于性能/数值对照。对照时一律通过
`scripts/openvino_baseline.py` **动态路由到 OpenVINO 最新稳定版**，不在代码或文档里
硬编码版本号：

```bash
python3 scripts/openvino_baseline.py resolve            # 打印当前最新稳定版
python3 scripts/openvino_baseline.py run --device GPU   # 隔离 venv 安装并跑基线
python3 scripts/openvino_baseline.py run --version X.Y.Z  # 仅用于复现历史基线
```

解析结果写入 `build-ct/openvino-baseline.json`。

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
