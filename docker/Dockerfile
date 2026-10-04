# infvino 开发/测试镜像
#
# 只包含 infvino 自身构建、测试与在 Intel iGPU 上运行所需的依赖，不依赖任何团队/
# 私有镜像或项目。构建后即为默认测试镜像 `infvino-dev:latest`。
# 自带 C++ 工具链 + Intel GPU runtime + Python 测试工具（requirements-dev.txt），
# 构建与测试可完全在容器内完成。
#
# 构建：
#   docker build -f docker/Dockerfile -t infvino-dev:latest .
# 运行（务必遵守 docs/benchmark_protocol.md 的资源限制规则）：
#   docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
#     --device=/dev/dri/renderD128 \
#     -v "$PWD":/workspace/infvino -w /workspace/infvino \
#     infvino-dev:latest bash -lc '<cmd>'

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# --- 构建与运行依赖（版本由发行版解析，不硬编码）---
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git \
      build-essential cmake pkg-config \
      libopencv-dev \
      ocl-icd-opencl-dev opencl-headers ocl-icd-libopencl1 \
      libyaml-cpp-dev \
      python3 python3-pip python3-venv \
      clinfo \
    && rm -rf /var/lib/apt/lists/*

# --- Intel GPU 计算运行时（OpenCL + Level-Zero）---
# 使用 Intel 官方 graphics apt 源；具体版本交给源解析，不在仓库里硬编码。
RUN curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key \
      | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy unified" \
      > /etc/apt/sources.list.d/intel-graphics.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      intel-opencl-icd libigc1 libigdfcl1 intel-level-zero-gpu level-zero \
 && rm -rf /var/lib/apt/lists/*

# --- Python 开发/测试工具 ---
# 让镜像自带完整测试环境：数值检验 / 计划生成 / 分析脚本（numpy、onnx、
# onnxruntime、opencv-python、PyYAML、packaging）可直接在容器里跑。版本锁定在
# requirements-dev.txt（见 docs/dependencies.md §4）。重型导出依赖
# （ultralytics/torch，requirements-export.txt）与 OpenVINO 对照基线
# （requirements-ov.txt，脚本动态建隔离 venv）刻意不装。
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r /tmp/requirements-dev.txt \
 && rm -f /tmp/requirements-dev.txt

# 与 CMakeLists.txt 一致，显式声明 OpenCL 目标版本。
ENV CL_TARGET_OPENCL_VERSION=300

WORKDIR /workspace/infvino
