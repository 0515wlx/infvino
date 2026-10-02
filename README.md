# infvino

**独立的自研推理后端**：用自研 OpenCL kernel（Intel iGPU）跑完整网络，**不依赖 OpenVINO / ROS**。
面向 ultralytics YOLO 系列与轻量分类网络，当前支持：

| 模型 | 任务 | 输入 |
|---|---|---|
| `yolov8n-pose` | 姿态（17 关键点） | 3×640×640 |
| `yolo11n-pose` | 姿态（17 关键点） | 3×640×640 |
| `mobilenetv3-small` | 分类（ImageNet 1000） | 3×224×224 |

> 定位：与团队项目解耦的推理后端。可通过 `onnx2plan.py` + 算子 kernel 继续扩展支持其他模型
> （detect / classify / pose 解码已具备，seg / obb 预留）。

## 特性

- **纯 C++ 库**：只依赖 OpenCV + OpenCL + yaml-cpp，**不依赖 ROS / OpenVINO**。
- **自研 kernel 执行**：ONNX → 执行计划(plan) → OpenCL kernel（GEMM / conv / 通用算子），
  权重 fp16 常驻设备，端到端数值对齐 onnxruntime。
- **后端无关的解码层**：detect / pose / end-to-end / classify 解码与后端解耦，
  输出统一为 `Tensor`（f32 主机内存）。
- **可扩展**：新增算子/模型只改 `PlanModel` 与 `onnx2plan.py`，上层 API 不变。

## 目录结构

```
include/infvino/
├── Types.hpp / Tensor.hpp / ModelInfo.hpp     # 结果类型 / 张量 / 模型元信息
├── Preprocess.hpp / Nms.hpp                   # letterbox 预处理 / NMS
├── Decoder.hpp / Decoders.hpp                 # 解码抽象与实现
├── ClRuntime.hpp / Tiles.hpp / Half.hpp       # OpenCL 运行时 / tile 配置 / fp16
├── PlanModel.hpp                              # 计划驱动的整网执行器（自研 kernel）
├── ClBackend.hpp                              # 推理后端（PlanModel 封装）
└── InferenceEngine.hpp                        # 门面 + Session
src/                                           # 对应实现 + tools/
kernels/*.cl                                   # 自研 OpenCL kernel 源码
scripts/                                       # 模型导出 / 计划生成 / 数值检验
config/models.yaml                             # 模型注册表
docs/                                          # 架构、基准、kernel 优化日志
```

## 依赖

| 依赖 | 版本 | 说明 |
|---|---|---|
| C++ | C++17 | |
| CMake | ≥ 3.16 | |
| OpenCV | 4.x（core/imgproc/dnn） | 预处理 |
| OpenCL | 1.2+（`ocl-icd-opencl-dev` + `opencl-headers`） | 自研 kernel |
| yaml-cpp | 系统版 | 模型配置 |
| Intel Compute Runtime / IGC | 见 `docs/dependencies.md` | iGPU 驱动栈 |

> 开发/测试使用的容器镜像：`infvino-dev`（含 OpenCV / OpenCL dev / yaml-cpp / cmake）。
> 不需要 OpenVINO。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 准备模型

```bash
# 1) 导出 ONNX（ultralytics 模型；需 ultralytics + onnx）
python3 scripts/export_models.py --out models

# 2) ONNX -> 自研 kernel 执行计划（每个模型一个目录，含 plan + 权重 .bin）
python3 scripts/onnx2plan.py --onnx models/yolov8n-pose.onnx --out-dir models/yolov8n-pose
python3 scripts/onnx2plan.py --onnx models/yolo11n-pose.onnx --out-dir models/yolo11n-pose
python3 scripts/onnx2plan.py --onnx models/mobilenetv3-small.onnx --out-dir models/mobilenetv3-small

# 3) 在 config/models.yaml 中登记 path / plan
```

## 使用

```cpp
#include "infvino/InferenceEngine.hpp"

infvino::ModelInfo info = infvino::ModelInfo::fromYaml("config/models.yaml", "yolov8n-pose");
infvino::InferenceEngine engine(info);        // 自动选择 GPU 设备
auto result = engine.infer(bgr_image);        // 或 engine.createSession() 每线程/每相机一个

for (const auto & d : result.detections)
  std::cout << d.class_id << " " << d.score << " " << d.box
            << " kpts=" << d.keypoints.size() << "\n";
```

命令行基准 / 调试：

```bash
./build/infvino_bench --config config/models.yaml --key yolov8n-pose --iters 50
./build/kernel_run   --plan models/yolov8n-pose/model.plan --report --iters 3
```

## 测试

所有测试都需要可用的 Intel iGPU（`/dev/dri`）与容器镜像（脚本用 docker 跑容器）：

```bash
# 算子级数值检验（vs numpy FP32）
python3 scripts/kernel_check.py   --repo $PWD --image infvino-dev:latest

# 整网端到端数值 + ops/EU/cyc（vs onnxruntime）
python3 scripts/model_check.py    --model yolov8n-pose --repo $PWD --image infvino-dev:latest

# 库后端数值检验（ClBackend，三个模型）
python3 scripts/numerical_check.py --repo $PWD --image infvino-dev:latest

# 引擎级检验（预处理 + ClBackend，喂图片）
python3 scripts/engine_check.py   --repo $PWD --image infvino-dev:latest
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 分层设计、API、执行计划、扩展点 |
| [`docs/xe-lp-isa.md`](docs/xe-lp-isa.md) | **Xe-LP(Gen12) ISA 逆向**：cache 层级与 EU 寄存器全貌 |
| [`docs/kernel.md`](docs/kernel.md) | 自研 kernel 优化日志与 ops/EU/cyc |
| [`docs/benchmark.md`](docs/benchmark.md) | 整网数值/性能基准与复现 |
| [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) | **GPU 基准安全协议**（防止开发板死机）|
| [`docs/dependencies.md`](docs/dependencies.md) | 依赖与版本清单 |
| [`docs/round22-status.md`](docs/round22-status.md) | **R22 现状分析**：1×1 kernel / OV conv3×3 / 融合 |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | 第三方（OpenVINO）代码归属与 Apache-2.0 合规 |

> **kernel 效率结论（R18–R21）**：本机（Iris Xe 80EU / 128 GRF / 无通用 L1）上，
> **direct conv、OpenVINO 式（lane=通道 + broadcast）、Winograd 三条数据通路的
> ops/EU/cyc 上限均为 ~16**（纯寄存器 FMA 结构上限 27.4–29.6，理论峰值 32）——
> 这是不换硬件能力时卷积复用的现实天花板。生产路径 direct conv 大层 ~10.3，
> 网格饥饿层经自适应分块 +20–56%。详见 [`docs/kernel.md`](docs/kernel.md)。
>
> **R22 更新**：把 OpenVINO `os_iyx_osv32` 的**真实数据通路**（lane=通道 +
> `intel_sub_group_block_read` 权重 + OSV swizzle，见 `kernels/conv_ov.cl`）移植进来后，
> conv3×3 大层从 ~10.3 提到 **12.6–13.6 ops/EU/cyc（+20–80%）**，全形状优于原 native
> direct conv；1×1 的 N=1 层改走 split-K GEMV（+5–27×）。三模型 kernel busy
> 1.10–1.81×、墙钟 1.08–1.57×（mobilenet 最大）。Apache-2.0 归属见
> [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 状态

- [x] OpenCL 运行时 + 设备探测 + kernel 构建缓存
- [x] 计划驱动整网执行（conv / gemm / 通用算子 / 融合）
- [x] 预处理 / NMS / detect+pose+classify 解码
- [x] 三个目标模型端到端数值对齐 onnxruntime
- [x] 算子级 + 整网 + 库后端三级数值检验
- [ ] 算子融合、内存复用、降低 launch 开销（整网墙钟）
- [ ] seg / obb 解码；多 Session 并行缓冲
- [ ] 支持更多模型（detect 系列、其他 backbone）
