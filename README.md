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
| [`docs/kernel.md`](docs/kernel.md) | 自研 kernel 优化日志与 ops/EU/cyc |
| [`docs/benchmark.md`](docs/benchmark.md) | 整网数值/性能基准与复现 |
| [`docs/dependencies.md`](docs/dependencies.md) | 依赖与版本清单 |

## 状态

- [x] OpenCL 运行时 + 设备探测 + kernel 构建缓存
- [x] 计划驱动整网执行（conv / gemm / 通用算子 / 融合）
- [x] 预处理 / NMS / detect+pose+classify 解码
- [x] 三个目标模型端到端数值对齐 onnxruntime
- [x] 算子级 + 整网 + 库后端三级数值检验
- [ ] 算子融合、内存复用、降低 launch 开销（整网墙钟）
- [ ] seg / obb 解码；多 Session 并行缓冲
- [ ] 支持更多模型（detect 系列、其他 backbone）
