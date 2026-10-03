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
scripts/                                       # 模型导出 / 计划生成 / 数值检验 / 自动调优
config/models.yaml                             # 模型注册表
config/tuning.json                             # 自动调优缓存（按设备/op/shape）
docs/                                          # 架构、基准、kernel 优化日志、自动调优
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

# 跨推理一致性（同进程 A→B vs 全新进程 B；抓缓存陈旧 bug）
python3 scripts/reuse_check.py    --model yolov8n-pose --repo $PWD --image infvino-dev:latest
```

> **测试约定（P0 起）**：整网数值检验都跑**两份不同输入**（`--input/--input2`、`--image/--image2`），
> 输出取**第二帧**。这样「把每帧都会变的激活错误地缓存成只算一次」这类 bug 会被立刻抓到——
> 单输入重复跑的测试发现不了（曾漏掉 `blkInput` 跨推理陈旧 bug，见 `docs/kernel.md`）。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 分层设计、API、执行计划、扩展点 |
| [`docs/autotuning.md`](docs/autotuning.md) | **自动调优体系（含 JIT 设计）**：TuningCache / 候选枚举 / 中间标准 expected_ops |
| [`docs/xe-lp-isa.md`](docs/xe-lp-isa.md) | **Xe-LP(Gen12) ISA 逆向**：cache 层级与 EU 寄存器全貌 |
| [`docs/kernel.md`](docs/kernel.md) | 自研 kernel 优化日志与 ops/EU/cyc |
| [`docs/benchmark.md`](docs/benchmark.md) | 整网数值/性能基准与复现 |
| [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) | **GPU 基准安全协议**（防止开发板死机）|
| [`docs/dependencies.md`](docs/dependencies.md) | 依赖与版本清单 |
| [`docs/round22-status.md`](docs/round22-status.md) | **R22–R23 现状分析**：1×1 kernel / OV conv3×3 / 融合 / 与 OV 对照 |
| [`docs/round24-analysis.md`](docs/round24-analysis.md) | **R24 conv3×3 逐 size 瓶颈分析**：ISA 配额证据 / 中间标准修正 / 两通路接入 |
| [`docs/round25-ovblocked.md`](docs/round25-ovblocked.md) | **R25 OV 阻塞式 conv 完整移植**：逐 size 对照 / 第三条 autotune 通路 |
| [`docs/round30-smallops.md`](docs/round30-smallops.md) | **R30 剩余 kernel（非 conv/gemm）的物理模型**：内存 roofline / ISA 配额 / 哪堵墙 |
| [`docs/openvino-gap-analysis.md`](docs/openvino-gap-analysis.md) | **infvino vs OpenVINO GPU 差距分析**：逐维对标 / 强项 / 学习清单（P0–P3） |
| [`docs/memory-reuse-design.md`](docs/memory-reuse-design.md) | **P0 激活内存池设计 + R-P0 实测**：生存期复用 / 视图并集 / 墙钟收益 |
| [`docs/register-model.md`](docs/register-model.md) | **7 线程 EU 寄存器限制的完整模型**：tile/ops 天花板推导 + 使用清单 |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | 第三方（OpenVINO）代码归属与 Apache-2.0 合规 |

> **kernel 效率结论（R18–R21，已被 R24 部分更正）**：本机（Iris Xe 80EU / 128 GRF / 无通用 L1）上，
> **direct conv、OpenVINO 式（lane=通道 + broadcast）、Winograd 三条数据通路**
> 曾被判定 ops/EU/cyc 上限均为 ~16（纯寄存器 FMA 结构上限 27.4–29.6，理论峰值 32）。
> **R24 用 ISA 反汇编更正**：移植路径的 `sub_group_broadcast` 被折进 `mad`，指令配额上限
> 实为 **~20.3**（见下方 R24 更新）。这仍是「不换硬件能力时的现实天花板」的一个更准确版本。
> 生产路径 direct conv 大层 ~10.3，网格饥饿层经自适应分块 +20–56%。
> 详见 [`docs/kernel.md`](docs/kernel.md)。
>
> **R22 更新**：把 OpenVINO `os_iyx_osv32` 的**真实数据通路**（lane=通道 +
> `intel_sub_group_block_read` 权重 + OSV swizzle，见 `kernels/conv_ov.cl`）移植进来后，
> conv3×3 大层从 ~10.3 提到 **12.6–13.6 ops/EU/cyc（+20–80%）**，全形状优于原 native
> direct conv；1×1 的 N=1 层改走 split-K GEMV（+5–27×）。三模型 kernel busy
> 1.10–1.81×、墙钟 1.08–1.57×（mobilenet 最大）。Apache-2.0 归属见
> [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
>
> **R23 更新**：`concat4` 改 3-D 网格（纯索引简化）+ 主机侧 kernel 缓存 →
> yolov8n busy **19.9→17.2 ms**、yolo11n **23.0→20.1 ms**（数值精确，三模型 PASS）。
> 与 OpenVINO 2025.2 对照，整网仍差 ~1.5–1.9×，差距集中在 conv3×3 网格饥饿
> （`os_iyx_osv32` 已到 ~8–13 ops）；阻塞式 conv 移植未打赢 osv32（负结果）。
> 详见 [`docs/kernel.md`](docs/kernel.md) Round 23 与
> [`docs/round22-status.md`](docs/round22-status.md)。
>
> **R24 更正（重要）**：用离线 `ocloc` 反汇编 `conv_ov.cl` 得到决定性证据——
> 主内循环 **288 packed mad / 453 指令 = 63.6% mad**，且 `sub_group_broadcast` **已被
> IGC 折进 `mad` 操作数**（没有独立广播指令）。所以本移植 kernel 的指令配额上限 ≈ **20.3**，
> **不是 R18/R20 的 ~16，也不是实测的 ~10–13**。实测 80×80 仅为配额的 68%、40×40 仅 42%，
> 缺口是**延迟/流水/占用**而非指令数。据此把中间标准 `expected_ops` 改为
> `32×0.636×prologue_amort×grid_factor`（上界 20.3），并让 **OV 与 native 两条通路在
> 全部 shape（含 stride=2）上都是 autotune 候选**。第一个 ILP 变体（`-DUK` 双累加集）
> 实测更慢并回退。逐 size 分析与下一步（split-K / 内部块 fast path / OSV64）见
> [`docs/round24-analysis.md`](docs/round24-analysis.md)。
>
> **R25 更新**：**完整移植了 OpenVINO 的阻塞式 conv**（`convolution_gpu_bfyx_f16`，
> lane=输出通道 + `b_fs_yx_fsv16` 输入 + `os_is_yx_isv16_osv16` 权重 + 向量 `mad`），
> 作为 `kernels/conv_blk.cl` + 第三条 autotune 候选（OBW=2/4/8，含输入/权重重排）。
> 逐 size 对照：blk 赢在 **s1 小空间/大通道**（20×20 系 +12–44%、40×40 C128 +16%），
> 输在 **stride-2**（−30–47%）与 **80×80 大层**（−16%），因此按 size 选而非替换。
> 整网强制全 blk 也 **−0.9%**（17.35→17.19 ms），强制 blk 与 default 均 vs onnxruntime PASS。
> **回答「能否到理论极限」：不能**——blk 指令配额 ≈17（实测达 72%）、osv32 ≈20（达 68%），
> 两条 OV 通路的现实天花板都在 **~12–14 ops**，差距是延迟/占用而非指令数。
> 详见 [`docs/round25-ovblocked.md`](docs/round25-ovblocked.md)。
>
> **R26（逐层重扫）**：用新候选集（OV 块 + blk OBW2/4/8 + native）重扫 31 个 conv3×3
> 签名，**13 条改选 blk**（其余 13 OV / 5 native），最大单层 **+87%**（20×20 s1 256→64）。
> 整网 kernel busy 同会话 A/B：**yolov8n 15.19→14.40 ms（−5.2%）**、
> **yolo11n 17.96→17.04 ms（−5.1%）**；`model_check` 三模型 **ALL PASS**。
> 纯配置收益（不改 kernel），来源就是让 OV 阻塞式 conv 按 size 可选。详见
> [`docs/round25-ovblocked.md`](docs/round25-ovblocked.md) §7。
>
> **自动调优（P0）**：引入分层自动调优体系（[`docs/autotuning.md`](docs/autotuning.md)）：
> `OpSignature` + `TuningCache`（按设备/op/shape）+ 候选枚举 + **中间标准 `expected_ops`**。
> 不改任何 kernel 源码，仅靠自动选择配置，yolov8n conv3×3 分项 **10.98→9.71 ms（−11.6%）**、
> 整网 busy **17.35→16.07 ms（−7.4%）**，输出**逐位不变**；中间标准自动定位出
> 「离物理极限最远」的层（如 `320×320 s2 Cin3 Cout16` ratio 0.17）。调优表见
> [`config/tuning.json`](config/tuning.json)。（R24 起中间标准按上面的 ISA 配额修正，
> ratio 数值随之更新。）
>
> **R-P0（激活内存池，结构改动）**：对齐 OpenVINO `memory_pool`——按**生存期复用**激活
> buffer（尺寸桶 + 冲突集 + reshape/flatten 视图并集），静态分配（`parse` 期一次，`run` 期
> 零开销）。不再「每个 tensor 一块常驻 `cl_mem`」：yolov8n 67.7→**26.1 MB**、
> yolo11n 76.8→**28.1 MB**、mobilenet 4.3→**0.7 MB**。同会话 A/B：
> busy y8 **−3.4%** / y11 **−4.9%** / mb **−4.3%**，墙钟 y8 **−5.2%** / y11 **−5.8%**；
> 三模型数值**逐位一致**。**新发现：内存池对墙钟的收益 > GPU busy**——墙钟里「busy 之外」
> 的开销部分随**常驻 `cl_mem` 数**增长，不只是 per-dispatch 固定成本。
> 见 [`docs/memory-reuse-design.md`](docs/memory-reuse-design.md) 与
> [`docs/openvino-gap-analysis.md`](docs/openvino-gap-analysis.md)。
>
> **R-P0b（连续 `copy_c` 别名 + 跨推理陈旧 bug 修复）**：8 条 `Split_output_1` 用
> `clCreateSubBuffer` 直接别名父张量的连续通道段（零 launch，消费者零改动）；busy 再
> −1~1.5%（y8 13.07→**12.98**、y11 13.87→**13.80**、mb 2.72→**2.66**）。
> 顺带**修掉一个跨推理陈旧 bug**：`blkInput`（conv_blk 的输入重排）按张量名缓存且只重排
> 一次 → 第 2 帧及以后用上一帧数据（单输入重复跑的测试发现不了）。修复后新增
> `reuse_check` 护栏：同进程 A→B 与全新进程 B 的输出必须一致。

## 状态

- [x] OpenCL 运行时 + 设备探测 + kernel 构建缓存
- [x] 计划驱动整网执行（conv / gemm / 通用算子 / 融合）
- [x] 预处理 / NMS / detect+pose+classify 解码
- [x] 三个目标模型端到端数值对齐 onnxruntime
- [x] 算子级 + 整网 + 库后端三级数值检验
- [x] concat 3-D 网格 + 主机侧 kernel 缓存（R23）
- [x] 分层自动调优体系（P0）：TuningCache + 候选枚举 + 中间标准 expected_ops + `kernel_autotune`
- [x] conv3×3 阻塞式 kernel（R25 完整移植 OV `convolution_gpu_bfyx_f16`）+ 接入 autotune 候选；两条 OV 通路现实上限 ~12–14 ops
- [x] conv3×3 逐层 autotune 重扫（blk 候选 + R24 中间标准）：yolov8n/yolo11n −5%；osv32 大层 ~8–13、blk 小层 +13–87%
- [x] 剩余 kernel（非 conv/gemm）物理模型：内存 roofline + ISA 配额；concat4 已到 DRAM 墙；标量广播快路径 + `expectedOps` 真实模型（R30）
- [x] 激活内存池（P0）：按生存期复用 + reshape 视图并集；wall −5%、数值逐位一致（R-P0）
- [ ] 算子融合、内存复用（byte-offset 子分配）、降低 launch 开销（整网墙钟；busy 13.1 vs 墙钟 16.7 ms）
- [ ] seg / obb 解码；多 Session 并行缓冲
- [ ] 支持更多模型（detect 系列、其他 backbone）
