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

> 开发/测试使用的容器镜像 `infvino-dev:latest` 由本仓库的
> [`docker/Dockerfile`](docker/Dockerfile) 构建；Python 工具依赖见
> [`requirements-dev.txt`](requirements-dev.txt)。**不需要 OpenVINO。**
> 与 OpenVINO 的对照一律经 `scripts/openvino_baseline.py` 路由到其**最新稳定版**（不硬编码）。

> 开发环境一键准备：
>
> ```bash
> docker build -f docker/Dockerfile -t infvino-dev:latest .
> ```
>
> `infvino-dev` 镜像已自带完整工具链（cmake / g++ / git / OpenCV / OpenCL / yaml-cpp +
> `requirements-dev.txt` 的 Python 测试工具），构建与测试可全在容器内完成。若要在**宿主**
> 直接跑 Python 工具，再建 venv：`python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt`。

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

### 离线单元测试（**不需要 iGPU / `/dev/dri`**）

可直接在构建同容器（或已装依赖的宿主）内运行，不需要 iGPU。改动 host 侧纯逻辑时的快速门——
覆盖调优缓存、布局最小割、物理标尺、算子族注册表这些「选型/记账/契约」层
（R42–R52 里反复出 bug、而 GPU 数值测试抓不到的地方）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target check        # 构建并跑全部离线测试（CTest）
# 或直接：ctest --test-dir build --output-on-failure
```

`tests/` 下的套件：`test_layout_solver`（随机 submodular 图 vs 暴力枚举的精确性）、
`test_tuning_cache`（ABI / 设备键 / 数值契约的 round-trip 与失效）、
`test_rulers`（`expectedOps` / 带宽曲线 / `occupancyPressure` / R55 `l3MlpFactor` 等）、
`test_kernel_families`（注册表不变量、`inIndex` 布局契约、`actMask` 过滤）、
`test_l3_model`（R55 全局 LRU 溢出模拟 + 有限差分逐节点价），
以及既有的 `tuning_test`。

### GPU 数值 / 整网回归

以下数值与整网检验需要可用的 Intel iGPU（`/dev/dri`）与容器镜像（脚本用 docker 跑容器）：

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
| [`docs/profiling-budget.md`](docs/profiling-budget.md) | **busy/net/e2e 预算与优化记分卡**：统一三态口径 + 消融归因（`kernel_run --profile-json` / `scripts/analyze_budget.py` / `scripts/profile_ablation.py`）|
| [`docs/command-buffer.md`](docs/command-buffer.md) | **`cl_khr_command_buffer` 可行性实测**：最新 26.35 runtime 仍不可用（LEO 门控 + 录制入口未实现）|
| [`docs/dependencies.md`](docs/dependencies.md) | 依赖与版本清单 |
| [`docs/round22-status.md`](docs/round22-status.md) | **R22–R23 现状分析**：1×1 kernel / OV conv3×3 / 融合 / 与 OV 对照 |
| [`docs/round24-analysis.md`](docs/round24-analysis.md) | **R24 conv3×3 逐 size 瓶颈分析**：ISA 配额证据 / 中间标准修正 / 两通路接入 |
| [`docs/round25-ovblocked.md`](docs/round25-ovblocked.md) | **R25 OV 阻塞式 conv 完整移植**：逐 size 对照 / 第三条 autotune 通路 |
| [`docs/round30-smallops.md`](docs/round30-smallops.md) | **R30 剩余 kernel（非 conv/gemm）的物理模型**：内存 roofline / ISA 配额 / 哪堵墙 |
| [`docs/round33-conv3x3-headroom.md`](docs/round33-conv3x3-headroom.md) | **R33 conv3×3 剩余空间**：三通路逐配额重算 / 「赢 OV」的两块区域 / CINC 候选（+45%，逐位一致） |
| [`docs/block-layout.md`](docs/block-layout.md) | **R36 持久 blocked 布局 + 布局自动化**：`b_fs_yx_fsv16` 链 / `OUT_FSV16` / 同帧去重 / autotune 驱动的规划器 |
| [`docs/openvino-gap-analysis.md`](docs/openvino-gap-analysis.md) | **infvino vs OpenVINO GPU 差距分析**：逐维对标 / 强项 / 学习清单（P0–P3） |
| [`docs/kernel-families.md`](docs/kernel-families.md) | **算子族管理与接入框架**：声明式 `KernelFamily` 注册表 / 布局图 / per-family 上限模型 / scenario→族 判定；含 **blocked 1×1 移植样例**（`conv1x1_blk`，pointwise 5–6×）|
| [`docs/memory-reuse-design.md`](docs/memory-reuse-design.md) | **P0 激活内存池设计 + R-P0 实测**：生存期复用 / 视图并集 / 墙钟收益 |
| [`docs/register-model.md`](docs/register-model.md) | **7 线程 EU 寄存器限制的完整模型**：tile/ops 天花板推导 + 使用清单 |
| [`docs/round44-global-objective.md`](docs/round44-global-objective.md) | **R44 autotune 目标函数修正 + 体系缺陷倒查**：隔离 min → 整网 busy 坐标下降回验（`--global`）；分层列出目标/表示/测量/安全/指标五类漏洞 |
| [`docs/round45-perplan-and-search.md`](docs/round45-perplan-and-search.md) | **R45 P0–P2 落地 + 整网搜索/offline 探索**：per-plan 选择覆盖、`choiceEntry` 统一、kernel 源指纹守卫、噪声地板/预算/top-K∪每族代表；两级缓存融合与回退快路径 |
| [`docs/round46-global-flow-findings.md`](docs/round46-global-flow-findings.md) | **R46 整网调优全流程实测**：暴露并修复 3 个基础设施 bug（`--global` 空操作/分批推进、N==1 GEMV 漏改、缺最终验收门）；实证「整网回验暂无可靠正收益」与 in-situ/稳态口径错配；**blocked chain 前置条件** |
| [`docs/round47-tvm-strategy.md`](docs/round47-tvm-strategy.md) | **R47 TVM meta_schedule 策略借鉴**：可加目标/代理剪枝/梯度预算三层对照；测量 min+median 双口径与交错；按 R46 前置排序的落地路线（blocked chain 暂缓） |
| [`docs/round47-full-flow-findings.md`](docs/round47-full-flow-findings.md) | **R47 三模型全流程实测**：mb −2.2%（仅布局耦合型模型有收益）、yolo ≈0；逐节点归因（收益全在 blk 族、reorder 反升）；倒查 8 项 bug/设计缺陷；判定 blocked chain 是 mb 的主矛盾、非 yolo |
| [`docs/round47-l3-model.md`](docs/round47-l3-model.md) | **R47 把 L3/DRAM 显式建模进标尺**：锁频重测 BW-足迹曲线（膝点 3→4MB）、通用内存 roofline、conv/gemm 软 expected 取 min；内存受限层从假余量纠正为贴墙 |
| [`docs/round48-candidate-expansion-plan.md`](docs/round48-candidate-expansion-plan.md) | **R48 系统性扩充算子候选集计划**：按物理瓶颈×契约分 6 维（输出 tiling/split-K/数据通路/布局守护/小算子融合/全核内建）+ 基础设施前置 + M0–M6 里程碑 + 风险边界 |
| [`docs/round49-layout-mincut-pilot.md`](docs/round49-layout-mincut-pilot.md) | **R49 布局最小割试点**：mincut 提案 + 验收门；逐节点 L3 定价的负结果（正确形态是整网模拟/评分器） |
| [`docs/round50-depthwise-yblock-and-mincut-fix.md`](docs/round50-depthwise-yblock-and-mincut-fix.md) | **R50 depthwise `Y_BLOCK` + mincut 正确性修复**：`C%16` 门 + `model_check` env 透传 + `--verify` 激活码修复 |
| [`docs/round51-cat4-registry-and-non16-fsv16.md`](docs/round51-cat4-registry-and-non16-fsv16.md) | **R51 cat4 进注册表 + 非 16 通道 fsv16 持久化**：`gemm_cat4_f16` 族；mb 默认 −2.7%；SE Mul→conv1x1 融合（opt-in，整网中性） |
| [`docs/round52-planner-graph-bug-and-gap-fsv16.md`](docs/round52-planner-graph-bug-and-gap-fsv16.md) | **R52 planner 图级 bug**：`gap` fsv16 分配/标记漂移修复 + 池/布局探针 |
| [`docs/round53-conv3x3-route-shape-layout.md`](docs/round53-conv3x3-route-shape-layout.md) | **R53 conv3x3 路线×shape×layout 系统分析**：fsv16 输出成本 + mincut 3x3 opt-in |
| [`docs/round54-chain-layout-default-and-ov-reorder.md`](docs/round54-chain-layout-default-and-ov-reorder.md) | **R54 链布局进 plan 期默认 + 移植 OV SLM-transpose reorder**（用户决策 A） |
| [`docs/round55-l3-coupling-calibration.md`](docs/round55-l3-coupling-calibration.md) | **R55 L3 occupancy↔miss 标定 + 可加定价主问题**（collective/pernode 负结果） |
| [`docs/round56-opset-completion-and-fusions.md`](docs/round56-opset-completion-and-fusions.md) | **R56 算子集收尾**：契约修复 + N=1 多输出候选；depthwise 向量 store / SPPF 融合负结果 |
| [`docs/round57-mincut-localize-and-reorder-objective.md`](docs/round57-mincut-localize-and-reorder-objective.md) | **R57 min-cut 回退局部化 + reorder 级结构价**：提案生成与接受分离 |
| [`docs/round58-component-localized-layout-gate.md`](docs/round58-component-localized-layout-gate.md) | **R58 布局验收门分量级局部化**（默认开） |
| [`docs/round59-l3-spill-calibration.md`](docs/round59-l3-spill-calibration.md) | **R59 L3 spill 实测重标定 + 建模验证** |
| [`docs/round60-l3-exact-spill-plru-and-reorder.md`](docs/round60-l3-exact-spill-plru-and-reorder.md) | **R60 精确 spill（复用/栈距离）+ pLRU 等价模型自标定 + reorder 可加性** |
| [`docs/round61-l3-line-granular-model-and-complexity.md`](docs/round61-l3-line-granular-model-and-complexity.md) | **R61 行粒度 1b-NRU 精确模型（公开 PRM）+ 成本函数复杂度分析** |
| [`docs/round62-l3-fidelity-geometry-correction.md`](docs/round62-l3-fidelity-geometry-correction.md) | **R62 硬件 L3 保真度/几何纠正**：GPU 私有 L3 = 3.75 MiB、膝点 ~4MB、周期 512 行 |
| [`docs/round63-address-mapping-reverse-engineering.md`](docs/round63-address-mapping-reverse-engineering.md) | **R63 地址→bank/set 逆向 + 两级层次澄清**：GPU 私有 L3 + 共享 LLC |
| [`docs/round65-l3-isolation-and-geometry-conclusion.md`](docs/round65-l3-isolation-and-geometry-conclusion.md) | **R65 L3 隔离尝试与几何结论**：地址→bank/set 只能逆到周期 512，正式接受近似 + 整网 A/B |
| [`docs/round66-two-dim-roofline-and-selection.md`](docs/round66-two-dim-roofline-and-selection.md) | **R66 二维有效带宽接进内核级内存 roofline**（诊断）+ 与选择的关系澄清 |
| [`docs/round67-capacity-term-in-layout-objective-negative.md`](docs/round67-capacity-term-in-layout-objective-negative.md) | **R67 容量项加进布局目标函数**（负结果，默认关） |
| [`docs/round68-l3-model-final-form.md`](docs/round68-l3-model-final-form.md) | **R68 L3 成本建模最终形态（收敛总结）**：可加主问题 + 全局非可加 spill |
| [`docs/round69-occupancy-contract-and-spill-interface.md`](docs/round69-occupancy-contract-and-spill-interface.md) | **R69 占用/访存契约声明化**：`KernelFamily::mem`（`MemContract`）把候选几何收进注册表，`occupancyPressure`/`buildL3Access` 注册表优先；选择逐位不变、`mean_rel` 与 R68 相同 |
| [`docs/open-items.md`](docs/open-items.md) | **开放项登记表**：系统「未完成/未决/暂缓/负结果」的单一权威清单（R70 起维护） |
| [`docs/round70-status.md`](docs/round70-status.md) | **R70 统一收尾**：逐节点 spill 归因 + `--global` NO-OP 硬信号 + 残差落盘；三模型 `model_check`/`reuse_check` 回归；全量 retune 激活 fixpoint 的负结果 |
| [`docs/round71-binding-wall-model.md`](docs/round71-binding-wall-model.md) | **R71 绑定墙分析框架**：带宽侧补 GPU 私有 L3/共享 LLC/DRAM 三级足迹与两级 spill（`memTierTime`/`l3TwoLevelSplit`）；计算侧补寄存器 ILP 与 SLM 带宽/容量（`computeWall`/`attributeWall`）；`kernel_autotune --wall-report` + `scripts/analyze_walls.py`（按调用加权）。**纯诊断，不参与选择** |
| [`docs/round71-1x1-depthwise-layout.md`](docs/round71-1x1-depthwise-layout.md) | **R71 三项实施 + blocked 链布局缺陷**：① 1×1 split-K 候选放开小空间（mb busy −4.7%）；② cat4→blocked 1×1 管道（concat `OUT_FSV16`）；③ **链感知布局定价**——修复「不动点卡在 NCHW、跨算子 fsv16 链断掉」的缺陷（对 OV 差距的核心；`INFVINO_NO_FUSE_CAT4` 对照） |
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
> 与 OpenVINO 最新稳定版对照（该次测量时为 2025.2；新对照经
> [`scripts/openvino_baseline.py`](scripts/openvino_baseline.py) 动态解析，不硬编码），
> 整网仍差 ~1.5–1.9×，差距集中在 conv3×3 网格饥饿
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
>
> **R-P1a（融合通用化，P1）**：把激活融合从「只覆盖 1×1/SiLU」扩到
> **通用/depthwise conv 的 epilogue**（内核本就支持 act=1..4）。mobilenet 的 11 条
> `depthwise→ReLU/HardSwish` 全部折进 conv，`ew_unary` **11→0**；补跑 9 条 autotune。
> mobilenet busy **2.65→2.59 ms（−2.4%）**，数值**逐位一致**。yolo 无此模式（不变量）。
> **R-P1 调研（布局）**：`conv_blk` 每层每帧重排输入，y8/y11 **24/23 次、0.66–0.68 ms
> （≈5% busy）**；但重排是 **launch floor 主导**（向量化 store 实测更慢，负结果）。
> 离线 ROI 定量：只有 **11/24（y8）、9/23（y11）** 条边可省（blk→blk 且单消费者），
> 至多 ~0.3 ms（<2% 墙钟），代价是大重构 → **P1-layout 正式关闭**。
>
> **R31（padded depthwise，A 方向，负结果）**：离线 ISA 证明 `depthwise_v` 的墙是
> 地址/边界谓词，零边预填充后指令 **−41%**（1063→627，配额 3.9→6.5）。完整落地了
> `depthwise_pad` + 无边界 `depthwise_vp`（与 `depthwise_v` **逐位一致**），并让 autotune
> 把额外 pad 一趟计入 vp 成本。但整网 A/B：depthwise 分项 **−7~11%**，而 pad 是带宽受限的
> 额外一趟（y11 +0.070 / mb +0.035 ms），**二者相抵**（y11 +0.6%、mb 持平）。
> **决策**：vp 默认不进候选（`INFVINO_DW_PAD=1` 才启用），内核与算子级测试保留。
> 详见 [`docs/kernel.md`](docs/kernel.md) Round 31。
>
> **P2（host 分段，先测）**：给 `PlanModel::run` 加了 host 分段计时（入队提交 / 同步 /
> 其余 host），实测 GPU busy 只占墙钟 **52%（mb）–74%（y8）**，缺口由
> `clEnqueueNDRangeKernel` 与 `setArg`/簿记两块平分。见
> [`docs/benchmark.md`](docs/benchmark.md) §2.4。
>
> **R33（conv3×3 剩余空间 + CINC 候选）**：按三条通路各自的 ISA 配额重算 `ratio`，指出
> 调优表对 blk/native 的 `expected` 口径失真；"赢 OV" 的窄通道区（native，真 ratio
> 0.23–0.42）候选集过窄——把 `CINC` 纳入 native 候选后，`160×160 s1 8→16`（Cin=8）
> **3.73→5.41 ops/EU/cyc（+45%）、数值逐位一致**，调优器端到端已选 `CINC8`。
> **R34**：CINC 泛化到小 `Cin`（精确 `CINC=Cin`），stem `320×320 s2 3→16` **2.97→3.56（+20%）**；
> 同时复核主导层 `40×40/80×80 C64` 已触顶（`OBW=10/12`、`kd` unroll 均无收益）。
> 详见 [`docs/round33-conv3x3-headroom.md`](docs/round33-conv3x3-headroom.md)。
>
> **R35（command buffer，负结果）**：CUDA-graph 类比（一次录制整帧、每帧重放）实测**不可行**——
> dev 镜像的 23.17 与最新 **26.35.39758.10**（需 Ubuntu 24.04）在本机都不暴露
> `cl_khr_command_buffer`；上游实现本身也**只完成一半**（无 `clCommandNDRangeKernelKHR`
> 等录制入口，且仅在实验性 LEO 驱动、默认关闭）。见
> [`docs/command-buffer.md`](docs/command-buffer.md)。
>
> **R36（block layout，P1-layout 重开落地）**：把 `conv_blk` 的输入重排从「每层每帧一次」
> 升级成**持久 blocked 链 + 同帧去重**，并让**布局决策由 autotune 自动驱动**：生产者
> （blocked conv）直接写 `b_fs_yx_fsv16`（`-DOUT_FSV16=1`），消费者直接读、零 reorder。
> 三模型数值**逐位一致**、`model_check` PASS、`reuse_check` PASS；yolov8n reorder
> **24→10/帧（−58%）**、busy **−2.2%**、墙钟 **−1.6%**，yolo11n reorder **23→13**、
> 墙钟 **−2.2%**（与当初 P1-layout「≈2%」的定量一致，但现在自动且零数值风险）。
> 离线规划/分析器：`scripts/analyze_layout.py`；运行时开关
> `INFVINO_NO_BLOCK_LAYOUT=1` / `INFVINO_NO_REORDER_DEDUP=1`。
> 详见 [`docs/block-layout.md`](docs/block-layout.md)。

> **三模型预算分析 + Tier 0–2 落地（2026-10）**：基于 `busy/net/e2e` 框架完整分析了
> yolov8n-pose / yolo11n-pose / mobilenetv3-small（逐 kernel + 总体 `ops/EU/cyc`、
> 缓存管理短板、系统缺陷、优化路线），报告见
> [`docs/budget-analysis-3models.md`](docs/budget-analysis-3models.md)。已落地：
> **(1) 自动调优覆盖** 224→**239** 条、签名覆盖 100%；**(2) 激活池 byte-offset 子分配**
> （默认开）y8 24.0→**20.6 MB**、y11 26.5→**23.1 MB**，busy −1.7%/−2.3%，逐位一致；
> **(3) 磁盘 kernel 二进制缓存**（`INFVINO_PROGRAM_CACHE`）把每进程冷启动 JIT
> **8.5–12 s → ~0.1 s**；**(4) 调优缓存 `cache_abi` 守卫** 防止 kernel 宏语义变化后
> 静默套用旧 options；**(5) host f32↔f16 批量转换（AVX2，逐位一致）** 把每次推理的
> 转换开销 1.59→**0.50 ms/帧**，net yolo **−5.9%~−6.7%**；**(6) 首层 Cin=3 专用 conv**
> （`kernels/conv_cin3.cl`）y8/y11 stem **−40%/−50%**，三模型 `model_check` PASS。
> 并在 `ops/EU/cyc` 分析框架前加入 **roofline 判断**提醒
> （[`docs/profiling-budget.md`](docs/profiling-budget.md) §3.0）；残余空间分析（结论：
> 大头是物理/结构受限，非候选缺失）与小算子缓存命中分析（`scripts/analyze_cache.py`）
> 见 [`docs/budget-analysis-3models.md`](docs/budget-analysis-3models.md) §9/§10。

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
- [x] 融合通用化（P1）：通用/depthwise conv 折入激活 epilogue；mobilenet `ew_unary` 11→0、busy −2.4%、逐位一致
- [x] P1-layout 调研收口：重排 launch floor 主导、可省边 <2% 墙钟（首轮暂缓）
- [x] block layout（R36）：持久 `b_fs_yx_fsv16` 链 + 同帧去重 + **autotune 驱动的布局自动化**；y8 reorder 24→10、busy −2.2%、逐位一致（`docs/block-layout.md`）
- [x] 算子族注册表（Phase 2）：`KernelFamily` 声明式管理（候选/布局/上限/激活契约）驱动选择；移植 **blocked 1×1（RES）+ blocked depthwise**，`1x1→depthwise→1x1` 持久 fsv16 链；canonical 激活码统一；mobilenet busy **2.62→1.72 ms（−34%）**、yolo11 −3%、三模型 `model_check` PASS（`docs/kernel-families.md`）
- [x] P2 host 分段实测：busy 占墙钟 52–74%，缺口 = 入队提交 + setArg/簿记（`docs/benchmark.md` §2.4）
- [x] 磁盘 kernel 二进制缓存（`INFVINO_PROGRAM_CACHE`）：冷启动 JIT 8.5–12 s → ~0.1 s，逐位一致
- [x] 激活池 byte-offset 子分配（默认开，`INFVINO_NO_POOL_OFFSET=1` 可关）：y8 24.0→20.6 MB、busy −1.7%
- [x] 调优缓存 `cache_abi` 守卫（kernel 宏语义变化即整份作废）
- [x] `ops/EU/cyc` 分析框架前置 roofline 判断提醒（`docs/profiling-budget.md` §3.0）
- [x] host f32↔f16 批量转换 AVX2（逐位一致）：yolo net −5.9~6.7%（`INFVINO_NO_SIMD_HALF=1` 对照）
- [x] 残余空间分析：conv3x3/depthwise/bmm 为物理/结构受限（`docs/budget-analysis-3models.md` §9）
- [x] 首层 Cin=3 专用 conv（`kernels/conv_cin3.cl`）：y8/y11 stem −40%/−50%，三模型 `model_check` PASS
- [x] 小算子缓存命中分析器 `scripts/analyze_cache.py`：多数已贴住「工作集+launch」上限，余额在 permute/resize（跨步）与 gap（网格）
- [x] depthwise padded（A 方向）：指令 −41%、逐位一致，但 pad 带宽相抵 → 整网负结果，默认关闭（Round 31）
- [x] 修复 `conv_ov` 的 `SLM_DIV` 数值 bug（工作组几何：子组须沿最快 local 维铺开）+ 重开 ov 的 SLM 候选；conv3×3 小网格再拿一块，busy **y8 −11.2% / y11 −8.1% / mb −3.2%**，三模型 `model_check`/`reuse_check` PASS（`docs/round39-convov-slm-fix.md`）
- [x] 分析框架反查：修 `conv3x3_ov` ceiling 通道粒度（16→32）、ov/native ceiling 的「WG vs sub-group」占用口径、`refresh-expected` 覆盖 conv3x3；记录内存族 roofline ceiling 偏低（未决）（Round 39）
- [x] conv kernel 全局 `ops/EU/cyc` × size 分析：ov+SLM 已通吃（24/32 签名）；推翻 R33「blk 赢小空间大通道」、修正 R24「40×40 无解」；补 native `CB=8`（Cout≤8，+21%）与 ov stride-2 `OBW=7` 候选（`docs/round40-conv-ops-size-analysis.md`）
- [x] conv 离天花板差距的**系统归因**：主因是「无 L1 + ~150cyc 访存延迟」被 128-GRF 墙卡死的单线程 ILP（软件流水/手工预取**无效**），DRAM/L3 带宽只在工作集>L3 与**输出写大**（stem、窄 Cout）的层主导；新增 `-DPROBE`/`-DPF` 诊断宏（默认关）；顺带修 `kernel_bench --verify` 的参考激活（此前 ACT≠0 全错，R33「native verify 坏」实为工具 bug）（`docs/round41-kernel-bottleneck-attribution.md`）
- [x] 标尺/噪声/计费治理：`ops/EU/cyc` 分层为 **hard_ceiling（只放 ISA 配额/roofline 下界）+ 软 expected + 告警**；autotune 改 **min 估计器** + spread 告警 + `scripts/gpu_clocks.sh` 锁频；`scripts/noise_check.sh` 判稳定；修 **reorder 折进 kernel ms** 的记账 bug（污染 ops/ratio）；确认噪声根因 = **LLC→DRAM 带宽断崖**（`docs/round42-rulers-noise-and-tooling.md`）
- [x] 硬/软标尺落地：`KernelFamily.hardCeiling` + `TuningEntry.hard_ceiling/hard_ratio`（向后兼容），`kernel_autotune --expected` 同打 soft/hard（`docs/round43-hard-ceiling-mdapi-jointlayout.md`）
- [x] dev 镜像加 MD API 依赖（`intel-metrics-discovery(+dev)`/`intel-gpu-tools`/`libdrm-dev`）并验证；新增 `gpu_metrics --list/--sample`（枚举 EuActive/EuStall/L3/SLM 计数器 + `CalculateMetrics` 解码；**OA 采样受本机内核 `CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS` 未开限制**）
- [x] `scripts/kernel_diag.sh`：一命令跑 feed/store 隔离探针 + `hard_ratio` + 墙判决（R41/R43 归因工具化）
- [x] 联合 (族,布局) 不动点从 conv3x3 **推广到 conv1x1/depthwise**（去保守计费、`choiceEntry`/`nodeFamily` 统一、修 conv1x1 输入下标）；三模型 retune 后 `#blk/#non/#reorder` 就位、`model_check`/`reuse_check` PASS，busy y8 **−0.9%** / y11 **−1.3%** / mb +1.6%；**发现「隔离 bench ≠ 流水线表现」**（候选选择目标函数问题）（`docs/round43-...md`）
- [x] **修 autotune 目标函数**（R44）：从「单节点隔离 `min(ms)`」改为「整网 busy 坐标下降回验」——保留隔离 top-K 短名单，在真实 plan 上逐签名回验、只有降低端到端 busy 才接受（`kernel_autotune --global`，默认关）；轻量 GPU 机制自验通过；并**分层倒查调优体系缺陷**（per-plan 选择存储缺失、`choiceEntry` 接入不统一、数值契约未进缓存 ABI、候选静默跳过、内存族 ceiling 偏低等）（`docs/round44-global-objective.md`）
- [x] **R45 P0–P2**：`choiceEntry` 统一所有可调族；**per-plan 选择覆盖**（`<plan>.tuning.json`，两级缓存让位置相关全局最优不被跨模型覆盖）；kernel 源指纹守卫（`INFVINO_TUNING_STRICT` 可强制作废）；候选跳过率/离散度告警；隔离 top-K 复测、整网短名单 = **top-K ∪ 每族代表**（隔离 top-K 可能整体漏掉某族）、噪声地板、`--global-budget`；`autotune.py --global --lock` 安全驱动（`docs/round45-perplan-and-search.md`）
- [x] **R46 整网调优全流程实测**：跑通三模型（锁频/分批/无 HANG），暴露并修复 3 个基础设施 bug（`--global` 在有条目缓存时空操作 + `--retune` 分批不推进、P0#6 漏掉 N==1 GEMV、缺最终验收门）；加门后 mb 回归 +5.8%→−0.2%、y8 +0.5%（均噪声内），但**暂无可靠正收益**；实证 in-situ `min` 与稳态口径错配（同改动 in-situ −3~5% vs 稳态 +12%）；给出 **blocked chain 前置条件**（`docs/round46-global-flow-findings.md`）
- [x] **R69 占用/访存契约声明化**：`KernelFamily::mem`（`MemContract`）把候选占用几何收进注册表（补全 conv3x3 四 kernel / gemm_sk / GEMV / 非 blk depthwise），`occupancyPressure` 与 `buildL3Access` 注册表优先（R48 §10.6-B 缺口）；三模型选择逐位不变、`model_check`/`reuse_check` PASS、`mean_rel` 与 R68 相同；`INFVINO_LEGACY_OCCUPANCY`/`INFVINO_NO_MEM` 可回退/逐族消融（`docs/round69-occupancy-contract-and-spill-interface.md`）
- [x] **R70 统一收尾**：逐节点 spill 归因暴露（`INFVINO_LAYOUT_REPORT=1` + `INFVINO_LAYOUT_SPILL=1`）；`--global` 空操作硬信号（`globalRetune` 返回 `-1` → `kernel_autotune` 退出码 3、`autotune.py` 检测停止）；整网回验残差数据集落盘（`--residual` / `INFVINO_GLOBAL_RESIDUAL`）；新增 [`docs/open-items.md`](docs/open-items.md) 开放项权威清单（`docs/round70-status.md`）
- [x] **R71 绑定墙分析框架**：带宽侧补 **GPU 私有 L3/共享 LLC/DRAM 三级足迹**与**两级 spill 拆分**（`memTierTime`/`l3TwoLevelSplit`）；计算侧补**寄存器 ILP**与**SLM 带宽/容量**（`computeWall`/`attributeWall`）；`kernel_autotune --wall-report` + `scripts/analyze_walls.py`（按调用加权墙预算）；**纯诊断、不改选择与数值**（三模型 `model_check` 逐位不变、`reuse_check` PASS、`ctest` 6/6）。把框架用于三模型与 OpenVINO 逐层对照的结论另文记录（未随本框架提交）
- [ ] 其余开放/暂缓项见 [`docs/open-items.md`](docs/open-items.md)：`KernelFamily::launch` 声明化（A2）、完整 blocked chain（B1）、conv3x3 布局链（B3）、epilogue 可组合化（D6）、降低 launch 开销（E1）、多 Session 并行缓冲（E3）、seg/obb 解码（F1）、更多模型（F2）等
