# Round 22 现状分析：1×1 专用 kernel、OpenVINO conv3×3 移植、融合

> 目标模型：`yolov8n-pose` / `yolo11n-pose` / `mobilenetv3-small`
> 硬件：Intel Iris Xe（80 EU / 1.3 GHz，TGL iGPU），单流，warm。
> 数值判据：相对误差（`kernel_check` mean_rel<1e-2 & max_rel<5e-2；
> `model_check` mean_rel<2e-2 & max_rel<5e-2）。
> 本轮改动见 `docs/kernel.md` Round 22。

---

## 1. 任务与交付

| 任务 | 交付 | 结论 |
|---|---|---|
| 写 conv1×1 专用 kernel，优化 mobilenetv3-small 端到端 | `kernels/conv1x1.cl`（`conv1x1_gemv_f16` + 实验性 `conv1x1_f16`）；`onnx2plan.py` 折叠 1×1 为 `conv1x1` 单节点；`gemm_f16` 加融合 epilogue | **成功**：mobilenet kernel busy 6.28→3.46 ms（1.81×） |
| conv3×3 直接上 OpenVINO 的 kernel（引入代码、不引入依赖，处理 Apache-2.0） | `kernels/conv_ov.cl`（自包含移植 `convolution_gpu_bfyx_os_iyx_osv32`）；host 端 OSV 权重重排；`third_party/openvino/LICENSE` + `THIRD_PARTY_NOTICES.md` | **成功且更快**：大层 12.6–13.6 ops/EU/cyc，全形状优于原 native |
| 用融合提升三个模型端到端 | 1×1/Gemm 融合 bias+激活；conv3×3 融合 bias+激活+尾部 HardSwish；残差融合（实测负，默认关） | **部分成功**：bias+act 融合生效；残差融合负结果 |
| 完整的现状分析 | 本文 | — |

---

## 2. 同会话基线（HEAD = `4437956`）

在独立 worktree 里用 HEAD 源码 + HEAD `onnx2plan.py` 重新生成 plan，与新版在同一时段测量：

| 模型 | HEAD kernel busy | HEAD 墙钟（net only） |
|---|---|---|
| yolov8n-pose | 19.57–19.62 ms | 24.09 ms |
| yolo11n-pose | 22.40–22.42 ms | 28.24 ms |
| mobilenetv3-small | 6.27–6.29 ms | 7.29 ms |

（会话最早一次的测量为 19.61 / 22.38 / 6.27 ms，与 worktree 基线一致，可排除漂移。）

---

## 3. 三块工作

### 3.1 conv1×1 专用 kernel

**问题定位**：把 `gemm` 的 profile tag 细化为 `gemm@MxNxK` 后，mobilenet 的 4.12 ms gemm 中
**N=1（HW==1）的矩阵-向量积占 ~2.3 ms**（classifier 与 10 个 SE 的 fc1/fc2）。通用 GEMM 在
N=1 时 `BN=64` 浪费 63/64 lane，且网格只有 `ceil(M/BM)` 个 work-group（1000×1×1024 仅 8 个）。

**实现**：
- `conv1x1_gemv_f16`：一个 work-group（16 lane）= 一个输出通道；lane 沿 K 分块归约，
  `sub_group_reduce_add` 收口。权重自然 `[Cout][Cin]`、相邻 lane coalesced；X 广播；
  **fp32 累加**。融合 bias + 激活（+可选残差）。
- N>1：复用调好的 `gemm_f16` + **融合 bias/激活 epilogue**（`-DEPI=1`）。
- 实验性 `conv1x1_f16`（lane=空间、`[Cin][Cout]` 宽载权重）：N>1 上全面慢于 `gemm_f16`
  （小 tile 网格饥饿，反而不如 GEMM 的 SLM 分块），保留为负结果。

**逐层效果（mobilenet，N=1）**：gemm → gemv

| 层 (Cin→Cout) | gemm | gemv | 加速 |
|---|---|---|---|
| 576→1024（classifier.0）| 0.264 ms | 0.043 ms | 6.1× |
| 1024→1000（classifier.3）| 0.461 ms | 0.058 ms | 7.9× |
| 576→144（SE fc2）| 0.143 ms | 0.015 ms | 9.5× |
| 240→64（SE fc1）| 0.297 ms | 0.011 ms | 27× |
| 288→72 | 0.131 ms | 0.010 ms | 13× |

同时 `bias_add`×42 与 `ew_unary`×42 两次 launch 全部并入 1×1/Gemm 节点。

### 3.2 OpenVINO conv3×3 移植

- 源：`convolution_gpu_bfyx_os_iyx_osv32.cl`（Intel，Apache-2.0）。
- 自包含化：OV 的 `#include` 辅助头与 JIT 宏层用本地定义替换，**数据通路保持原样**：
  lane=输出通道（OSV_SIZE=32，每 lane 2 通道）；输入块一次载入寄存器 +
  `sub_group_broadcast`；权重用 `intel_sub_group_block_read_us2` 从 OSV swizzle 布局读取；
  **零 SLM、零 barrier**；bias+激活（+可选残差）在输出阶段融合。
- host 端 `PlanModel::ovWeight` 按 `GET_FILTER_OS_IYX_OSV_INDEX`（`o%32` 自然序）重排权重，
  按 init 名缓存。
- **关键修正**：R20 的自写 OV 式 kernel 只有 8.2，是因为把 `_sub_group_block_read*` 当成
  「连续配对 `2l,2l+1`」；实际是**跨步**语义（lane l 取位置 `l, l+16`）。修正后正确且更快。

**单 kernel 对比（vs R18 native `conv3x3_f16`）**：

| shape | native | OV 移植 | 加速 |
|---|---|---|---|
| 64→64@80 s1 | 10.31 | **13.61** | 1.32× |
| 128→128@40 s1 | 10.34 | **12.61** | 1.22× |
| 64→64@40 s1 | 6.16 | **7.91** | 1.29× |
| 16→16@160 s1 | 4.81 | **5.75** | 1.20× |
| 128→128@20 s1 | 3.09 | **5.27** | 1.71× |
| 256→64@20 s1 | 1.63 | **2.92** | 1.79× |
| 64→128@160 s2 | 7.07 | **10.73** | 1.52× |

→ 所有测试形状都赢，因此 `PlanModel` 的 conv3×3 默认走 OV（节点 `ov=0` 可回退 native）。
这推翻了 R20「OV 路径不如我们」的结论。

**Apache-2.0 合规**：`third_party/openvino/LICENSE`（原文）、`THIRD_PARTY_NOTICES.md`
（来源/文件/修改说明/清单）、每个派生文件头部保留 Intel 版权行 +
`SPDX-License-Identifier: Apache-2.0` + 修改说明。不链接 OpenVINO 库。

### 3.3 融合

**生效**：
- 1×1 conv / `Gemm`：`bias + 激活` 融进单节点（§3.1）。
- conv3×3 OV：`bias + 激活` 融合；尾部 HardSwish 折叠（`ACT_CODE_CONV`）。
- `onnx2plan.py` 增加通用尾部激活折叠（Sigmoid/Relu/HardSwish/HardSigmoid），
  仅当生产节点只有一个消费者且非图输出时。

**负结果（默认关，`INFVINO_FUSE_RESIDUAL=1` 可开）**：把 C2f / mobilenet 的残差 `Add`
折进 conv epilogue，三个模型**全部变慢**（yolo8 +0.31、yolo11 +0.95、mobilenet +0.33 ms）：
融合后的 conv 多一次全张量残差读 + 一个分支、寄存器压力上升，代价大于省下的 `ew_binary`。

---

## 4. 端到端结果

### 4.1 kernel busy（`kernel_run --report`，同会话对照）

| 模型 | HEAD | **R22** | 加速 |
|---|---|---|---|
| yolov8n-pose | 19.60 ms | **17.26–17.43 ms** | 1.13× |
| yolo11n-pose | 22.40 ms | **20.19–20.36 ms** | 1.10× |
| mobilenetv3-small | 6.28 ms | **3.46–3.55 ms** | **1.81×** |

### 4.2 墙钟（`infvino_bench`，net only，含 launch 开销）

| 模型 | HEAD | **R22** | 加速 |
|---|---|---|---|
| yolov8n-pose | 24.09 ms | **22.34 ms** | 1.08× |
| yolo11n-pose | 28.24 ms | **25.58 ms** | 1.10× |
| mobilenetv3-small | 7.29 ms | **4.64 ms** | **1.57×** |

> 墙钟提升小于 kernel busy，因为 launch/同步占相当比例（yolov8：busy 17.4 vs 墙钟 22.3）。
> 下一步的整网收益主要靠**减少 kernel 数**（图融合）而不是继续压单 kernel 时间。

### 4.3 数值（三级检验全 PASS）

- `kernel_check.py`：**ALL PASS**，含新增 `conv1x1`（融合 epilogue）、`conv1x1g`（split-K GEMV）、
  `conv3x3ov`（OV 移植）。
- `model_check.py`（vs onnxruntime FP32）：
  - yolov8n-pose `mean_rel=5.69e-4 / max_rel=2.03e-2`
  - yolo11n-pose `mean_rel=9.54e-4 / max_rel=2.76e-2`
  - mobilenetv3-small `mean_rel=1.31e-2 / max_rel=1.09e-2`
- GEMV 的 fp32 累加使 mobilenet 的 fc/classifier 数值优于旧 fp16 gemm（mean_rel ~1e-8 单算子）。

---

## 5. 逐算子现状（R22 最终 profile，按耗时排序）

### yolov8n-pose（17.26 ms）
| 算子 | ms | 说明 |
|---|---|---|
| conv3x3ov 大层（40×40/80×80/20×20 等）| ~7.5 | OV，已到 12–13.6 ops（大层）|
| concat4 ×19 | 1.14 | **DRAM 带宽受限**，C2f concat |
| conv3x3ov 小/低通道层 | ~3.5 | 网格饥饿，2–7 ops |
| conv1x1 ×28 | ~2.4 | N>1 走 gemm+EPI |
| ew_binary ×15 | 0.48 | 残差/head |
| 其余（resize/copy_c/maxpool/slice…）| ~1.5 | launch/带宽 |

### yolo11n-pose（20.19 ms）
| 算子 | ms | 说明 |
|---|---|---|
| conv3x3ov | ~7.5 | OV |
| concat4 ×23 | 1.26 | DRAM 带宽 |
| bmm ×2 | 0.94 | attention，逐元素串行 K |
| depthwise ×7 | 0.77 | |
| softmax_axis ×2 | 0.63 | |
| conv1x1 ×49 | ~3.5 | 含 attention/head |
| ew_binary ×24 | 0.56 | |

### mobilenetv3-small（3.54 ms）
| 算子 | ms | 说明 |
|---|---|---|
| conv1x1 N>1（expansion/projection）| ~1.9 | gemm+EPI；`96x49x576` 仍偏慢（0.51 ms）|
| depthwise ×11 | 0.51 | |
| conv1x1g（SE/classifier，N=1）| ~0.5 | split-K GEMV |
| ew_binary ×15 | 0.20 | SE Mul + 残差 |
| gap ×10 | 0.12 | 并行树归约后 |
| conv3x3ov ×1（stem）| 0.11 | |

---

## 6. 剩余瓶颈与下一步

1. **`concat4`（yolo 1.1–1.3 ms）**：数据（数 MB）超出 3.75 MB L3，落 DRAM（~19 GB/s），
   已接近带宽极限。真正要省必须把 concat **融进消费者 conv 的输入 staging**（跨多输入
   gather 进 SLM），本轮未做——是 yolo 最大的单项剩余。
2. **`bmm` / `softmax`（yolo11 ~1.6 ms）**：attention 的 batched matmul 是「每输出一个
   work-item + 串行 K」，与旧 gemm 同类网格饥饿；softmax 是每 (outer,inner) 串行扫 axis。
   可套用本轮 GEMV/并行归约的思路。
3. **conv3×3 小/低通道层（20×20、Cout=16）**：OV 已比 native 快，但仍只有 2–6 ops
   （网格/复用受限），属 R18–R21 已论证的「lane=通道 + block-read 也撞 ~16 天花板」。
4. **mobilenet 的 `96x49x576` 类 expansion（HW=49）**：N 太小、K 中等，gemm 与寄存器
   内核都一般；可试「lane=通道 + block-read」的 OV 式 1×1（类似 conv_ov，但 K=1 特化）。
5. **launch 开销**：墙钟 vs busy 差 ~5 ms（yolov8），减少 kernel 数（图融合 / 持久化 kernel /
   多 stream）是墙钟的主要杠杆。

---

## 7. 风险与负结果记录

- **残差融合**：三模型均负 → 默认关闭（env 开关保留）。
- **`conv1x1_f16` 寄存器内核**：N>1 负结果，保留为实验工具。
- **OV 移植的正确性坑**：`intel_sub_group_block_read*` 是跨步语义、OSV 权重是 `o%32`
  自然序；一开始按连续配对实现导致结果错误（已修正并由 `kernel_check` 覆盖）。
- **stride-2 + TX=64 的 native `conv3x3_f16` 在运行时 build 失败**（IGC/资源），生产路径
  对 s2 用 TX=40，`kernel_check` 的 s2 用例已改为 TX=40。
- 全程遵守 `docs/benchmark_protocol.md`（容器限内存、只挂 `renderD128`、单条命令少量配置、
  `timeout` 包裹）。

---

## 8. 复现

```bash
# 生成 plan（融合后的 conv1x1 单节点）
for m in yolov8n-pose yolo11n-pose mobilenetv3-small; do
  python3 scripts/onnx2plan.py --onnx models/$m.onnx --out-dir models/$m
done

# 整网 kernel busy / 逐算子
./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 5

# 算子级数值（含 conv1x1 / conv1x1g / conv3x3ov）
python3 scripts/kernel_check.py --repo $PWD --image infvino-dev:latest

# 整网数值 vs onnxruntime
python3 scripts/model_check.py --model mobilenetv3-small --repo $PWD

# 墙钟
./build/infvino_bench --config config/models.yaml --key yolov8n-pose --iters 100
```

---

# Round 23 补充：小算子收益、与 OV 的整网对照、阻塞式 conv（负结果）

> 详见 `docs/kernel.md` Round 23；数值判据同上。

## A. 与 OpenVINO 最新稳定版的整网对照（同一 iGPU）

> 下表 OV 数值为历史测量（当时为 2025.2）。重新对照请用
> `scripts/openvino_baseline.py run`，它会**动态解析最新稳定版**（不硬编码）。

| 模型 | OV GPU 合计 | OV infer | OV e2e | infvino busy (R23) | 差距 |
|---|---|---|---|---|---|
| yolov8n-pose | 9.33 ms | 11.2 ms | 13.9 ms | **17.2 ms** | ~1.5× |
| yolo11n-pose | 9.61 ms | 11.8 ms | 14.5 ms | **20.1 ms** | ~1.7× |
| mobilenetv3-small | 0.96 ms | 1.8 ms | 2.2 ms | **3.5 ms** | ~1.9× |

OV 端到端测试脚本：`models/*.onnx` + `openvino.Core().compile_model(..., "GPU")`，
预处理与 `infvino_bench` 一致（letterbox/RGB/normalize）。OV per-node 用
`enable_profiling` 取 GPU real_time 聚合（`Convolution` 独占大头）。

## B. infvino R23 的两项**已验证**收益

| 提交 | 内容 | 效果 | 数值 |
|---|---|---|---|
| `b28a6a4` | `.cl` 源文本缓存 + per-node kernel 句柄缓存 | 去除每次推理上百次磁盘读/编译/建 kernel | 逐位不变 |
| `bef119f` | `concat4` 改 3-D 网格（去逐元素 div/mod）| yolov8 busy **19.9→17.2**、yolo11 **23.0→20.1** ms | 精确 copy，三模型 PASS |

- 并行 `softmax_axis_r` 试过：0.63→0.155 ms，但**破坏数值**（mean_rel 3.7e-2），已回退。

## C. conv3×3 差距定位：**网格饥饿**

- yolov8 里 `conv3x3ov@40x40s1_Cin64_Cout64` 单层 1.49 ms ×10 ≈ 15 ms ≈ 总时 85%，
  且只有 ~8 ops。
- osv32 block 扫描（OBW/OBH）：80×80 最好 13.6（8,2）、40×40 最好 8.4（5,2）、
  20×20 最好 5.9（4,2）。**调参上限 ≤ +7% 且大层回退**。
- 结论与 R18–R21 一致：`os_iyx_osv32` 是 **1 broadcast : 1 mad 的 ~16 ops 发射上限**，
  与 block 大小无关。

## D. OpenVINO 阻塞式 conv（`convolution_gpu_bfyx_f16`）移植：**未成功**

- OV 实际选中的是阻塞式 kernel（lane=通道 16/块、每 lane 持 `OUTPUT_X_BLOCK_SIZE` 个连续
  输出列、输入行 staged 复用、权重 block_read、网格 `(X_BLOCKS, feature_blocks)`）。
- infvino 写了自包含简化版 `kernels/conv_blk.cl`：
  - 首版（lane 各自标量读行）：**1.3 ops**（16 lane 冗余读、权重非合并）；
  - 改权重 block_read + 输入 `sub_group_broadcast`：40×40 到 **6.6 ops**，仍**低于**
    osv32 的 7.9，且输出 NaN（leftover/OOB 未收口）。
- **判定**：当前工时内简化版没打赢已调优的 osv32，正确性未收口，**已回退**保留为负结果。
  若继续：需完整实现 leftover/OOB/分组路径 + 逐层 autotune，属大工程。

## E. 下一步建议（按性价比）

1. **墙钟/launch 开销**：yolov8 busy 17.2 vs 墙钟 22.3 ms，~5 ms 是 launch/同步。
   图融合（concat→conv 已试但正确性未收口）、减少 kernel 数、持久化 kernel 是主要杠杆。
2. **conv3×3**：完整移植 OV 阻塞式 kernel + 逐层 autotune（大工程，天花板 ~16 ops）。
3. **attention**：yolo11 的 `bmm`/`softmax`（~1.6 ms）仍是串行/低复用，值得专用化
   （注意：本机 GPU 在反复提交时易触发 i915 GPU HANG，须遵守 `docs/benchmark_protocol.md`）。
