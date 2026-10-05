# R48：系统性扩充算子候选集 —— 计划

> 背景：R47 把 L3/DRAM 显式建模进标尺与选择目标（[`round47-l3-model.md`](round47-l3-model.md)），
> 并落地「模型驱动选择」（[`round47-full-flow-findings.md`](round47-full-flow-findings.md) §4.8）。
> 最终判读是：**整网性能的下一步增量不在「选谁」，而在「有什么可选」** —— 模型（正确地）
> 判断当前候选集里没有比现状更优的选项。本文件给出**扩充候选集**的系统性计划。

---

## 0. 修订（R49 布局一等公民后，**立项标准已变**）

> R49（[`round49-layout-mincut-pilot.md`](round49-layout-mincut-pilot.md)）把**布局**提升为系统层
> 一等公民，用**精确最小割**做全局布局标注，并确立：**隔离 ms 只用于生成提案，接受/拒绝以
> 外部稳态 A/B 为准**。据此，本计划做三处修订：

1. **每个新候选必须声明布局契约**（否则 mincut 看不见它）：
   - `Candidate.canOutFsv16`：能否**直接产出** `b_fs_yx_fsv16`（生产者直写，省一趟 reorder）。
   - 族级 `KernelFamily.layout`：`in/out/canOutFsv16/inIndex`（激活槽）。**新增族/候选若不声明，
     会被布局规划当作 NCHW-only 处理**（R49 两次翻车即此类静默损失）。
2. **D4（reorder 融合）不再是独立 kernel 任务**：它已被布局 mincut 消费为「生产者直写能力」。
   后续「consumer prologue 直读 NCHW / producer epilogue 直写」都应以**布局契约字段**表达，
   而不是散落的 op 判断。
3. **验收口径**：新候选先以**隔离 `min`** 评估（候选生成），整网收益**必须**用 R49 的
   **门**（交错 median）或外部稳态 A/B 裁决；**不得**用 in-situ 单口径下结论（R44–R48 教训）。

> 另：R49 已证明「大 spatial 1×1 的最优族是 NCHW GEMM，不是 blocked」（§6.2），因此 D1（blocked
> 输出 tiling）这类**同族候选**对改变选择无帮助——立项前先过「是否改变布局图/是否跨物理瓶颈」
> 这一关（R48 §7.3-E、R49 §9.5）。

---

---

## 0. 目标与判据

- **目标**：让每个签名（op × shape × act）都有**跨越不同物理瓶颈**的候选，且每族有独立的上限
  模型与布局契约；新增候选**不破坏逐位数值契约**（或明确记录误差口径）。
- **判据**：
  1. 每个「离硬上限最远」的签名，至少有 **≥2 个不同瓶颈类别**（Fma/Instruction/Memory/Launch）
     的候选；
  2. 新候选经 **autotune 在隔离 `min` 口径**上可复现地更快（≥2% 且 spread 稳定），或
  3. 新候选**改变整网 L3/布局收益**（`predictNet` 预测下降且外部稳态 A/B 不回归）；
  4. 三模型 `model_check` / `reuse_check` PASS。

---

## 1. 现状盘点（单一真相源 = `src/KernelFamilies.cpp` 注册表）

| op | 现有族（数据通路）| 缺口（候选同质化的地方）|
|---|---|---|
| conv3x3 | `conv3x3_ov`（osv32）、`conv3x3_blk`（OBW/SLM）、`conv3x3_f16`（native）、`conv3x3_cin3` | 无 **输出 tiling / split-K**；大层受延迟/占用限（R41），无跨瓶颈候选 |
| conv1x1 | `gemm`（BM/BN/BK/TM/TN）、`gemm_sk`（split-K）、`conv1x1_blk`、`conv1x1_gemv`（N=1）| N 中等时无 **direct conv1x1**（非 im2col）；小 N 的 tiling 单一 |
| depthwise | `depthwise_v`、`depthwise_blk`、`depthwise_pad`(opt-in) | 无 **向量化/多输出-per-WI** 变体；K=5 的专用谱系缺失 |
| gemm | `gemm`、`gemm_sk` | 无 **SLM staging / 双缓冲 K** 变体谱系 |
| 小算子 | ew_binary/bcast/unary/concat4/copy/slice/pool/resize/permute/bmm/gap/softmax | 多数只有「标量 vs 向量」两变体；缺 **launch 融合**（把相邻小算子折进消费者）|

> 扩展基础设施已就位：`KernelFamily`（supports/candidates/layout/ceiling/hardCeiling/actMask）
> 是声明式的；加族 = 加声明 + 自包含 `.cl`，**不改** Autotuner/dispatch/布局规划/expectedOps
> （见 [`kernel-families.md`](kernel-families.md)）。R47 的 L3 建模让新候选可被**显式评估**。

---

## 2. 扩充维度（按「物理瓶颈 × 契约」组织）

### D1. 输出/空间 tiling 维度（Memory + Occupancy）
- **动机**：R47 标定出「占用 → 有效 L3 容量/带宽断崖」（§9.1），且大层受占用限。对
  conv3x3/conv1x1 增加**输出空间 tiling** 候选（一次只驻留一个子块的工作集），让工作集落在
  ~1MB 有效容量内 → 命中 R47 模型里的「低占用、高带宽」区间。
- **产物**：`conv3x3_ov` 的 `OBH` 谱系扩展 + 新 `-DTILE_OUT=HxW` 变体；`conv1x1` 的 N 维切块。
- **验收**：大 spatial 层（W80/H80）隔离 min 改善；整网 `predictNet` 的 L3 spill 下降。

### D2. split-K / 归约维度（Latency + Grid）
- **动机**：网格饥饿层（20×20 系）受延迟限（R24/R40）。
- **产物**：把现有 `gemm_sk` 思路推广到 conv3x3（**split-K conv**），以及 conv1x1 的多级 split。
- **验收**：小网格层隔离 min 改善 ≥2%。

### D3. 数据通路维度（Fma + Memory）
- **动机**：conv3x3 现在只有 ov/blk/native/cin3；WINograd、direct-vector（CINC 泛化）未覆盖宽通道区。
- **产物**：评估 **Winograd F(2×2,3×3)**（曾判定上限低，用 R47 的内存/占用模型复核是否值得）；
  `CINC` 从 Cin≤4 泛化到任意 Cin%（R33 方向）。
- **验收**：宽通道层隔离 min 改善；数值契约标注（Winograd 改变累加顺序）。

### D4. 布局守护维度（Launch + L3）
- **动机**：R47 §4.6 reorder 负收益的根因是「每层一趟 reorder」；减少 launch 次数是确定性杠杆。
- **产物**：**决策级**先做（已具备 chain move，§4.3 S3）；**内核级**再做「producer 直接写消费者
  所需布局」的**融合 reorder**（把 reorder 折进 conv 的 epilogue/prologue）。
- **验收**：reorder 调用数与 ms 下降；整网 A/B 不回归。

### D5. 小算子融合维度（Launch）
- **动机**：小算子占 mb busy ~50%（§4.7）；多为 launch/带宽受限。
- **产物**：把「P1 epilogue 激活融合」推广到**相邻小算子**（如 `concat4→conv1x1` 已有）；
  候选层面：`ew_binary_bcast` 的通道特化已存在，补 **NHWC 连续读写**变体。
- **验收**：dispatch 数下降；小算子分项 ms 下降。

### D6. 全核内建候选项（Coverage）
- **动机**：`conv1x1` N>1 无 direct；`depthwise` 无多输出-per-WI。
- **产物**：`conv1x1_direct`（非 im2col，直接滑窗）、`depthwise_v2`（多输出/WI + 向量 store）。
- **验收**：在「gemm 不适用的窄通道/小 N」区间拿到隔离 min 改善。

---

## 3. 基础设施前置（做 D1–D6 之前）

1. **候选规模与安全**：候选数增长会放大 IGC JIT 与 GPU HANG 风险。需在
   `candidatesXxx`/registry 层加**每签名候选上限 + 族配额**，并保证 `autotune.py` 分批。
2. **数值契约标记**：`TuningEntry` 需能标注「本候选是否仍逐位一致 / 允许的误差上限」，
   让 `model_check` 能按候选验收（R44 #7 的延伸）。
3. **L3/占用模型接入候选**：D1 的 `TILE_OUT` 必须让 `occupancyPressure` 能解析（否则模型看不到）。
4. **隔离口径为准**：新候选先以**隔离 `min`** 评估（R42 口径，稳定），整网用 `INFVINO_GLOBAL_MODEL`
   + 外部稳态 A/B 裁决（R47 §4.8）。
5. **文档/上限模型**：每个新族须带 `ceiling`/`hardCeiling`，否则 `ratio` 排名失真（R39 教训）。

---

## 4. 里程碑（建议顺序，每步独立可验收）

| M | 内容 | 依赖 | 验收 |
|---|---|---|---|
| M0 | 候选上限/配额 + 数值契约标记基础设施 | — | `tuning_test` PASS；候选数可控 |
| M1 | D1 输出 tiling（conv3x3/conv1x1） | M0 | 大 spatial 层隔离 min↑；L3 spill↓ |
| M2 | D2 split-K conv | M0 | 小网格层隔离 min↑ |
| M3 | D5 小算子融合/向量变体 | M0 | dispatch↓；小算子分项↓ |
| M4 | D3 数据通路（Winograd/CINC 泛化，先离线评估） | R47 模型 | 宽通道层隔离 min↑ 或明确负结果 |
| M5 | D4 内核级 reorder 融合 | M1、S3 chain move | reorder ms↓；整网不回归 |
| M6 | D6 全核内建（direct conv1x1 / depthwise_v2） | M0 | 窄通道/小 N 区间改善 |

> 每步完成后：`model_check`/`reuse_check` 三模型回归 + 锁频稳态 A/B；负结果如实归档。

---

## 5. 风险与边界

- **数值**：Winograd / 融合 / 不同累加顺序会改变逐位结果 → 必须标注并放宽到明确误差口径。
- **GPU 安全**：候选数增长 = IGC JIT 暴露量增长（历史 HANG 主因）→ 严格分批 + `gpu_guard`。
- **收益不确定**：R40/R41 已证 conv3x3 大层是「无 L1 + 128-GRF ILP」的结构墙；D3 可能仍负结果。
  R47 的模型让**负结果可提前离线判断**，减少无效 GPU 实验。
- **不要重蹈 R46**：新候选的整网收益必须用**外部稳态 A/B**（非 in-situ）裁决，避免噪声主导。

---

## 6. 与既有工作的接口

- **注册表**：[`kernel-families.md`](kernel-families.md) 的 `KernelFamily`（加族即加声明）。
- **选择**：R47 模型驱动（`INFVINO_GLOBAL_MODEL`）在候选扩充后才有用武之地。
- **L3 模型**：[`round47-l3-model.md`](round47-l3-model.md) 的占用/带宽曲线用于 D1 的 tiling 目标。
- **reorder 负收益**：[`round47-full-flow-findings.md`](round47-full-flow-findings.md) §4.1/§4.6 是 D4 的动机。
- **分段/预算**：[`profiling-budget.md`](profiling-budget.md) 的三态口径用于验收。

---

## 7. 进展（首轮：M0 + D1 首个候选）

见 [`round48-candidate-expansion-findings.md`](round48-candidate-expansion-findings.md)。

- **M0 已落地**：候选预算（每签名上限 + 族配额，确定性截断 + 审计命令
  `kernel_autotune --candidates`）、数值契约标记（`Candidate`/`TuningEntry`/缓存 JSON +
  `model_check` 按候选放宽）、L3/占用模型解析新 tiling 几何。`tuning_test` PASS。
- **D1 首个候选已落地**：`conv1x1_blk` 的 `-DY_BLOCK`（输出行 tiling，逐位一致）。
  yolo 大 spatial 精确签名隔离 min 比最优既有候选快 **1.0–1.3×**（vs XB4/YB1 基线
  1.2–1.6×）；小 H 变慢（按场景分族）。
- **整网外部稳态 A/B 已完成（见 findings §6）**：yolo **−0.2%**、mb **+0.2%**，均噪声内
  → **收益未确认**。根因（findings §6.2）：大 spatial 1×1 上 **NCHW GEMM 本就快于 blocked**
  （1.1–1.6×），YB 无从被选中——与 reorder 无关。
- **系统设计倒查（见 findings §7）**：reorder 税真实存在（mb ~6% busy、19 次/帧），但根因是
  **布局规划器漏洞**——持久化判据硬编码「消费者输入槽 0」，而 conv1x1/gemm 的激活在槽 1，
  导致 **conv1x1_blk 消费者永不持久化**；叠加 NCHW 的 conv3x3 主族，布局图被切碎。
- **D4 已落地并确认收益（findings §10）**：`LayoutReq.inIndex` 修正激活槽 + SE `Mul`
  （`ew_binary_ch`）生产者直写 fsv16。mobilenet **−2.8%**（交错 8/8 全胜），yolo 噪声内；
  三模型 model_check / reuse_check PASS。**建议继续 D4（conv 生产者直写），暂缓 D1/D2/D3/D6**。
- `config/tuning.json` 未改动（D4 是布局/内核级改动，不改选择）。

---

## 8. 进展（R50：D6 depthwise）

见 [`round50-depthwise-yblock-and-mincut-fix.md`](round50-depthwise-yblock-and-mincut-fix.md)：
`depthwise_blk` 新增 `-DY_BLOCK`（多行/WI，逐位一致）+ 布局契约成本 `#blkfsv16`
（输出 fsv16 比 bfyx 快 ~2–3×）+ 修复 R49 mincut 漏 `Cout%16` 门导致的数值错乱
（并修复 `model_check` 未透传 env 的假 PASS）。mincut（opt-in）mb −3.9% / y11 −2.1%。

