# 算子族管理与接入框架（model-driven kernel families）

> 目标：把「加 kernel」从**散落的 if/else + 手写候选**，变成**声明式的算子族注册表**，
> 让「选哪个族 / 配哪个 config / 用什么布局 / 离物理极限多远」由**统一的物理模型 + 代价**
> 决定。回答两个问题：
> 1. **管理**：一个 kernel 的所有「接口」（支持的 op、布局、约束、knob、权重重排、
>    网格、上限模型）如何在一处声明，被 autotune / dispatch / 布局规划**共同消费**；
> 2. **判断**：对每个 (op, shape)，**哪个族在什么场景更接近物理极限**——是当前族还可以
>    继续压，还是需要换一个新族。
>
> 本文件配套一个**已落地的样例**：移植 OV `convolution_gpu_bfyx_f16_1x1`
> → `kernels/conv1x1_blk.cl`，接入 `kernel_bench --op conv1x1blk`。实测见 §7。

---

## 0. 结论先给

- **价值单元是「算子族」，不是「kernel」**。一个族 = 一个数据通路 × 一套布局契约 × 一段
  可枚举的配置空间。pointwise 1×1 的 3–5× 差距，缺的不是「第 12 个 gemm 变体」，而是
  **blocked 布局下的 pointwise 族**。
- **管理的关键是「声明式 + 单一真相源」**：kernel 的布局要求、约束、knob、权重重排、
  网格、上限模型全部在注册表里声明一次；`Autotuner`、`PlanModel` dispatch、
  布局规划器、报告器都读它。现在这些信息散在 `candidatesXxx` / `PlanModel::run` 的
  if/else / `expectedOps` / 布局规划四处，是**半套框架**。
- **判断「是否到极限」要两把尺子**：当前族的 `familyCeiling`（本族物理上限）与**跨族
  `bestCeiling`**。`measured/currentCeiling` 低 = 本族没写满；`bestCeiling > currentCeiling`
  = 该换族。R33 之前的 `ratio` 只有第一把尺子（且对 blk/native 口径还是坏的）。
- **不要学 OV 的堆法**。OV 的 `kernel_selector` 是「ParamsKey 过滤 + 优先级列表」，它的
  势力范围靠几百个模板覆盖；它没有公开的 per-family 物理模型。infvino 的差异化优势
  （ISA 标定 + 中间标准）恰好可以做成**模型驱动**的选择：**先算上限，再决定值不值得写**。

---

## 1. 为什么 OV 是「堆」而我们不该堆

| | OpenVINO `kernel_selector` | 建议的 infvino |
|---|---|---|
| 选择依据 | `ParamsKey::Support` 布尔过滤 + `KernelsPriority` 静态排序 + cache.json | **约束过滤 + 实测 ms**，且记录每族的 ceiling |
| 布局 | 独立的 `layout_optimizer` + 20+ reorder pass | 族的布局契约 + 一个**总代价最小化**的布局规划 |
| 特化 | 源码级 JIT（宏注入），每个 shape 一套 | 命令行宏 + 离线 autotune（保持可复现/可审计） |
| 物理模型 | 无（有 occupancy/estimate，但不是公开的 per-kernel 指令/roofline 模型） | **有**：ISA 配额 / roofline / launch 地板 |
| 代价 | 几百模板的维护与组合爆炸 | 每族一个自包含 kernel，靠**模型**决定边界 |

**结论**：我们要的不是「更多 kernel」，而是**更少的族 + 每族一个可信的上限模型 +
一个全局选择器**。标准是「这个 shape 上，有没有一个族的 ceiling 明显更高」——有就补族，
没有就别写 kernel。

---

## 2. 现状盘点：已有的「半套框架」

| 机制 | 位置 | 缺什么 |
|---|---|---|
| `OpSignature` | `include/infvino/Tuning.hpp` | 已是稳定 key，✓ |
| `TuningCache` / `Candidate` | `Tuning.hpp` / `Autotuner.cpp` | `Candidate` 只有 kernel/source/options/config；**没有布局、约束、族 id、上限** |
| 候选枚举 | `Autotuner.cpp::candidatesXxx` | 每个 op 一个手写函数，**加族要改多处** |
| `expectedOps` | `Tuning.cpp` | 单函数，对 blk/native 口径失真（R33 §2）；**无「按族」** |
| dispatch | `PlanModel::run` 的巨大 if/else | kernel → 参数绑定/网格硬编码在分支里 |
| 布局 | `PlanModel::planBlockedLayout` | **只覆盖 conv3×3**；1×1/depthwise 不在链里 |
| 权重重排 | `ovWeight` / `blkWeight` / `cin3Weight` | 每个族一段，散落 |

**判断**：基础设施（签名/缓存/计时/中间标准）是齐的，缺的是把「族」抽象出来并把上述
四处收敛到一处。

---

## 3. 数据模型：`KernelFamily` 声明式规格

一个族的所有元信息（建议新文件 `include/infvino/KernelFamily.hpp` + `src/KernelFamilies.cpp`）：

```cpp
enum class Layout { NCHW, FSV16 };          // 可扩展 bfyx/bsv/... 
enum class Bottleneck { Fma, Memory, Launch, Latency, Instruction };

struct FamilyModel {                        // 中间标准（per-family）
  // 该族在 (sig, dev) 上可达的 ops/EU/cyc 上界，及瓶颈类别（决定用哪把尺子）
  std::function<double(const OpSignature&, const ClDeviceInfo&)> ceilingOps;
  std::function<Bottleneck(const OpSignature&)>                  bottleneck;
};

struct Knob { const char* macro; std::vector<int> values; };  // -DOBW=2/4/8 ...

struct KernelFamily {
  std::string name;                          // "conv1x1_blk"
  std::string source;                        // kernels/conv1x1_blk.cl
  std::vector<std::string> ops;              // {"conv1x1"}
  Layout in_layout, out_layout;              // FSV16 / FSV16
  std::function<bool(const OpSignature&)> supports;   // 约束：对齐/stride/groups
  std::vector<Knob> knobs;                   // 枚举轴
  std::function<std::string(const OpSignature&, const Config&)> options;
  RepackKind weight_repack;                  // None | OsIsYxIsv16Osv16 | OsIyxOsv32 | OutCout
  // 网格 + 参数绑定（dispatch 用它，bench 也用同一份）
  std::function<Launch(const OpSignature&, const Config&, ArgSink&)> launch;
  FamilyModel model;
};

const std::vector<KernelFamily>& kernelFamilies();   // 单一真相源
```

- `Config` = 每个 knob 取一组值（现有 `Candidate` 的 config/options 由它生成）。
- `Layout` 是**一等的**：族声明它要什么输入、能产出什么输出；布局规划器据此决策（§4）。
- `supports()` 取代散落的 `if (sig.Cin%16==0)` 之类。
- `model` 取代 `expectedOps` 的单函数分支（§5）。

**加一个新族 = 往 `kernelFamilies()` 加一条 + 一个自包含 `.cl`。** 不碰 dispatch、
不碰 autotuner、不碰布局规划。

---

## 4. 布局：从「per-node kernel」到「全局布局图」

为什么这次非做不可：blocked 1×1 只有**输入输出都是 `b_fs_yx_fsv16`** 时才快（§7）。
若每层都 bfyx↔fsv16 往返，一次 reorder ≈28 µs，会把 0.04 ms 的收益吃掉。所以
「用哪个族」和「张量用什么布局」必须**联合决定**。

建议把 R36（conv3×3 专用）推广成通用布局规划：

```
输入：节点图 + 每族声明的 (in_layout, out_layout)
目标：给每个张量选 Layout，最小化  Σ reorder_cost + Σ Σ_family 布局不匹配惩罚
约束：网络输入/输出固定 NCHW；每族只能吃它声明的布局
```

因为族数量有限、布局二值，可以先用**贪心 + 局部搜索**（保持 R36 的「同帧去重」语义）：
- 一条 A→B→C 的链，若 A、B、C 都只有 blocked 族且通道对齐 → 整条 FSV16，零 reorder；
- 任一张量有非 blocked 消费者 → 该张量保持 NCHW（否则要反向物化）；
- 决策**由 tuning 驱动**（族是否被选中），与 R36 一致，保证「规划=执行」。

> 这正是 OV `layout_optimizer` 做的事，但我们**只维护族声明的布局**，不引入上百 format。

---

## 5. 物理上限模型：per-family 的「中间标准」

现状 `expectedOps` 对 conv3×3 一律用 osv32 几何，导致 blk/native 的 ratio 不可比
（R33 §2 已记录）。模型驱动后的形态：

```cpp
// 每族自己给 ceiling；标尺按 bottleneck 选：
//  Fma        -> 32 × mad_frac（该族 ISA 实测的 mad 占比）
//  Instruction-> 32 × mad_frac（depthwise：mad 本就稀）
//  Memory     -> roofline: 2·out / (EU·clk·(launch + bytes/BW(footprint)))
//  Launch     -> 由 dispatch 数 + launch 地板决定
double familyCeiling(const KernelFamily&, const OpSignature&, const ClDeviceInfo&);
```

对每个节点同时算两把尺子：

| 量 | 含义 | 用途 |
|---|---|---|
| `measured/currentCeiling` | 当前族写满了多少 | <1 → 本族还有工程空间（ILP/占用/指令） |
| `bestCeiling/currentCeiling` | 有没有族上限更高 | >1 → **该换族**（1×1：gemm ceiling ~4 vs blocked ~?） |
| `bestCeiling/`peak | 该 shape 离硬极限多远 | 决定是否值得投入 |

**「scenario → closer to limit」** 即：按 `OpSignature` 的特征（HW 大小、K、通道对齐、
layout 是否已 blocked）分桶，记录每桶的**胜出族 + 其 ratio**。例如 1×1 的经验分桶（§7）：

| 场景 | 特征 | 胜出族 | 原因 |
|---|---|---|---|
| 小 HW、大 K、通道对齐 | HW≤~64, K≥128, C%16=0 | **blocked 1×1** | 复用 + coalesced，网格够 |
| 大 HW、小 K | HW≥~1600, K≤64 | NCHW GEMM | blocked 网格/每 WI 工作量不占优 |
| N=1 | HW=1 | GEMV split-K | lane 沿 K 归约 |

这个桶表本身就是「系统性调优」的产物：**不是给每个 shape 试所有 kernel，而是先按特征
定位场景，再用模型判断该场景该归哪个族。**

> **基础设施缺口**：当前 dev 镜像的 `ocloc` 缺 IGA（`couldn't load iga`），无法离线
> 得到 blocked 1×1 的 mad 占比 → `ceilingOps` 暂时标「待标定」。**这是框架的前置依赖**：
> 要么在镜像里补 IGA，要么恢复 R24 的离线反汇编环境。否则新族没有可信上限。

---

## 6. 选择流程（把上面拼起来）

```
对每个 op 节点：
  1. layoutPlanner 先定每个张量的布局（依赖各族 in/out_layout + tuning 选择）
  2. candidates = ∪_family { f.supports(sig) ? f × knobs(sig) : ∅ }
  3. autotune：逐候选实测 ms，取最小 / 或缓存命中
  4. 记录：family、measured、familyCeiling(current)、bestCeiling
  5. 报告：ratio_self = measured/familyCeiling；ratio_x = bestCeiling/familyCeiling
```

- 第 3 步仍是**实测选优**（保持现有可靠性）。
- 第 4/5 步是新增的**诊断输出**：`kernel_autotune --report` 直接打印
  「本层：当前族 X，self ratio 0.42，跨族上限 Y/Z 使 ratio_x=3.1 → 建议补 blocked 族」。
  这让「该不该写新 kernel」变成**数据**而不是直觉。

---

## 7. 已落地样例：blocked 1×1（`kernels/conv1x1_blk.cl`）

**移植**：OV `convolution_gpu_bfyx_f16_1x1`（lane=输出通道 / `b_fs_yx_fsv16` I/O /
`os_is_yx_isv16_osv16` 权重 / `X_BLOCK` 连续列 / 可选 `SLM_DIV` split-K）。自包含，头部
保留 Intel 版权 + Apache-2.0，见 `THIRD_PARTY_NOTICES.md`。

**数值**（`kernel_bench --op conv1x1blk --verify`，vs FP32 numpy 口径）：

| config | mean_rel | max_rel(amax) | 判定 |
|---|---:|---:|---|
| XB=2/4/8，act=0，Cin=576,Cout=96,7×7 | 3.13e-3 | 5.43e-3 | PASS |
| XB=4，act=3(HardSwish) | 3.66e-3 | 6.70e-3 | PASS |
| Cin=51（非 16 倍数），Cout=64 | 9.26e-4 | 2.21e-3 | PASS |
| SLM_DIV=2（split-K） | 2.33e-3 | 2.92e-3 | PASS |

**性能 A/B**（同会话，`--iters 20`；blocked 的输入/权重已预重排、不计 reorder）：

| shape (Cin,Cout,H,W) | M,N,K | blocked | NCHW gemm（现生产路径） | 加速 |
|---|---|---:|---:|---:|
| 576,96,7,7 | 96,49,576 | **0.041 (XB2)** | 0.208 | **5.1×** |
| 96,576,7,7 | 576,49,96 | **0.030 (XB4)** | 0.085 | **2.8×** |
| 240,40,14,14 | 40,196,240 | **0.022 (XB2)** | 0.140 | **6.4×** |
| 128,64,40,40 | 64,1600,128 | **0.071 (XB4)** | 0.135 | **1.9×** |
| 64,64,80,80 | 64,6400,64 | 0.214 (XB4) | **0.134** | 0.63×（**gemm 仍胜**） |

读法：
- mobilenet 的 pointwise（小 HW、K 大）上 blocked **5–6×**；与上一轮 OV 逐算子剖析
  （OV pointwise 0.42 ms vs infvino 2.0 ms）一致。
- **大 HW、小 K** 时 gemm 仍胜——所以这是**按场景分族**，不是「替换」。
- 不计 reorder 是**故意的**：它正是 §4 布局规划要解决的问题，也说明**单看 kernel 会高估收益**。
- `SLM_DIV=4` 在同一 shape 上进一步到 **0.024 ms**（比 XB2 再快 ~1.7×）——**split-K 也是
  按场景的 knob**，说明「每族的 knob 空间」本身要被注册表声明和枚举。

**仍未做**：`PlanModel` 的 dispatch/权重重排/布局规划尚未接入这个族；`ocloc` 缺 IGA 无法
标定它的 `ceilingOps`。

---

## 8. 接入路线图（按 ROI，可分阶段）

**Phase 1 — 注册表与报告（低风险，不改运行时数值）**
1. 新增 `KernelFamily.hpp` + `KernelFamilies.cpp`：把现有 `conv3x3`（ov/blk/native）、
   `gemm`、`conv1x1`（kernels gemm/gemv）、`depthwise`、小算子**声明**进去；
   `candidatesXxx` 改为 `candidates(sig)` 读注册表。
2. `expectedOps` → `familyCeiling(family, sig, dev)`；`TuningEntry` 增
   `family`、`familyCeiling`、`bestCeiling` 字段；`kernel_autotune --report` 打印 §6 的
   诊断。**先只报告，不改变选择**，用现有缓存回归验证。

**Phase 2 — blocked 1×1 端到端接入**
3. `PlanModel::blk1x1Weight`（`os_is_yx_isv16_osv16`，1×1 特例）+ dispatch 分支
   （读缓存选中的 `conv1x1_blk`）+ `candidatesConv1x1` 增族。
4. 布局规划器从「conv3×3 only」推广到「按族 in/out_layout」，先只接 1×1 链。
5. 逐层 `kernel_autotune --retune` + `model_check` 回归（mobilenet 优先，它 76% 是 1×1）。

**Phase 3 — depthwise blocked + 全链持久布局**
6. 移植 OV `convolution_gpu_bfyx_f16_depthwise`；把 1×1→depthwise→1×1 连成持久 fsv16 链。

**每阶段验收**：`model_check` PASS + A/B 逐位一致 + `reuse_check` PASS。

---

## 9. 不做 / 风险

- **不引入 OV 的 format/reorder 全家桶**。布局是二值（NCHW/FSV16），决策由族声明驱动。
- **不做在线 JIT 特化**（保持离线 autotune 的可复现/可审计取舍）。
- **不一次重构 `PlanModel::run`**。先让注册表成为「真相源」，再逐族把 dispatch 搬过去；
  避免大爆炸式改动。
- **布局规划数值风险**：任何持久布局改动都必须配「独立数值用例 + A/B 逐位一致」，
  吸取 R30 §7.7 多消费者 alias FAIL 的教训。
- **`ocloc`/IGA 缺失**是 Phase 1 的硬前置：没有它，新族的上限模型只能拍脑袋。

---

## 10. 待决策

1. 注册表落地深度：只做「报告用」（Phase 1）还是直接驱动选择？
2. blocked 1×1 先接 mobilenet 还是先做通用布局规划？
3. depthwise 是否同期移植（它和 1×1 共享布局，一起做才连成链）？
4. IGA 环境：补镜像，还是先把 ceiling 模型降级为「实测 + roofline」？

---

---

## 12. Phase 2 进展（本轮）

按「注册表直接驱动选择 + 先做通用布局」落地了第一个垂直切片。**默认运行时数值零回归**
（未重扫的缓存下输出与基线逐位一致；`model_check` PASS）。

### 12.1 已落地

| 项 | 内容 |
|---|---|
| **注册表** | `include/infvino/KernelFamily.hpp` + `src/KernelFamilies.cpp`：族声明（op/布局契约/瓶颈/`supports`/`candidates`/`ceiling`）。`Autotuner::candidatesConv1x1` 追加 `candidatesFromRegistry(sig)` —— **新族只在这里声明**，不再改候选函数。 |
| **per-family 中间标准** | 每族自带上限模型（conv3x3 ov/blk/native、gemm、depthwise、conv1x1_blk），修 R33 §2 的「blk/native 用 osv32 几何」失真。 |
| **通用布局规划** | `planBlockedLayout` 改为 `nodeFamily(node)`（由 tuning 选中的族决定 in/out 布局契约）+ 「所有消费者都吃 FSV16 才持久化」，R36 从 conv3x3-only 推广到任意声明了布局的族。 |
| **blocked 1x1 端到端** | `PlanModel::blk1x1Weight` + `run()`/`autotune()` dispatch 分支 + `blkInput` 复用；候选/权重/网格全通。 |

### 12.2 发现并修掉一个真 bug（值得记）

启用 blocked 1x1 后整网 `model_check` **FAIL**（mean_rel 1.48）。逐层定位：
- 该层单独 bench **PASS**，且单层输出与 gemm **逐位一致** → 一度排除 kernel；
- 关 `NO_BLOCK_LAYOUT`/`NO_POOL`/`NO_LAUNCH_CACHE` 仍 FAIL → 排除布局/池/缓存；
- 真正根因：`parse()` 的 **`fuseResidualAdd()` 会把 `conv1x1→Add` 折进 conv**，节点输出被改名成 `Add_output_0` 且 `ins[3]` 带上残差；**`conv1x1_blk` 不支持 RES**，而 gemm 是运行时动态加 RES epilogue。于是带残差的 pointwise 丢了残差。
- 教训：**用 plan 文本解析来核对运行时节点是不可靠的**（fusion 发生在 parse 期）；必须用运行时 `outs[0]`/debug 打印。

**修复**：`conv1x1_blk.supports` 排除 `groups==2`（残差）；`run()` 里若同签名缓存给了 blk 而节点带残差，退回启发式 gemm。

### 12.3 实测（mobilenet，同会话 A/B，`--iters 10`）

只把 2 个**非残差** pointwise 层切到 blocked 1x1：

| 项 | baseline | Phase 2 | |
|---|---:|---:|---|
| busy | 2.658 ms | **2.394 ms** | **−9.9%** |
| `conv1x1@576x49x96` | 0.189 | **0.094** | 2.0× |
| `conv1x1@96x49x288` | 0.131 | **0.025** | 5.2× |
| reorder(blk) | 0 | 0.023（×4） | 可忽略 |
| `model_check` | PASS | **PASS**（mean_rel 1.01e-2） | |

带残差的投影层（`Cout96_N49_Cin576` 等）仍走 gemm（blocked 1x1 暂不支持 RES）。

### 12.4 下一轮（Phase 2 收尾）

1. **`conv1x1_blk` 支持 RES**（复用 gemm 的动态 RES 语义）——放开残差 pointwise，mobilenet 收益会明显更大。
2. 移植 **`depthwise_blk`**（OV `convolution_gpu_bfyx_f16_depthwise`），把 `1x1→depthwise→1x1` 连成持久 fsv16 链（当前每个 blocked 1x1 仍付一次 `reorder(blk)`）。
3. 注册表**接管现有族的候选 config**（现在仍在 `Autotuner.cpp`，属迁移余量）。
4. 用注册表统一 **activation 码**（conv3×3 与 1×1 的 act 码不一致，是隐式接口）。
5. 全量 `kernel_autotune --op conv1x1 --retune` 重扫（本轮的 A/B 只重扫了 2 层，缓存未落库）。

---

## 13. Phase 2 完成（items 1–5）

按顺序做完 5 项，全部 `model_check` PASS、`reuse_check` PASS。

| # | 项 | 结果 |
|---|---|---|
| 1 | **`conv1x1_blk` 支持 RES** | 加残差参数 + `-DRES`（激活后加，语义同 gemm）；算子级 act0/3/4、`OUT_FSV16`、split-K 全 PASS。带残差的 pointwise（`Add` 折入）现在也能走 blocked。 |
| 2 | **移植 `depthwise_blk` + 连链** | 移植 OV `convolution_gpu_bfyx_f16_depthwise`（lane=通道、fsv16 I/O、`[C/16][K][K][16]` 权重）。**关键优化**：每行输入只加载一次到寄存器、跨 K 个 kw 复用（初版每 tap 重读，慢 2×）；改用后与 `depthwise_v` 互有胜负。布局规划器新增 depthwise 族，`1x1→depthwise→1x1` 可持久 fsv16。 |
| 3 | **注册表接管候选 config** | `src/KernelFamilies.cpp` 成为**单一真相源**：conv3x3（ov/cin3/blk/native）、gemm/gemm_sk、conv1x1（gemv/blk）、depthwise（f16/v/vp/blk）全部声明在注册表；`Autotuner` 的 `candidatesXxx` 退化为 `candidatesFromRegistry` 薄包装。`autotuneOp` 的 `expected` 改用**胜出族**的上限模型（修 R33 口径）。 |
| 4 | **激活码契约** | `KernelFamily.actMax`：conv3x3 {0..2}、depthwise {0..4}、其余 {0..5}。注册表据此拒绝「act 码超出本族语义」的节点，杜绝把 1x1 的 ReLU(=2) 当 conv3x3 的 HardSwish(=2) 之类的静默错误。完整语义统一（canonical 码 + onnx2plan 映射）见 §14。 |
| 5 | **全量 retune conv1x1** | 三模型 `--op conv1x1 --retune`（+depthwise）。**mobilenet 大赢、yolo 回退**，见下。 |

### 13.1 关键工程发现：blocked 族的「reorder 税」

blocked 族的 net 收益 = kernel 增益 − 每帧 `bfyx→fsv16` 重排（未持久化时）。tuner 只按
kernel ms 选，会在 reorder 主导时误选。**落地了「保守计费」**：`conv1x1_blk`/`depthwise_blk`
候选一律加上实测的一趟 reorder 成本后再与 NCHW 族比较（不依赖构造期布局，因为 retune 后
布局会变）。

即便如此，**全量 retune 暴露了场景边界**：

| 模型 | baseline busy | +blk | 判定 |
|---|---:|---:|---|
| mobilenetv3-small | 2.62 ms | **1.68 ms（−36%）** | ✅ 1×1 是小 HW(7×7/14×14)、大 K，kernel 增益 ≫ reorder |
| yolov8n-pose | 13.17 ms | 13.30 ms | ❌ 大 N(1600/6400/25600) 1×1：gemm 已饱和，reorder 主导 → **保留 baseline** |
| yolo11n-pose | 14.15 ms | 14.18 ms | ❌ 同上 |

**决策**：`config/tuning.json` 只落 mobilenet 的 blocked 收益（+depthwise），yolo 保持基线
（其 1×1 需要**更广的 fsv16 持久链**或 reorder-aware 的布局联合搜索才能翻正，属 §14）。

```bash
# 最终 A/B（同会话，--iters 15）
mobilenetv3-small  2.622 -> 1.677 ms
yolov8n-pose      13.170 -> 13.296 ms（噪声内）
yolo11n-pose      14.152 -> 14.182 ms（噪声内）
```

### 13.2 §12.4 旧清单的收口

- ① RES：**已做**（本 §1）。
- ② depthwise_blk + 链：**已做**（本 §2），但 yolo 场景 reorder 主导，默认只对 mobilenet 生效。
- ③ 注册表接管候选：**已做**（本 §3）。
- ④ 激活码：**部分**（契约边界，见 §4）；canonical 化留 §14。
- ⑤ 全量 retune：**已做**，结论见 §5/13.1。

---

## 14. items 1–4 收口（本轮）

| # | 项 | 结果 |
|---|---|---|
| 2 | **reorder-aware 布局联合** | **根因是注册表的一个 bug**：`candidatesFromRegistry` 按 `f.op == sig.op` 过滤，把 `gemm_f16`（op="gemm"）从 conv1x1 候选里排除了 → conv1x1 丢失最强族、blk 被误选 → yolo 回退。改为**只按 `supports` 过滤**（一个族可服务多 op）后，yolo 回退消失。配合「blk 候选保守计费一趟 reorder」+ autotune 末尾重新规划布局，选择即安全。**实测 2-pass 布局不动点在本机无额外收益**，故保留保守单遍。最终：mb **2.62→1.72 ms**、y8 13.23（持平）、y11 **14.15→13.69 ms**。 |
| 4 | **容器内 IGA** | 根因：`ocloc`（经 `libocloc.so`）`dlopen("libiga64.so")`（**无版本号**），而镜像只有 `libigc1` 提供的 `libiga64.so.1`；host 能反汇编只是因为有 oneAPI debugger 带了无版本号那份。**Dockerfile 加 `libigc-dev`**（提供 `libiga64.so -> libiga64.so.1`）。据此离线标定新内核：`conv1x1_blk` 主循环 mad 占比 **36%（XB8）→ 配额 11.5**；`depthwise_blk` K5 **12.6%**，替换掉此前的估计值。 |
| 1 | **canonical 激活码** | 统一到 conv1x1/gemm 空间：**1=SiLU 2=ReLU 3=HardSwish 4=HardSigmoid 5=Sigmoid**。改 `onnx2plan`（`ACT_CODE_CONV`/`ACT_CODE_GENERAL`）+ `conv.cl`/`conv_ov.cl`/`conv_blk.cl`/`conv_cin3.cl`/`conv_general.cl`/`depthwise_blk.cl`。注册表用 `actMask` 精确声明每族支持集（conv3x3 `{0,1,3}`、depthwise `{0..4}`、其余 `{0..5}`）。三模型 `model_check` PASS；旧缓存因 sig 键里的 act 码变化自然失配（无 ABI 破坏），重扫 depthwise 后性能保持。 |
| 3 | **注册表接管小算子** | `smallCandidates`（ew_binary/ew_unary/bcast/concat4/copy_c/slice_axis/maxpool/resize_nn/permute/bmm/softmax/gap）迁入注册表，`candidatesSmall` 退化为薄包装。**收益**：候选枚举 100% 单一真相源（加族/改候选不改 Autotuner）。**管理**：小算子无布局/权重重排，作为 `bottleneck=Memory`、`ceiling=expectedOps(sig)` 的族登记即可，成本很低。 |

## 15. 仍然开放的项

1. **blocked 族的全局布局联合搜索**：现在是「保守计费 + 选后规划」；理论最优需要 (选族, 布局) 联合搜索（2-pass 不动点本机无收益，但更大模型/更多 blocked 族时可能不同）。
2. **动态 shape / i8-u8 / USM/remote tensor**：当前非瓶颈。
3. **kernel 二进制缓存与 registry 的 `source_hash` 联动**：已有 program cache，未与族声明哈希绑定。

---

## 11. 复现

```bash
# 构建（容器，CPU）
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake -S . -B build-blk -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build-blk -j4 --target kernel_bench'

# 数值（遵守 benchmark_protocol.md）
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'export LD_LIBRARY_PATH=$PWD/build-blk; \
  ./build-blk/kernel_bench --op conv1x1blk --conv1x1blk 4,1,0,0 \
    --conv-shape 576,96,7,7 --iters 3 --verify'

# 性能 A/B（blocked vs NCHW gemm）
#   blocked: --op conv1x1blk --conv1x1blk 2,1,0,0 --conv-shape 576,96,7,7
#   gemm   : --op gemm --shape 96,49,576
```
