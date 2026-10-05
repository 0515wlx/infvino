# Round 47 全流程实测：效果、系统缺陷倒查、与「主要矛盾是否在 blocked chain」的判定

> 承接 [`round47-tvm-strategy.md`](round47-tvm-strategy.md)（TVM 借鉴 + 实现）。本轮在锁频
> 开发板上跑通**三模型全流程**（隔离重扫 + R47 整网回验 + per-plan 工件），做稳态 A/B、
> 逐节点归因与数值/复用回归，并倒查系统 bug/设计缺陷。
>
> 结论先说：**R47 的框架跑通、无数值回归、无 GPU HANG**；但**全局步本身仍无可靠正收益**
> （与原 R46 一致）。唯一有可复现收益的是 **mobilenetv3-small（−2.2%）**，而它恰好是布局耦合
> （reorder/fsv16）最大的模型——**blocked chain 的假设对「布局耦合型模型」成立，但对 yolo 不成立**
> （yolo 的 reorder <2%、fsv16≈0，全局搜索结构上无空间可挖）。

---

## 1. 实验设置

- 锁频：`scripts/gpu_clocks.sh lock`（min=max=RP0=1300）全程。
- 安全：每个容器 `--memory=2~3g --pids-limit=256 --device=/dev/dri/renderD128` + `timeout`；
  分批（`--batch 2`）；跑后 `gpu_guard.sh after`。**全程 0 次 GPU HANG**。
- 二进制缓存：`INFVINO_PROGRAM_CACHE` 挂持久目录（消除每进程冷 JIT 8–12s）。
- 生产缓存 `config/tuning.json` **不动**；实验用临时缓存 `models/tuning_r47_<m>.json`（从生产拷贝），
  per-plan 工件写入 `models/<m>/model.plan.tuning.json`（实验后删除）。
- 命令（安全驱动）：
  ```bash
  python3 scripts/autotune.py --model <m> --batch 2 --iters 4 \
      --global --global-topk 2 --global-iters 4 --global-rounds 2 \
      --global-budget 1500 --cache models/tuning_r47_<m>.json
  ```
- 稳态 A/B：`kernel_run --report --iters 15`，4 组**交错**（base/iso/pponly/full 轮转）取 busy。

---

## 2. 结果

### 2.1 稳态 A/B（锁频，4 交错 rep，ms）

| 模型 | base(med) | iso(med) | pponly(med) | full(med) | full vs base |
|---|---:|---:|---:|---:|---:|
| mobilenetv3-small | 1.6505 | 1.6165 | 1.6155 | **1.6145** | **−2.2%** |
| yolov8n-pose | 10.527 | 10.529 | 10.5425 | 10.5475 | +0.2%（噪声）|
| yolo11n-pose | 11.519 | 11.549 | 11.521 | 11.548 | +0.3%（噪声）|

- `iso`=仅隔离重扫（`--retune` 更新共享缓存）；`pponly`=仅 per-plan 工件；`full`=两者。
- **mb：`full ≈ iso ≈ pponly`（都在 −2% 带内）**→ 收益来自**隔离重扫刷新缓存**，全局步边际 ≈0。
- y8/y11：四者完全重合，全局步既无收益也无损失。

### 2.2 逐节点归因（`kernel_run --profile-json`，base→full）

| 模型 | blk 族 Δ | 非 blk Δ | reorder（占比）| fsv16 张量 |
|---|---:|---:|---:|---:|
| mb | **−4.6%**（0.613→0.585）| +0.0% | **+4.6%**（8.3%→8.9%）| 3→5 |
| y8 | +3.0%（仅 5 节点，2.3%）| −0.8% | ≈0（0.6%）| 0→0 |
| y11 | −1.9%（16 节点，7%）| −0.9% | ≈0（1.8%）| 0→1 |

- **mb 的收益全部来自 blk 族**；但 **reorder 同步上升**，抵消了一部分——这正是跨 op 布局耦合。
- `iso→full`（全局步的边际）在 mb 上反而把 blk +0.9%、reorder +10.8% 改差 → **全局步测不准这个残差**。
- y8/y11 的 reorder 占比 <2%、fsv16≈0 → **没有可挖的耦合残差**。

### 2.3 数值 / 复用回归

三模型 `model_check` **PASS**、`reuse_check` **PASS**；mb 在 R47 选择下单独数值亦 PASS。

---

## 3. 倒查：系统 bug 与设计缺陷

| # | 类型 | 问题 | 状态 |
|---|---|---|---|
| 1 | 工具 bug | `autotune.py` grep 了 `global-retune` 却从未 `export INFVINO_GLOBAL_RETUNE_REPORT` → 整网回验日志**从不出现**（可观测性黑洞）| **已修** |
| 2 | 工具 bug | `autotune.py` 绝对 `--cache /tmp/x.json` 被拼成容器内 `/workspace/infvino//tmp/x.json` → 缓存**写到错误位置甚至静默丢失**（R46 文档正是这么用的）| **已修**（bind-mount 目录 + 原样传路径）|
| 3 | 设计缺陷 | **可加目标未真正减少测量**：非耦合签名仍逐候选端到端回验；`predictNet` 只用于排序/分组。y8/y11 无可加残差却仍花整网预算 | 待落地（见 §5）|
| 4 | 测量缺陷 | **in-situ 与稳态口径仍错配**：mb 批内 `base(med)` 1.62–1.80（波动 ±5–9%），外部稳态 1.65；真实收益 ~2% 淹没在噪声内。min+median+交错只缓解**相对**漂移，绝对口径仍不可比 | 部分（R47 双口径已落；绝对口径待解）|
| 5 | 噪声 | 隔离候选离散度大：y8 `gemm` winner spread **17–19%**、2/12 候选 >15%；mb depthwise **48–54%**。选择本身不可靠 | 待治理 |
| 6 | 设计张力 | `globalRetune` 把**所有** target（含未改变者）写进 per-plan 工件作为分批进度标记，会**冻结隔离默认**、掩盖后续 fixpoint 变化 | 待评估 |
| 7 | 两套口径 | 全局步回验期间绕过 `resolveLayoutChoices`（只 `planBlockedLayout`，直接赋 `node_choice_`），持久化后运行时又交回不动点 → 仍是两套决策口径（R44 #5 未根治）| 待评估 |
| 8 | 文档/语义漂移 | `--global-budget` 语义在 R47 改为「整网执行次数」，与 `round45` 的「测量次数」描述不一致 | 已在 R47 doc 注明 |

> 其中 **#3、#4 是「全局步选不出净收益」的直接机制**：不是算法不够聪明，而是
> (a) 该少测的地方没少测，(b) 测出来的是不可比的量。

---

## 4. 主要矛盾是否集中在 blocked chain？

**条件成立**：

- **mobilenetv3-small：是。** 它是唯一有可复现收益的模型（−2.2%），且收益全部来自 **blk 族**；
  其 reorder 占 busy **8–9%**、fsv16 持久张量 3→5。全局步能改 blk 内核，却**无法消掉随之而来的
  reorder**——因为缺的正是 **持久 fsv16 blocked chain**（让 blk 消费者整链直接读写 fsv16、
  reorder 归零）。reorder 的 8–9% 就是 blocked chain 的机会上限（全消则 −8%；部分则可观）。
- **yolov8n / yolo11n：否。** reorder 仅 0.6%/1.8%、fsv16≈0，耦合残差可忽略；主体是
  conv3x3 的 native/OV 通路（占比 >65%），瓶颈是**微内核天花板**（R40/R41 已定量为
  无 L1 + ~150cyc 访存延迟卡在 128-GRF 的单线程 ILP），**与 blocked chain 无关**。

**综合判断**：blocked chain 是 **「布局耦合型模型」（mb）的主矛盾**，也是整网搜索唯一
「算法上说得通、且有量化机会上限」的方向；但对 yolo 不是主矛盾。这也解释了 R46/R47
全局步的整体中性——**平均效应被两个「无耦合」模型稀释，而唯一有耦合的 mb 又因缺
blocked chain 只能拿到部分收益**。

---

## 4.1 reorder 为什么会产生负收益（实测 + 代码机理）

### 机制

infvino 全图默认保持普通 **bfyx(NCHW)**；blk 族（`conv3x3_blk / conv1x1_blk / depthwise_blk`）
需要 **`b_fs_yx_fsv16`**。因此每个「输入不是已持久 fsv16」的 blk kernel，每帧都要插一趟
`reorder_bfyx_to_fsv16`——它是**纯 gather/scatter：整张量 1 读 + 1 写 + 一个独立 dispatch**
（`kernels/conv_blk.cl:300`），**与 kernel 的计算增益无关**，只随张量字节数和 launch 走。

持久化规则是**全或无**（`planBlockedLayout`，`src/PlanModel.cpp:1003-1009`）：张量只有当
**生产者能写 fsv16 且其所有消费者都只吃 fsv16（且读 slot 0）**时才持久；否则保持 bfyx，
于是**每个 blk 消费者都要自己付 reorder**。

### 实测（mb，`kernel_autotune --retune` 写出的 `#blk/#non/#reorder`）

| 签名 | blk(ms) | non(ms) | reorder(ms) | blk+reorder | 判定 |
|---|---:|---:|---:|---:|---|
| `conv1x1\|Cout240_N196_Cin40` | 0.0253 | 0.0387 (gemm) | 0.0042 | 0.0295 | blk 划算 |
| `conv1x1\|Cout576_N49_Cin96` | 0.0291 | 0.0604 (gemm) | 0.0047 | 0.0339 | blk 划算 |
| `conv1x1\|Cout24_N784_Cin72` | 0.0180 | 0.0454 (gemm) | 0.0114 | 0.0294 | blk 划算 |
| **`depthwise\|W14H14s1p2_Cin240_Cout240_K5`** | 0.0296 | 0.0348 (`depthwise_v`) | **0.0085** | **0.0381** | **净负** |

**结论**：负收益只在「**带宽受限 op + blk 增益小**」时出现。1×1 的 non-blk 回退是 `gemm`（慢 1.5–2×），
blk+reorder 仍赢；**depthwise 的 non-blk `depthwise_v` 本就不慢（内存墙为主，地址/边界开销小）**，
blk 只快 ~0.005ms，而 reorder 要 0.0085ms → **净 +0.0033ms**。即：**blk 的收益跑不过一趟额外全张量访存**。

### 为什么系统还会选中净负的 blk（真正的病灶）

1. **族间计费口径不一致（已定位，S1 修正）**：`conv3x3` 的 base 判据是 `min(blk, non)`（`PlanModel.cpp:3263-3264`，**不计 reorder**）；
   而 `conv1x1`（`3591-3592`）与 `depthwise`（`3775-3776`）早已用 `eff = blk + reorder` 计费。
   **先前误写为「conv3x3/conv1x1 都不计」，实际只有 conv3x3 漏计**——已按 S1 对齐（见 §4.3）。
2. **布局 fixpoint 常常不生效**：reorder-aware 的联合不动点 `resolveLayoutChoices` **要求缓存里有
   `#blk / #non / #reorder`**（`1047-1055`）；而**生产 `config/tuning.json` 里 513 条全是 base，一条 `#` 都没有**
   → `any=false` → 退化成纯 `planBlockedLayout`，**reorder 从不参与决策**。
3. **`globalRetune` 绕过 fixpoint**：回验期直接 `node_choice_=assign` + `planBlockedLayout`（`3963-3964`），
   只按**端到端 busy** 接受；而端到端噪声 **±5%** 远大于 reorder 的符号量级（~2–4%）
   → 会接受「kernel 更快但净负」的 blk（正是 R46 实测的 `depthwise_v → depthwise_blk`、reorder +0.036）。
4. **耦合（用户担心的点）**：持久化是 producer→consumers 的**全或无**联合条件，而成本模型是**逐节点**
   的（`blkCost = blk + reorder`，不知邻居的选择会让 reorder 变免费）。这是**反协同博弈**：贪心
   best-response 可能停在「付了 reorder 却永远摊不掉」的混合解，**命中不了「整条链一起切 blk → reorder=0」的联合最优**。

### 对 blocked chain 的判定

- blocked chain **是正确方向**：若让生产者直接写 fsv16 并把链传下去，depthwise 的输入 reorder 归零，
  `depthwise_blk`（0.0296）就重新赢过 `depthwise_v`（0.0348）——**把净负翻成净正**；mb 的 reorder
  占 busy **8.3%**，即机会上限。
- 但**耦合风险确实是真实的、且是核心难点**（全或无 + 跨 op 联合），并且**收益是模型相关的**
  （mb ~8%、yolo <2%）。因此**不建议直接上完整 blocked-chain kernel 改造**，按下面分期。

## 4.2 分期决策（先低复杂度，再决定是否动 kernel）

| 阶段 | 动作 | 复杂度 | 收益预期 |
|---|---|---|---|
| **S1** | **统一计费口径**：让 `conv3x3/conv1x1` base 也按 `blk+reorder` 计费（与 depthwise 一致）；保证 fixpoint 所需 `#blk/#non/#reorder` 始终可得（缺失时即时估算）| 低 | 消除「不计 reorder」导致的净负 blk；让生产缓存也能正确退 blk |
| **S2** | **`globalRetune` 尊重布局契约**：不再用「绕过 fixpoint 的裸 end-to-end」拍板；改成对**耦合连通分量**做联合 move，接受前用 fixpoint 的布局模型核对「是否维持/建立持久化」，端到端只做最终验收 | 中 | 让已有分组骨架真正表达「整链一起切」 |
| **S3** | **决策级 chain move（opt-in，按 reorder 占比门控）**：把一条 producer→consumers 链整体切 blk 作为**一个 move** 端到端验证；mb 开启、yolo 跳过 | 中 | 拿到 §4.1 的联合最优，量化真实收益 |
| **S4** | **完整 blocked chain（kernel/图改造）**：仅当 S3 在稳态 A/B 上给出可靠正收益再上 | 高 | 消除残余 reorder（≤8%）|

> **据此的建议**：**暂不上完整 blocked chain**。先做 **S1+S2**（低成本、直接堵住负收益来源），
> 用 **S3** 的 opt-in chain move 在 mb 上量化「整链持久化」的真实端到端收益；只有在 S3
> 于锁频稳态 A/B 上可复现为正、且结算清耦合风险后，才投入 S4。

## 4.3 实施记录（S1–S4）

### S1 统一计费口径 —— 已落地（待全量 retune 验证）

- **修改**：`src/PlanModel.cpp` conv3x3 分支——先把该节点输入的一趟 `reorder` 测出来，
  再按 **`blk+reorder`** 选 base（存 kernel-only ms），与 conv1x1/depthwise 完全一致。
- **发现问题（记录）**：先前 findings 误写「conv3x3/conv1x1 都不计 reorder」；实测代码
  `conv1x1` 早已按 `blk+reorder` 计费（`3591-3592`），**只有 conv3x3 漏计**。已更正文档。
- **验证（GPU 定向 retune, y11 conv3x3×3）**：base 选择已与计费规则一致——三签名
  `blk+reorder` 均 > `non`，base 都选了 non（`conv3x3_f16 / conv3x3_ov / conv3x3_cin3`），
  `#blk/#non/#reorder` 三键齐全。构建通过、`tuning_test` PASS。
- **发现（记录）**：conv3x3_blk 普遍慢于 OV（实测 `0.125 vs 0.108`、`0.313 vs 0.257`、
  `0.884 vs 0.213`），所以该修复对 **yolo 实际影响有限**（与 yolo `fsv16=0` 一致）；
  真正的负收益来自 conv1x1/depthwise + `globalRetune` 绕过布局契约（→ S2）。
- **发现（记录，独立 bug 线索）**：`reorder_bfyx_to_fsv16` 按 fsv16 把通道**补齐到 16 的倍数**，
  对 `Cin<16` 的输入有**写放大**：`Cin=3` 的 320×320 输入 reorder 实测 **3.55 ms**（≈写的
  1.6M half vs 读的 0.31M），是 stem conv 本身（0.21 ms）的 16 倍。小 `Cin` 层应避免经由
  fsv16 reorder（除非上游能直接产出 fsv16）。
- **残留（S1 的边界）**：`#blk/#non/#reorder` 只在 `autotune(merge)` 时写入；生产
  `config/tuning.json` 无这些键 → 布局 fixpoint 仍退化。S1 保证 **base 条目本身**已
  reorder-aware（即使无 `#`），但要让**跨节点** fixpoint 生效需一次**全量 retune**
  重新生成缓存（S2/S3 会用到）。

### S2 globalRetune 尊重布局契约 —— 已落地

- 新增 **契约门** `kPredTol = 1%`：候选的**可加目标预测净**（含 reorder）若比现状预测差 >1%，
  直接拒绝、**不做端到端测量**。这堵住「多付 reorder、全靠端到端噪声过关」的净负 blk
  （R46 的 `depthwise_v→blk`）。
- 设计要点：该门只**拒绝**、不放宽接受，因此**不会**重蹈 R44「隔离快、流水线慢」的旧病
  （那类候选预测更好，仍会被端到端回验裁决）；它是对「端到端噪声 ±5% ≫ 接受门槛 1%」的护栏。
- 构建通过、`tuning_test` PASS。

### S3 决策级 chain move + reorder 占比门控 —— 已落地并实测（结论：暂不足以支撑 S4）

- 对每个布局耦合分量，显式构造两种**整分量联合赋值**：（**整链一起切 blk**）/（整链一起切 non），
  各作为一个 move 端到端回验（先过 S2 契约门），取 median 最优者。整族候选**按 kernel ms 选**
  （不看 reorder）——reorder 是整链的联合属性，必须在整链赋值上评估；逐节点用 `predictNet` 会被
  reorder 卡住而无法翻转（顺序贪心陷阱）。
- **发现并修复的 bug**：`predictNet / predReorder` 此前**只对 target 节点求和**（`assign` 对
  非 target 为空），整网代理/reorder 占比被严重低估（一度误算成 0% → chain move 被错误跳过）。
  已改为按 `choiceEntry` 解析非 target 节点的实际选择。
- **实测（mb，单进程 all-conv1x1，33 签名同属 1 个连通分量，锁频）**：
  - `chain-diag` 显示 **1 个耦合分量**，chain move **被评估但从未被接受**（未超过 median 门槛）。
  - 逐节点贪心：**6 个节点各报 1.7–4.1% 改善，最终净 +0.8%（变慢）**——因低于 1% 回退地板而
    **未被拦下**；另一次运行净 **+2.9% → 被门回退**。
- **结论（关键）**：**逐节点的 in-situ「改善」不组合**——每个候选是相对「当时」的全局赋值交错测量的，
  当邻居随后移动时，之前的改善会蒸发/反转；再叠加 ±5% 噪声。**即便有 min+median+交错，per-node
  贪心目标也非可组合**。这是比「缺 blocked chain」更底层的瓶颈。

### S4 完整 blocked chain —— **暂缓（preconditions 未满足）**

- 按 §4.2 的定义，S4 的前置是「S3 在稳态 A/B 上给出可靠正收益」——**未满足**（S3 chain move 从未被接受）。
- 更重要的：blocked chain 会**增加**跨 op 布局耦合，而 S3 实测显示当前瓶颈正是
  **「逐节点贪心不可组合 + 噪声」**——在搜索本身修好前上 S4，会放大而非缓解失败模式。
- **S4 的正确前置**（建议）：
  1. **整赋值验收**：不再逐节点贪心提交；某个分量/整轮的所有改动**只有在该整赋值的稳态口径改善时
     才提交**（把「提交 + 事后回退」改成「先评估整赋值再提交」）。
  2. **外部稳态裁决**：以 `kernel_run --report` 交错 A/B（典型口径）为最终判据，in-situ 只做候选生成。
  3. **收紧回退地板 / 提高重复**：本轮 +0.8% 漏网说明 1% 地板相对噪声偏松。
  只有这三条让「整链持久化」的收益稳定可测后，才值得投入 S4 的 kernel/图改造（消除残余 reorder ≤8%）。

## 4.4 前置①：整赋值验收（已落地并实测）

**改动**：`globalRetune` 的每轮不再逐节点提交；而是相对**同一个固定上下文** `committed`
评估每个 target 的候选、汇总成一个**整赋值候选** `proposal`，再用 `measurePair(committed,
proposal, max(reps,3))` 对**整网稳态**（median）验收——通过才提交整轮，否则整轮拒绝。

**实测（mb 单进程 all-conv1x1，33 签名，锁频）**：

```
round0 propose Cout24_N784_Cin72   blk->blk  (target med 1.6503 -> 1.6160)   # 各报改善 1.6–2.8%
round0 propose Cout288_N196_Cin48  blk->blk  (target med 1.6520 -> 1.6209)
round0 propose Cout40_N196_Cin240  blk->blk  (target med 1.6400 -> 1.5946)
round0 propose Cout96_N49_Cin288   blk->blk  (target med 1.6575 -> 1.6352)
round0 whole-net(median) 1.6597 -> 1.6483 ms (-0.7%) REJECT (no commit)      # 整赋值只有 -0.7%
net(median) 1.6389 -> 1.6416 ms (+0.2%)                                      # 重复测同一赋值=±噪声
```

- **整赋值验收直接暴露了非组合性**：4 个 target 各报 1.6–2.8%，合起来只有 **−0.7%**。
- 因整轮 −0.7% 未过 1% 地板 → **拒绝、不提交**（此前逐节点版会提交并最终 +0.8%）。
- 末尾 `1.6389 -> 1.6416 (+0.2%)` 是**同一赋值**的两次测量 → 说明这个量级就是**噪声**。
- **A/B（锁频 3 rep）**：base median 1.658 → full 1.627（≈−1.9%，来自 retune 缓存刷新），
  global 步贡献 ≈0，**无回归**。

**结论**：前置①达成目标——**消除「逐节点小改善累积成净回归」**，把 globalRetune 变成
**保守/近无操作**。这也**再次确认整网全局步当前没有可靠正收益**，因此 **S4 仍不具备前置**。

## 5. 下一步（按 ROI）

> 实现顺序以 **§4.2 的 S1–S4** 为准（先统一计费口径 → 尊重布局契约 → 决策级 chain move →
> 完整 blocked chain）。

1. **确认 blocked chain 的机会上限**：离线算 mb 若 reorder→0（全部 fsv16 持久化）能省多少
   busy（当前估计 ~8%），再决定是否投入（对齐 R46「前置就位后另开」）。
2. **落地 #3（可加目标真正省测）**：对**无耦合连通分量**的签名，直接用 `predictNet` argmin
   决定、**跳过端到端回验**；只对耦合分量与最终门做整网测量。这同时降低 mb/yolo 的 GPU 暴露。
3. **解 #4（绝对口径）**：把最终裁决改为**外部稳态 A/B**（`kernel_run --report` 交错）而非
   in-situ `measurePair`；in-situ 只用于生成候选/排序。
4. **#5 噪声**：对 `spread>阈值` 的签名强制 top-K 复测或直接不发言（返回隔离默认）。
5. 正式 retune 应与本次实验一致地「锁频 + 分批 + 临时缓存」，结论以**交错稳态 A/B**为准。

---

## 6. 复现

```bash
scripts/gpu_clocks.sh lock
mkdir -p /tmp/opencode/pc   # INFVINO_PROGRAM_CACHE

# 全流程（每模型；安全驱动）
python3 scripts/autotune.py --model mobilenetv3-small --batch 2 --iters 4 \
    --global --global-topk 2 --global-iters 4 --global-rounds 2 \
    --global-budget 1500 --cache models/tuning_r47_mb.json

# 稳态 A/B（4 交错 rep）
for i in 1 2 3 4; do
  INFVINO_TUNING_CACHE=config/tuning.json INFVINO_PLAN_TUNING=none \
    ./build-ct/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 15
  INFVINO_TUNING_CACHE=models/tuning_r47_mb.json \
    INFVINO_PLAN_TUNING=models/mobilenetv3-small/model.plan.tuning.json \
    ./build-ct/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 15
done

# 逐节点（离线）
python3 scripts/analyze_budget.py --model mobilenetv3-small \
    --plan models/mobilenetv3-small/model.plan --profile-json /tmp/prof_full.json

scripts/gpu_clocks.sh unlock
# 回归
python3 scripts/model_check.py  --model mobilenetv3-small --repo $PWD
python3 scripts/reuse_check.py --model mobilenetv3-small --repo $PWD
```
