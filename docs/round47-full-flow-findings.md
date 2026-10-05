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

## 5. 下一步（按 ROI）

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
