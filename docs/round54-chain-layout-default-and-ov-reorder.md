# R54：链假设进入 plan 期默认路径 + 移植 OV 的 SLM-transpose reorder

> 用户决策：**A**（先做「求解器/模型」层的通用改进——把 R53 §4.3 的「链假设」并进 plan 期），
> **如果 OV（~/openvino）有相关 kernel 就直接搬过来**。
>
> 本轮两件事：
> 1. **把精确最小割（链假设）从 opt-in 提升为 plan 期默认求解器**——旧的不动点存在
>    chicken-and-egg（R53 §4.3），无法从 NCHW 基线发现 blocked 链；mincut 是唯一能
>    「假设整条链都 blocked」的求解器。
> 2. **移植 OV 的 `reorder_data_bfyx_to_blocked_format.cl`**（SLM transpose）：修掉我们
>    per-element reorder 在大空间小 Cin 上的稀疏写；按 `W` 分派（网络实际 shape 仍走旧
>    kernel）。

---

## 0. 结论先给

| 项 | 结果 |
|---|---|
| **链假设默认化** | 外部稳态 A/B（交错 8×，`--iters 4` busy）：mb **−11.2%（8/8）**、y11 **−1.4%（8/8）**、y8 **−0.2%（噪声，5/8）**。 |
| **为何现在安全** | R49 的「yolo 过度持久化 +3%」前提已消失：布局契约声明化（R49 §9.1）+ conv3x3 排除（R49 §9.3）+ **R52 图级数值修复**。当前 mincut 在 yolo 上不再过度持久化（y8 `fsv16=0`、`0.0%`）。 |
| **开关** | 默认开；`INFVINO_NO_LAYOUT_MINCUT=1` 回退旧不动点；`INFVINO_LAYOUT_MINCUT_GATE=1` 仍可做整网 A/B 验收（R49 §9.2）；`INFVINO_LAYOUT_MINCUT_3X3=1` 仍把 conv3x3 纳入（R53）。 |
| **OV reorder 移植** | 忠实移植 SLM-transpose（`reorder_bfyx_to_fsv16_slm`），**逐位一致**（bench verify）；大空间小 Cin **2.6×**（Cin3 640²: 8.5→22.4 GB/s）、Cin16 322² **+8%**；小空间 `W<256` 旧 kernel 更优 → 按 `W>=256` 分派。 |
| **数值** | 三模型 `model_check`（默认=链布局）全 **PASS**（mb 1.215e-2、y8 4.114e-4、y11 7.908e-4）；`reuse` diff=0；`tuning_test` PASS。 |
| **config/tuning.json** | 未改动（链布局是求解器行为；OV reorder 是 kernel 级、按 shape 分派）。 |
| **模型缺口（未修）** | 软标尺 `expectedOps` 仍低估 `20×20 256→64`（ratio 1.53）等大 K 小空间层——留 R47 L3 模型复核（见 §4）。 |

---

## 1. 链假设默认化（plan 期求解器）

### 1.1 缺口（R53 §4.3）

`resolveLayoutChoices` 的联合不动点：`planBlockedLayout` 只有在「生产者族 canOutFsv16
（即已选 blk）」时才把输出标 fsv16；而不动点要「输出 fsv16」才会用 `blkFsv16` 给 blk 定价。
初始全 NCHW ⇒ `blkCost = blk.ms + reorder` ⇒ 永远选 non ⇒ **死锁**。精确最小割
（`LayoutSolver`，二元标注）可以**同时**决定整条链的布局，是唯一能跳出该死锁的求解器。

### 1.2 改动

`src/PlanModel.cpp`：`mincutOn` 从 `getenv("INFVINO_LAYOUT_MINCUT")` 改为
`getenv("INFVINO_NO_LAYOUT_MINCUT") == nullptr`（默认开）。其余不变：`resolveLayoutMinCut`
仍是「有 `#blk/#non` 备选 + 布局契约」的节点集（conv1x1 + depthwise），conv3x3 仍默认排除
（`INFVINO_LAYOUT_MINCUT_3X3` opt-in），成功则 `mincut_active_` 阻止 `planBlockedLayout`
覆盖。

### 1.3 为什么 R49 的「默认关」前提不再成立

R49 §5 的 yolo +3% 来自**首轮**（只覆盖 conv1x1、且漏了生产者直写能力，两次翻车）。
R49 §9.4 步骤 1–3 后：y8 mincut **无门** +2%、**有门**回退；y11 −1.5%；mb −14%。
本轮的差别：
- **R52** 修了图级 correctness bug（gap_fsv16 分配/标记漂移）与池/布局探针；
- conv3x3 明确排除、布局契约声明化；
- 实测（本轮，交错 8×）**y8 不再回归**。

### 1.4 实测（锁频非全程；`kernel_run --report --iters 4`；交错 8×）

| 模型 | chain（默认） median | fixpoint（opt-out） median | Δ | 胜场 | fsv16 |
|---|---:|---:|---:|---:|---:|
| mobilenetv3-small | **1.454** | 1.637 | **−11.2%** | 8/8 | 23 vs 13 |
| yolo11n-pose | **11.30** | 11.46 | **−1.4%** | 8/8 | 9 vs 2 |
| yolov8n-pose | 10.52 | 10.54 | −0.2%（噪声） | 5/8 | 0 vs 0 |

结构（mb）：reorder 调用 10→9、ms 0.079→0.060；持久 fsv16 张量 13→23。
数值：mb `mean_rel` 1.215e-2（mincut 口径，与 R49/R51 一致）**PASS**；y8/y11 逐位同基线。

> 保留 opt-out（`INFVINO_NO_LAYOUT_MINCUT=1`）与验收门（`INFVINO_LAYOUT_MINCUT_GATE=1`），
> 符合 R49「不在噪声上下结论」的纪律：保守场景可开锁频外部 A/B 验收。

---

## 2. OV 相关 kernel：移植 SLM-transpose reorder

### 2.1 动机（R53 §5）

我们的 `reorder_bfyx_to_fsv16` 是 per-element（1 WI/元素）：对**大空间小 Cin**（Cin3
640×640，写只占 3/16 lane）写稀疏、8.5 GB/s。OV 的
`reorder_data_bfyx_to_blocked_format.cl` 用 **SLM 做 8×8/16×16 transpose**，读写都合并。

### 2.2 移植（`kernels/conv_blk.cl`，自包含简化）

一个 work-group 处理一个 `(16-channel block, row y, 16-wide x tile)`：16 lane = 16 个 x；
每个 lane 对 16 个通道各做一次 coalesced 读（lane=x），转置进
`__local half tile[16][16]`，再各写一条连续 32 B（16 通道）的 vector store。两侧都合并。
（OV 原版用 TILE_SIZE=8 + `fetch_data.cl` 宏层；此处等价简化为 16×16、无 offsets/pitches。）

### 2.3 实测（`kernel_bench --op reorder --iters 20`，同会话）

| Cin | H×W | per-element | **SLM（新）** | 备注 |
|---:|---|---:|---:|---|
| 3 | 640² | 8.5 GB/s | **22.4 GB/s** | **2.6×** |
| 16 | 322² | 53.5 | **58.0** | +8% |
| 64 | 162² | 59.3 | 57.6 | −3% |
| 16 | 162² | 49.6 | 44.4 | −10% |
| 8 | 162² | 58.3 | 41.7 | −28% |
| 120 | 28² | 36.1 | 23.0 | −36% |
| 16 | 14² | 5.5 | 1.1 | −80%（lane 空转） |

**正确性**：两 kernel 输出 **逐位一致**（bench verify：Cin3 640² / Cin16 322² / Cin120 28² /
Cin8 162² 全 `0 mismatches`）。

### 2.4 落点：按 `W` 分派

`planReorder(Cin,H,W)`：`W >= 256 && H >= 16` → SLM 变体；否则 per-element（网络实际的
reorder 是 small-spatial，旧 kernel 更好）。用于 `blkInput` 与三处 autotune 的 `#reorder`
测量，保证成本模型与实际 kernel 一致。

> 现网络的大 W reorder 极少（stem 走 `conv3x3_cin3`，无 reorder），故当前整网基本无差异；
> 这是**为大空间 blocked 链将来可用**的 kernel 级准备，并修掉了 autotune 里 Cin3 640²
> `#reorder` 3.56→~1.4 ms 的成本失真。

---

## 3. 复现

```bash
# 链布局 A/B（默认 vs opt-out）
for i in 1 2 3 4 5 6 7 8; do
  ./build-blk/kernel_run --plan models/mobilenetv3-small/model.plan --iters 4 --report 2>/dev/null | grep 'total kernel'
  INFVINO_NO_LAYOUT_MINCUT=1 ./build-blk/kernel_run --plan models/mobilenetv3-small/model.plan --iters 4 --report 2>/dev/null | grep 'total kernel'
done

# reorder：per-element vs OV SLM（含逐位 verify）
./build-blk/kernel_bench --op reorder --iters 20 \
  --conv-shape 3,0,640,640 --conv-shape 16,0,322,322 --conv-shape 120,0,28,28

# 数值（默认=链布局）
python3 scripts/model_check.py --model mobilenetv3-small --repo "$PWD"
```

---

## 4. 模型缺口（未修，留文档）

- **软标尺低估大 K 小空间 conv3x3**：`20×20 s1 256→64` 实测 11.61 ops、软 `expected`
  7.61 → **ratio 1.53>1**；`256→51` 1.24。`hard_ceiling`（ISA 配额）正确，问题在
  `expectedOps` 的 `n_wg/gridFactor` 与 R47 内存 roofline 的组合。影响仅排名。
  下一步：按 R47 §10 的正解把 L3 会计做成**整网评分器（零 GPU 预筛）**，而非逐节点项。
- `#reorder` 仍是单趟张量级成本，未建模多消费者共享（mincut 局部队列例已按张量级处理，
  但 greedy 路径仍按节点）。

---

## 5. 与 R48 §6 的关系

R48 §6 的三项：①conv3x3 布局链（R53 分析：不成立）②direct conv1x1 ③split-K conv。
本轮完成**用户选定的 A**（求解器/模型层通用改进）——它让**所有 op** 的布局决策摆脱
chicken-and-egg，是 ②③ 与新 kernel 上线前的「决策正确性」前置。②③ 仍需新 kernel。
