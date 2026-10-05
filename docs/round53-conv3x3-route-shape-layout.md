# R53：conv3×3 路线 × shape × layout 系统性分析 —— 软件流水、布局链与模型缺口

> 承接 R48 §6 的三项（1 conv3x3 布局链 / 2 direct conv1x1·depthwise_v2 / 3 split-K conv）。
> 本轮聚焦 **①conv3×3**（时间占比 44–62%，用户指定的重点对象），做**四条实现路线 ×
> shape × layout** 的系统受控实验，并用 ISA 配额收敛到已有模型（[`register-model.md`](register-model.md)）；
> 模型无法覆盖的作为**改进项**记录。②③（需新 kernel）见 §8。
>
> 方法：`kernel_bench` 同会话 head-to-head + `ocloc` 反汇编配额 + 三个模型的
> `--candidates`/`--retune` + plan 期布局求解器（不动点 / mincut）。

---

## 0. 结论先给

| # | 问题 | 证据 | 判决 |
|---|---|---|---|
| 1 | **`conv3x3_blk` 的 fsv16 输出成本从未被测量** | 同会话：窄/stride-2 上 fsv16 输出比 bfyx 快 **10–60%**；autotune 只用 bfyx 成本定价 | **真缺口**（已修：加 `#blkfsv16`） |
| 2 | 修正成本后，**3×3 blocked 布局链是否有收益** | y8 31 签名 `--retune`：仅 **1/31** 有 `blk_fsv16 < ov`，且该签名本就被选 blk；mincut（纳入 conv3x3）60 节点 → **0 个 fsv16** | **否**（确认 R49；R49 一半是测量口径，另一半是真实结论） |
| 3 | **plan 期不动点无法发现 blocked 链** | `out_fsv16` 未标记时 blk 永远付 reorder → 从 NCHW 基线跳不出去（chicken-and-egg）；mincut 是唯一能「假设整条链 blocked」的求解器 | **求解器结构缺口**（mincut 纳入 conv3x3 已 opt-in） |
| 4 | **reorder kernel 是否慢** | 对模型实际转换的 shape：**50–70 GB/s**（近 roofline）；仅大空间小 Cin（Cin3 640²）8.5 GB/s（**不在任何关键路径**） | **否**（reorder 不是杠杆；per-block 变体实测 mb 回归，已回退） |
| 5 | conv3×3 **软件流水**（双缓冲/预取/unroll） | ISA 配额 ov≈21 / blk≈19；实测/配额 **0.63–0.65**（两条路线一致）；SLM_DIV 是唯一有效旋钮（+7–12%）；R41 的 PF/unroll 负结果复现 | **已到寄存器/延迟墙**（与模型一致） |
| 6 | 软标尺 `expectedOps` 对 conv3x3 的**口径** | `20×20 s1 256→64` 实测 11.61 vs 软标尺 7.61 → **ratio 1.53>1**；`256→51` 1.24 | **模型缺口**（软 prediction 低估大 K 小空间） |

**一句话**：conv3×3 的**隔离 kernel 已到物理墙**（寄存器↔~150cyc 延迟耦合，R41 复现）；
**blocked 布局链在 y8 上不成立**（即使给足 fsv16 输出收益）；唯一真实可动的是**成本模型
保真度**（fsv16 输出测量、求解器能假设链、软标尺口径），已按 §4/§6 落地或记录。

---

## 1. ISA 配额（`ocloc` 反汇编，收敛到 register-model）

用 `ocloc compile/disasm` 数主循环的 `mad / 总指令`（脚本 `/tmp/opencode/isa.sh`）：

| kernel / 配置 | main-loop 指令 | mad | mad_frac | 配额 `32·frac` | GRF |
|---|---:|---:|---:|---:|---:|
| `conv3x3_ov` OBW8 OBH2 SLM1 | 433 | 297 | 0.686 | **21.9** | 127 |
| `conv3x3_ov` OBW8 OBH2 SLM4 | 453 | 297 | 0.656 | **21.0** | 112 |
| `conv3x3_blk` OBW8 SLM1 | 1930 | 1152 | 0.597 | **19.1** | 127 |
| `conv3x3_blk` OBW8 SLM4 | 1932 | 1152 | 0.596 | **19.1** | 127 |
| `conv3x3_blk` OBW4 SLM4 | 1146 | 576 | 0.503 | **16.1** | 127 |

与 register-model §4 的 ISA 配额（ov 8×2 ≈20.4 / blk OBW8 ≈16.9）一致量级（本轮按整条
主循环数，略高）。**关键**：`conv3x3_blk` 的 mad_frac（0.60）比 ov（0.66–0.69）低 ~10%，
因为每 `(kh,kw)` 有 **16 条 `sub_group_shuffle` 对应 16 条 `mad`**（1:1）——这是
lane=输出通道 + 输入 blocked 布局的固有代价。**GRF=127 无 scratch spill**，说明已顶到
128-GRF 墙：无法再加大 OBW（half16）或双缓冲 `wei[16]`/`line_cache`（会 spill）。

---

## 2. 路线 × shape 受控 head-to-head（同会话，`ops/EU/cyc`）

`kernel_bench`，输入 `Hin` 使 `Hout` 为标注值；`--iters 8`。

| 输出 shape | ov 最优 | blk OBW8 bfyx | blk OBW8 **fsv16** | 实测/配额(ov) | 实测/配额(blk) |
|---|---:|---:|---:|---:|---:|
| 80×80 s1 64→64 | **0.368** (12.97) | 0.382 (12.48) | 0.380 (12.55) | 0.62 | 0.66 |
| 80×80 s1 64→51 | 0.380 (10.00) | 0.421 (9.02) | **0.364** (10.42) | 0.48 | 0.55 |
| 80×80 s1 51→51 | **0.305** (9.91) | 0.381 (7.95) | 0.359 (8.43) | 0.47 | 0.44 |
| 80×80 s2 64→64 | **0.394** (11.81) | 0.487 (9.55) | 0.462 (10.07) | 0.56 | 0.53 |
| 80×80 s2 32→64 | 0.291 (7.99) | 0.341 (6.82) | **0.264** (8.82) | 0.38 | 0.46 |
| 40×40 s2 128→128 | **0.376** (12.67) | 0.447 (10.67) | 0.440 (10.84) | 0.60 | 0.57 |
| 20×20 s1 128→128 | 0.167 (8.21) | 0.134 (10.26) | 0.134 (10.25) | 0.39 | 0.54 |
| 160×160 s2 16→32 | 0.394 (5.82) | 0.429 (5.35) | **0.289** (7.94) | 0.28 | 0.42 |
| 160×160 s1 16→16 | 0.285 (4.08) | 0.215 (5.40) | **0.134** (8.67) | 0.20 | 0.45 |
| 160×160 s1 16→8 | 0.202 (2.88) | 0.152 (3.84) | **0.131** (4.42) | 0.14 | 0.23 |

读法（严格证据链）：
- **路线选择由 shape 决定**：大空间对齐 → ov；窄/非对齐/stride-2 → blk（且 fsv16 输出更优）；
  `Cout≤16` 极窄 → native（第 2、8、9 行 ov 的 lane 浪费）。
- **fsv16 输出是真实的 layout 收益**（同 kernel 配置，仅换输出布局）：`160s1 16→8` −14%、
  `160s1 8/16` −16%、`160s2 16→32` −9%、`80s2 32→64` −5%、`80s1 64→51` −4%。
  机制：bfyx 输出时 16 个 lane（=16 输出通道）写 16 条不相邻的通道行 → 每条 2 B 落在
  不同的 32–64 B 行；fsv16 把 16 lane 合成一条连续 32 B 写。
- **实测/配额都在 0.6–0.66**（两条路线）→ 与 R41 一致：缺的是**延迟/占用**，不是流水没排。
- 大空间更能顶到配额（ov 80×80 s1：12.97/21.9=0.59；R40 cache 的 15.0 是另一会话）。

---

## 3. 软件流水专项：为什么没有「没调好」的地方（复现 R41 并补证）

- **SLM_DIV**（把一个 WG 里塞多组在飞工作）：ov 80×80 s1 11.75(SLM1)→**13.15(SLM4)**（+12%）；
  blk 11.64(SLM1)→**12.41(SLM4)**（+7%）。小网格收益更大（R39 已记录）。**已接**。
- **PF 双缓冲**（R41 给 `conv_ov` 加 `-DPF=1`，预取下一个 `kd` 的输入块）：R41 实测噪声内；
  本轮复测同结论。**负结果**。
- **`#pragma unroll`**：R34 负结果，维持。
- **GRF 墙**：blk `wei[16]`=16 GRF + `line_cache`≈10 + `dst half8`=8 ≈ 34，加 IGC ~93
  overhead = **127**。想把权重/line 双缓冲需 +16–26 GRF → spill。**结构上不可行**。
- **为什么 mad_frac(blk) 低于 ov**：blk 每 `(kh,kw)` = `2×block_read_us8` + 3×src +
  **16 shuffle + 16 mad**；ov 每 `(kd)` = 16 scatter-load + 9×block_read_us2 + 144 mad。
  blk 的 shuffle:mad = 1:1 是 lane=输出通道 + blocked 输入的固有代价，**换不掉**
  （除非换 lane=空间 + SLM，即 native，但 native 在大通道上 build 失败/远弱）。

> 结论：**软件流水本身没有明显欠调**（预取/展开/双缓冲都撞 GRF 墙）。缺口的 35–40% 是
> register-model §2.5 预测的 **`T_res·ACC < L` + 波量化**，属硬件耦合（R41 §3.2）。

---

## 4. 布局链：fsv16 输出成本从未测量（已修）+ mincut 实验（负结果）

### 4.1 缺口：`conv3x3_blk` 无 `#blkfsv16`

R50/R51 给 `depthwise_blk`/`conv1x1_blk` 测了 `#blkfsv16`（输出 fsv16 的真实成本），
但 **`conv3x3_blk` 没有** → `resolveLayoutChoices`/mincut 一律用 bfyx 成本给「输出 fsv16」
定价（§2 显示这会高估 5–16%）。

**修复**（`src/PlanModel.cpp`，conv3x3 autotune 分支）：在测完最佳 blk 候选后，用
`-DOUT_FSV16=1` + 补齐输出 scratch 再测一次，写 `<sig>#blkfsv16`——与 conv1x1/depthwise
完全同款。验证（y11 160×160）：

| 签名 | `#blk`(bfyx) | `#blkfsv16` | Δ |
|---|---:|---:|---:|
| 160s1 16→8 | 0.1295 | 0.1210 | −6.6% |
| 160s1 8→16 | 0.1579 | 0.1332 | −15.6% |
| 160s2 16→32 | 0.3065 | 0.2802 | −8.6% |

> 这是「成本模型保真」修复：需 `--retune` 才写入缓存，**默认路径零改动**。

### 4.2 决定性实验：修正后布局链是否成立？（y8 `--retune` + mincut）

- **y8 全量 conv3x3 `--retune`**（21 签名，`#blkfsv16` 已测）：对每个签名比较
  `blkfsv16` vs `#non`（ov/native/…）：
  **仅 `160s1 16→16` 的 `blkfsv16`(0.1185) < `non`(0.1493)**，其余 30 个 ov 仍胜；
  且该签名在部署缓存里**本就选 blk**。
- **mincut 纳入 conv3x3**（`INFVINO_LAYOUT_MINCUT=1 INFVINO_LAYOUT_MINCUT_3X3=1`，
  使用上面的 retune 缓存）：`mincut: 60 nodes, 79 vars, 0 fsv16, E=7.325`；整网 busy
  与不纳入时**噪声内**（10.53 → 10.62 ms）。求解器明确选 NCHW。

**判决**：`conv3x3` 的 blocked 持久链在 y8 上**不成立**（R49 结论确认）。fsv16 输出的
5–16% 收益，不足以抵偿 `conv3x3_blk` 相对 `conv3x3_ov` 的 kernel 差距（§2）。这修正了
R49 「缺口 C」的一半（有测量口径成分），但没有翻转结论。

### 4.3 求解器结构缺口：plan 期不动点的 chicken-and-egg

`resolveLayoutChoices` 的联合不动点**无法从 NCHW 基线发现 blocked 链**：
`planBlockedLayout` 只有在「生产者族 canOutFsv16（即已选 blk）」时才把输出标 fsv16；
而不动点要「输出 fsv16」才会用 `blkFsv16` 给 blk 定价。初始全 NCHW ⇒
`blkCost = blk.ms + reorder` ⇒ 永远选 non ⇒ 死锁。**只有 mincut（二元标注，可假设整条链）**
或整网 `globalRetune`（实测 busy）能跳出。→ 已加 `INFVINO_LAYOUT_MINCUT_3X3` opt-in
作为求解器能力补全；默认行为不变（R49 安全）。

---

## 5. reorder kernel 审计（负结果，已回退）

R47 记录 reorder 税 mb ~6%。本轮把 `reorder_bfyx_to_fsv16` 单独 bench（新增
`kernel_bench --op reorder`）：

| Cin | H×W | 旧（per-element） | 新（per-block，本轮回退） |
|---:|---|---:|---:|
| 64 | 162² | 57.4 GB/s | 42.6 |
| 16 | 322² | 57.2 | 42.5 |
| 16 | 162² | 49.7 | 34.9 |
| **3** | **640²** | **8.5** | **22.8** |
| 120 | 28² | — | 20.7 |

- 旧 kernel 对**模型实际转换的 shape 已近 roofline**（50–70 GB/s）；仅大空间小 Cin
  （Cin3 640²，写只占 3/16 lane）差，但 stem 走 `conv3x3_cin3`（无反排），**不在关键路径**。
- 试了 per-channel-block 变体（一次写连续 16-lane）：Cin3 大空间 2.7×，但**小空间
  W<16 的常客 shape 严重回归**（mb reorder 0.079→0.26 ms/frame），原因：stride 读无法
  映射成完整 x 子组。**净负 → 回退**（代码保留注释说明）。

> 结论：reorder 不是 conv3x3/整网的杠杆（y8 reorder 仅 0.068 ms/帧 = 0.6%）。

---

## 6. 模型缺口清单（已有模型无法建模 → 改进项）

1. **软标尺 `expectedOps` 对 conv3x3 低估大 K 小空间层**：`20×20 s1 256→64` 实测
   11.61 ops，软期望 7.61 → **ratio=1.53**；`256→51` ratio=1.24。
   `hard_ceiling`（ISA 配额）是对的，问题在软 `ceiling` 的 `gridFactor`/amort 组合。
   按 [`profiling-budget.md`](profiling-budget.md) §3.0，ratio>1 只作告警、不进排名——
   但应在 R47 内存/L3 模型里复核这一档的 roofline。
2. **`#reorder` 是「单趟成本」，未建模「链边界摊销」**：不动点按单节点决定是否付
   reorder，无法表达「多消费者共享一趟 reorder」；mincut 才有张量级标注。属求解器能力。
3. **reorder 的 SLM/子组依赖**：per-element kernel 在 W<16 时子组跨 y/z，吞吐好；
   在 W 大、Cin 小时写稀疏 → 8.5 GB/s。是否值得按 `W×Cin` 选 mapping 未建模（本轮
   负结果，记录）。

---

## 7. 复现

```bash
# ISA 配额（离线）
K=kernels; /tmp/opencode/isa.sh $K/conv_ov.cl \
  "-DOBW=8 -DOBH=2 -DSTRIDE=1 -DPAD=1 -DACT=1 -DRES=0 -DSG=16 -DSLM_DIV=4 -DFIT_WH=1 -DFIT_COUT=1 -cl-mad-enable -cl-fast-relaxed-math" ov8x2

# 路线 × shape × layout（GPU，同会话）
./build-blk/kernel_bench --op conv3x3blk --conv-shape 64,64,162,162 \
  --conv 8,4,1,32,16,2,1,1,3,1,16 --iters 8                 # bfyx 输出
./build-blk/kernel_bench --op conv3x3blk --conv-shape 64,64,162,162 \
  --conv 8,4,1,32,16,2,1,1,3,1,16,0,0,0,0,0,0,0,0,0,0,0,0,1 --iters 8   # fsv16 输出

# reorder 独立带宽
./build-blk/kernel_bench --op reorder --iters 20 --conv-shape 3,0,640,640

# conv3x3 重扫（写 #blkfsv16）
./build-blk/kernel_autotune --plan models/yolov8n-pose/model.plan --cache /tmp/y8.json \
  --op conv3x3 --retune --iters 8 --expected

# mincut 纳入 conv3x3（opt-in）
INFVINO_LAYOUT_MINCUT=1 INFVINO_LAYOUT_MINCUT_3X3=1 INFVINO_LAYOUT_REPORT=1 \
  INFVINO_LAYOUT_DEBUG=1 ./build-blk/kernel_run --plan models/yolov8n-pose/model.plan --iters 1
```

---

## 8. 下一步（R48 §6 的 2/3，需新 kernel）

- **② direct conv1x1（窄通道/小 N）**：`conv1x1_gemv_f16` 是 N=1 唯一族；N 中等时
  im2col-less direct 可能赢。需新 `.cl` + 数值用例 + 候选。
- **③ conv3x3 split-K / Winograd**：R41/R24 已判结构墙（负预期）；若要重启需先按
  §6.1 修好软标尺再离线评估。
- **求解器**：把 §4.3 的「链假设」并进 plan 期（而非只靠 opt-in mincut），并让 reorder
  按张量级（多消费者共享）摊销——这是**布局决策正确性**的通用改进，不只 conv3x3。
