# 三模型端到端性能分析（busy/net/e2e 预算 + 逐 kernel ops/EU/cyc）

> 工具链：`kernel_run --profile-json`（L0）→ `scripts/analyze_budget.py`（L1/L2）
> → `scripts/profile_ablation.py`（优化消融）。口径见
> [`profiling-budget.md`](profiling-budget.md)。
> 标尺是硬件极限 / 中间标准 `expected_ops`，**不以 OpenVINO 为参照**。
>
> 采样：2026-10-04，Iris Xe 80EU / 1.3GHz（fp16 峰值 32 ops/EU/cyc = 3328 GFLOP/s），
> `kernel_run --iters 20` + `infvino_bench --iters 80`；消融矩阵 `--iters 40 / report 20`。
> 每次 GPU 命令限内存/限时，跑后 `gpu_guard.sh after` 无 HANG。

---

## 1. 三层预算瀑布（本次实测）

`e2e = pre + net + post`，`net = busy + launch/copy/f16/bubble`。

| 模型 | e2e (ms) | fps | pre | net | **busy** | net−busy | busy/net | post |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolov8n-pose | 19.27 | 51.9 | 2.05 | 17.11 | **13.57** | 3.53 | 79.3% | 0.015 |
| yolo11n-pose | 20.71 | 48.3 | 2.33 | 18.54 | **14.61** | 3.93 | 78.8% | 0.015 |
| mobilenetv3-small | 3.85 | 259.8 | 0.38 | 3.46 | **2.69** | 0.77 | 77.8% | 0.014 |

**读法**：

- GPU kernel busy 只占 net 的 **78–79%（三个模型几乎一致）**；剩下 **21–22% 是非 GPU 段**
  （launch / H2D-D2H / f16 转换 / kernel 间气泡 / host 簿记）。
- 按 dispatch 摊：**y8 117 次/帧 → ≈30 µs/次；y11 159 → ≈25 µs；mb 73 → ≈10.5 µs**。
  而本机实测 launch floor ≈ **3.5 µs/dispatch**（R30）。即**每次 dispatch 有 3–9× 的
  结构性开销**，mobilenet 尤其被 launch 支配（2.69 ms busy vs 3.85 ms e2e）。
- 预处理固定 0.38–2.33 ms（1080p），占 e2e 10–11%，与内容无关。

> **预算结论**：*busy 的天花板* 由 kernel 能力决定（下一节）；*墙钟的天花板* 由
> **dispatch 数 + 常驻 buffer 数 + host 提交**决定。两条曲线要分开优化。

---

## 2. 逐 kernel / 数据通路 ops/EU/cyc

> **读之前先做 roofline 判断（见 [`profiling-budget.md`](profiling-budget.md) §3.0）。**
> `ops/EU/cyc` 只对**算术强度高、以 packed FMA 为主**的算子有效。本机很多算子受
> **内存 roofline / 指令配额 / launch 地板**约束（depthwise、逐元素、搬运、归约、
> M=1 GEMV…），其低 `ratio` **不等于**有同比例的可用时间。而且很多 kernel 只是更大
> 模式的一部分、**可以被融合**（激活/残差 epilogue、concat→conv、gap+分类头…），
> **只看单个 kernel 是不合理的**：必须回到**端到端 + 具体模型**判断该层是否值得优化、
> 能否融合。下面的 `ops/EU/cyc` 与 `ratio` 仅用于**同 family 内排序**，`headroom_ms`
> 是相对量；**不要**把其中任何数字当成硬编码的硬件极限。

`measured = FLOPs /(ms·1e-3)/(EU·clk)`；fp16 峰值 32。计算类=conv3x3+conv1x1+depthwise+conv_general。

### 2.1 逐算子汇总

| 模型 | 算子 | ms | 占比 | calls | GFLOPs | ops/EU/cyc | 通路拆分 |
|---|---|---:|---:|---:|---:|---:|---|
| y8 | conv3x3 | 8.99 | 66.2% | 45 | 7.57 | **8.10** | native 0.35 / blk 3.60 / ov 5.04 |
| y8 | conv1x1 | 3.41 | 25.1% | 28 | 1.61 | 4.54 | |
| y8 | reorder(blk) | 0.33 | 2.4% | 10 | — | — | |
| y8 | smallops | 1.18 | 8.7% | 34 | — | — | ew_binary/concat/maxpool/resize/… |
| y11 | conv3x3 | 6.87 | 47.0% | 41 | 5.18 | 7.25 | native 0.70 / blk 2.51 / ov 3.66 |
| y11 | conv1x1 | 5.22 | 35.7% | 49 | 2.20 | 4.06 | |
| y11 | depthwise | 0.54 | 3.7% | 7 | 0.024 | **0.42** | |
| y11 | reorder(blk) | 0.54 | 3.7% | 13 | — | — | |
| y11 | bmm | 0.32 | 2.2% | 2 | — | — | attention |
| y11 | smallops | 1.98 | 13.6% | 60 | — | — | |
| mb | conv1x1 | 2.04 | 76.0% | 42 | 0.087 | **0.41** | |
| mb | depthwise | 0.39 | 14.7% | 11 | 0.015 | **0.36** | |
| mb | smallops | 0.19 | 7.2% | 20 | — | — | ew_binary/gap |

**总体**（计算类 FLOPs / 计算类时间）：

| 模型 | compute GFLOPs | compute ops/EU/cyc | % of 32 | 若摊到整帧 busy |
|---|---:|---:|---:|---:|
| yolov8n-pose | 9.178 | **7.12** | 22.2% | 6.50 |
| yolo11n-pose | 7.406 | **5.64** | 17.6% | 4.87 |
| mobilenetv3-small | 0.113 | **0.44** | 1.4% | 0.40 |

- **yolo 是「conv 主导、离峰值还远」**：conv3x3 ≈ 7.2–8.1，处于 OV 移植通路现实天花板
  （~12–14）的 55–65%；conv1x1 ≈ 4.1–4.5，偏 GEMM 整核上限 13.7 的一半。
- **mobilenet 是「小算子/内存主导」**：0.44 ops/EU/cyc（1.4%）不代表 kernel 差——它的
  FLOPs 本来就极小，时间花在 GEMV 归约、depthwise 地址谓词、GAP 归约和 launch 上。
  **对 mb 用 ops/EU/cyc 评kernel 会误导**，应看「内存 roofline + 每 dispatch 地板」。

### 2.2 中间标准记分卡（ratio = 实测/期望，headroom = Σ ms·(1−ratio)）

| 模型 | tuning 命中/未命中 | headroom 合计 | 最大 headroom family |
|---|---|---:|---|
| y8 | 102 / 5 | **5.66 ms** | conv3x3 4.74（n=45，ratio 中位 0.43） |
| y11 | 138 / 8 | **5.72 ms** | conv3x3 3.87 + conv1x1 0.71 + depthwise 0.46 + bmm 0.24 |
| mb | 64 / 9 | **1.52 ms** | conv1x1 1.05 + depthwise 0.34（+ gap 0.08） |

**最差节点（按 headroom，含热层）**：

- y8：`320×320 s2 Cin3→Cout16` **0.486 ms，ratio 0.188**（3.03/16.07）；`160×160 s2 16→32`
  ratio 0.350；`40×40 s2 64→128`（blk）ratio 0.446；`20×20 s2 128→256` ratio 0.395。
  主导层 `80×80 s1 64→64` ×4 各 0.37 ms、ratio **0.677**（13.6 ops，接近该通路实际上限）。
- y11：`40×40 s2 128→128`（blk）**0.693 ms ratio 0.489**；stem `320 s2 3→16` 0.538/0.188；
  `80×80 s2 64→64` 0.483/0.518；**depthwise 7 层 ratio 0.076–0.194**（headroom 0.46 ms）；
  `bmm` ratio 0.26（0.21 ms）。
- mb：`96×49×576` gemm 两个 0.24 ms、ratio 0.50；**head `1000×1×1024` GEMV 0.106 ms ratio 0.04**；
  20+ 个 N=1 GEMV 与 `gap` ratio 0.003–0.1；depthwise ratio 0.085–0.235。

> **标尺注意**：N=1 GEMV / `gap` / 极小 `concat` 的 ratio 是 0.0x，但这是
> `expected_ops` 对**归约延迟/固定 launch 地板**建模不足导致的「期望虚高」，
> 不代表这些层有对应的可挖时间。**ratio 只用于同 family 内排序**。

---

## 3. 缓存管理有哪些不足

系统现有 6 层缓存，逐一给出证据。

### 3.1 kernel program 缓存：**只有内存缓存，没有磁盘二进制缓存**（最大冷启动缺陷）

`ClRuntime::programs_` 按 `source|options` 缓存 `cl_program`，但只有
`clCreateProgramWithSource` + `clBuildProgram`，**没有 `clCreateProgramWithBinary`**，
也没有落盘（`grep` 全仓库无命中）。且缓存是 **per-`ClRuntime` 实例**。

实测**冷启动 JIT 固定成本**（`kernel_run` 1 iter vs 20 iter 外推）：

| 模型 | 进程总耗时 (1 iter) | 每帧边际 | **固定 JIT/初始化** |
|---|---:|---:|---:|
| yolov8n-pose | 8 498 ms | ≈15.5 ms | **≈8.5 s** |
| yolo11n-pose | 12 021 ms | ≈17.2 ms | **≈12.0 s** |
| mobilenetv3-small | 9 053 ms | ≈3.6 ms | **≈9.0 s** |

→ **每次进程启动花 8.5–12 s 做 IGC JIT**（本机 20+ 个 kernel 变体 × 各自 options）。
对照：OV 有 `kernels_cache` 二进制缓存。**这是纯缓存缺失，不是算子慢。**
多 Session / 多次 `createSession()` 若各自持 `ClRuntime`，会重复付这个成本。

### 3.2 激活内存池（P0）：复用率 60–78%，**离理论 87% 有实质差距**

`ActPool`：生存期冲突集 + 尺寸桶「最小的够用块」+ 静态分配。实测：

| 模型 | 朴素总量 | 实际分配 | 复用率 | 离线理论峰值 | 理论复用率 | 超额常驻 |
|---|---:|---:|---:|---:|---:|---:|
| y8 | 60.5 MB | **24.0 MB** | 60% | 8.9 MB | 87% | **+15.1 MB** |
| y11 | 69.0 MB | **26.5 MB** | 62% | 10.6 MB | 87% | **+15.9 MB** |
| mb | 3.5 MB | 0.8 MB | 78% | 0.9 MB | 77% | ≈0 |

差距根因（`memory-reuse-design.md` §5.1 已记录，本次确认）：

1. **没有 byte-offset 子分配**：只能「整块复用」，大 buffer 的空闲区切不给小张量
   （OV 的 padded pool 会按偏移子分配）。yolo 的密集小张量因此各自占块。
2. **view 生存期并集**延长了部分区间（reshape/flatten/sub-buffer alias 共存储）。
3. 影响不只是显存：Iris Xe 只有 **3.75 MB LLC**，多出的 ~15 MB 常驻缓冲把热数据挤出
   L3，直接抬高 concat/bmm/ew 这类贴 DRAM 墙算子的成本（R30 的「热 56 GB/s → 冷 24 GB/s」）。

> 对照消融：`INFVINO_NO_POOL=1`（每张量一块，60–70 MB）使 e2e 变慢
> **y8 +5.1% / y11 +6.7% / mb +3.2%**，且 **Δnet > Δbusy**（y11 +7.5% vs +5.8%），
> 印证「常驻 `cl_mem` 数本身就在拉墙钟」。**池是第 2 大杠杆。**

### 3.3 自动调优缓存（`config/tuning.json`）：覆盖不全 + 无失效校验

- 全库 224 条，本次**命中率 y8 95% / y11 94% / mb 88%**；**未命中节点回退内置启发式**
  （可能不是最优通路）。未命中集中在少数 shape。
- `TuningEntry.source` 字段被保存/读取，但 **`TuningCache::lookup()` 不校验 source**
  （也不校验 kernel 源码哈希 / options 语义版本）。kernel 源码改名或宏语义变化后，
  **旧条目会被静默套用**；只有 `expected` 会用新公式重算，options 仍是旧的。
- device key：本机 PCI id 未取到，退化成 **name+eu+clk 字符串**
  （`Intel(R) Iris(R) Xe Graphics_eu80_clk1300`）；换型号/换 EU 数会全 miss（可接受），
  但**同名不同 driver 版本不会区分**。
- 中间标准本身有偏差：见 §2.2 注。

> 对照消融：`INFVINO_TUNING=off` 使 e2e 变慢 **y8 +17.6% / y11 +23.1% / mb +7.7%**，
> 是**第一大的单项杠杆**。把未命中补齐 + 对热点层 retune，收益直接。

### 3.4 每节点 dispatch 缓存（P2 `node_cmds_`）：收益小且噪声内，且语义脆弱

首帧录制「克隆 kernel + 已设参数 + 网格 + tag」，之后每帧 replay。实测消融
`INFVINO_NO_LAUNCH_CACHE=1`：

| 模型 | Δbusy | Δnet | 判读 |
|---|---:|---:|---|
| y8 | **+0.491 (+3.6%)** | −0.075 | busy 有意义，net 在噪声内 |
| y11 | −0.064 | +0.121 | 噪声 |
| mb | −0.014 | −0.012 | 噪声 |

- 说明「逐节点 setArg/字符串/签名」这部分在**非 profiling 的墙钟里不是大头**；
  真正的瓶颈是 `clEnqueueNDRangeKernel` 提交 + dispatch 数（见 §4）。
- **脆弱点**：录制结果依赖首帧的内存指针/布局；tuning 变更需显式
  `captured_=false; node_cmds_.clear()`（自动调优路径已处理），但没有通用的
  「plan/布局/调优任一变更 → 失效」守卫。`blkInput` 的跨推理陈旧 bug 就是这类
  缓存语义踩过的坑（已修 + `reuse_check` 护栏）。

### 3.5 布局 / reorder 缓存（R36）：有效但余额小，去重是「录制期烘焙」

- `reorder(blk)` 每帧 y8 10 次 0.330 ms（2.4% busy）、y11 13 次 0.542 ms（3.7%）。
- `INFVINO_NO_BLOCK_LAYOUT=1`：y8 busy **+2.8%**、net +0.4%；y11 net +0.8%；mb ≈0。
  小但真实。
- `reordered_frame_` 同帧去重**只在 capture（首帧）生效**，之后靠重放里「少录了一个
  reorder」体现；逻辑正确但**不可从 replay 单帧复核**，属于「隐式状态」。
- `blk_in_` 每名一块 scratch，与池分离（`owned_blk_`），未纳入 liveness 复用；
  `ov_w_`/`blk_w_` 对同一权重可能各存一份重排。

### 3.6 无跨进程 / 跨 Session 预热

没有 program 二进制、没有 `clSVM`/remote tensor；`createSession()` 多 Session 目前
串行（mutex），共享同一 `ClRuntime` 时能共享 `programs_`，但不共享则重复 JIT。

---

## 4. 现有系统的缺陷（按对墙钟的影响排序）

1. **dispatch 多 + 每次 dispatch 结构开销大**。117/159/73 次/帧，每次约
   10–30 µs（对比 3.5 µs 地板）。其中大量是**本可零 launch 的 view/shape 节点**：
   y8 `concat4`×6、`slice_axis`×4、`permute`×1、`maxpool`×3、`resize`×2；
   y11 另有 `bmm`×2、`concat4`×6、`slice`×7。**smallops 已占 y11 13.6% busy。**
2. **单 in-order 队列 + 逐节点同步（profiling）**。没有 event 依赖图、OOO 队列、
   多 stream；host 不能持续投递，kernel 间存在气泡。
3. **内存池只做到整块复用**，无 byte-offset 子分配（§3.2），多 ~15 MB 常驻。
4. **无磁盘 kernel 二进制缓存**（§3.1），冷启动 8.5–12 s。
5. **自动调优覆盖不全 + 无失效校验**（§3.3）；中间标准对极小/内存受限算子虚高。
6. **数值/精度面窄**：仅 fp16 plan、f32 输出，无 i8/u8；静态 shape。
7. **`--dump-tensor` 语义受池限制**（run 后读只能读到最后一个写者），诊断体验受限。
8. **可选在线调优默认关闭**（开发板 GPU HANG 历史 → 安全取舍），意味着新 shape
   在没有离线 retune 时只能吃启发式。

> 反过来说，硬件能力不是主因：conv3x3 大层已达 13.6（68% 指令配额），
> 继续微调单 kernel 收益递减；**结构性缺口在 dispatch/内存/提交三段**。

---

## 5. 继续优化三个模型端到端的路线（按 ROI）

### Tier 0 — 复用现成机制，先把「已证明有效」的吃满

1. **补齐自动调优覆盖**（消融最大项，e2e 敏感 +17.6/+23.1/+7.7%）：
   把未命中签名（y8 5 / y11 8 / mb 9）补进 `config/tuning.json`，并对
   §2.2 的热点 shape **retune**（尤其 stem、小空间大通道、`bmm`、depthwise）。
2. **内存池做到 byte-offset 子分配**（消融第 2 项 +5.1/+6.7/+3.2%）：
   目标把 y8 24.0→~9 MB、y11 26.5→~10.6 MB。收益一半来自 L3 命中（busy），
   一半来自常驻 `cl_mem` 数下降（net/墙钟）。

### Tier 1 — 按 headroom 打 kernel 热点

3. **conv3x3 网格饥饿层**（y8 headroom 4.74 ms / y11 3.87 ms 的主体）：
   `320 s2 3→16`（0.19）、`160 s2 16→32`（0.35）、`40 s2 64→128`（0.45）、
   `20 s2 128→256`（0.40）。方向：CINC 精确对齐（R33 已证 +20–45%）、
   更大 WG / 更好 tile、`FIT_WH`/`FIT_CIN` 编译期特化。
4. **80×80 Cin64→64 ×4（y8）**：已 13.6 = 68% 配额，剩余是延迟/占用 →
   试 OSV64 / 更深 unroll / ILP；预期边际小。
5. **y11 depthwise（headroom 0.46 ms，ratio 0.08–0.19）**：R30 证明墙是地址/边界
   谓词；padded depthwise 因额外一趟带宽相抵（R31）。**重试方向**：把 pad 折进
   生产者 epilogue，或换 tile 减少边界谓词。
6. **y11 `bmm`（headroom 0.24 ms）**：attention 的 `1×2×64×400×400` 与
   `1×2×400×32×400`，调 tile/layout，或与前后 `permute/softmax` 融合。
7. **mb 1×1 GEMV + depthwise**：mb busy 的 76% 是 conv1x1。head
   `1000×1×1024` GEMV 0.106 ms、`1024×1×576` 0.057 ms 属**内存受限**，
   可与 `gap` 融合、或把 head 走一次 batched GEMM；`96×49×576` ×2（0.24 ms）
   试更大 N-tile / split-K。

### Tier 2 — 引擎结构（收益最大，风险也最大）

8. **减少 dispatch**：把 view/shape 节点零 launch 化（`slice_axis`/尾部 `concat4`/
   `permute` 的单消费者 alias，延续 P0 第二步与 Route A 的思路）；对 y11 的
   smallops 1.98 ms 直接收益。**每个新 alias 必须配独立数值用例 + `reuse_check`。**
9. **提交侧**：非 profiling 也已去掉逐节点等待；进一步可试 **out-of-order 队列 +
   event 依赖** 与**多 Session 并行**（每 Session 一套激活池），把 kernel 间气泡压掉。
10. **磁盘 kernel 二进制缓存**：按 `source_hash + options + device` 落盘
    `cl_program`，把冷启动 8.5–12 s 降到接近 0；同时让多 Session/多进程共享。
    需要同时给 `TuningCache` 加 **source 哈希校验**，避免旧条目静默误用。

### Tier 3 — 精度/后端（按需）

11. i8/u8（若目标模型量化）、USM/remote tensor；动态 shape。当前非瓶颈。

### 预期

- Tier 0 两项（补齐调优 + 池子分配）机制已在，落地风险低：**e2e 有 8–15% 空间**。
- Tier 1 命中 y8/y11 conv3x3 网格饥饿层与 y11 depthwise/bmm：**busy 再降 10–20%**。
- Tier 2 是拉平「busy vs 墙钟」两条曲线的关键；`command buffer` 本机不可用
  （R35），故走「减 dispatch + OOO/多 Session」而非整帧重放。

---

## 6. 附：优化消融矩阵（本次实测，Δ = 关闭该优化 − baseline；正 = 该优化有收益）

单位 ms（括号为相对 baseline 的 %）。`--iters 40 / report 20`，单开关独立容器。
**e2e 抖动约 ±0.1–0.2 ms，小于此不下结论。**

| 开关 | y8 Δbusy | y8 Δe2e | y11 Δbusy | y11 Δe2e | mb Δbusy | mb Δe2e |
|---|---:|---:|---:|---:|---:|---:|
| `TUNING=off`（自动调优） | +3.71 (+27.6%) | **+3.45 (+17.6%)** | +4.76 (+32.4%) | **+4.78 (+23.1%)** | +0.26 (+9.5%) | **+0.30 (+7.7%)** |
| `NO_POOL=1`（激活池） | +0.83 (+6.1%) | **+0.99 (+5.1%)** | +0.85 (+5.8%) | **+1.38 (+6.7%)** | +0.14 (+5.1%) | **+0.13 (+3.2%)** |
| `NO_BLOCK_LAYOUT=1`（R36 布局） | +0.37 (+2.8%) | +0.09 (+0.4%) | +0.05 | +0.16 (+0.8%) | +0.01 | −0.06 |
| `NO_LAUNCH_CACHE=1`（P2 dispatch） | +0.49 (+3.6%) | −0.14 | −0.06 | +0.09 | −0.01 | −0.02 |
| `NO_REORDER_DEDUP=1`（同帧去重） | +0.21 (+1.6%) | −0.06 | −0.01 | +0.19 | −0.01 | −0.02 |
| `NO_FUSE_RES=1`（残差融合） | +0.01 | −0.26 | −0.09 | −0.10 | −0.05 | −0.04 |
| `DW_PAD=1`（padded depthwise，需 retune） | +0.02 | −0.26 | −0.05 | −0.04 | +0.00 | −0.04 |

**判读**：自动调优与内存池是**仅有的两个显著项**；布局/launch 缓存是小的、方向正确但
接近噪声的项；残差融合与 padded depthwise 在本代三模型上暂不显著。
baseline（本表同会话）：y8 busy 13.46 / net 17.26 / e2e 19.56；y11 14.70 / 18.51 / 20.75；
mb 2.70 / 3.49 / 3.90。

---

## 7. 复现

```bash
# L0 采样（安全容器）
docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc '
    export LD_LIBRARY_PATH=$PWD/build
    ./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 20 \
        --profile-json /tmp/prof.json --profile-label yolov8n-pose
    ./build/infvino_bench --config config/models.yaml --key yolov8n-pose --iters 80 > /tmp/bench.txt'

# L1/L2 预算与记分卡
python3 scripts/analyze_budget.py --model yolov8n-pose --plan models/yolov8n-pose/model.plan \
  --profile-json /tmp/prof.json --bench /tmp/bench.txt --tuning config/tuning.json

# Phase 3 优化消融矩阵
python3 scripts/profile_ablation.py --repo $PWD --image infvino-dev:latest \
  --models yolov8n-pose yolo11n-pose mobilenetv3-small \
  --iters 40 --iters-report 20 --out /tmp/opencode/ablation.json

# 离线内存复用理论上限（不碰 GPU）
python3 scripts/analyze_memreuse.py models/*/model.plan
```

---

## 8. Tier 0–2 落地结果（本轮实施）

> 逐项都做了**数值回归**（随机输入 A/B 逐位比对 + 跨推理一致性 + 调优 A/B），
> 全部 GPU 命令遵守 `benchmark_protocol.md`，跑后无 HANG。

### Tier 0（已落地，默认生效）

| 项 | 改动 | 实测 |
|---|---|---|
| **T0.1 自动调优覆盖** | `kernel_autotune` 补齐未命中签名（小算子），`config/tuning.json` 224→**239**，三模型签名覆盖 **100%**（原 y8 5 / y11 8 / mb 7 缺失） | busy y8/mb **−0.5%**（y11 噪声内）；消除「未命中回退启发式」的不确定性 |
| **T0.2 激活池 byte-offset 子分配** | `ActPool` 增加 arena+偏移打包（`INFVINO_NO_POOL_OFFSET=1` 可退回整块复用）；`copy_c` 别名改为从 arena base 建子 buffer，失败则退化为不别名（保数值） | 分配 y8 **24.0→20.6 MB**、y11 **26.5→23.1**、mb 0.8→0.7；arena 数 32→24 / 40→31；busy y8 **−1.7%** / y11 **−2.3%**；**逐位一致 + 跨推理一致** |

### Tier 1（首层 Cin=3 专用 conv 已落地；其余记录负结果）

- **首层 Cin=3 专用 conv（本轮落地）**：新增 `kernels/conv_cin3.cl` —— lane=空间列、
  每 WI 算全部 Cout、权重 host 重排为 `[Cin*9][Cout]`（通道连续、跨 lane 广播、无 SLM/
  barrier），接入 `candidatesConv3x3`（`Cin≤4`）+ autotune + dispatch。三模型 stem：
  **y8 0.36→0.217 ms（−40%）**、**y11 0.42→0.212（−50%）**、mb 0.055→0.045；
  ops 3.0→3.9/4.0/2.3。三模型 `model_check` **全部 PASS vs onnxruntime**
  （y8/y11 stem **逐位一致**；mb 因累加顺序不同有 ≤3.1e-2 的 fp16 舍入，远在容差内）。
- **小算子 retune**（`depthwise`/`bmm`，`--retune`）：只有 1 条 `depthwise` 换了通路、
  ops 0.34→0.35（噪声）。与 R30 一致：这些算子已贴**指令配额 ~2.8**，不是带宽问题。
- **conv3x3 网格饥饿层**（其余，y8 headroom 4.74 ms / y11 3.87 ms 的主体）：候选已有
  OV/blk/native + CINC，R24/R25/R33/R34 已逐配额扫描；缺口是**延迟/流水/占用**而非
  指令数或候选缺失。**未新增数据通路**（属 multi-hour 内核重构，风险与 ROI 不匹配）。

### Tier 2（部分落地）

| 项 | 改动 | 实测 |
|---|---|---|
| **T2.10 磁盘 kernel 二进制缓存** | `ClRuntime` 按 `device+source_hash+options` 落盘 `cl_program` 二进制（`clCreateProgramWithBinary` 加载，原子写；`INFVINO_PROGRAM_CACHE` 覆盖目录，`none` 关闭） | **冷启动 8.5/3.7/6.2 s → 0.12/0.13/0.05 s**（≈98%↓）；50 个文件 / 5 MB；cache on/off 输出**逐位一致** |
| **T2.10 调优缓存 ABI 守卫** | `config/tuning.json` 增加 `cache_abi`；内核宏语义变化时 bump `kTuningCacheAbi` ⇒ 旧条目整体作废（回退启发式），杜绝静默套用旧 options | 向后兼容（未标注=接受）；`tuning_test` PASS |
| **T2.11 host f32↔f16 批量转换（AVX2）** | `ClBackend` 每次推理对 1.23M 输入 + 0.47M 输出做转换；新增 `f32_to_f16_bulk`/`f16_to_f32_bulk`（AVX2 无分支，**与标量逐位一致**；`INFVINO_NO_SIMD_HALF=1` 对照）。项目编译选项下是净耗时里一块未被优化的固定开销 | 1.59→**0.50 ms/帧**（y8）；net y8 **16.9→16.1**、y11 **18.3→17.3**（≈5%）；逐位一致 |
| T2.8 减少 dispatch / T2.9 OOO+多 Session | **暂缓**：需 view/skip alias 扩到 `slice`/`concat`/`permute` 或引擎调度重构，数值风险高；R35 已证 command buffer 本机不可用（本轮复查容器 runtime 23.17 仍无 `cl_khr_command_buffer`） | — |

### 综合（相对本轮起点）

| 模型 | busy | net | e2e* | 激活分配 |
|---|---:|---:|---:|---:|
| yolov8n-pose | 13.574 → **13.29（−2.1%）** | 17.108 → **16.1（−5.9%）** | 19.27 → **~17.8** | 24.0 → **20.6 MB** |
| yolo11n-pose | 14.612 → **14.36（−1.8%）** | 18.543 → **17.3（−6.7%）** | （pre 受宿主抖动，见下） | 26.5 → **23.1 MB** |
| mobilenetv3-small | 2.687 → **2.65（−1.3%）** | 3.455 → **3.29（−4.8%）** | 3.85 → **~3.7** | 0.8 → **0.7 MB** |

> net 的降幅主要来自 **T2.11 host 转换**（yolo −0.8~1.0 ms；mb −0.1 ms）。

> \* `e2e` 的 `pre` 是宿主侧预处理，实测会随宿主负载在 1.8–4.5 ms 抖动
> （同一二进制、同一空图）：**不要用单次 e2e 判断优化**，看 `busy`/`net`。
> 冷启动收益不在稳态 fps 里，但把每个进程 8–12 s 的 JIT 变成 ~0.1 s。

### 新增诊断开关

| 开关 | 作用 |
|---|---|
| `INFVINO_NO_POOL_OFFSET=1` | 关闭激活池 byte-offset 子分配，退回整块复用（对照） |
| `INFVINO_PROGRAM_CACHE=<dir>` | 指定磁盘 program 二进制缓存目录；`none` 关闭 |
| `kernel_run --report` 新增 `program cache: hits/misses` 行 | 观测冷/热启动 |
| `INFVINO_NO_SIMD_HALF=1` | 关闭 host f16 批量转换的 AVX2 路径（对照；逐位一致） |

---

## 9. 残余空间分析（做完 Tier 0–2 之后，还剩多少）

> 原则：把「中间标准 headroom」和「物理/结构上**可达**的余量」分开。前者是模型
> 算出来的上限，后者才是能拿到的 ms。**不假设任何未验证的新数据通路。**

### 9.1 逐桶可达性（结论：大头是物理/结构受限，不是候选缺失）

| 桶 | 现值 | 中间标准 headroom | 可达性判断 |
|---|---:|---:|---|
| y8 `conv3x3` | 9.12 ms | 4.60 | **低**：OV/blk/native + CINC 候选已逐配额扫（R24/R25/R33/R34）；缺口是**延迟/流水/占用**（大层 13.6 = 68% 配额，小层 42%）。换数据通路才可能，属 multi-hour 重构 |
| y11 `conv3x3` | 7.32 ms | 3.77 | 同上 |
| y8 `conv1x1(cat4)` | 3.38 ms | 0.80 | **低-中**：cat4 中位 ratio 0.85（已近期望）；剩余在少数网格饥饿 shape |
| y11 `conv1x1(cat4)+conv1x1` | 5.19 ms | 1.04 | 低-中；同理 |
| y11 `depthwise` | 0.54 ms | 0.46 | **低**：R30 ISA 证墙是地址/边界**指令**（配额 ~2.8），R31 padded 因带宽相抵为负结果 |
| y11 `bmm` | 0.32 ms | 0.24 | 低：网格 + K 依赖链；retune 无变化 |
| mb `conv1x1` | 2.03 ms | 1.04 | **低**：多为 N=1 GEMV / 7×7 gemm，**内存受限**；headroom 被中间标准高估 |
| mb `depthwise` | 0.39 ms | 0.34 | 低：同 y11 |
| y8/y11 小算子合计 | ~1.0 / ~1.6 ms | ~0.2 | 低：R30 已到 roofline / launch 地板 |
| **host 转换** | 1.59→**0.50** | — | **已落地**（AVX2） |
| **`net−busy`（enqueue+bubble）** | ~2.6–3.0 ms | — | **结构**：command buffer 本机不可用（R35）；靠减 dispatch，而可减的正是已试过的 view/skip |
| y8/y11 **首层 Cin=3**（`320 s2 3→16`） | 0.36 / 0.42 ms，ratio 0.19 | 4.6 / 4.5 | **中**：唯一「形状特殊到值得专用 kernel」的层（16 通道补零浪费）；专用 3→16 直接卷积可能 ~2×，但收益 ~0.2 ms |

### 9.2 结论（诚实的期望值）

- 去重后**真正还可能有正收益**的只剩两块：**host 转换**（已做，yolo ≈5% net）与
  **首层 Cin=3 专用 kernel**（≈0.2 ms，需新 kernel + `kernel_check`）。
- 其余（conv3x3、conv1x1、depthwise、bmm、小算子）的「headroom」绝大部分是
  **延迟/指令/内存 roofline** 的物理必然，或**中间标准对内存受限算子高估**的部分；
  **没有候选缺失**，再加同族候选不会赢。
- `net−busy` 的 ~2.6–3.0 ms 是**引擎结构**（117 次 dispatch 的提交 + kernel 间气泡），
  唯一的大杠杆是 command buffer（本机 runtime 不支持，R35）；其次是把 view/shape
  节点零 launch 化——即用户已指出「试过」的方向。
- 因此：**单模型端到端再压 1–3% 有把握（首层 + 少量 retune）；>5% 需要换数据通路或
  换 runtime，ROI/风险不匹配。** 与你的判断一致。

### 9.3 如果只做一件事

做 **首层 Cin=3 专用 conv**（y8+y11 stem，占 ratio 0.19、共 ~0.8 ms）：它是唯一
「Math 上明显可赢、且边界清晰」的 kernel；其余方向的边际收益都已接近噪声。
→ **本轮已做**（见 §8 Tier 1，y8/y11 stem −40%/−50%）。

---

## 10. 非 compute-bound 小算子的缓存命中分析（`scripts/analyze_cache.py`）

### 10.1 体系

对算术强度 ~0.25–0.5 FLOP/byte 的小算子，标尺是**内存 roofline**，而 roofline 依赖
**工作集是否命中缓存**（Iris Xe 只有 ~3.75 MB LLC，无独立大 L2）。`analyze_cache.py`
逐节点（profile-json 精确 join）算：

- `bytes`：读+写总字节（与 R30 / `expectedOps` 同模型）；
- `tier`：工作集 ≤ L1 / ≤ LLC / > LLC；
- `t_lnch = 3.5µs × 节点数`（launch 地板）、`t_mem = bytes / BW(footprint)`；
- `eff = (t_lnch + t_mem) / ms`，`bottleneck` 区分 **launch/网格** 与 **内存/缓存**。

`eff≈1` = 已贴住「该工作集 + launch 地板」的现实上限。中间标准 `expectedOps` 本就用
同一条 footprint→BW 曲线，所以 autotune 已经在隐式地选缓存友好变体（`EW_VEC`、grid3）。

### 10.2 当前水平（本轮实测，小算子部分）

| 模型 | 小算子 ms | MB(rw) | 整体 achieved | 结论 |
|---|---:|---:|---:|---|
| yolov8n-pose | 0.79 | 65.9 | 83 GB/s | 多数 eff 0.9–1.3 |
| yolo11n-pose | 1.31 | 73.9 | 57 GB/s | 多数 eff 0.9–1.3 |
| mobilenetv3-small | 0.18 | 2.7 | 15 GB/s | 2 类，均近地板 |

逐算子（eff = 实测 / 现实上限）：

| op | y8 eff / 瓶颈 | y11 eff / 瓶颈 | mb eff / 瓶颈 | 判读 |
|---|---:|---:|---:|---|
| ew_binary | 1.23 / mem | 1.26 / mem | 0.83 / mem | **已贴住**（甚至优于 copy 曲线） |
| maxpool | 1.02 | 1.02 | — | 已贴住 |
| concat4 | 5.3 | 7.5 | — | 源数据在 L3、纯写 → **远超 copy 曲线**（模型偏差，说明缓存命中好） |
| softmax_axis | 1.8 | 1.6 | — | 3 趟读命中缓存 → 超曲线 |
| slice_axis | 0.87 | 0.88 | — | 近地板 |
| ew_unary | 0.88 | 0.94 | — | 近地板 |
| **permute_0213** | 1.25 | **0.58** | — | y11 的 attention 转置：**跨步访问**，有 ~1.7× 空间 |
| **resize_nn** | **0.69** | **0.69** | — | 最近邻上采样读放大/跨步，有 ~1.4× 空间 |
| **gap** | — | — | **0.65 / launch/grid** | 10 个极小归约，**网格受限**，非缓存 |

### 10.3 针对残余项的两个尝试（均负结果）

按 §10.2 的残余项各做了一个变体并 `--retune` 实测：

| 尝试 | 做法 | 结果 |
|---|---|---|
| **`resize_nn_v`** | 向量写（每 WI 8 个输出像素、half8 store），降 store 指令 | **未胜**：`resize_nn3` 仍最优（0.029/0.052 ms） |
| **`permute_0213_t`** | SLM 16×16 tile，读/写两侧都合并（标准 transpose） | **未胜**：`permute_0213_3d` 仍最优（y11 0.011/0.076 ms） |

> 两个变体已**回退**（不留永远选不中的候选，避免 autotune JIT 放大）。结论：这些
> 残余项的低 eff **不是「可修的缓存/访问问题」**——`resize` 的读放大在 lane 内已被
> 硬件广播吸收，`permute` 的 400×400 转置工作集 320 KB 本就命中 L3，SLM tile 只多付
> barrier。**没有空间。**

### 10.4 结论

- **「极致的缓存命中」在多数小算子上已经达到**：`ew_binary/maxpool/slice/ew_unary`
  的 eff ≈ 0.85–1.3；`concat4/softmax` 甚至超过 copy 曲线（源热在 L3）。
- 残余项（`permute`/`resize`/`gap`）经**变体实测**确认无可落地空间（见 §10.3）；
  `gap` 是网格受限（每节点只有 C 个工作组），不是缓存。
- 小算子再压空间 **<1% e2e**，与 §9 一致。

### 10.5 附：分类头 f16 logits 对 softmax 无影响（另一尝试）

「尾部 f16→f32 应在 softmax 之前」这一要求**当前已满足**：网络输出 `logits`(f16) →
`ClBackend` 转 f32 → `ClassifyDecoder` 在 f32 上 softmax。唯一可议的是 **logits 先存成
f16 再转**。离线实测（mobilenet，20 组随机输入，ORT f32 logits 量化到 f16 再 softmax）：

- **argmax 翻转 0/20**，最大概率差 **9.5e-5**。

→ 存 f16 logits 对分类结果**无影响**，不需要 f32 输出通路（该改动会牵动「全 f16 plan」
的 dtype 体系，收益为零）。
