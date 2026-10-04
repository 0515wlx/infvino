# Round 42：标尺构建 · 墙→候选工具 · 噪声治理 · reorder/expectedOps 复盘

> 承接 R41（conv 离天花板差距的系统归因）。本轮回答四个问题：
> 1. roofline/延迟标尺**怎么建**（保留 `ops/EU/cyc` 作为上限，符合项目哲学）；
> 2. 「由墙反推该造什么候选」**怎么工具化 + 怎么准测**（含 Intel 工具调研）；
> 3. 噪声**怎么治**（固定频率、估计器、bug 排查）；
> 4. `reorder`/`expectedOps` 到底是**固有成本还是设计失误**。
>
> 已落地：`benchCandidate` 改**最小值**估计器、`scripts/gpu_clocks.sh`、`reorder` 不再
> 折进 kernel ms（修一个真 bug）、R41 的 `-DPROBE`/`-DPF` 诊断宏。

---

## 1. 标尺：`ops/EU/cyc` 作**硬上限**，其余墙做**告警/证据**

### 1.1 原则（修订 R42 初版）

初版把「经验 derate」也折进 `min(...)` 当硬上限——这是错的，会重犯 R33「把估计当极限」的病。
正确的层次：

| 层 | 含义 | 用法 | 例子 |
|---|---|---|---|
| **hard_ceiling** | 物理上**不可越**：ISA 指令发射配额 `32·mad_frac`（或唯一字节 roofline 下界） | `hard_ratio = measured/hard_ceiling`，**唯一能 capping 的尺子** | ov 20.3、blk 16.9、native 16.4、gemm 13.7 |
| **expected（软预测）** | 「我们认为这层能到多少」：含 amort/gridFactor/占用等经验 derate | 仅用于**同族排序**与相对比较；不 capping | ov 8–20 随 shape |
| **alarms（告警）** | 非硬性模型的**证据**：延迟 derate、L3 复用、footprint>L3、lane 浪费、网格饥饿 | 只**标注**，不下判决 | "可能受 L3 边界影响" |

**为什么**（用户两点批评都成立）：
- 盲目把软模型当上限**违背「追求极限」的设计哲学**——hard 必须留出「接近物理」的空间；
- 软模型一旦被证伪（measured>soft）会**连带污染**同一条证据链；分层后，**一次证伪只修正那一层**，hard 尺子不动。

已落地（R43）：`KernelFamily.hardCeiling` + `TuningEntry.hard_ceiling/hard_ratio`（向后兼容，
旧缓存回退软值）；`kernel_autotune --expected` 同时打印 soft/hard；`refresh-expected` 一并重算。

### 1.2 多墙只用来**解释 hard_ratio 的缺口**，不参与 capping

```
hard_ratio = measured / 32·mad_frac          # 离物理极限的真实距离
gap        = 1 − hard_ratio                  # 待解释
alarms     = { W_lat_model, W_l3_reuse, footprint>L3, lane_waste, grid_starvation, store_share }
```
- 用 R41 的 `-DPROBE` 剖面给 `hard_ratio` 的缺口**归因**（输入 feed / 权重 / store / lane）。
- 归因结论是**证据**：可以多族交叉印证；某条被证伪（如「权重是瓶颈」）只需改该条，
  不牵动 hard 尺子，也不影响其它族的结论。
- 经验 derate（η）只进 `alarms`，可带置信度；`kernel_diag`（见 §2）输出它。

### 1.3 实证：为什么字节/BW 不能当硬上限

用 R41 的字节模型跑 naive roofline，会把**所有层**判成内存墙（W_mem 3.3–8.3 ≪ measured 5–20）。
反证：`80×80 s1 64→64` 去掉权重（占建模流量 **80%**）只涨 **+15%**，去掉输入（18%）涨 **+28%**。
→ 权重流量不是瓶颈；「按字节/BW」的 roofline 只能作**告警**，不能作硬上限。缺口是
load→broadcast 依赖链延迟，由占用（SLM_DIV）部分遮盖并被 128-GRF 墙封顶（R41）。


---

## 2. 墙 → 候选：工具化 + 准确测量

### 2.1 现状：这是已经在做、但没自动化的活

R24→R42 的每次「换数据通路 / 加候选」都是人工走了一遍：剖面 → 猜墙 → 造候选 → 实测。
把它固化成 **`kernel_diag`**：

```
输入：一个 OpSignature（+ 一个族/候选列表）
步骤：
  1. 对每个 family 的候选：build baseline + probe 变体(noIn/noW/noStore/PF)
  2. 计时（min 估计器）→ wall profile:  W_issue / full / noIn / noW / noStore
  3. 分类 binding wall（阈值化 profile，而不是靠公式）
  4. 输出「墙 → 建议候选轴」
```

### 2.2 「墙 → 候选轴」映射（来自 R41/R42 的证据）

| binding wall（profile 特征） | 证据 | 该怎么造候选 |
|---|---|---|
| **lane 浪费**（`noBoth` ≈ quota·used_lanes/total） | `Cout=8`：noBoth=5.4≈22·8/32 | 降 `CB`/`OSV`（`CB=8` 已加）、或 **lane=空间** 变体 |
| **输入 feed 延迟**（`noIn` 大涨、`noW` 小、`PF` 无效、SLM 有效） | 多数 s1/s2 层 | 提高占用（`SLM_DIV`，已接）；**降 broadcast 依赖**（把输入块一次性载入寄存器/SLM 后跨 lane 复用，而非逐 mad shuffle） |
| **权重/输入 feed 带宽**（`noW` 也大涨、且流量模型自洽） | 目前**未观测到**（`noW` 只 +6–21%） | 更大输出 tile（每权重多算几个输出）、权重常驻寄存器、lane=空间 |
| **store/DRAM 带宽**（`noStore` 大涨） | stem `+178%`、`16→16` `+32%` | **融合下游**（不物化）、输出持久布局、写合并 |
| **DRAM 唯一字节**（footprint>3.75MB 且 `noIn` 大） | `80×80 s2`、`160² s2` | 减小工作集（切空间/通道 tile）、布局持久化 |
| **固定开销/短 K**（`noBoth`≪quota 且与 feed 无关） | `20×20 51→51`、stem 一部分 | 全展开 / 专用 kernel / 减少 prologue 谓词 |
| **issue 饱和**（`noBoth≈quota` 且 full≈quota） | 大网格对齐层 | 不再加同族候选 |

### 2.3 准确测量：Intel 工具有什么（调研结论）

本机能用 / 可装的：

| 工具 | 能力 | 获取 | 备注 |
|---|---|---|---|
| `intel/pti-gpu` → **`unitrace`** | kernel 时间线 + **硬件指标（含 instruction-level EU stall）**，OpenCL/L0 皆可 | GitHub 构建/apt | 最贴近「自动判墙」 |
| **Metrics Discovery API** (`intel-metrics-discovery`) | 原始计数器：EU busy、L3/DRAM 字节、cache 命中、stall | `apt install` | unitrace/GPA 的底座；可自己读 |
| **`intel_gpu_top`**（IGT） | PMU：engine busy、**内存带宽（IMC/RAPL）**、频率 | `apt install intel-gpu-tools` | 我们走 i915，支持；Xe 驱动才需 gputop |
| **Intel VTune**（**已装** `intel-oneapi-vtune`） | GPU Compute/Media Hotspots：EU stall 占比、内存 | 已装 | 需 `CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS=y` + 驱动权限 |
| **Intel GPA** | System/Frame Analyzer，EU/L3/memory 指标 | 官网 | 图形向，计算也能看 |
| **`ocloc`/IGA** | ISA、mad_frac、GRF、spill | 已有 | 已用 |
| **`intel_gpu_frequency`**（IGT） | 锁频 | `apt install intel-gpu-tools` | 或 sysfs（见 §3） |
| `clGetKernelWorkGroupInfo` + `CL_KERNEL_SPILL_MEM_SIZE_INTEL` | GRF/spill/SLM 占用 | OpenCL 内建 | 可无依赖拿到占用 |

**建议分层**：
1. **in-house probe（主力）**：`-DPROBE` + `min` 计时，可移植、无依赖、可进 CI；
2. **unitrace / MD API（交叉验证）**：拿 EU stall 原因与 L3/DRAM 字节，替代「猜」；
3. **VTune/GPA（深挖个别热点）**：一次性确认。

> 已有先例说明这条路可行：汇编（ocloc）已在用；`clinfo` 已确认设备参数；
> 缺的只是把 probe + min + 指标读回来接进 `kernel_diag`。

---

## 3. 噪声：根因、修复、bug 排查

### 3.1 实测：噪声 = LLC→DRAM 带宽断崖（已确认）

同一 binary、同 shape、连跑：正常值 ±2%，但**偶发 −15%**（0.33→0.39 ms）。用
`kernel_bench --op bandwidth` 扫 footprint（各 5 次，copy 读写 GB/s）：

| footprint | 1 MB | 2 MB | 3 MB | 4 MB | 5 MB | 8 MB | 16 MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| BW min–max | 130–152 | **108–140** | **90–114** | 49–56 | 32–38 | 22–23 | 19.4–19.9 |
| 离散度 | ~15% | **~30%** | **~25%** | ~13% | ~18% | <5% | <3% |

- **断崖在 4–8 MB**（L3 3.75 MB / CPU-LLC 8 MB 之间），且**离散度在断崖处最大**；
  深 DRAM（≥8 MB）反而稳定。**噪声主因 = 工作集跨 L3 边界时 DRAM 带宽骤降**，
  且被宿主/其它分配的占用情况（hit 比例）调制。
- 环境排查：无其它容器、load 0.18、温度 28–33°C、turbo on → 排除争用/热/CPU。
- 相反地，**conv 层的 min 很稳**（边界层 4.2 MB：连跑 min=0.404–0.410，±1.5%），
  因为 kernel 内复用让实际 compute 时间对 hit 比例不敏感，但**慢尾（p90/median）**
  会偶尔抬高——所以问题主要在「用 median 选择候选」。

### 3.2 防噪声机制（已落地）

| 措施 | 说明 |
|---|---|
| **`scripts/gpu_clocks.sh lock/unlock/status`** | 无免密 sudo 也能用（把 sysfs 文件 rw bind-mount 进容器写，已验证）。跑基准前 `lock`，跑完 `unlock`。 |
| **`benchCandidate` 用 min** | 外部干扰只能加时间 → **min 是内禀成本的无偏（偏小）估计**；±2% vs median 的 ±10–20%。warmup 3→5。 |
| CPU governor | 本机已是 `performance`（无需改） |

### 3.3 还没做但建议

1. **候选交错 + top-K 复测**：现在 autotune 是顺序扫候选，热漂移会系统性偏向先测的；
   改成 round-robin，或对 top-2/3 结束时复测取 min。
2. **`ops/EU/cyc` 的时钟一致性**：`opsPerEuCycle` 用固定 `clock_mhz=1300`；只有锁频后
   实测与理论才自洽（否则降频会把 ops 算小）。**锁频是 ops 口径正确的前置。**
3. `perf_event_paranoid` / 绑核 / 隔离容器（可选，深测时用）。

### 3.4 bug 排查结论

- `ClRuntime::timeMs` **本身没有计时 bug**（profiling queue 正确开启、逐次
  `clWaitForEvents`、事件释放正确）。
- 真正的「噪声问题」是**估计器选择**（median）×**顺序扫描**×**未锁频**三者叠加，
  不是数据竞争或错误采样。
- R37 里「autotune 静默吞 enqueue 失败」是**另一类** bug（已修 + `INFVINO_AUTOTUNE_DEBUG`）；
  建议把「候选跳过率」作为 retune 的告警（跳过 >20% 说明几何/几何解析有问题）。

---

## 4. `reorder` / `expectedOps`：是设计失误，不是固有成本

### 4.1 现状（本轮实测）

y8 `kernel_run --report`：`reorder(blk) 0.071 ms ×5`，占 busy **0.66%**（R37 时是
0.33 ms/10 次）。R38 的**联合 (族,布局) 不动点**把 conv3x3 的 reorder 基本消掉了。

### 4.2 发现的两个问题

**(a) 联合不动点只覆盖 conv3x3（设计不完整）。**
`resolveLayoutChoices()` 的 `Alt{blk,non,reorder}` 只对 `conv3x3` 生效；
`conv1x1_blk` / `depthwise_blk` 仍走 **保守计费**：`eff = kernel.ms + 一趟 reorder`，
**无论输入是否会持久化**。这会把「本可持久化、reorder=0」的 blk 误判为贵。
→ 正确做法是把 R38 的不动点推广到所有声明了 `in/out_layout` 的族
（`KernelFamily.layout` 已有一等公民，规划器 `nodeFamily` 也已通用，只差 conv1x1/depthwise
的 `#blk/#non/#reorder` 记账与 `Alt` 泛化）。

**(b) 真 bug：reorder 被折进 kernel 的 `ms`。**
保守计费处：

```cpp
eb.ms = eff;                       // ← eff = kernel + reorder（错）
```

而运行时 reorder 是**独立 dispatch**（`timed("reorder(blk)")`），本来就会单独计入 busy。
把 reorder 折进 `eb.ms` 导致：
- 该节点存进 `tuning.json` 的 `ms` 偏大 → `ops` 偏小、`ratio` 偏低；
- 预算/记分卡把 reorder **重复计一次**（节点内 + 独立 dispatch）；
- 表现为「中间标准对 blk 节点系统性偏高」——**这正是被误当成 `expectedOps` 缺陷的那部分**。

**已修**：选择仍用 `eff`（保持不选亏本的 blk），但**存储 kernel-only `ms`**，与 conv3x3
的 `#blk/#reorder` 分离口径一致。→ `expectedOps`/ratio 不再被 reorder 污染。

### 4.3 结论

- **reorder 不是「该计费的成本」**，它是「布局没持久化」的症状；根治靠联合布局选择，
  不是加罚金。当前的保守计费是**过渡手段**，且已引入 §4.2(b) 的记账 bug（已修）。
- **expectedOps 不是罚金**；它的偏差来自**单墙尺子**（§1），由多墙模型解决。
- 两者叠加造成的「blk 明明更快却选不中」/「某层 ratio 莫名低」——一半是设计不完整，
  一半是记账 bug，**不是物理必然**。

---

## 5. 复现 / 使用

```bash
# 锁频（跑基准前），无免密 sudo 也可
scripts/gpu_clocks.sh lock
scripts/gpu_clocks.sh status
# ... 跑 kernel_bench / kernel_autotune ...
scripts/gpu_clocks.sh unlock

# 噪声对照：同 config 连跑，看 min vs median
#   （autotune 已改用 min；kernel_bench 打印的是 median，用于与历史文档对齐）

# 诊断宏（R41）：--conv 第 18 字段=PROBE，第 23 字段=PF
scripts/gpu_guard.sh run docker run --rm --memory=2g --device=/dev/dri/renderD128 \
  -v "$PWD":/workspace/infvino -w /workspace/infvino infvino-dev:latest bash -lc '
    for p in 0 1 2 3; do OV_SLM=4 ./build/kernel_bench --op conv3x3ov \
      --conv-shape 64,64,80,80 --conv 8,2,1,32,16,1,1,1,3,1,16,0,0,0,0,0,0,$p --iters 10; done'
```

---

## 6. 待办（按 ROI）

1. **多墙 ceiling**：先加 `W_dram`（唯一字节）与 `W_lat`（经验 derate），`binding` 输出。
2. **`kernel_diag`**：把 R41 的 probe + min + 分类 + 候选建议做成一个命令。
3. **锁频 + min + top-K 复测**进 `autotune.py` 默认流程。
4. **推广联合布局不动点**到 conv1x1/depthwise（去掉保守计费）。
5. 接 `unitrace`/MD API 做 L3/DRAM 字节与 EU stall 的交叉验证（可选，增强）。
