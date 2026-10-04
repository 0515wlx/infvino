# Round 41：conv kernel 离天花板的差距——系统归因（流水 / GRF-SLM / 带宽）

> 问题：R40 找出「离指令天花板较远」的 conv 层（stride-2 大层、`Cout≤16`、`Cin≤16`、
> 小空间、stem）。本轮要**判定差距落在哪堵墙**：
> (a) 软件流水没做好，(b) GRF/SLM 容量与带宽的耦合，(c) DRAM/L3/LLC 带宽。
>
> 方法：`ops/EU/cyc` + **指令配额**（`ocloc` 反汇编）+ **feed 隔离探针**（`conv_ov.cl`
> 新增 `-DPROBE`：去掉输入 load / 去掉权重 load / 去掉 store）+ **手工软件预取实验**
> （`-DPF`）+ **存储层级实测**（`kernel_bench --op bandwidth/memlat`）。全部默认关闭，
> 不影响生产；`model_check`/`verify` 无回归。

---

## 0. 判决先给（一句话）

**主因既不是「软件流水没写好」，也不是「LLC/DRAM 带宽不够」，而是「无 L1 + ~150 cyc 的
访存延迟被 GRF 上限卡死的单线程 ILP / 受 7 线程/EU 限制的并发」——即 `(b)` 的
GRF↔延迟耦合；DRAM/L3 带宽只在两类层上是主因：`(1)` 工作集 > 3.75 MB 的层（stride-2
大层）与 `(2)` **输出写很大**的层（stem、窄输出）。**

三条独立证据：

1. **去掉两个 global feed 后，多数层到指令配额的 87–100%**（80×80 64→64：13.6→**20.2**，
   配额 22）→ 计数流水本身没问题，卡在「喂」。
2. **手工软件预取（PF，双缓冲 `in[]`）零收益**（13.59→13.63，全在噪声内）→ 不是
   「流水没排好」，IGC 已经在排；再往前排也没有更多寄存器/并发可用。
3. **去掉 store**：stem `3→16` **1.65→4.58（+178%）**、`16→16` 5.55→7.35（+32%）→
   这些层是**写带宽**主导，纯物理。

---

## 1. 实测基线（本轮，同机 Iris Xe 80EU/1.3GHz）

### 1.1 存储层级（`kernel_bench --op bandwidth / memlat`）

| footprint | copy 读写 | read-only |
|---:|---:|---:|
| 1 MB | 136.5 GB/s | 81.8 GB/s |
| 2 MB | 107.4 | 89.7 |
| 4 MB | 54.2 | 94.0 |
| 8 MB | 22.1 | 80.0 |
| 16 MB | 19.1 | 29.0 |
| 64 MB | 19.5 | 22.6 |

- L3(3.75 MB) 内 copy ~107–136 GB/s，>4 MB 断崖到 54→22 → **DRAM ≈ 19–22 GB/s**。
- 只读带宽在 ≤8 MB 保持 ~80–94 GB/s（读比读写便宜得多）。

### 1.2 访存延迟（pointer-chase）

| 工作集 | 4 KB | 256 KB | 4 MB | 16 MB |
|---|---:|---:|---:|---:|
| cyc/access | 138 | 199 | 148 | 150 |

**~150 cyc 且几乎与工作集无关**（无 L1 → 与 FMA 延迟 ~30 cyc 相差 5×）。这是本轮所有
归因的物理常数。

### 1.3 ISA（`ocloc` 反汇编，OBW8/OBH2/SG16/SLM_DIV=1）

主循环（`L1608`）**297 `mad` / 432 指令 = 0.688**，13 条 send，2 个 `(16|M0)` 累加器组；
**GRF max = r127，无 scratch spill**。→ 指令配额 = 32×0.688 ≈ **22.0**；
寄存器已顶到 128 GRF 墙但不 spill。

---

## 2. 归因矩阵（feed/store 隔离探针）

`ops/EU/cyc`；配额 ≈ 22。探针：`noIn`=去掉输入 global load，`noW`=去掉权重 block read，
`noBoth`=两者都去，`noStore`=只写一次（谓词恒假）。

| 层（输出） | full | noIn | noW | noBoth | noStore | noBoth/配额 | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| 80×80 s1 64→64 | 13.6 | 17.5 | 15.6 | **20.2** | 14.1 | **0.92** | **feed 延迟**（输入 > 权重） |
| 80×80 s2 64→64 | 11.0 | 16.3 | 11.8 | **19.9** | — | 0.90 | **输入 feed 延迟** |
| 20×20 s2 128→256 | 9.8 | 16.0 | 11.5 | **19.1** | — | 0.87 | **输入 feed 延迟**（SLM 正在治） |
| 160×160 s2 16→32 | 5.4 | 6.7 | 6.0 | 8.1 | — | 0.37 | 混合：**工作集 4.9MB>L3** + Cin 小 |
| 20×20 s1 51→51 | 3.4 | 5.3 | 4.7 | 4.8 | — | 0.22 | **固定开销**（K 短 / 网格小） |
| 160×160 s1 16→8 | 3.4 | 4.6 | 4.2 | 5.4 | 3.7 | 0.25 | **lane 浪费**（8/32）+ feed |
| 160×160 s1 16→16 | 5.6 | — | — | 8.8 | **7.4** | 0.40 | **store 带宽** + lane 浪费 |
| stem 320×320 s2 3→16 | 1.6 | 1.8 | 1.7 | 1.9 | **4.6** | 0.09 | **store/DRAM 带宽**主导 |

读法（三个决定性行）：

- 60×80 族：**noBoth 到 19.9–20.2 ≈ 配额 22 的 90%** ⇒ 数据通路的计数/流水没有浪费，
  差的全部是「等 load / 等 weight」。
- **stem 反例**：去掉 store 才暴涨，去掉 feed 没用 ⇒ 它不是算力问题，是**写 3.28 MB
  输出**打满了 ~23 GB/s 的写通道（3.28 MB / 144 µs ≈ 23 GB/s，占 225 µs 的 ~64%）。
- **Cout=8 反例**：noBoth 只有 5.4 ≈ 配额的 0.25 = 8/32 的 lane 利用率 ⇒ **结构 lane 浪费**。

---

## 3. 三堵墙逐条判决

### 3.1 软件流水：**否**

- 手工把下一 `kd` 的输入块预取进第二组寄存器（`-DPF=1`，双缓冲 + 旋转）：

  | 层 | PF=0 | PF=1 |
  |---|---:|---:|
  | 80×80 s1 64→64 SLM4 | 13.59 | 13.63 |
  | 80×80 s2 64→64 SLM4 | 10.92 | 11.05 |
  | 20×20 s2 128→256 SLM8 | 10.39 | 10.33 |
  | 160×160 s2 16→32 SLM2 | 5.31 | 5.63 |

  **全部噪声内**（且 PF 输出与 PF=0 **逐位一致**）。R34 的 `#pragma unroll 4` 同样负结果。
- 结论：IGC 已把 load 提前调度；差距不是「没排流水」，而是**排了也填不满**——要同时
  在飞的东西超过了单线程能给的数量。

### 3.2 GRF / SLM 容量-带宽耦合：**是（根因）**

- 访存延迟 ~150 cyc（§1.2），FMA ~30 cyc。要在单线程里盖住一次 load，需要 ~150 cyc 的
  **独立**工作量；而每线程独立累加链 `ACC = OBW×OBH`（8×2 只有 16）。
- 每 EU 7 线程 ⇒ 聚合独立链 `7×ACC ≈ 112`（8×2），**仍 < 150**，所以第一个 mad 之后仍有
  气泡；这正是 full/配额 ≈ 0.6 的来源。
- 想加大 `ACC` 就撞布局墙：`half2` 累加器 **2 GRF/个**，`OBW×OBH=16` 已吃 32 GRF，
  加上 IGC 的 ~85–95 GRF overhead 与输入缓冲 → **GRF max=127（§1.3）**。**ACC 被
  128-GRF 硬墙封顶在 ~16–19，刚好盖不住 150 cyc。**
- 逃生口是**线程级并发**而非单线程 ILP：`SLM_DIV`（一个 WG 塞多组在飞工作）确实有效
  （R39 已接），但总线程/EU 仍是 7，收益有上限——这解释了为什么 SLM 在小网格 +20–50%
  却在大网格饱和。

> 也就是说：这一档的「离天花板」是 `128 GRF 数据寄存器 ≈ 40` 与 `访存延迟 150 cyc`
> 两个硬件事实耦合出的**结构性 ~30–40% 缺口**，不是编码缺陷。

### 3.3 DRAM / L3 / LLC 带宽：**是，但只在两类层**

- **工作集 > 3.75 MB 的层**：`80×80 s2 64→64`（输入 3.28 MB + 输出 0.82 MB ≈ 4.2 MB）、
  `160×160 s2 16→32`（4.9 MB）→ 输入读溢出 L3 到 DRAM；再叠加 feed 延迟。这类层
  `noIn` 增益最大（+45%/+23%）。
- **输出写主导的层**：stem（store 3.28 MB）、窄 Cout（`16→16` store +32%）→ 写通道
  ~20–23 GB/s 打满。kernel 内无法减少字节（输出是契约），**只能靠融合/持久布局**。
- 其余（80×80 s1 64→64 = 1.7 MB；20×20 s2 128→256 = 0.6 MB）**全程 L3 命中**，是纯延迟，
  与带宽无关。

---

## 4. 逐 regime 结论（决定「还该不该写 kernel」）

| regime | 主墙 | kernel 内可动？ | 动作 |
|---|---|---|---|
| 大空间 s1·对齐 | feed 延迟（GRF↔latency） | 无（PF/unroll 已否） | 认账；靠 SLM/occupancy 已吃到的部分 |
| stride-2 大层 | 输入 feed 延迟 + 溢出 L3 | 无（PF 否） | 认账；或减少 footprint（布局/融合） |
| 小空间大 K | 输入 feed 延迟 | 已被 SLM 治（R39） | 已尽 |
| `Cout≤16` / `Cout%32` | lane 浪费（+ 窄层 store） | 是：`CB=8`/OSV=16 | **R40 已加 `CB=8`** |
| stem `Cin=3` | **store/DRAM 带宽** | 否 | 靠**融合**或持久布局 |
| `Cin%16`、小 K | 固定开销 | 边际 | 认账 |

**结论**：conv kernel 侧已基本触及物理墙。唯一还能靠 kernel 候选拿的是 **lane 浪费**
（R40 的 `CB=8`、以及未做的 OSV=16），其余要靠 **融合 / 布局**（引擎级）或换数据通路
（会撞同一组 GRF/SLM/延迟约束）。

---

## 5. 本轮已做的 kernel 尝试与结果

| 尝试 | 做法 | 结果 |
|---|---|---|
| **软件预取 `PF`** | `conv_ov` 双缓冲输入块，提前一个 `kd` 发 load | **负结果**（噪声内）；证明流水不是缺口 |
| **feed 隔离探针 `PROBE`** | 1=no input / 2=no weight / 3=both / 4=no store | **诊断工具**（默认关）；给出 §2 归因 |
| **`CB=8` native 候选** | `Cout≤8` 时加 `CB=8`（R40） | **正**：`16→8` 4.16→5.03（+21%） |
| **stride-2 `OBW=7`** | R40 候选 | 正（~+4%） |
| **修 `--verify` 参考激活** | `benchConv`/`benchConvBlk` 的参考只算裸卷积、且 blk 把 HardSwish 写成旧码 `ACT==2` | **修复**：两者统一按 canonical conv3x3 码（1=SiLU, 3=HardSwish）算参考。修后 ov/blk/native 的 `--verify` 在 ACT=1/3 下 **PASS**（mean_rel ~1–5e-3） |

> **反查结论**：R33 §5 记录的「`kernel_bench --op conv3x3 --verify` 对 native 是坏的
> （mean_rel≈0.5）」是**工具 bug**（参考没加激活），不是 native kernel 的数值问题。
> 修参考后 native `64→64 80×80` ACT=0/1 均 `mean_rel≈3e-3` PASS。此前用 `--verify`
> 得出的「native 数值不可信、必须走 numtest」的结论应撤回。

> 生产默认零改动（`PROBE=0`/`PF=0`；`CB=8`/`OBW=7` 是候选，待 retune）；`verify`
> （SLM=1/2/4）与 `tuning_test` 无回归，PF 与 PF=0 输出逐位一致。

---

## 6. 回看 autotune 的不足（基于上面的墙分类）

1. **只有一把尺子（FMA 指令配额）**。stem（store 带宽）、`Cout=8`（lane 浪费）的
   `ratio` 是 0.09/0.26，会被当成「巨大 headroom」——但**没有候选能拿到**。
   注册表里有 `Bottleneck` 枚举，但 conv3x3 各族**一律标 `Fma`**，没有 roofline/延迟尺子。
   → 应按 `OpSignature` 特征给该层**分类墙种**，再选对应尺子（计算/roofline/launch/延迟）。
2. **候选空间对撞上的墙无能为力**：没有减 store 的候选、没有 OSV=16、没有 lane=space
   variant。autotune 只能「在给定候选里选最快」，不能**由墙反推该造什么候选**。
   → 缺一层「瓶颈 → 候选生成」的反馈（本轮的 `PROBE` 正是这个反馈的原型）。
3. **无逐候选墙遥测**：`TuningEntry` 只有 ms/ops/expected；没有 GRF/SIMD/SLM/占用。
   → 无法回答「这层为什么慢」，只能回答「谁最快」。
4. **近噪声取舍**：feed 受限层里多个 config 相差 <5%（如 stride-2 的 6×2/7×2/5×2），
   单次计时会把噪声当胜负。→ 热点层应 top-2 复测 / 加 iters。
5. **计费不全**：blocked 族的 reorder 税没进候选比较（R37 §8）；`refreshExpected`
   曾漏 conv3x3（R39 已补）；`expectedOps` 的 conv3x3 兜底仍是 osv32 几何。
6. **缓存跨模型共享**：同一 shape 在 y8/y11 的布局持久化不同时无法各存一份（R37 §8）。

> 一句话：**autotune 已经能「在候选里选最优」，但在 kernel 触到物理墙之后，它的价值
> 取决于「候选集是否覆盖对的轴」和「尺子是否对得上下面的墙」——这两点它目前都没有。**

---

## 7. 复现

```bash
# 存储层级
scripts/gpu_guard.sh run docker run --rm --memory=3g --device=/dev/dri/renderD128 \
  -v "$PWD":/workspace/infvino -w /workspace/infvino infvino-dev:latest bash -lc '
    ./build/kernel_bench --op bandwidth --mb 4 --iters 8
    ./build/kernel_bench --op memlat --sizes 4,256,4096,16384'

# feed 隔离探针（--conv 第 18 字段 = PROBE；第 23 字段 = PF）
#   no input : PROBE=1   no weight: PROBE=2   both: PROBE=3   no store: PROBE=4
scripts/gpu_guard.sh run docker run --rm --memory=2g --device=/dev/dri/renderD128 \
  -v "$PWD":/workspace/infvino -w /workspace/infvino infvino-dev:latest bash -lc '
    for p in 0 1 2 3; do OV_SLM=4 ./build/kernel_bench --op conv3x3ov \
      --conv-shape 64,64,80,80 --conv 8,2,1,32,16,1,1,1,3,1,16,0,0,0,0,0,0,$p --iters 6; done'

# 手工软件预取（PF=1）
#   --conv 8,2,1,32,16,1,1,1,3,1,16,0,0,0,0,0,0,0,0,0,0,1

# ISA 配额 / GRF
ocloc compile -file kernels/conv_ov.cl -device tgllp \
  -options "-DOBW=8 -DOBH=2 -DSTRIDE=1 -DPAD=1 -DACT=1 -DRES=0 -DSG=16 -DSLM_DIV=1 -cl-mad-enable -cl-fast-relaxed-math" \
  -output /tmp/ov
ocloc disasm -file /tmp/ov_tgllp.bin -device tgllp -dump /tmp/ov_isa
```
