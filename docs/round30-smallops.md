# Round 30：剩下的 kernel（非 conv/gemm）的瓶颈定位与物理模型

> 目标：把 R1–R29 用来优化 GEMM/conv 的那套流程——**profile → 离线 ISA → 硬件模型 →
> 上限标定 → 判断撞哪堵墙**——原样套到「剩下的 kernel」：`concat4 / copy_c / slice_axis /
> permute / resize_nn / maxpool / ew_binary / ew_unary / softmax / gap / depthwise / bmm`。
>
> 结论先给：这些算子的算术强度只有 ~0.25–0.5 FLOP/byte，**根本碰不到 FMA 发射**，
> 所以「ops/EU/cyc」不是它们正确的标尺；正确的上限是**内存 roofline（bytes/EU/cyc）
> 再叠加每 dispatch 的固定开销**。在 Iris Xe（80 EU / 单通道 LPDDR / 19–23 GB/s DRAM）
> 上，搬运类算子的现实天花板只有 FP16 FMA 峰值的 **~1–5%**。**concat4（占整网 ~10%）
> 已经贴住 DRAM 墙**；`copy_c/slice/maxpool/softmax` 也已到「launch floor + L3 copy」的
> 模型上限；真正还有 1.5–5× 空间的是 **`depthwise`（地址/边界指令）、`bmm`（网格饥饿）、
> `ew_binary_bcast`（整数 div/mod）、`permute/resize`（跨步访问）**，而不是带宽。
>
> 本轮**不改数值语义**：只改 dispatch（标量广播快路径，输出逐位不变）+ 补上「中间标准」
> 的真实模型（`expectedOps`，见 §7）。

---

## 0. 口径与数据来源

- 时间：`kernel_run --plan models/<m>/model.plan --report --iters 3`（OpenCL event，
  只计 kernel 自身，与 `infvino_bench` “net only”一致）。设备：
  `Intel Iris Xe Graphics [OpenCL 3.0 NEO] EU=80 clk=1300MHz`。
- 内存 roofline：`kernel_bench --op membw/scanbw/bandwidth`（本轮实测，§2）。
- 指令配额：离线 `ocloc compile/disasm`（**不需要 GPU**），数主循环的 `mad` 占比。
- ops/EU/cyc 的定义沿用 `docs/kernel.md`：`flops × fps / (EU × clk)`。
  对 conv/gemm 用真实 FLOP；对小算子，调优器传 `flops = 2 × 输出元素数`，本文称
  该 proxy 为 **E**（`2·out / EU / cyc`），保证与 `TuningEntry.ops` 同量纲。

---

## 1. 账本：剩下多少、谁最大

三模型 `--report` 的非 conv/gemm 部分（只列本轮关心的算子；conv/conv1x1/depthwise 中
`conv1x1` 属 GEMM 体系已优化）：

| 算子 | yolov8 nodes / ms | yolo11 nodes / ms | mobilenet nodes / ms |
|---|---:|---:|---:|
| `concat4` | 19 / **1.400** | 23 / **1.567** | — |
| `ew_binary`（含 bcast） | 15 / 0.503 | 24 / 0.585 | 15 / 0.120 |
| `depthwise` | — | 7 / 0.562 | 11 / **0.407** |
| `maxpool` | 3 / 0.153 | 3 / 0.153 | — |
| `resize_nn` | 2 / 0.115 | 2 / 0.122 | — |
| `softmax_axis` | 1 / 0.070 | 2 / 0.127 | — |
| `slice_axis` | 4 / 0.040 | 7 / 0.056 | — |
| `permute` | 1 / 0.040 | 3 / 0.150 | — |
| `ew_unary` | 2 / 0.028 | 2 / 0.030 | 11 / 0.065 |
| `copy_c` | 16 / 0.183 | 18 / 0.196 | — |
| `gap` | — | — | 10 / 0.090 |
| `bmm` | — | 2 / 0.300 | — |
| **小计** | **2.53 ms** | **3.85 ms** | **0.68 ms** |
| 整网 kernel busy | 14.57 | 15.97 | 2.81 |
| **占比** | **17.4%** | **24.1%** | **24.3%** |

- **`concat4` 是最大单项**（yolov8 1.40 ms、yolo11 1.57 ms，占整网 ~10%）。
- mobilenet 的剩余开销几乎就是 **depthwise（0.41）+ ew（0.19）+ gap（0.09）**。
- 节点数 91–199，`launch floor`（每 dispatch ~3.5 µs）本身就贡献 ~0.3–0.7 ms。

---

## 2. 内存 roofline 本机标定（本轮新增，kernel_bench）

conv/gemm 的墙是 ALU/寄存器；这些 op 的墙是内存。先把本机的内存层级重新量一遍
（此前 R10/R18 只有大致值）：

| 足迹 | copy(read+write) | read-only |
|---:|---:|---:|
| 4 KB | 3.0 GB/s | 15.3 GB/s |
| 64 KB | 22.7 | 165.7 |
| 256 KB | 46.4 | 233.8 |
| 512 KB | 63.3 | 252.3 |
| **1 MB** | **89.7** | **320.2** |
| 2 MB | 62.4 | 264.1 |
| 4 MB | 44.6 | 157.8 |
| 8 MB | 21.0 | 75.8 |
| 64 MB（DRAM） | **19.4** | 22.8 |

读数：

- **LLC(≈3.75 MB) 内 copy 只有 ~90 GB/s**（不是 300+），因为 copy 同时占读+写端口；
  read-only 才能到 ~320 GB/s。
- **DRAM 墙 19.4 GB/s**（copy）；这与 CPU memcpy 19.9 GB/s 一致，是单通道上限。
- 4–16 KB 的 “带宽” 只有 3–8 GB/s，本质是 **launch/尾延迟 floor ≈ 3.5 µs/dispatch**：
  4 KB copy = 0.003 ms，与数据量无关。

**把 DRAM/L3 墙换算成 ops/EU/cyc 尺度**（fp16，读+写 4 B/elem）：
`E_ceiling = BW / (2·EU·clk)` → L3 90 GB/s ⇒ **0.43**；DRAM 19.4 GB/s ⇒ **0.093**。
而 FP16 FMA 峰值是 32。→ **搬运算子的物理上限只有峰值的 0.3–1.3%**。
read-only（2 B/elem）则 `E = BW/(EU·clk)`：L3 ⇒ 3.1，DRAM ⇒ 0.22。

---

## 3. 为什么 conv/gemm 的模型套不上

| 算子 | 算术强度 | 撞的墙 |
|---|---:|---|
| conv3x3 64→64@80×80 | ~276 FLOP/B | ALU 发射 / 寄存器（R1–R24） |
| gemm | 高（tile 复用） | 同上 |
| `concat/copy/slice/permute` | **0** | 内存带宽 |
| `ew_binary` | 0.5 FLOP/B（1 mad : 3 访存） | 内存 / 索引 |
| `softmax` | ~0.5 | 内存 + 3 趟串行 |
| `depthwise` | 4.5 FLOP/B（K²=9，理想） | 地址/边界**指令数** |
| `bmm` | 高但网格极小 | 网格饥饿 / 延迟 |

所以对大多数剩余算子，`ops/EU/cyc` 的“上限”按 FMA 算（32）毫无意义，必须换成
**bytes/EU/cyc**（或等价地：launch + 字节/带宽 的最小时间）。

---

## 4. 逐类模型（measured vs 物理上限）

### 4.1 纯搬运：`concat4 / copy_c / slice_axis / permute / resize_nn`
**模型**：`T_min = launch(≈3.5 µs) + bytes / BW_copy(footprint)`，bytes = 读+写（各 2 B/elem）。
脚本 `scripts/analyze_smallops.py` 按**逐节点**足迹累加上限；下表 `eff = 上限 / 实测`，
eff≈1 表示已到该模型的上限，eff<1 表示还有余量（实测比模型慢）。

| 算子 | 实测 (y8/y11) | 模型上限 | eff | 判定 |
|---|---:|---:|---:|---|
| `concat4` | 1.400 / 1.567 ms（**22–24 GB/s**）| 0.72 / 0.84 ms（按 L3 46 GB/s）| 0.52–0.54 | 见下：**DRAM 级，已到实际顶** |
| `copy_c` | 0.183 / 0.196 | 0.207 / 0.224 | **1.13** | 已到顶（launch+L3 copy）|
| `slice_axis` | 0.040 / 0.056 | 0.042 / 0.065 | **1.05–1.15** | 已到顶 |
| `maxpool` | 0.153 | 0.158 | **1.03** | 已到顶（按每输出 K² 读模型）|
| `softmax_axis` | 0.070 / 0.127 | 0.108 / 0.158 | **1.24–1.54** | 已到/超过模型 |
| `permute` | 0.040 / 0.150 | 0.039 / 0.066 | 0.44–0.98 | attention 转置还有 ~2× |
| `resize_nn` | 0.115 / 0.122 | 0.052 | 0.42–0.45 | 还有 ~2× |
| `ew_binary` | 0.503 / 0.585 / 0.120 | 0.249 / 0.326 / 0.111 | 0.49–0.93 | 还有 1–2×（广播见 4.2）|
| `ew_unary` | 0.028 / 0.030 / 0.065 | 0.018 / 0.018 / 0.084 | 0.60–1.29 | 还有 ~1.6× |
| `gap` | 0.090 | 0.064 | 0.71 | 还有 ~1.4× |

**`copy_c` 是模型拟合的漂亮验证**：单节点 ~0.61 MB，`3.5 µs + 0.61 MB/68 GB/s ≈ 12.5 µs`
（脚本按足迹插值得 ~11.5 µs），实测 11.4 µs。说明这一类**既不是带宽也不是索引问题，
而是「launch floor + L3 copy」两段之和**；再压只能减少 dispatch 数（融合）或提高 L3 copy
带宽（硬件，做不到）。`slice/maxpool/softmax` 同理，实测已在模型上限附近或以上。

**`concat4` 的判定**：整网里读 33.6 MB、写 33.6 MB，1.40 ms ⇒ 24 GB/s。它**低于**纯 L3
模型（46 GB/s，eff 0.52），但**高于**实测 DRAM copy 墙（19.4 GB/s，纯 DRAM 下界
= 33.6MB/19.4 ≈ 1.73 ms，比实测 1.40 ms 还慢）。也就是说：concat 的输入是刚算完的
conv 输出、很快被后续 conv 挤到 DRAM，**in-situ 落在 DRAM 与 L3 的混合区，已经贴着
实际内存墙**——它不是指令/索引问题。单节点 autotune 时输入还热、能到 ~56 GB/s，
这正是同一 kernel 在部署时“掉一半”的原因（调优器测热缓存、部署是冷缓存）。
**要再快只能减少物化**（把 concat 融进下游 conv，或让 Split 分支直接写进 concat 目标
缓冲），属结构改动，不在本轮。

### 4.2 逐元素：`ew_binary / ew_unary / ew_binary_bcast`
**模型**：同 4.1，但 3 条内存流（a+b 读、y 写）。理想 L3 上限 ≈ 60–90 GB/s。
实测：

- `ew_binary` 非广播：21 GB/s（占 L3 上限 ~1/3）——受 launch + 每 WI 处理元素数限制，
  VEC=2 已由调优器选中。
- `ew_unary`：20–25 GB/s，同上。
- **`ew_binary_bcast` 的大节点是明确的“非物理”瓶颈**：pose 解码的三条
  `[1,17,2,8400]`（n=285600）实测 **~12 GB/s、≈35 cyc/element**。离线 ISA 显示通用
  `ew_binary_bcast` 每元素要跑 **4 组 `math.iqot/irem`（整数除/取模）** 的 rank 循环，
  而 footprint 只有 ~1.1 MB（本该 L3 级）。ISA：190 条指令里 0 mad、4 iqot + 4 irem。
  **这是整数除法延迟 bound，不是带宽。**

  三条 op 的广播模式（`bdims`）：
  1. `Slice_2 * Const_24`：b 是 **真标量**（b_st 全 0），却仍走通用 rank 核（`b_scalar=1`）；
  2. `Mul_3 + Const_25`：b 在 d1×d3 上变化（等价 `[1,17,1,8400]`）；
  3. `Add_3 * Const_26`：b 只沿最后一轴（`[8400]`）。

**已落地（第 1 种）**：`PlanModel::smallSig/smallLaunch` 现在识别 `b_scalar=1`——即使
plan 带了 `bdims`，也走标量 `ew_binary` 核（a 与 out 同形连续 ⇒ aidx=i、bidx=0，
数值逐位相同）。同会话 A/B（`kernel_run --report --iters 5`，输出逐位相同）：

| 模型 | `ew_binary` old → new | 整网 busy old → new |
|---|---|---|
| yolov8n | 0.500 → **0.430** ms（−14%）| 14.584 → **14.539** ms |
| yolo11n | 0.594 → **0.500** ms（−16%）| 15.974 → **15.720** ms（−1.6%）|

第 2/3 种可用「3-D 网格 + 运行期 stride」去掉 div/mod（同 R23 concat4、R28 `_3d` 的
思路），是剩余算子里下一个明确的可优化点（约 0.19 ms/模型）。

### 4.3 归约：`softmax / gap`
- `softmax_axis`（dfl，`[1,16,33600]`）：三趟串行 + exp，实测 27–31 GB/s（L3 上限
  320 的 ~10%）。网格只有 `outer×inner` 个 WI，且每 WI 串行扫 `axdim` → **延迟/网格饥饿**。
  长轴（attention `800×400`）已由 `softmax_axis_r`（SLM 树归约）接管（R29，4.25×）。
- `gap_r`：一个 WG 一个通道，网格只有 C（10–576）→ **网格饥饿**；实测 7 GB/s。
  模型 `T = launch + bytes/BW` 下它已经被 launch 完全支配（E floor 0.01）。

### 4.4 `depthwise`：不是带宽，是**地址/边界指令**
R29 已用 `depthwise_v`（滑窗寄存器复用）把标量版的重复全局读去掉，整网 1.38×/1.13×。
但离线 ISA 显示瓶颈已转移到**指令数**：

| kernel | 指令 | mad | mad 占比 | 每输出指令 |
|---|---:|---:|---:|---:|
| `depthwise_f16`（标量 K=3） | 523 | 16 | 3.1% | — |
| `depthwise_v`（TW=8,K=3） | **1485** | **128** | **8.6%** | **≈185 / 8 输出** |

`depthwise_v` 每 8 个输出要执行 ~1485 条指令（大量 `mov/add/cmp/sel/send` 来自
`xx>=0 && xx<W` 的逐 tap 边界谓词与地址计算），只有 128 条 mad。因此：

> **指令配额上限 = 32 × mad_frac ≈ 32 × 0.086 ≈ 2.8 ops/EU/cyc**（FLOPs 已含 K²）。
> 实测最好 ~0.3–0.5，即 **只有配额的 ~11–18%**；但**配额本身就低**——所以 depthwise
> 的现实上限不是 32，而是 ~2.8。要突破必须减少非 mad 指令（向量载入 + 去边界谓词 /
> padding 预处理），而不是加寄存器 tile。

### 4.5 `bmm`：FP32 配额 8.1，实测 2（网格饥饿）
`bmm_t`（寄存器分块，R29）K 循环 ISA：**260 条指令 / 132 mad = 50.8% mad**。
FP32 峰值 16 → **配额 = 16 × 0.508 ≈ 8.1 ops/EU/cyc**。
attention 两个 shape（`[1,2,400,32,400]` 用 TM8×TN4、`[1,2,64,400,400]` 用 TM4×TN4）
的网格 `N/TN × M/TM × B` = **10 000 / 3 200 个 WI（≈7.8 / 2.5 sub-group/EU）**，
后者占用不足；且每 WI 是长 K 依赖链、A/B 从 L3 载入 → **实测 ~2.0–2.1 ops/EU/cyc
（配额的 ~26%）**。R29 已把 bmm 从 0.73 提到 2.1（2.66×）。剩余空间在
**split-K / 更小 tile 提并行度**，但会改变浮点累加顺序（破坏与 `bmm` 的逐位一致），
本轮不动。

---

## 5. 上限汇总：每个 kernel 的上限与“哪堵墙”

| 算子 | 实测 | 物理/结构上限 | 具体限制 | eff=上限/实测 |
|---|---:|---|---|---:|
| `concat4` | 22–24 GB/s | **DRAM 19.4 / L3 混合** | LPDDR 带宽（冷缓存） | **≈0.5→已到实际顶** |
| `copy_c` | 52–54 GB/s | 3.5µs + bytes/L3-copy | L3 copy + launch | **1.13（到顶）** |
| `slice_axis` | 40–46 | 同上 | L3 + launch | **1.05–1.15（到顶）** |
| `maxpool` | 0.153 ms | K² 读 + launch（0 mad） | issue（K² 谓词） | **1.03（到模型顶）** |
| `softmax` | 0.070/0.127 | 3 趟读 + 写 | 串行/网格饥饿 | **1.24–1.54（到顶）** |
| `permute`（attention） | 0.150 | L3 + launch | 跨步访问/div | 0.44（~2× 空间）|
| `resize_nn` | 0.115–0.122 | in+out @ L3 | 读放大 + launch | 0.42（~2× 空间）|
| `ew_binary` | 0.50/0.59/0.12 | 3 流 @ L3 | launch + 每 WI 元素数 | 0.49–0.93（1–2×）|
| `ew_binary_bcast`（大 n） | ~12 GB/s | 3 流 @ L3 | **整数 div/mod 延迟** | **~0.3（3–5×）** |
| `ew_unary` | 0.028–0.065 | 2 流 @ L3 | launch | 0.60–1.29（~1.6×）|
| `gap` | 0.090 | launch/latency | 网格饥饿（C 个 WG） | 0.71（~1.4×）|
| `depthwise` | 0.3–0.5 ops | **2.8 ops（ISA 配额）** | 地址/边界指令 | ~0.15（5–9×）|
| `bmm` | 2.1 ops | 8.1 / 网格 ~2 | 网格饥饿 + K 依赖链 | 0.26（1–4×）|

`eff` 用 `scripts/analyze_smallops.py` 的逐节点上限/实测汇总。maxpool 按每输出 K² 次
带边界读建模（ISA 0 mad），实测已到该模型顶；要再快只能减少 tap 的指令数。

**哪些已经是物理/结构极限**：
1. **`concat4` → DRAM 墙（19–24 GB/s）**，LPDDR 单通道硬上限，除减少物化（融合）外无解。
2. **`copy_c/slice_axis/maxpool/softmax` 的 launch floor + L3 copy**：单节点只有
   0.1–0.6 MB，3.5 µs 的 dispatch 已占一半时间，实测已到模型顶（eff 1.0–1.5）。
   除减少 dispatch 数（融合）外无解。
3. **`depthwise` 的 ~2.8 ops 指令配额**：结构（地址/边界指令）决定，属“换数据通路”级。

**还有空间，但不是带宽**：`permute`（跨步/div，~2×）、`resize`（读放大，~2×）、
`ew_binary_bcast`（去整数 div/mod，~3×）、`gap`（网格，~1.4×）、`bmm`（并行度，~1–4×）。

---

## 6. 与整网的账

- 剩余 kernel 合计 **17–24%** 的 GPU busy；其中 **concat4 ≈ 10% 且已到 DRAM 顶**，
  `copy/slice/maxpool/softmax` 已到模型顶——这几项没有余量。
- 因此“把剩余 kernel 优化到极限”对整网的乐观收益约 **3–4%**（剩余可压项：
  pose 广播 ~0.19ms、permute/resize ~0.1ms、ew_binary ~0.25ms、bmm/depthwise），
  **不是 2× 空间**。
- 真正的大头仍是 conv（占 yolov8 ~64%，见 R24/R25/R26）。

---

## 7. 已落地（本轮）

### 7.1 标量广播快路径（kernel 侧，正向）

见 §4.2：`b_scalar=1` 不再走通用 rank 核。输出逐位不变；`ew_binary` −14%/−16%。

### 7.2 `expectedOps` 的真实模型（替换 `return 8.0`）

`src/Tuning.cpp::expectedOps` 此前对小算子一律返回占位 `8.0`，使中间标准 `ratio` 无意义。
本轮按上面的物理模型改写（**只影响报告/ratio，不参与选优**——选优仍按最小 ms）：

- 新增 `kBwCurve`（§2 实测 copy 带宽-足迹）→ `copyBwGbps(footprint)`；
- 新增 `smallMemCeiling(out, bytes) = 2·out / (EU·clk·(launch + bytes/BW))`，`launch=3.5 µs`；
- 按 op 计算 bytes（读+写、fp16）：
  - `copy_c/slice/concat/permute`：4·out；`ew_unary` 4n；`ew_binary` 2(2n 或 3n)；
    `ew_binary_bcast` 2(2n+C 或 3n)；`resize/maxpool` 2(in+out)，maxpool 读按 out·K²；
    `softmax` 8n（三趟读+写）；`gap` 2(C·HW+C)；
  - `bmm`（计算受限）：`16 × 0.508 / K × gridFactor`；
  - `depthwise`（指令受限）：`32 × 0.086 × gridFactor`（≈2.8）。
- `tuning_test` 增加 6 条断言（单调性、量纲范围、depthwise 配额）。

对缓存里既有签名的复算（参考实现）显示：搬运类 `ratio` 落在 0.7–1.5（模型与实测同量级），
`depthwise/bmm/softmax/maxpool` 落在 0.15–0.6，正好把“离上限最远”的层标出来。

---

## 8. 下一步（按预期收益）

1. ~~pose-decode 广播的标量分支~~（**已落地**，−0.07/−0.09 ms）；剩余两条混合 stride
   广播用 3-D 网格 stride 核去掉 `iqot/irem`（预计再 −0.19 ms/模型，~1.2%）。
2. `maxpool` 向量化 tap（去边界谓词 + 一次读 K² 个连续 half）。
3. `softmax` 的 2 趟（在线 max+sum）或减少写回读。
4. `concat` 融合进下游 conv（结构改动，收益最大但风险高，属独立一轮）。
5. `depthwise` 去边界谓词的 interior fast path（R28 FIT 思路）。

---

## 9. 复现

```bash
# profile（每模型一条，遵守 benchmark_protocol.md）
docker run --rm --memory=3g --memory-swap=3g --pids-limit=256 --device=/dev/dri/renderD128 \
  -v "$PWD":/workspace/infvino -w /workspace/infvino infvino-dev:latest bash -lc '
  for m in yolov8n-pose yolo11n-pose mobilenetv3-small; do
    timeout 80 ./build-ct/kernel_run --plan models/$m/model.plan --report --iters 3; done'

# 内存 roofline（本机标定）
./build-ct/kernel_bench --op scanbw --sizes 4,16,64,256,512,1024,2048,4096,8192
./build-ct/kernel_bench --op membw  --sizes 4,16,64,256,512,1024,2048,4096,8192 --iters 200
./build-ct/kernel_bench --op bandwidth --mb 64 --iters 30

# 离线 ISA（不需要 GPU）
ocloc compile -file kernels/ops.cl -device tgllp -options \
  "-DEW_VEC=4 -DSM_WGS=64 -DGAP_WGS=64 -DBMM_TM=8 -DBMM_TN=4 -DBMM_UK=4 -cl-mad-enable -cl-fast-relaxed-math" \
  -output /tmp/ops
ocloc disasm -file /tmp/ops_tgllp.bin -device tgllp -dump /tmp/opsd
#   bmm_t 主循环 260 指令 / 132 mad = 50.8%；ew_binary_bcast 每元素 4×iqot+4×irem
ocloc compile -file kernels/conv_general.cl -device tgllp -options \
  "-DDW_K=3 -DDW_S=1 -DDW_P=1 -DDW_ACT=0 -DDW_TW=8 -cl-mad-enable -cl-fast-relaxed-math" -output /tmp/dw
ocloc disasm -file /tmp/dw_tgllp.bin -device tgllp -dump /tmp/dwd
#   depthwise_v 1485 指令 / 128 mad = 8.6% -> 配额 2.8

# 中间标准自检（不需要 GPU）
cmake --build build-ct --target tuning_test && ./build-ct/tuning_test
```
