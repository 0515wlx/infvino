# Xe-LP (Gen12) ISA 逆向：cache 层级与 EU 寄存器全貌

> 对象：本机 **Intel i5-1135G7 / Iris Xe 80EU（Tiger Lake，Gen12LP，`tgllp`）**。
> 目的：把「能控制哪些 cache 层级」和「每 EU 到底有多少通用/专用寄存器」查清楚，给 kernel
> 分块与寄存器预算一个**硬件事实**基准。
>
> 方法 = **官方文档** + **离线 ISA 逆向**（`ocloc` 反汇编，不需要 GPU，遵守
> [`benchmark_protocol.md`](benchmark_protocol.md) 的 R3）+ 本项目已有的微基准实测
> （[`kernel.md`](kernel.md) R10–R17）。
>
> 标注约定：`[官方]` = Intel 公开文档；`[ISA逆向]` = 本机 `ocloc` 反汇编实证；
> `[本项目实测]` = `kernel.md` 微基准；`[推断]` = 由上述证据推导。

---

## 0. 一句话结论

1. **Xe-LP 没有通用（可读可写）L1 数据 cache**（Intel 自己的架构表：Xe-LP 每 Xe-Core
   `L1 = 0 KB`）。全局 load/store 经 Data Port 直达 GPU L3（软件常称 L2/LLC）。因此
   **任何全局数据的复用只能显式放进 SLM 或寄存器**——这从硬件层面解释了 R17「SLM 是
   延迟隐藏的载体，不是可绕开的开销」和 conv 的 staging 瓶颈。`[官方]+[ISA逆向]`
2. 真正可被 kernel 作者**直接控制**的只有：**寄存器(GRF)**、**SLM**、以及
   **只读/常量/采样器**路径；L3 的 cacheability/逐出策略要被 **MOCS（驱动）** 和
   **L3ALLOCREG（特权寄存器）** 控制，kernel 只能发**显式 fence/flush**。`[ISA逆向]`
3. **每 EU 寄存器**：`128 GRF/线程 × 32 B × 7 线程 = 28 KB/EU`；另有 ARF 专用寄存器
   （8 个累加器、2 个标志寄存器、8 个 message 寄存器等）。Xe-LP **只有 128-GRF 模式**，
   没有 Xe-HPG 的 256-GRF large mode。`[官方]+[ISA逆向]`

---

## 1. 本机身份与可查询参数

`[官方/clinfo]`（`clinfo` 实测，2026 驱动栈）：

| 项 | 值 |
|---|---|
| CPU/SoC | 11th Gen Intel Core i5-1135G7（Tiger Lake-U） |
| iGPU | Intel Iris Xe Graphics（Gen12LP，`tgllp`） |
| EU / DSS | **80 EU**（=5 个 16-EU Dual Subslice） |
| GPU 时钟 | 1300 MHz（`gt_RP0`；clinfo `Max clock 1300 MHz`） |
| 执行宽度 | sub-group 8 / 16 / 32；**每 EU 发射 1 条向量指令**（R13） |
| OpenCL | 3.0 NEO，driver `23.17.26241.33`，IGC `1.0.13822.8`，ocloc 同源 |
| 每线程 GRF | **128 × 32 B** |
| 线程/EU | **7** |
| 每 WG SLM | **64 KiB**（`CL_DEVICE_LOCAL_MEM_SIZE = 65536`） |
| 最大 WG | 512 |
| 全局 cache（=GPU L3） | **3 932 160 B = 3.75 MiB**，line = 64 B |
| 主机 L3（LLC） | **8 MiB**（i5-1135G7），CPU 与 iGPU 共享 |
| 内存 | 7.5 GB 与 iGPU 共享；实测 DRAM ≈ **19–23 GB/s**（单通道） |

---

## 2. 完整存储层级（外部 → 内部）

Intel 文档里同一个结构有**三套名字**，先做消歧：

| PRM 里的名字 | 计算 API 里的名字 | 本机实际 | 谁控制 |
|---|---|---|---|
| GFX **L3** Cache | **L2 / LLC / "global memory cache"** | 3.75 MiB | 硬件 + MOCS + 驱动 |
| **LLC** (CPU host last level) | LLC | 8 MiB | CPU/硬件，不可控 |
| **L1**（Xe-LP 无） | — | **0 KB** | — |

> 本项目 `kernel.md` 用的「LLC（片上共享，~4 MB 内持平、8 MB 崩塌）」= GPU L3(3.75 MiB)
> 的 4 MB 平台期 + CPU LLC(8 MiB) 的过渡段 + DRAM 崩塌，三段合一。

### 2.1 层级表（本机 80EU 实例）

| # | 层级 | 容量 | 带宽（实测/官方） | 延迟 | 控制方式 | 来源 |
|---|---|---|---|---|---|---|
| 0 | **GRF** 通用寄存器 | 128×32 B/线程；**28 KB/EU**；80EU ≈ 2.24 MB | 每周期喂 3 操作数 mad 到 ~86% | 1 cyc | 编译器/源码寄存器压力 | `[官方][ISA]` |
| 0' | **ARF** 专用寄存器 | 见 §3.2（很小） | — | 1 cyc | 编译器（隐式） | `[ISA逆向]` |
| 1 | **SLM**（软件 scratchpad） | **64 KB/WG**；**128 KB/16-EU DSS**；80EU ≈ 640 KB | **~555–780 GB/s** 读（配额 ≤4 KB/WG 上限） | 低 | **`__local`，完全可控** | `[官方][实测]` |
| 1' | 只读 / 常量 / 采样器 cache | 小、与 sampler 共享 | — | 低 | MOCS / `__constant` / `read_image` | `[官方]` |
| 1'' | **HDC L1**（Data Port L1） | 小；**仅 MOCS[6:1]=48…59 命中时启用**，否则 bypass | — | 低 | **驱动 MOCS**（kernel 不可直接控） | `[官方 PRM Vol6]` |
| 2 | **GPU L3**（软件 L2/LLC） | **3.75 MiB 可用**（物理 8 bank × 480 KB = 3.84 MB） | 读 **~270–345 GB/s**，写 ~330 GB/s；64 B/bank/cyc | ~几十 cyc | MOCS（UC/streaming/WB/WT）、L3ALLOCREG、显式 flush | `[官方][实测]` |
| 3 | GTI fabric | — | 172.8 GB/s；128 B/cyc 双向 | — | — | `[官方 Advisor]` |
| 4 | **CPU LLC** | 8 MiB（i5-1135G7） | 共享，实测 4–8 MB 段衰减 | ~40 cyc | 不可控（MOCS LeCC 略可影响） | `[官方][实测]` |
| 5 | **DRAM** | 7.5 GB 共享 | **~19–23 GB/s** | ~200–400 cyc | 不可控 | `[实测]` |

> **SLM 容量口径**：Intel oneAPI 文本写「每个 DSS 128 KB SLM」，架构表又写「每 Xe-Core
> 64 KB」。本机 OpenCL 报 `LOCAL_MEM_SIZE = 64 KiB`（per work-group 上限），Advisor perf
> model 报 96EU 整片 `SLM_size = 768 KB` = 6×128 KB。三者调和为：
> **16-EU DSS 有 128 KB，单个 work-group 最多吃 64 KB**，所以一个 DSS 可同时驻留 2 个
> 用满 SLM 的 WG。80EU（5 DSS）≈ 640 KB 总 SLM。`[官方][推断]`

### 2.2 为什么「没有 L1」是硬事实

- Intel oneAPI《Intel Xe GPU Architecture》架构表（2025-2）明确列出每 Xe-Core 的
  `L1 cache size`：Xe-HPG/HPC = 192–512 KB，**Xe-LP (Tiger Lake) = 0 KB**。`[官方]`
- Xe-HPG 白皮书说 Xe-HPG **新引入** read/write L1，可与 SLM 动态划分（最多 192 KB L1
  或 128 KB SLM）；这句话反过来说明 **Xe-LP 没有这个 read/write L1**。`[官方]`
- 早期媒体（AnandTech/Tom's）说「Xe-LP 新增 L1 cache」，指的是
  **只读/texture cache** 与 **SLM**，不是通用数据 L1；按 Intel 架构表应记为 0。`[官方]`
- 本项目实测侧证：全局 streaming 的 `scanbw` 在 256 KB–2 MB 只有 270–293 GB/s 且随
  footprint 单调，看不到一个独立的 L1 平台期；而 SLM 带宽要显式写 `__local` 才有
  555–780 GB/s。`[本项目实测]`

> 对优化的直接含义：**全局数据没有「白送」的 L1 复用**。想要复用必须
> (a) 放寄存器（受 128 GRF 限制）、(b) 放 SLM（受 64 KB/WG 与带宽-occupancy 耦合限制）、
> (c) 指望 3.75 MB 的 L3——而 L3 对 kernel 是**不可编程的**，只能靠数据局部性撞。

---

## 3. 寄存器架构

### 3.1 GRF（General Register File，通用寄存器）

`[官方][ISA逆向]`

- **每线程 128 个 GRF，每个 256 bit = 32 B**，记作 `r0 … r127`；
  每个 GRF 可切成子寄存器 `rN.0 … rN.7`（8×32b），支持 `.f/.d/.ud/.w/.uw/.hf` 等类型。
- **每 EU 7 个硬件线程**（SMT），故 `128 × 32 B × 7 = 28 KB/EU`。
  与 Gen11 官方 `224 KB / 8 EU = 28 KB/EU` 完全一致。`[官方]`
- 80EU 整片 GRF ≈ **2.24 MB**。
- **只有 128-GRF 模式**：Xe-HPG 才支持 256-GRF large mode（占用率减半）；
  本机 `ocloc -device tgllp -options -cl-intel-256-GRF-per-thread` 能编译但
  **ISA 与默认逐字节相同**（说明被忽略），且 oneAPI 架构表 Xe-LP 一列固定 128。`[官方][ISA逆向]`
- 实测：`gemm_f16`/`conv3x3_f16` 编译后 `grf ≤ 112–127`，一旦数据 tile 顶到 ~126–128
  就 spill 或掉 SIMD8（R13/R14/R16），这是「128 GRF 墙」的物理来源。
- 线程启动时，**thread payload 预载入 `r0` 起始的 GRF**（PRM 前言：
  *"data is pre-loaded into the thread's GRF (starting at r0)"*），因此每个 kernel 开头都能
  看到 `mov r3.0 r0.0`——线程/工作组 ID 等来自 payload，而不是 ARF。`[官方][ISA逆向]`

### 3.2 ARF（Architecture Register File，专用/架构寄存器）

以下清单来自 **IGA（Intel Graphics Assembler，与 ocloc 同一套解码表）的寄存器规格**
（`intel-graphics-compiler/visa/iga/.../Models.cpp`，XE/Gen12 部分），并用本机反汇编交叉验证。

| 寄存器 | 编码 `RegNum[7:4]` | 宽度 | 作用 | 本机观察到 |
|---|---|---|---|---|
| `null` | 0x0 | 32 b | 丢弃/常量源 | ✅ 大量 |
| `a0` | 0x1 | 32 b | 地址寄存器（间接寻址 / send 地址基址） | — |
| `acc0 … acc7` | 0x2 | **每个 256 b（8×32b）** | 累加器；`mul/addc {AccWrEn}` 隐式写入 | ✅ `acc0.0`, `acc1.0` |
| `mme0 … mme7`（别名 `acc8…acc15`） | 0x2 | 每个 256 b | 数学宏只读结果 | — |
| `f0`, `f1` | 0x3 | 每个 64 b（`.0`/`.1` 各 32 b） | **标志寄存器**：谓词/条件码 | ✅ `f0.0`, `f0.1`, `f1.0` |
| `ce0` | 0x4 | 32 b | Channel Enable（执行掩码） | — |
| `msg0 … msg7` | 0x5 | 每个 32 b | message 控制 | — |
| `sp` | 0x6 | 16 B（`sp.0/.1` 各 8 B） | 栈指针 | — |
| `sr0` | 0x7 | 16 B（`sr0.0…sr0.3`） | 状态寄存器：EUID/TID、dispatch mask、System IP 等 | —（ID 走 payload） |
| `cr0` | 0x8 | 12 B（`cr0.0…cr0.2`） | **控制寄存器**：FP 模式/例外/线程优先级 | ✅ 每个 prologue `or cr0.0, 0x4C0` |
| `n0` | 0x9 | 12 B（`n0.0…n0.2`） | 通知寄存器（`sync.bar`/`sync.host` 隐式引用） | — |
| `ip` | 0xA | 32 b | 指令指针 | — |
| `tdr0` | 0xB | 16 B | Thread Dependency Register（软件 scoreboard） | — |
| `tm0` | 0xC | 20 B（`tm0.0…tm0.4`） | 时间戳 | — |
| `dbg0` | — | — | debug | — |

要点：

- **8 个累加器 `acc0–acc7`，每个 32 B**（Gen11 只有 2 个）；`acc8–15` 被 `mme0–7`
  复用。IGC 常把 `mul`+`mach` 的中间积放进 `acc0`，把 `addc {AccWrEn}` 的进位放进
  `acc0/acc1`——反汇编里 `acc0.0`/`acc1.0` 出现非常频繁。`[ISA逆向]`
- **2 个标志寄存器 `f0/f1`（4 个 32b 标志位）**，供 `cmp/if/while/csel/sel/break` 的谓词。
- `cr0.0` 在每个 kernel 前导被 `or` 上 `0x4C0`（设置 IEEE/denorm/线程模式等 FP 控制）。
- `sr0` 在 IGC 生成代码里**没有直接出现**：线程 ID 通过 `r0` 起始的 payload 获得
  （见 §3.1）。`sr0` 更多用于异常/系统例程与架构上可读线程状态。
- 本项目 R12 说的「Gen12 移除硬件 scoreboard，改用 SWSB/软件 scoreboard」对应
  `sync.nop`（SWSB 等待）与 `tdr0`；反汇编里 `sync.nop ... {$0.dst}` 就是编译器插入的
  寄存器/发送依赖同步。`[官方][ISA逆向]`

### 3.3 寄存器预算速查（本机）

| 单位 | GRF 个数 | 字节 |
|---|---|---|
| 每线程 | 128 | 4 KB |
| 每 EU（×7 线程） | 896 | **28 KB** |
| 每 16-EU DSS | 14336 | 448 KB |
| 整片 80EU（5 DSS） | 71680 | **2.24 MB** |

---

## 4. ISA 与访存指令

### 4.1 指令大类（本机反汇编实际出现）

| 类 | 指令 | 说明 |
|---|---|---|
| ALU/FP | `mov add addc sub mul mach mad dp4a cmp csel sel shl shr and or` | 8-wide FP/INT 向量 ALU；`mad`=FMA，`dp4a`=INT8 点积（khr_integer_dot_product） |
| 扩展数学 EM | `math.exp math.log math.rsqt math.irem` | 2-wide EM 流水（Port 1）；`rsqrt`→`math.rsqt` |
| 流控 | `if else endif while break` | 谓词化 SIMD，配合 `f0/f1` |
| 同步 | `sync.bar`（WG barrier）、`sync.nop`（SWSB 依赖等待）、`sync.allwr`（等所有写） | Gen12 用软件 scoreboard |
| 消息 | `send.dc0` `send.dc1` `send.smpl` `send.gtwy` `send.ts` | 全部访存/同步/结束都走 SEND |
| 粒子 | `(W)`、`(16\|M0)`、`{Compacted}`、`{@N}` | 写使能 / SIMD 宽度·通道偏移 / 压缩编码 / 依赖计数 |

> Xe-LP EU 每个周期可 co-issue：1 条 8-wide FP/INT + 1 条 2-wide EM，**Send/Branch 可与
> ALU 并行发射**（Hot Chips 2020 Xe-LP 幻灯片 / oneAPI）。`[官方]`

### 4.2 SEND 消息端口（ISA 逆向——这是「能控制哪些 cache」的落点）

本机反汇编（probe + 生产 kernel）出现的全部消息类型与描述符：

| 端口 | 描述符示例 | IGA 解码含义 | 方向 |
|---|---|---|---|
| `send.dc1` | `0x04205E00` / `0x04205E01` | untyped surface read with x | HDC Data Port（全局读） |
| `send.dc1` | `0x04025E00` | untyped surface write with x | HDC Data Port（全局写） |
| `send.dc1` | `0x0420A7FE` / `0x0414A7FF` | untyped / a64 atomic int32 add | 全局原子 |
| `send.dc1` | `0x04048AFF` | a64 atomic int32 signed max | 全局原子 |
| `send.dc0` | `0x04210500` / `0x04030500` | byte gathering read / byte scattering write **16b** | **SLM 读写** |
| `send.dc0` | `0x02484401` | oword aligned **block read x8** | SLM/块读（sub-group block IO） |
| `send.dc0` | `0x0219E000` | **synchronized global fence flushing** | 刷到 L3/内存（末尾） |
| `send.dc0` | `0x0219E100` | **synchronized global fence flushing L1** | 刷 **L1**（`mem_fence(CLK_GLOBAL)`） |
| `send.dc0` | `0x0219E0FE` | **synchronized SLM fence** | 刷 **SLM**（`mem_fence(CLK_LOCAL)`） |
| `send.smpl`/`smp` | `0x04258000` | sampler sample（lod=0，sampler 0） | 只读/texture cache |
| `send.gtwy` | `0x02000004` | signal barrier | Message Gateway（跨 EU 同步） |
| `send.ts` | `0x02000010` | end of thread (EOT) | Thread Spawner |

即：**Xe-LP 的 load/store/fence/atomic/barrier/EOT 全部是 SEND 消息**，没有独立的
`ld/st` 标量访存指令；cache 策略编码在**消息描述符**里。

### 4.3 描述符里的 cache 控制字段（IGA 解码字典）

从本机 `libiga64.so` 抽出的 ISA cache 选项字符串：

| 字段 | 可选值 | 层级 |
|---|---|---|
| L1 cache policy | `uncached (bypass)`、`streaming`、`writeback`、`writethrough`、`use state settings for both L1 and L3`（默认） | **L1** |
| L3 / 逐出策略 | `evict (dirty lines invalidated and evicted)`、`invalidate (all clean lines, but do not evict)`、`discard (dirty and clean lines invalidated without eviction)` | **L3/L1** |
| flush 消息 | `Flush L1`、`L1Flush`、`L3 implies L1 flush` | L1 → L3 |
| 约束 | `SLM forbids cache control options`、`atomic L1 must be an uncached option`、fence 的 BTI 只能是 `0x0` 或 `0xFE` | — |

→ **ISA 层面可表达的 cache 控制层级 = L1（cache / uncached / streaming / WB / WT）、
L3（cacheability + evict/invalidate/discard）、SLM（只能 fence，不能设 cache 策略）。**
L2/CPU-LLC/DRAM 在 GTI 之后，ISA 不直接暴露。`[ISA逆向]`

---

## 5. 「能控制哪些 cache 层级」——按控制主体分层

| 层级 | 谁能控制 | 在 OpenCL/本项目里怎么用 | 备注 |
|---|---|---|---|
| **GRF** | 编译器（受源码寄存器压力影响） | 减小 live 变量、控制 tile；`intel_reqd_sub_group_size` 钉 SIMD | 128 GRF 硬墙（R13/R14） |
| **SLM** | ✅ **kernel 作者直接控制** | `__local` + `barrier`；`async_work_group_copy`；block read/write | 唯一完全可编程的片上存储 |
| **只读/常量/采样器 cache** | 半可控 | `__constant`、`read_image`、`read_only`/`__global const`、`restrict` | 路由由 IGC 决定 |
| **HDC L1**（Data Port） | ❌ 驱动（MOCS） | **仅 MOCS[6:1]=48…59 才启用**；否则 bypass，直达 L3 | PRM Vol6；kernel 无 API |
| **GPU L3 (3.75 MB)** | ❌ 驱动 + 特权寄存器 | MOCS：`UC / WT / WB`（`L3CC`/`LeCC`）；`L3ALLOCREG` 按 way 分区（DC/RO/URB/Command） | kernel 只能用**局部性**去撞 |
| 显式刷/逐出 | ✅（间接） | `barrier` / `mem_fence` → ISA 的 **SLM fence / L1 flush / global flush** | R17 已观察 |
| **CPU LLC (8 MB)** | ❌ | — | iGPU 与 CPU 共享 |
| **DRAM** | ❌ | — | ~19 GB/s 单通道 |

补充 `[ISA逆向/libigdfcl strings]`：IGC 内部有
`LscLoadCacheControlOverride` / `LscStoreCacheControlOverride` /
`TgmLoadCacheControlOverride` / `TgmStoreCacheControlOverride`、
`ForcePrefetchToL1Cache`、以及 `-cl-intel-256-GRF-per-thread` 等开关；这些是
**编译器/驱动内部**选项，不是标准 OpenCL API——所以从 kernel 源码**无法**直接给
某个访问打 L1/L3 cache hint，只能靠限定符与算法结构。

---

## 6. 对本项目 kernel 优化的直接结论

1. **没有通用 L1** ⇒ 全局复用只能靠 SLM/寄存器。R17 的结论（`conv3x3_f16` 是
   staging（global→SLM + barrier）主导，SLM 是延迟隐藏载体）从架构上被坐实：
   去掉 SLM（`GN`/`send.sg` 直读全局）必然暴露 L3/DRAM 延迟。
2. **L3 只有 3.75 MB**，且不可编程 ⇒ 3×3/GEMM 的 tile 工作集应尽量落在 3.75 MB 内；
   超过就掉到 CPU LLC(8 MB) 再掉到 DRAM(~19 GB/s)，这与 R10 的
   「≤2 MB 快、4 MB 过渡、≥16 MB 崩」完全一致。
3. **SLM 64 KB/WG、128 KB/DSS**，且带宽 ≈ 与常驻 WG 数耦合（R10/R11）⇒
   「容量↔带宽↔并发」是同一约束；双缓冲/SLM 足迹不是免费。
4. **128 GRF/线程 × 7 线程**，每周期只发 1 条向量指令 ⇒ 可达 tile 算术强度被卡死
   （R13/R14 的 128-GRF 墙、SIMD8 悬崖）。
5. **ARF 只有 8 个累加器 + 2 个标志寄存器**：`acc` 适合做 `mul/mach` 与进位链的临时
   累加，但**不是**可用作大 tile 的通用存储（每个 32 B，且被 `mach/addc` 隐式占用）。
   真正可用的数据寄存器就是 128 个 GRF。

> 一句话：这台机器上「缓存不够」不是软件没优化好，而是 **Xe-LP 的设计里就没有通用
> L1**；可编程的只有 GRF + SLM，硬件缓存只有一块不可编程的 3.75 MB L3。所有分块
> 策略本质都是在 **128 GRF / 64 KB SLM / 3.75 MB L3** 三个容量之间做取舍。

---

## 7. 复现（全部离线，不碰 GPU）

```bash
# 1) 获得 ISA（不需要 GPU，遵守 benchmark_protocol R3）
cd /tmp/opencode/xelp
ocloc compile -file /path/to/kernel.cl -device tgllp -options "-cl-std=CL1.2" -out_dir ./out
ocloc disasm  -file out/kernel_tgllp.bin -device tgllp -dump ./isa
ls ./isa/*_KernelHeap.asm        # 每个 kernel 一个反汇编

# 2) 枚举本机 kernel 用到的 send 消息 / ARF / cache 选项
cat isa/*_KernelHeap.asm | grep -oE 'send\.[a-z0-9]+ .*//.*' | sort -u
cat isa/*_KernelHeap.asm | grep -ohE '\b(null|acc[0-9]+(\.[0-9]+)?|f[0-9]+\.[0-9]+|cr[0-9]+(\.[0-9]+)?|sr[0-9]+|a[0-9]+)\b' | sort -u

# 3) 挖 IGA 的 cache 选项字典
strings /usr/lib/x86_64-linux-gnu/libiga64.so.1 | \
  grep -iE "uncached|streaming|writeback|writethrough|evict|invalidate|discard|flush L1|SLM forbids"

# 4) 设备侧 cache 容量（需要 GPU，短命令）
clinfo | grep -iE "Global Memory cache size|Local memory size|Max compute units|Sub-group sizes"

# 5) 官方文档
#   oneAPI: Intel Xe GPU Architecture（架构表：Xe-LP L1=0KB, L3=3.75MB, SLM, 7 threads/EU）
#   Tiger Lake Open Source PRM Vol.7 Memory Cache（GFX L3 480KB/bank）
#   Tiger Lake PRM Vol.6 Memory Views（MOCS：HDC L1 48–59、L3CC/LeCC）
#   Gen11 Architecture（RF=28KB/EU、SLM=64KB/subslice、历史对照）
#   IGA Models.cpp（ARF/GRF 寄存器规格）
```

---

## 8. 官方来源

- Intel oneAPI GPU Optimization Guide — *Intel Xe GPU Architecture*（2025-2 / 2024-x）：
  架构对比表（Xe-LP `L1=0 KB`、`L3=3.75 MB`、`SLM per Xe-core`、`7 threads/XVE`、`128 GRF`）。
- Intel oneAPI GPU Optimization Guide — *Intel Iris Xe GPU Architecture*（2023-0）：
  DSS=16 EU、SLM 128 KB/DSS、7 threads、sub-group 8/16/32。
- Intel oneAPI GPU Optimization Guide — *Registers and Performance* / *Small vs Large Register Mode*。
- *Intel Processor Graphics Xe-LP API Developer and Optimization Guide*（ID 676369）：
  「>73 B SLM/lane 会掉 SIMD 宽度」、寄存器压力与 SIMD 宽度、L3 带宽建议。
- *Tiger Lake Open Source PRM*：Vol.7 Memory Cache（GFX L3，480 KB/bank、64 B line、1b LRU、
  按 way 分区 L3ALLOCREG）、Vol.6 Memory Views（MOCS：`L3CC` UC/WB、`LeCC` UC/WT/WB、
  `HDC L1` 48–59 索引）、Vol.1 Preface（GRF/ARF/Thread Payload 定义）。
- *The Architecture of Intel Processor Graphics Gen11*：RF 224 KB/8EU=28 KB/EU、SLM 64 KB/subslice。
- Hot Chips 2020 *Intel Xe*（Xe-LP EU 发射端口、Subslice 组成）。
- Intel *Introduction to the Xe-HPG Architecture*：对照 Xe-HPG 新引入 read/write L1。
- intel-graphics-compiler / IGA `Models.cpp`：Gen12 ARF/GRF 寄存器规格（acc0–7、f0/f1、sr0、cr0…）。
- 本机 `libiga64.so.1`（`1.0.13822.8`）与 `ocloc` 反汇编：cache 选项字典与真实消息描述符。
