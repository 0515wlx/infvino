# 自研 GPU kernel 优化日志（ops/EU/cycle）

> 目标：为 `yolov8n-pose` / `yolo11n-pose` / `mobilenetv3-small` 手写 OpenCL kernel，
> 在 **Intel Iris Xe（80 EU / 1300 MHz）** 上逐轮提升硬件效率，指标
> **`ops/EU/cycle = GFLOPs × fps / (EU × clock)`**（单位 FLOPs/EU/cycle）。
> 硬件上限：**FP32 = 16**（8 FMA/EU/cyc），**FP16（packed half2）= 32**（16 FMA/EU/cyc）。
> 每一轮都必须通过 `scripts/kernel_check.py` 数值检验（vs numpy FP32 参考）。

口径：单流、只计 kernel 自身时间（OpenCL event profiling），与 `infvino_bench`
的 “net only” 一致。

## 硬件与理论峰值

| 项 | 值 |
|---|---|
| GPU | Intel Iris Xe Graphics（TGL iGPU） |
| EU | 80 |
| 频率 | 1300 MHz（`gt_RP0_freq_mhz`；i5-1135G7 官方 Graphics Max Dynamic Frequency = 1.30 GHz） |
| FP32 峰值 | `80 × 1.3e9 × 16` = **1.664 TFLOP/s**（8 FMA/EU/cyc） |
| FP16 峰值 | `80 × 1.3e9 × 32` = **3.328 TFLOP/s**（16 packed FMA/EU/cyc，需 half2/half8） |
| 运行环境 | `infvino-dev`（在 develop 镜像上补 `ocl-icd-opencl-dev` + `opencl-headers`） |

> ⚠️ **口径修正**：早期文档把 “16” 当作 **FP16** 理论上限，实为 **FP32** 上限。
> Xe-LP 的 EU 是 8 条 FP32 通道；FP16 通过 **packed half2** 提供 **2×** 速率
> （16 FMA/EU/cyc = 32 FLOPs/EU/cyc）。所以下面历史表格里的 `ops/EU/cyc` 数值不变，
> 但相对上限的百分比改为对 **32** 计算（= 旧 “% of 16” 的一半）。
>
> 注意：`kernel_bench` / `scripts/model_check.py` 目前仍按 16 打印百分比（数值不变）。

### 硬件实测（寄存器 FMA + 各层 cache 带宽）

> **ISA 逆向全貌**（cache 层级、EU 寄存器、SEND/cache 控制字段）见
> [`xe-lp-isa.md`](xe-lp-isa.md)。关键事实：Xe-LP **没有通用 L1**，可编程缓存只有
> GRF(128/线程) 与 SLM(64 KB/WG)，硬件缓存只有一块不可编程的 3.75 MB L3。

用独立 OpenCL 微基准测得（寄存器内多条独立 FMA 链 + 大 work-item 数；访存用 float4
扫工作集，每趟固定 ~512 MB 流量以消除计时误差）：

| 项 | 理论（FLOPs/EU/cyc） | 实测 | 备注 |
|---|---|---|---|
| FP32 FMA | 16 | **15.0**（~1565 GFLOP/s） | 94% |
| FP16 FMA（显式 half2/half8 + 循环展开） | 32 | **29.4**（~3053 GFLOP/s） | 92%，≈ FP32 的 2× |
| FP16 FMA（标量 half） | 32 | 22–26 | 只部分打包；要吃到 2× 必须显式向量化 |

| 存储层级 | 聚合带宽（80 EU） | 备注 |
|---|---|---|
| L1 / SLM | ~650–780 GB/s（read），~650 GB/s（write） | 16 KB/WG 最好 |
| L2 / LLC（片上共享，~4 MB 内持平，~8 MB 后崩塌） | ~330–345 GB/s | ~8 MB 与 CPU 8 MiB L3 共享 |
| DRAM（≥64 MB） | **~19 GB/s** | 与 CPU `memcpy` 19.9 GB/s 一致 → 单通道内存上限 |

> 因此「读一次用一次」（AI≈1）的算子在这台机器上必然被 DRAM 卡死（19 GB/s）；
> 计算要想跑满，数据必须留在 LLC（8 MB）内做复用。

## 三条数据通路的 ops/EU/cyc 现实上限（R18–R21 结论）

> **本机（Iris Xe 80EU / 128 GRF、7 线程 per EU / 无通用 L1）上，三条数据通路的
> ops/EU/cyc 上限都是 ~16，只是各撞各的墙；direct conv 的 ~10.3 是当前最优。
> ~16 不是「没优化好」，而是不换硬件能力时卷积复用的现实天花板。**

| 通路 | 结构 | 上限 | 撞的墙 |
|---|---|---|---|
| **direct conv**（`conv3x3_f16`，生产）| lane=空间，通道在寄存器 + SLM staging | staging-free **~16.4**，整核 **~10.3** | global→SLM staging + 索引/边界/SLM 写指令 |
| **OpenVINO 式**（`conv3x3_sg`）| lane=输出通道，输入块驻寄存器 + `sub_group_broadcast` | **~16.0** | 1 条 broadcast : 1 条 mad |
| **Winograd F(2×2,3×3)** | 用变换把乘数减少 2.25× | **< 16** | 每 (tile, 通道) 需同时持有 16 个 m 值 → 128 GRF 放大 |

- 参照：纯寄存器 FP16 FMA 的结构上限是 **27.4–29.6（86–92%）**；理论峰值 32。
- 推导见：Round 14/18（direct 的 staging 与寄存器墙）、Round 19–20（L3 / OpenVINO 通路）、
  Round 21（Winograd）。数据通路只有 **GRF + SLM** 两级可编程低延迟存储，是共同根因。

## Round 0 —— 历史基线（OpenVINO 2023.0.2，已移除）

infvino 已不依赖 OpenVINO；下表为早期对照基线（整网 GPU，单流，net only）：

| 模型 | infer ms | fps | ops/EU/cyc | % of 32（FP16） |
|---|---|---|---|---|
| yolov8n-pose | 11.68 | 85.6 | 7.66 | 23.9% |
| yolo11n-pose | 12.41 | 80.6 | 5.89 | 18.4% |
| mobilenetv3-small | 1.86 | 539 | 0.31 | 1.0% |

## Round 1 —— FP16 tiled GEMM（conv 的公共底座）

**代码**：`kernels/gemm.cl`、`src/tools/kernel_bench.cpp`、`src/tools/kernel_numtest.cpp`。
**做了什么**：
1. OpenCL 运行时封装（`ClRuntime`）：设备枚举 / kernel 构建缓存 / event 计时。
2. FP16 tiled GEMM（`C[M,N]=A[M,K]·B[K,N]`），编译期 tile 可经 `-D` 扫参。
3. 正确的 **`half2` K-pair 打包（VEC2）**：B 在 SLM 里**转置**为 `BsT[N][K]`，
   使 `(k,k+1)` 沿 K 连续，`half2` 载入配对正确。
4. tile 扫描 + 数值检验。

**关键结论（踩坑/调优）**：

- **B 转置进 SLM** 是最大收益点：修掉 SLM bank conflict 后，小 N（YOLO head 常见）shape
  从 ~3.0 提升到 ~5–6 ops/EU/cyc。
- **`VEC2`（half2 mad）虽然数值正确但更慢**（见下），标量 `half` mad + `-cl-mad-enable` 更快；
  最终默认 `VEC2=0`。（注：这是本 GEMM 的**布局/转置开销**结论，并不否定 half2 的
  ALU 吞吐优势——纯寄存器 FMA 实测 half2 才能到 32 的上限。）
- `BK=8` 优于 `BK=16/32`；`BM=128` 优于 `64/256`。
- **最终默认 tile：`BM=128, BN=64, BK=8, TM=8, TN=4, VEC2=0, PAD=0`**（见 `Tiles.hpp`）。

**Result（kernel_bench，默认 tile）**：

| GEMM shape (M×N×K) | ms | GFLOP/s | ops/EU/cyc | % of 32（FP16） |
|---|---|---|---|---|
| 6400×64×64 | 0.083 | 629 | **6.05** | 18.9% |
| 1600×128×128 | 0.099 | 532 | **5.11** | 16.0% |
| 400×256×256 | ~0.13 | ~400 | ~4.0 | ~12.5% |
| 1024×1024×1024 | 2.56 | 838 | **8.05** | 25.2% |
| 4096×512×512 | 2.50 | 860 | **8.27** | 25.8% |

tile 扫描关键点（`kernel_bench --op gemm --shape 4096,512,512 --tiles ...`）：

| tiles (BM,BN,BK,TM,TN) | 4096×512×512 | 6400×64×64 | 说明 |
|---|---|---|---|
| 64,64,16,8,4（初版） | 5.02 | 3.67 | 无 B 转置 |
| 64,64,8,8,4 | 7.37 | 4.97 | BK=8 |
| 128,64,8,8,4 **VEC2=1** | 4.56 | 3.38 | half2 反而更慢 |
| **128,64,8,8,4 VEC2=0（默认）** | **8.27** | **6.05** | 当前最优 |
| 128,128,8,8,4 VEC2=0 | 8.52 | 3.70 | 大 N 好，小 N 差 |

**数值检验**（`scripts/kernel_check.py`，vs numpy FP32；
判据 = **相对误差** mean_rel 与 max_rel(amax)，不再使用余弦）：

```
gemm 6400x64x64      mean_rel=7.31e-04 max_rel(amax)=3.20e-03 PASS
gemm 1600x128x128    mean_rel=1.02e-03 max_rel(amax)=5.18e-03 PASS
gemm 400x256x256     mean_rel=1.49e-03 max_rel(amax)=6.44e-03 PASS
gemm 1024x1024x1024  mean_rel=6.93e-03 max_rel(amax)=2.05e-02 PASS
gemm 4096x512x512    mean_rel=2.61e-03 max_rel(amax)=1.22e-02 PASS
```

> 说明：FP16 累加 + 长 K 使 `max_rel` 升到 ~2%；`mean_rel` 仍在 1e-3–7e-3。

## Round 2 —— 原生直接卷积（roofline 决策）

用户建议：iGPU 是 SIMD、无 MMA 单元且带宽低，提示「抬高计算强度」；若 im2col-GEMM 的
计算强度也够高就不必。**先做 roofline，再决定。**

**实测机器平衡**：DRAM 约 **19–23 GB/s**（`kernel_bench --op bandwidth`；独立微基准
~19 GB/s，与 CPU `memcpy` 19.9 GB/s 一致 → 单通道内存上限）。完整层级见
「硬件实测（寄存器 FMA + 各层 cache 带宽）」。

→ FP16 ridge point = `3328 GFLOP/s ÷ 19 GB/s` ≈ **175 FLOP/byte**
（FP32 口径为 `1664 ÷ 19` ≈ 88 FLOP/byte）。

**逐层 AI**（`scripts/analyze_conv.py`，yolov8n-pose 73 个 conv / 9.18 G conv-FLOPs）：

- 主力 3×3 层（Cin=Cout=64，80×80）**完美复用 AI ≈ 276** → **compute-bound**
  （276 > 175，但余量比按 16 估计时小）。
- 若把这些层**物化 im2col**：A = M×K = 6400×576 fp16 = 7.4 MB 写+读 → **AI ≈ 30 < 175
  → bandwidth-bound**。✅ 用户判断成立。

**结论：采用原生直接卷积**（输入 halo 留在 SLM，不物化 im2col）。

**实现**：`kernels/conv.cl` → `conv3x3_f16`（groups=1，stride 1/2，fusion ACT：SiLU/Hardswish）。
结构：work-group 负责 `TX×TY` 空间 × `CB` 输出通道；输入 halo 每 Cin 分块只 stage 一次并
被全部 CB 通道复用；**VECC=1** 把 2 个输出通道打进 half2 累加器（权重通道连续存储），
在同样寄存器预算下把 CB 抬到 16。

**关键调优发现**：

1. **累加器寄存器预算 ≈ 32（half2）**：`TM×CB` 超过即溢出到私有内存，性能断崖式下跌
   （`TM=4,CB=16` → 32 half 溢出 → **24 ms / 0.14**；`TM=2,CB=16` → 16 half2 → **0.73 ms / 6.4**）。
2. **VECC（half2 通道打包）**：正确且把可用的 CB 抬高一档。
3. **输入条带复用**：每行一次性载入 `TM+KW-1` 连续输入，跨 kw 复用（省 1/3 SLM 读）。
4. `UNROLL_CI=3` 最佳；`CINC=32/64` 编译器直接 **build-fail**（本地内存/寄存器上限）。
5. **输入重读放大 = Cout/CB**：真实瓶颈是 x 方向未被完全复用（大 Cin/Cout 层尤甚）。

**Result（s1，`kernel_bench --op conv3x3`）**：

| 层 (Cin→Cout@HW) | 最佳 tile | ms | GFLOP/s | ops/EU/cyc | % of 32（FP16） |
|---|---|---|---|---|---|
| 64→64@80×80 | 128,8,2,16,16,u3 | 0.705 | 669 | **6.43** | 20.1% |
| 32→64@80×80 | 同上 | 0.391 | 604 | **5.80** | 18.1% |
| 64→128@40×40 | 同上 | 0.694 | 340 | 3.27 | 10.2% |
| 128→128@40×40 | 同上 | 1.345 | 351 | 3.37 | 10.5% |
| 256→256@20×20 | 同上 | 3.115 | 152 | 1.46 | 4.6% |
| 16→32@160×160 **s2** | 32,8,2,16,16,u3 | 4.611 | 102 | 0.98 | 3.1% |

**数值检验**（`scripts/kernel_check.py --skip-gemm`，vs numpy FP32；相对误差 mean/amax）：

```
conv 64x64x80x80 s1   mean_rel=3.08e-03 max_rel(amax)=8.19e-03 PASS
conv 32x64x80x80 s1   mean_rel=2.18e-03 max_rel(amax)=5.82e-03 PASS
conv 64x64x40x40 s1   mean_rel=3.06e-03 max_rel(amax)=7.23e-03 PASS
conv 16x32x160x160 s2 mean_rel=1.54e-03 max_rel(amax)=4.21e-03 PASS
```

> 对照：OpenVINO 整网 yolov8n-pose = 7.66 ops/EU/cyc。**主力 s1 3×3 层已到 6.43，
> 但整网还差 conv 覆盖度（stride2 / 大通道 / 1×1 / depthwise / 算子集）。**

## Round 3 —— 1×1 卷积 = 复用 GEMM

**关键点**：1×1 卷积 **就是** `gemm_f16`：`M=Cout, N=H·W, K=Cin`，权重 `[Cout,Cin]`、
输入 `[Cin,HW]`、输出 `[Cout,HW]` —— **布局天然吻合，无需任何转置**。
因此不写新 kernel，直接把 1×1 路由到已调好的 `gemm_f16`（`kernel_bench --op conv1x1`）。

**Result**（默认 GEMM tile `128,64,8,8,4`）：

| 1×1 层 (Cin→Cout@HW) | M×N×K | ms | ops/EU/cyc | % of 32（FP16） |
|---|---|---|---|---|
| 384→128@40×40 | 128×1600×384 | 0.273 | **5.54** | 17.3% |
| 192→128@40×40 | 128×1600×192 | 0.142 | 5.31 | 16.6% |
| 128→128@40×40 | 128×1600×128 | 0.099 | 5.08 | 15.9% |
| 96→64@80×80 | 64×6400×96 | 0.197 | 3.83 | 12.0% |
| 64→64@80×80 | 64×6400×64 | 0.138 | 3.65 | 11.4% |
| 512→256@20×20 | 256×400×512 | 0.270 | 3.73 | 11.7% |
| 256→64@20×20 | 64×400×256 | 0.132 | 0.95 | 3.0% |

**数值检验**（相对误差 mean/amax）：conv1x1 全部 PASS（`mean_rel ≤ 2.9e-03`）。

- 小尾巴：`M=Cout<BM=128`（head 的 64/51 通道）或 `N=HW` 很小（20×20）时，
  固定 block 浪费 → 需要按 M/N 自适应选 block（记为后续优化）。

## Round 4 —— stride2（部分完成）与 depthwise（未开始）

**stride2**：输入 tile 是输出的 2×2 空间，staging 成本天然高；调 `CINC`（受 SLM 限制，
stride2 下 `TX=64,CINC≤8`）后从 0.98 → **1.61**。

| stride2 层 (Cin→Cout@HW) | 最佳 tile | ms | ops/EU/cyc | % of 32（FP16） |
|---|---|---|---|---|
| 16→32@320×320 | 64,8,2,16,8,s2 | 1.227 | **1.85** | 5.8% |
| 64→128@80×80 | 同上 | 1.343 | 1.69 | 5.3% |
| 32→64@160×160 | 同上 | 1.412 | 1.61 | 5.0% |
| 3→16@640×640 | 同上 | 0.849 | 1.00 | 3.1% |
| 128→256@40×40 | 同上 | 2.793 | 0.81 | 2.5% |

数值检验：stride2 PASS（`mean_rel=1.54e-03, max_rel=4.21e-03`）。

- **depthwise（mobilenet 5×5/3×3）**：整网里先走通用 `conv_general`（正确性优先），
  性能待优化。

## Round 5 —— 三个模型端到端（ONNX → 自研 kernel）

**新增**：`kernel_run`（计划驱动的图执行器）、`scripts/onnx2plan.py`（ONNX→计划）、
`scripts/model_check.py`（端到端数值检验 + ops/EU/cyc）、`kernels/ops.cl`、
`kernels/conv_general.cl`。

**结果（单流，kernel 自身时间之和）**：

| 模型 | mean_rel | max_rel(amax) | kernel total | ops/EU/cyc | % of 32（FP16） |
|---|---|---|---|---|---|
| yolov8n-pose | 5.22e-04 | 1.03e-02 | 55.7 ms | **1.58** | 4.9% |
| yolo11n-pose | 8.19e-04 | 1.92e-02 | 61.4 ms | **1.16** | 3.6% |
| mobilenetv3-small | 1.08e-02 | 1.08e-02 | 6.5 ms | **0.16** | 0.5% |

**全部 PASS**（相对误差判据）。三个模型完整跑通（含 YOLO11 的 attention/PSA、
mobilenet 的 SE/depthwise/5×5）。

**端到端暴露的问题（下一步优化点）**：

- **`conv_general` 是最大瓶颈**：5×5/depthwise 与 yolo11 的 s2/大 K 层走的是
  「correctness-first」朴素 kernel（一个 work-item 一个输出）→ yolov8 里 30.2/55.7 ms。
  需要一个 tuned 版本。
- **整网远低于算子级**：算子级 3×3 有 6.43，整网只有 1.58——大量小算子
  （concat/copy/ew/bias_add；`bias_add` 在 yolo11 里 48 次）+ 每层一次 kernel 启动/同步。
  **融合（conv+bias+act + 消除 concat/copy）是下一步关键**。

## Round 6 —— Xe-LP 逆向：half2/half8 到底有没有用？+ conv 路由修正

**背景**：上一轮怀疑「GEMM/conv 没上 half2/half8」。用 `ocloc`（离线编译+`disasm`，不占 GPU）
反汇编真实 kernel，并用独立微基准逐层定位瓶颈。结论出人意料：**half2 不是关键**。

### 6.1 纯寄存器 FP16 吞吐（微基准，`cl_khr_fp16`）

8 条独立累加链、大 work-item 数、循环内无访存：

| 内核 | 实现 | 实测 ops/EU/cyc | 说明 |
|---|---|---|---|
| `fma_f32` | 标量 float | **15.3** | FP32 上限 16，96% |
| `fma_h1` | 标量 half | **28.8** | IGC **自动**打成 half2 |
| `fma_h2` | 显式 half2 | 29.5 | |
| `fma_h8` | 显式 half8 | 29.6 | |
| `fma_h16` | 显式 half16 | 30.1 | 逼近 32 上限 |

→ **硬件确实是 2× FP16，而且标量 half 也会被 IGC 自动向量化**。所以「朴素 half 代码 = 没享受到
2×」这个前提不成立；`mad ...:hf`（SIMD16）本身就是 packed 的。

### 6.2 逐层逼近：瓶颈其实在 SLM/调度，不在 FPU

| 实验 | 配置 | ops/EU/cyc |
|---|---|---|
| 纯寄存器 outer-product（N 打包） | 4×8 tile, ILP1 | 24.9 |
| 同上 ILP4 | | 28.8 |
| SLM 内层循环（`slm_prod`） | TM4 TN8, 标量 a + half2 b | 14.2 |
| SLM 内层 + half8 宽载 + A 转置（`slm_prod2`） | TM8 TN8 | 17.0 |
| **真实 `gemm_f16`** | 默认 tile | **8.0** |
| `gemm_np`（N 打包 + 宽载 + A 转置） | 64,64,32,8,8 | 5.05（更慢） |

- 真实 GEMM 的 `disasm`：每个 k-tile 256 条 `mad :hf`（对应 256 次标量 half FMA），
  即该布局下编译器**没有**把不同 `(i,j)` 的 FMA 两两打包；但即便如此，FPU 也不是瓶颈。
- 加大 tile（BM/BN）、换 N 打包、加宽 SLM 载入，都不能把真实 GEMM 拉高到微基准的 17 ——
  说明瓶颈是**全局/SLM 访存延迟 + 调度/occupied work-group 数**，而非 ALU。
- `DBUF`（k-tile 双缓冲）实测**负优化**（8.05 → 4.94）：即使把两个 SLM buffer 的索引做成
  编译期常量，仍变慢，说明问题不是「staging 延迟未被隐藏」。相关代码保留在 `gemm.cl`
  （`-DDBUF`，默认 0），作为已证伪的实验记录。

> **教训**：这台 iGPU（Iris Xe 80EU，单通道 DDR ~19GB/s）上，盲目追 half2/half8 收益有限；
> 真正的大头是**数据复用/分块**与**避免朴素 kernel**。

### 6.3 真正的端到端加速点：`conv_general` 占了一半

用逐节点计时（`kernel_run --report` 按节点 tag）定位 yolov8：

| 节点 | ms | 说明 |
|---|---|---|
| `conv_general` × 7 | **9.44** | K3S2 G1 的 stride2 层，朴素 kernel（~0.6–0.8 ops/EU/cyc）|
| `conv3x3` × 12 | 6.42 | s1 主力层 |
| 其余小算子 | ~2.7 | concat/bias/ew/gemm |

yolo11 里 `conv_general` 更是 12.24/20.41 ms（60%）。**修正**：这些 K=3、groups=1 的层
之前只把 **stride1** 路由到 `conv3x3_f16`，stride2 全走了朴素 kernel；而 `conv3x3_f16`
本来就支持 stride。于是：

1. `scripts/onnx2plan.py`：`use_direct = (kh==3 and groups==1 and sh in (1,2))`。
   - **s2** 用 `cfg=64,8,1,32,8,2`（TX=64,TY=8,**TM=1,CB=32**,CINC=8,s2）：halo 是 s1 的两倍，
     用 TM=1/CB=32 降输入重读放大（`Cout/CB`），CINC=8 保证 halo 不超 SLM。
   - **s1** 默认 `Conv3x3Cfg` 从 `TX128 TY8 TM2 CB16 CINC16` 改为 **`TX64 TY8 TM1 CB32 CINC16`**。
2. 结果（**逐层**）：低通道 s2 层最高 **4.5×**，高通道 s1 层约 **1.9–2×**：

| 层 (Cin→Cout@Hout, s) | 旧 conv_general/默认 | 新 cfg | 加速 |
|---|---|---|---|
| 32→64@160 s2 | 1.77 ms | 0.55 ms | 3.2× |
| 64→128@80 s2 | 1.37 ms | 0.54 ms | 2.6× |
| 128→256@40 s2 | 1.55 ms | 1.19 ms | 1.3× |
| 64→128@40 s1 | ~3.3 | 0.67 ms→6.46 ops | ~2× |
| 128→128@40 s1 | 3.5 ops | 6.44 ops | 1.9× |
| 256→256@20 s1 | 1.46 ops | 3.02 ops | 2.0× |

### 6.4 修正 `kernel_run` 的耗时统计 bug（重要）

`PlanModel::run()` 每次开头 `tprof_.clear()`，而 `kernel_run` 又把累计值 **除以 iters** →
「total kernel time」被低估 `iters` 倍（`1/iters` 缩放），`benchmark.md` 的旧数字（如
yolov8 18.61 ms / 4.74 ops）都是这个 bug 的产物。修正：`run()` 不再清空，改由
`clearProfile()` 在统计循环前显式清空；`kernel_run` 先跑一次 warmup 再计时。
修好后数值与 iters 无关（yolov8 恒为 ~27.0 ms）。

**真实结果**（单流、kernel 自身时间之和，warm）：

| 模型 | 旧（修正后）| 新 | 加速 | 新 ops/EU/cyc |
|---|---|---|---|---|
| yolov8n-pose | 55.8 ms / 1.58 | **27.0 ms** | **2.07×** | **3.26** |
| yolo11n-pose | 61.2 ms / 1.16 | **28.6 ms** | **2.14×** | **2.49** |
| mobilenetv3-small | 6.54 ms / 0.16 | 6.37 ms | 1.03× | 0.17 |

数值检验：`scripts/model_check.py` 三模型全部 **PASS**（相对误差判据不变）。
mobilenet 几乎没动：它的瓶颈是 576×49×576 这类 1×1 pointwise GEMM（N=HW 只有 49，
work-group 少、AI 低），换 tile 无效（~1.0 ops/EU/cyc），需要后续专门处理。

## Round 7 —— 纯 GEMM 瓶颈定位（结论：不是 FPU 流水，是 barrier/SLM/occupancy）

目标：把 `gemm_f16` 单独拆开，找真正的瓶颈。方法：给 `gemm.cl` 加临时探针宏
（`SKIP_STAGE`：只 stage 第一块、保留 barrier；`SKIP_COMPUTE`：只 staging、不做 mad），
配合 `ocloc disasm` 与独立微基准（`GBM` 默认 tile `128,64,8,8,4`，shape `4096×512×512`）。

**分解结果（旧结构，transposed-B, VEC2=0）**

| 版本 | ops/EU/cyc | 时间 | 含义 |
|---|---|---|---|
| normal | 8.37 | 2.46 ms | 完整 |
| `SKIP_STAGE`（只算+barrier）| 8.23 | 2.50 ms | **staging 在 BK=8 下几乎免费** |
| `SKIP_COMPUTE`（只 staging+barrier）| 24.25 | 0.85 ms | staging 本身 ~0.85 ms |
| normal, BK=16 | 4.82 | 4.28 ms | 加 staging 后崩 |
| `SKIP_STAGE`, BK=16 | **12.85** | 1.61 ms | 无 staging 时 BK=16 好得多 |

**关键结论**

1. **inner loop 不是 FPU 上限**：去掉 staging 后 `BK=16` 能到 12.85；纯寄存器外积微基准
   24.9–28.8；标量 half 已被 IGC 自动打包。ALU 有富余。
2. **BK=8 的「compute」主要被 barrier 限制**：每个 k-tile 只有 8 个 kk 的计算夹在 barrier 之间，
   编译器无法跨 barrier 预取 SLM。微基准里在循环内插 barrier 会把 13.2 打到 10.07。
   `BK≥16`（更多计算/barrier）可恢复，但**真实 kernel 的转置 B staging 在 BK≥16 时急剧变慢**
   （0.85→1.1→2.5 ms），抵消收益。
3. **staging 与 compute 是相加而非重叠**：normal ≈ staging + compute（2.46 ≈ 0.85 + 1.61 量级），
   因为二者被 barrier 串行化。
4. **ISA 佐证**：default kernel 静态指令 256 `mad` vs 252 `mov` + 143 `add` + 76 `mul`+76 `mach`
   + 67 `shl` + 91 `cmp` + 49 `csel` + 46 `if`——大量非 FPU 指令（staging/地址/边界）。

**试过但未取得正收益（都已实测，故未保留）**

| 尝试 | 结果 |
|---|---|
| N 打包 half2 + B 行主序（不转置）| 微基准 inner loop 14.08，但真实 kernel 仅 5.8（TN=8）/8.2（TN=4）|
| A 转置 + half8 宽载 | staging 转置有 bank conflict，5.05 |
| `-DDBUF`（编译期常量索引双缓冲）| 负优化（8.05→4.94）|
| `-DPIPE`（寄存器预取软流水）| 负优化（8.4→2.3，寄存器压力/occupancy 下降）|
| `BK=16/32/64` | 无 staging 时 12.8，有 staging 时 4–5 |
| 32-bit 索引（去掉 `size_t`）| 无变化 |
| `PAD`（SLM padding）| 无改善甚至更差 |
| `VEC2`（沿 K 打包）| 更慢（与历史结论一致）|

> **微基准的坑**：当 SLM 在循环内不变（只 stage 一次）时，编译器会把 SLM 载入**提升到循环外**，
> 于是测到的是「纯 mad」而非「SLM 载入 + mad」。所以 13–14 的「inner loop 上限」被高估；
> 真实 kernel 里 SLM 每 tile 变化，载入无法提升。Round 6 里的 17 也受此影响。

**下一步方向（未做）**：真正需要的是「更大的 BK + 廉价 staging + 足够 occupancy」的组合，
比如：(a) 用向量化 staging（`vload`/half2/half4）把 staging 指令数降 2–4×；
(b) 用 `async_work_group_copy`/子组 block IO 做无寄存器占用的异步 staging；
(c) 或干脆放弃大 K 的通用 tiling，针对 1×1 conv 的 (M=Cout, N=HW, K=Cin) 做专用内核。
当前 `gemm_f16` 仍是已验证的最优（~8.4 ops/EU/cyc @ 4096×512×512）。

## Round 8 —— GEMM staging：行主序 B + 向量化 staging（+1.2~1.33×）

Round 7 定位到「BK≥16 时 staging 崩掉」是最大症结。进一步确认**元凶是转置 B 的 SLM 写
bank conflict 随 `BKP=BK+PAD` 线性放大**（步长 `BKP` halfs → BK=8/16/32 时 4/8/16 路冲突）。
本轮做了两件事：

**(1) 行主序 B（不转置）+ half4 向量化 staging —— 正收益**

- `Bs[BK][BN]` 行主序：staging 写变得**连续、无冲突**；inner loop 的 `b[]` 也变成连续
  `half4` 载入（`b[j]=Bs[kk][lx*TN+j]`）。
- staging 用 `half4` 向量载入/存储，指令数降 ~4×；`K%4`/`N%4` 不满足时自动回退标量。
- 结果（`kernel_bench`，单配置）：

| shape | 旧 | 新 BK=8 | 新 BK=32 |
|---|---|---|---|
| 4096×512×512 | 8.37 | 10.02 | **11.09 (1.33×)** |
| 1024×1024×1024 | 8.05 | 9.79 | **10.91 (1.36×)** |
| 6400×64×64 | 6.04 | **6.93** | 5.25 |
| 1600×128×128 | 5.11 | 5.44 | 4.19 |
| 128×1600×384 | 5.54 | **6.65** | 5.23 |
| 400×256×256 | ~4.0 | 4.34 | 3.84 |

- **大 K 用 BK=32、小 K 用 BK=8**：已在 `PlanModel` 的 gemm 分派里按 `K>=256` 选（每个 gemm
  节点按需 build/cache kernel）。

**(2) 异步 staging（`async_work_group_copy`）—— 负收益**

- A 有 BM=128 行，逐行 async copy 不现实；只对 B 的 BK 行做 `async_work_group_copy`。
- 实测**显著更慢**（4096×512×512：10.11 → 3.86 ops/EU/cyc）——这台 iGPU 上逐行 async copy
  的开销远大于收益。保留为 `-DASYNC`（默认 0）作为已证伪记录。

**端到端**（kernel busy，warm）：yolov8 27.06→**26.38 ms**、yolo11 28.55→**27.91 ms**、
mobilenet 6.31→**6.10 ms**（GEMM 占比已很小，故整网提升仅 ~2–3%）。
三模型 `model_check` 仍全部 **PASS**。

到此 `gemm_f16` 从 ~8.4 → ~11 ops/EU/cyc（@4096×512×512）；剩余 gap 仍是 staging 未与
compute 重叠（v2 的思路是 `async`/子组 block IO 的**双缓冲**，本轮单缓冲 async 无效）。

## Round 9 —— GEMM 双缓冲软件流水 + 自适应 tile（+6~75%，端到端 gemm −18%）

Round 8 把 staging 做廉价后，最大症结是 **staging 与 compute 被 barrier 串行化**（Round 7：
`normal ≈ staging + compute`）。本轮真正把软件流水做出来，并发现**双缓冲的胜负完全由
SLM/WG 预算决定的 occupancy 支配**，因此最终落成**按 shape 自适应分派**。

### 9.1 双缓冲流水线：先在 ISA 上踩了「编译器把整个 compute 消掉」的坑

朴素写法（`cur = kt & 1` 运行时选 buffer）**编译出 0 条 `mad`**：`As_ + (kt&1)*BM` 让 SLM
基址数据相关，IGC 无法解析别名，直接把循环体删了（`ocloc disasm` 实测 mad=0，kernel 假快到
240 ops/EU/cyc、结果全错）。修法：**k-tile 循环显式 2× 展开**，让 buffer 下标成为编译期常量：

```
stage(tile0 -> buf0); barrier;
for (kt = 0; kt+1 < kTiles; kt += 2) {
  stage(tile kt+1 -> buf1);  compute(buf0); barrier;   // 预取下一块，算当前块
  stage(tile kt+2 -> buf0);  compute(buf1); barrier;
}
```

一个**正确性坑**：`As_` 是 `half[2][BM][AR]`，取缓冲区要 `(__local half (*)[AR])As_ + pb*BM`
（按 **half 元素**偏移），第一版写成 `As_ + pb*BM`（按 `[BM][AR]` 行偏移）→ 结果错。

### 9.2 关键约束：SLM 容量 ≈ 16 KB/WG 决定 occupancy，不是 BK 越大越好

`DBUF=1` 让 SLM/WG 翻倍。显式 2× 展开后，**双缓冲本身是对的**，但：

- `BK=32, DBUF=1`（24.5 KB/WG）→ **5.3**（occupancy 崩），尽管它 barrier 最少。
- `BK=16, DBUF=1`（12 KB/WG）→ **11.8**：与 `BK=32, DBUF=0`（11.9）持平，但 **比
  `BK=16, DBUF=0`（6.75）高 +75%** —— 这 +75% 就是纯粹「staging 被 overlap 掉」的收益。

探针（`-DSKIP_STAGE`/`-DSKIP_COMPUTE`，本轮已接进 `Tiles` 以便复现）显示：`BK=32, DBUF=0`
的 **compute-only 上限 ≈ 13.66 ops/EU/cyc（85%）**；`BK=16, DBUF=1` 已把 staging 藏干净，
瓶颈回到 **barrier 频率 × occupancy**。16 KB/WG 是硬墙：`(BM+BN)*BK*2(bytes)*2(buf) ≤ 16K`。

### 9.3 自适应 tile：双缓冲只在「网格足够大」时才赢

把候选配置在整批真实 shape 上实测，规律非常清楚（ops/EU/cyc）：

| shape (M×N×K) | BK8 DBUF0 | BK16 DBUF1 | BK16 DBUF1 BM64 | 网格 WG |
|---|---|---|---|---|
| 4096×512×512 | 10.89 | **11.80** | | 256 |
| 1024×1024×1024 | 10.63 | **11.70** | | 128 |
| 1024×576×576 | 9.91 | **10.77** | | 72 |
| 1000×1024×1024 | 10.40 | **11.32** | | 128 |
| 128×6400×192 | 9.85 | **10.04** | 9.79 | 100 |
| 256×400×384 | **4.87** | 4.81 | | 14 |
| 128×1600×192 | **6.85** | 5.11 | | 25 |
| 128×1600×384 | **7.26** | 5.73 | | 25 |
| 64×6400×64 | 4.47 | 3.93 | **5.66** | 100 |
| 64×6400×192 | 5.02 | 4.83 | **8.46** | 100 |
| 240×196×40 | **1.21** | 0.62 | | 8 |
| 576×49×96 | **1.03** | 0.74 | | 5 |
| 72×3136×16 | **2.11** | 1.74 | | 49 |

规律：**双缓冲把 SLM 翻倍 → 常驻 WG 变少 → 只有网格够大（≈≥64 个 WG）时才有足够的
work-group 去填满 80 EU、隐藏长延迟**。网格小（M/N 很窄，如 `M=128`、`N=49`）时，`BK=8
DBUF=0` 的「小 SLM、高常驻」反而赢；`K` 很小时（≤一个 k-tile 量级）流水的 prologue/epilogue
也亏。`M≤64`（YOLO head / mobilenet 尾层）时 `BM=64` 把 A tile 减半，occupancy 上来，DBUF=1
反而大幅领先（+20~70%）。

**落地分派**（`PlanModel`，按节点）：

```cpp
Tiles t;                                   // 默认 BM=128, BN=64, BK=16, DBUF=1
long grid = ceil(M/BM) * ceil(N/BN);
if      (M <= 64)                  t.BM = 64;         // 小 M：减半 A tile
else if (K >= 192 && grid >= 64)   t.BK = 16;         // 大网格：双缓冲流水
else                             { t.BK = 8; t.DBUF = 0; } // 否则回退单缓冲小 tile
```

每个节点按需 build/cache 对应 kernel（编译期常量 tile，无运行时开销）。

### 9.4 结果

**纯 GEMM（`kernel_bench`，4096×512×512）**：8.37 → 11.09（R8）→ **11.80**（R9）；
1024³：8.05 → 10.91 → **11.70**。官方 `kernel_check.py` 全部 PASS（相对误差判据）。

**端到端（kernel busy，warm）**：

| 模型 | R8 | R9 | gemm 分项 R8→R9 |
|---|---|---|---|
| yolov8n-pose | 26.38 ms | **26.07 ms** | 4.62 → **3.79 ms（−18%）** |
| yolo11n-pose | 27.91 ms | **27.48 ms** | — |
| mobilenetv3-small | 6.10 ms | **6.09 ms** | 4.56 → **3.91 ms** |

三模型 `model_check` 全部 **PASS**。整网总时提升有限（gemm 只占 ~15%），但 gemm 自身
**−18%**，且拿到了可迁移到 conv3x3 的硬经验（见下）。

### 9.5 给 conv3x3 的迁移经验

1. **软件流水的敌人是 occupancy，不是分支**：`DBUF` 让 SLM 翻倍，必须同步缩小 tile（这里
   `BK 32→16`）把 WG 压回 ~16 KB；否则越「优化」越慢。
2. **编译期常量下标是硬要求**：运行时选 buffer 会让 IGC 删体/放弃优化；循环要显式展开，
   缓冲区下标必须是字面量。
3. **按 grid 大小而不是按「理论更优」分派**：同一份 kernel，窄 M 形状必须回退到小 tile。
4. **staging 与 compute 重叠可白赚 1.75×**（6.75→11.8），但前提是别把 SLM 撑爆。

**未做/已证伪**：`async_work_group_copy`（R8 已证伪）；非对称缓冲（只双缓冲 B、A 单缓冲）需要
每 k-tile 再 stage 一次 A → 多一道 barrier，分析上是**净负**，未保留代码；更深流水（3 buffer）
会再翻 SLM，按 9.2 的墙可判定不划算，未试。

## Round 10 —— 指令延迟与各级缓存/带宽实测（为分块策略定标）

Round 9 确认「DBUF 让 SLM 翻倍 → occupancy 掉 → 流水填不平延迟」。本轮把平台的关键常数
测出来，给分块策略一个**理论标尺**。新增 `kernels/micro.cl` + `kernel_bench` 的
`--op fmalat / memlat / membw / scanbw / slmbw`（全部短核，遵守 `docs/benchmark_protocol.md`）。

### 10.1 FP16/FP32 FMA 延迟与吞吐（`--op fmalat`）

方法：`DEPTH` 条独立累加链 × `ITERS` 次；`DEPTH=1` 是纯依赖链（测延迟），`DEPTH` 大测吞吐。
额外用整数依赖 `add` 链测出**循环开销 = 20.4 cyc/iter**，从 `DEPTH=1` 结果里减掉即得真实延迟。

| 指令 | depth=1 cyc/iter | **净延迟（−20.4）** | depth=8 吞吐 ops/EU/cyc | 备注 |
|---|---|---|---|---|
| `add`（int，基准）| 20.4 | ~0 | 40 | 纯循环开销 |
| **half mad（h1）** | 49.5 | **≈29 cyc** | 23.8 | 标量 half |
| half2/half4 mad | 49.5 | ≈29 cyc | 25.1 / 27.1 | 与 h1 同延迟 |
| **half8 mad** | 49.5 | **≈29 cyc** | **27.7** | 同延迟，吞吐最高 |
| **float mad** | 74.4 | **≈54 cyc** | 14.9 | FP32 延迟约为 FP16 的 1.9× |

结论：**(a) FP16 FMA 延迟 ~29 cyc，FP32 ~54 cyc；(b) half2/4/8 相比 half 标量「零延迟代价」**，
向量化只涨吞吐（29→27.7 ops/EU/cyc 里 h8 比 h1 高 ~16%）。(c) 延迟 29 cyc 意味着
**一条依赖链要 29 条独立链才能填满**——这正是 GEMM 需要大 tile / 多累加器的量化依据。

### 10.2 存储层级的大小与带宽（`membw` / `scanbw` / `slmbw`）

**读带宽 vs footprint**（`scanbw`，只读、核内多次 pass 以压过 launch 开销）：

| footprint | 读带宽 | 层级 |
|---|---|---|
| 256 KB – 2 MB | **270–293 GB/s** | **LLC（片上共享）** |
| 4 MB | 180 GB/s | LLC→DRAM 过渡 |
| ≥16 MB | **16–20 GB/s** | **DRAM** |

**拷贝带宽 vs footprint**（`membw`，read+write）：1 MB 达 92 GB/s，2 MB 掉到 37，≥16 MB 稳定
**19.7 GB/s**（与 DRAM 只读一致）。→ **LLC 边界 ≈ 2–4 MB**（与旧文档「~4 MB 内持平、8 MB 崩塌」一致）。

**SLM（片上 L1 scratchpad）读带宽**（`slmbw`，uint4 向量读）：

| 每 WG SLM 配额 | 读带宽 |
|---|---|
| 2–4 KB | **~555 GB/s**（上限） |
| 8 KB | 541 GB/s |
| 16 KB | **301 GB/s** |
| 32 KB | 168 GB/s |

→ **SLM 带宽与「每 WG 配额」强耦合**：配额翻倍，常驻 WG 减半，聚合带宽近似腰斩。
这就是 Round 9「16 KB/WG 是墙」的物理来源：**SLM 容量 ↔ 带宽 ↔ occupancy 是一个耦合约束**，
不是独立的三件事。

### 10.3 用这套常数做分块理论分析（关键）

**GEMM 内循环的 SLM 喂给强度**：每个 kk，读 `TM` 个 A + `TN` 个 B（half），做 `TM×TN` 次 FMA。
→ **每 FLOP 的 SLM 字节 = (TM+TN)/(TM·TN)**：

| tile | B/FLOP | 满速(27.7 ops/EU/cyc)所需 SLM 带宽 |
|---|---|---|
| TM8×TN4（当前）| 0.375 | **~1080 GB/s** |
| TM16×TN4 | 0.3125 | ~900 GB/s |
| TM8×TN8 | 0.25 | ~720 GB/s |
| TM16×TN8 | 0.1875 | ~540 GB/s |

把 27.7 ops/EU/cyc、80 EU、1.3 GHz 代进去：`BW = ops/2×80×1.3e9×(TM+TN)/(TM·TN)`。
实测 SLM 上限 ~550 GB/s（且只有配额 ≤4–8 KB/WG 时才有）。**当前 TM8×TN4 的理论需求 1080 GB/s
远超 550** → 说明当前 GEMM 的 **inner loop 是被「SLM 喂给 + barrier 串行」共同卡住**，
而不是纯 FPU。要往 13.7（compute-only 上限）甚至更高走，方向是：

1. **加大 TN 到 8**（B/FLOP 0.375→0.25）——但寄存器会涨，且 R9 实测 TN=8 慢于 TN=4（occupancy）。
2. **更关键：把 SLM 配额压到 ≤8 KB/WG**（带宽 2×），再谈大 tile。
3. **彻底绕开 SLM 喂给**：把 A 或 B 的复用放到**寄存器**里（外层 k 循环多累加），或改用
   `sub_group` 内共享（Xe 上 sub-group 寄存器交换不占 SLM）——这是下一步。

> **物理结论（用户观点证实）**：真正的约束是 **「容量 ↔ 带宽 ↔ 并发」的耦合**：
> - LLC 只有 ~2–4 MB、DRAM ~19 GB/s → 大 K 的数据复用必须靠**分块在 LLC 内做**，
>   否则一掉出 LLC 就直接 DRAM 限速（19 GB/s，AI≈1 的算子必死）。
> - SLM 容量小（~16 KB/WG 级）→ 一旦为了双缓冲/大 tile 把配额推高，**SLM 带宽随常驻 WG
>   减少而腰斩**，软件流水也填不平。
> - 因此「缓存不够」只能靠**软件流水**缓解；但流水的收益又受 SLM 带宽-occupancy 耦合上限——
>   两者是同一个约束的两面。
>
> ⚠️ **R11 更正**：本节「inner loop 被 SLM 喂给卡住」的推断**已被证伪**（加宽 A 读无效、且
> SLM 无 bank conflict/占用耦合已被直接测出）。真正的剩余瓶颈是 **occupancy + work-group barrier**，
> 见 Round 11。

### 10.4 复现

```bash
# FP16/FP32 延迟与吞吐（depth=1 延迟；depth=8 吞吐）
./build/kernel_bench --op fmalat --width h1  --depth 1
./build/kernel_bench --op fmalat --width h8  --depth 8
./build/kernel_bench --op fmalat --width f32 --depth 1
./build/kernel_bench --op fmalat --width add --depth 1     # 循环开销基准
# 存储层级
./build/kernel_bench --op scanbw --sizes 256,1024,2048,4096,16384,65536
./build/kernel_bench --op membw  --sizes 1024,2048,8192,65536,262144
./build/kernel_bench --op slmbw  --slm-kb 4 --width v4
./build/kernel_bench --op slmbw  --slm-kb 16 --width v4
```

## Round 11 —— SLM 带宽到底是 bank conflict 还是 occupancy？+ 三个分块候选实测

Round 10 留下两个问题：(1) SLM 带宽随配额腰斩，是 **bank conflict / miss** 还是 **occupancy**？
(2) R10.3 提的三个候选（加大 TN、压 SLM 配额、绕开 SLM 喂给）实际怎样？本轮把两个都做了。

### 11.1 SLM 带宽：不是 bank conflict，是 occupancy（证伪）

在 `kernels/micro.cl` 加了**可控访存模式**的探针 `slm_conf`（`-DMODE`）：
- MODE 0：标量、线程 t 读 word t（**无冲突**）
- MODE 1：标量、所有线程同 bank（**32 路冲突**）
- MODE 3：uint4、连续（可能 4 路冲突）
- MODE 4：uint4、跨 bank 错开

16 KB/WG、nwg=64 实测：

| MODE | 访存 | 带宽 |
|---|---|---|
| 0 | 标量无冲突 | 99.8 GB/s |
| 1 | 标量 **32 路冲突** | **100.0 GB/s** |
| 3 | uint4 | 298.8 GB/s |
| 4 | uint4 错开 | 299.8 GB/s |

**无冲突与 32 路冲突完全同带宽**（99.8 vs 100.0）→ 该探针下冲突被完全隐藏，SLM 带宽
**根本不是 bank conflict 限制**（SLM 也不会 miss）。

再用 `--nwg` 直接扫**并发数**（同一份 MODE 3、16 KB/WG）：

| nwg（活跃 WG） | 带宽 |
|---|---|
| 16 | 154.7 GB/s |
| 64 | 298.8 GB/s |
| 128 | 300.9 GB/s（饱和）|

→ 带宽随活跃 WG 数上升并饱和。**结论：SLM 带宽低是 occupancy 耦合**——每 WG 占的 SLM 越大，
能常驻的 WG 越少，聚合带宽越低。16 KB/WG 下机器只能维持 ~64 WG，所以只有 ~300 GB/s；
4 KB/WG 能塞更多 WG，到 **558 GB/s**。这**坐实了「容量↔带宽↔并发」是同一个物理约束**，
而不是冲突或 miss。

### 11.2 三个分块候选实测（都不如现状）

基线：`128,64,16,8,4 DBUF=1` @ 4096×512×512 = **11.90 ops/EU/cyc**。

| 候选 | 配置 | 结果 | 结论 |
|---|---|---|---|
| **A：加大 TN** | `128,64,16,8,8`（TN8）| **7.13** | 寄存器压力 ↑、occupancy ↓，负 |
| **B：压 SLM 配额 ≤8KB/WG** | `64,64,16,8,4 DBUF=1`（8 KB/WG）| **11.42** | BM 减半的复用损失 > 带宽收益，略负 |
| **C：绕开 SLM 喂给（A 转置向量读）** | `AT=1`（`128,64,16,8,4`）| **9.80** | A 的 inner 读从 8 标量→1 half8，但**转置 staging 写**更贵，净负 |
| C2：A 加 PAD 消冲突 | `PAD=8` | 10.74 | 同样负；印证冲突不是瓶颈 |
| C3：VEC=2 | `128,64,16,8,4,···,VEC=2` | 10.54 | 负 |

**为什么都不行**：11.1 已证明瓶颈不是「SLM 事务数 / 冲突」，而是 **occupancy 与 barrier 串行**。
A/B/C 都在动 SLM 的形状或事务数，没动这两个根因，所以最好也只是打平。
`AT=1` 代码保留（默认 0，已加 `#if AT && PAD #error` 防护），作为**已证伪记录**。

### 11.3 对下一步的判断（纠正 R10.3 的方向）

R10.3 假设「inner loop 被 SLM 喂给卡住」——**本轮证伪**：加宽 A 读（C）无效，说明事务数不是瓶颈。
真正剩下的两个根因：

1. **occupancy**：`128,64,16 DBUF=1` 用 12 KB/WG，机器装不满 → 需要**更小 SLM 足迹**同时**不牺牲复用**。
   候选是「A 留在寄存器 + 只 stage B」（A 的复用靠寄存器而非 SLM），但 A 的寄存器量 = TM×BM/LX
   需仔细设计。
2. **barrier 串行**：每 k-tile 两道 barrier。真正要的是**子组级同步**（不需要 work-group barrier），
   把 tile 缩小到 sub-group 能覆盖的程度。

> 这两条都指向「**减少 work-group 级同步 + 把复用在寄存器/sub-group 内做**」，而不是继续折腾 SLM 布局。
> 下一步优先验证 **sub-group tiling**。

## Round 12 —— 突破：IGC 的 SIMD8 悬崖（11.9 → 13.7 ops/EU/cyc，+15%）

本轮本来是冲着「主动突破 SLM 容量↔带宽↔并发耦合」去的：试了寄存器预取（`PF`）和彻底无
SLM（`GN`）。两者都失败，但在排查 `PF` 为什么失败时**挖出了真正的瓶颈**——不是 SLM，也不
是 barrier，而是 **IGC 在 staging 的寄存器压力下偷偷把计算降到 SIMD8**。把这个悬崖堵上后，
整核从 **11.89 抬到 13.70 ops/EU/cyc（+15%）**，且达到该内循环的 compute-only 上限。
新增开关：`PF` / `GN` / `SKIP_BARRIER`（SB，诊断）/ `SG`（强制子组宽度）。

### 12.1 先量化 barrier，再发现「barrier 成本」其实是 SIMD 假象

用 `DBUF=0` 单缓冲路径分解（`SKIP_STAGE` 只在该路径生效），@4096×512×512：

| 配置 | 时间 | ops/EU/cyc |
|---|---|---|
| BK16 DBUF0 full（SG=0）| 3.371 ms | 6.13 |
| BK16 DBUF0 **skipstage**（SG=0）| 2.327 ms | 8.87 |
| BK32 DBUF0 full（SG=0）| 1.746 ms | 11.83 |
| BK32 DBUF0 skipstage（SG=0）| 1.514 ms | 13.64 |

一度以为「BK16 compute-only 只有 8.87」是 barrier 太贵。**反汇编证伪**：BK16 skipstage 在
有 barrier 时被编译成 **SIMD8**，去掉 barrier 后变回 SIMD16——8.87 是 SIMD8 的产物，不是
barrier 的代价。用 `SKIP_BARRIER`（SB）直接对照：BK32 skipstage barrier ON=13.66、OFF=13.36，
**barrier 在 BK32 只占 ~2%**。

`--op barrier` 微基准（依赖链延迟，含 ~20 cyc 循环开销）：

| mode | 内容 | cyc/iter | 推论 |
|---|---|---|---|
| 4 | ALU + `barrier(0)` | 179.8 | 纯同步 barrier ≈ 160 |
| 2 | ALU + `barrier(LOCAL_FENCE)` | 206.5 | 内存 fence +27 |
| 1 | SLM 存+读，无 barrier | 330.4 | SLM round-trip ≈ 310 |
| 0 | SLM 存+fence barrier+读 | 389.6 | 组合 |

结论：**barrier 不是剩余瓶颈**；它被多 WG 并发隐藏得差不多。真正的杀手是 SIMD 宽度。

### 12.2 真凶：IGC 的 SIMD 悬崖，用 `SG` 堵上

`intel_reqd_sub_group_size(SG)`（`-DSG=`）强制子组宽度。把它加到**完整** kernel（含 staging）
后，之前被测到「SIMD8」的配置全部回血：

| 配置 | SG=0 | **SG=16** | 备注 |
|---|---|---|---|
| 基线 `DBUF1 BK16` | 11.90 | **13.18** | 默认路径本身就掉过 SIMD8 |
| `DBUF0 BK16` | 6.13 | **13.08** | 之前一直是 SIMD8 |
| `DBUF0 BK32` | 11.83 | **13.67** | 本轮的赢家 |
| `PF BK32` | 7.45 | **13.33** | 预取不再需要 |

子组宽度扫描（`DBUF0 BK32`）：SG=8 → 7.16，**SG=16 → 13.70**，SG=32 → 1.19（溢出/崩）。
所以 **Xe-LP 上 SG=16 是甜点**；SG=32 会打爆寄存器。

> 修正 R10/R11：之前所谓「compute-only 上限 13.7」以及「inner loop 被卡住」的推断，
> 都建立在**完整 kernel 已是 SIMD16** 的隐含假设上——实际它常常掉到 SIMD8。把 SIMD
> 钉死后，单缓冲 BK32 的**整核**（13.70）已经追平当初的 skipstage 上限，staging 被并发
> 完全藏住。

### 12.3 结果：+15% 来自 `SG=16`，不是来自 BK32

SG=0 → SG=16，各测两次，非常稳定：

| 配置 | SG=0 | **SG=16** | 提升 |
|---|---|---|---|
| `DBUF1 BK16`（原默认）| 11.83 / 11.79 | **13.73 / 13.67** | **+15~16%** |
| `DBUF0 BK32` | 11.89 / 11.78 | **13.72 / 13.64** | **+15~16%** |

**关键更正**：两条流水线在 SG=16 下**打平**（都 ~13.7）。也就是说，SG=16 修好 SIMD 之后，
R9 引入的「单缓冲 BK32 ⇄ 双缓冲 BK16」取舍**变得无关紧要**——双缓冲不再必要，单缓冲的
staging 被并发藏住。`PlanModel` 对大 grid 取 `BK=32 DBUF=0` 只是取个边际（在 256×400×512
略好、128×1600×384 略差，属平局内噪声），真正的收益全部来自 **逼 IGC 用 SIMD16**。

反汇编（`BK32 DBUF0 SG=16`）：1024 条 **SIMD16** `mad`，SLM 读全部 `rd:4`（一次 8 个 half，
沿 k 自动向量化），`scratch=0`，GRF≤112。数值：非整除（带尾巴）形状 `--verify` 通过
（mean_rel≈2.2e-3）；`numerical_check.py` 三个模型 **ALL PASS**。

### 12.4 仍然失败的候选（保留为已证伪记录）

| 候选 | 结果 | 原因 |
|---|---|---|
| `PF=1` 寄存器预取 | 13.33 | SG=16 后不再需要；且预取寄存器会顶掉 SIMD16（SG=0 时 7.45）|
| `GN=1` 无 SLM | 1.89 | global 延迟直接压在 mad 依赖链上；SLM 是延迟隐藏载体，不是可绕开的开销 |
| `TN=8`（128,64,16,8,8）| 7.13 / SG16 4.77 | 寄存器压力 |
| `AT=1`（转置 A 向量读）| 11.06 | 转置 staging 写太贵；且 IGC 已自动沿 k 向量化 A 读 |
| `DBUF1 BK32`（24 KB）| 5.53 | occupancy 崩 |
| `DBUF0 BK64`（24 KB）| 11.01 | 同上 |
| `ASYNC=1` | 7.32 | 无收益 |

→ 「加大 tile 提升算术强度」「去掉 SLM」「更多缓冲」都不行；12 KB/WG 的容量墙依然是真的，
只是**它不是本轮的主瓶颈**——主瓶颈是编译器 SIMD 选择。

### 12.5 库 dispatch 与端到端

`PlanModel` 的 gemm dispatch 更新（`Tiles::SG` 默认 16）：
- `M<=64` → `BM=64, BK=16, DBUF=1, SG=16`；
- `K>=192 && grid>=64` → `BK=32, DBUF=0, SG=16`（本轮赢家）；
- 其余 → `BK=8, DBUF=0, SG=16`（小 grid / 中等 K）；
- `K<32` 时回落 `SG=0`（极小 K 上 SG=16 略负）。

**端到端（kernel_run `--report`，5 iters）**：yolov8n-pose 整网 25.97 → 26.10 ms、
mobilenetv3-small 6.13 → 6.18 ms，**gemm 部分基本不变**。原因是这两个模型里
**conv3x3 占 73%（18.97/26.16 ms），gemm 只占 ~14%**，且模型里的 gemm 单节点很小
（28 节点合计 ~0.75 ms/iter ≈ 27 µs/节点），**是 launch/小 grid 受限而非算力受限**，
所以 kernel 级 +15% 在整网里被摊平。**收益主要体现在大 GEMM（benchmark/大 batch）**。

### 12.6 结论

1. **本轮最大杠杆是「把编译器钉死在 SIMD16」**：`SG=16` 单独就让整核 **+15~16%**
   （11.9 → 13.7 ops/EU/cyc），两条流水线（双缓冲 BK16 / 单缓冲 BK32）修好 SIMD 后打平。
   这是一个此前完全没被注意到的 codegen 悬崖——远端反汇编显示 staging 的寄存器压力会让
   IGC 把计算体降到 SIMD8。
2. **SLM 墙（容量↔occupancy）是真的**，但取消 SLM / 换更大 tile 都不能突破它；
   真正要做的是**降低每 WG 的 SLM 足迹同时保住 SIMD 宽度与 k-tile 摊销**。
3. barrier 在 BK32 下只占 ~2%，不是瓶颈；R11「barrier 串行是根因」需要下调权重。
4. 对整网的意义取决于 gemm 占比：本项目两模型被 conv3x3 主导，gemm 优化对端到端近中性。

### 12.7 复现

```bash
# 关键对照：SG=0 vs SG=16
./build/kernel_bench --op gemm --shape 4096,512,512 --iters 30 --tiles 128,64,32,8,4,0,0,0,4,0,0,0,0,0,0,0,0
./build/kernel_bench --op gemm --shape 4096,512,512 --iters 30 --tiles 128,64,32,8,4,0,0,0,4,0,0,0,0,0,0,0,16
# barrier 分解（--mode 0..4）
./build/kernel_bench --op barrier --wg 256 --nwg 64 --iters 20000 --mode 0
# 已证伪的候选
./build/kernel_bench --op gemm --shape 4096,512,512 --tiles 128,64,16,8,4,0,0,1,4,0,0,0,0,1,0,0,16   # PF
./build/kernel_bench --op gemm --shape 4096,512,512 --tiles 128,64,16,8,4,0,0,1,4,0,0,0,0,0,0,0,16   # baseline+SG16
```

## Round 13 —— 物理极限标定：IGC 掉 SIMD 的真因是「寄存器大小」，不是带宽

R12 发现 `SG=16` 让 GEMM +15%，但没说清 IGC 为什么会掉到 SIMD8。结论先行：**是每线程
128 GRF 的寄存器分配上限**（`28 KB/EU` = 7 个线程上下文 × 4 KB），既不是 RF 读带宽，也不是
「耦合导致」的间接效应。顺带修正了一个**差 2× 的峰值归一化错误**。

### 13.1 FP16 峰值修正（重要）：是 32，不是 16

给 `--op fmalat` 加了 `--sg`（强制子组宽度）。纯 FMA 吞吐：

| 探针 | depth | ops/EU/cyc |
|---|---|---|
| `h1`（scalar half）| 4 / 8 / 16 / 32 / 48 / 64 | 25.9 / 26.3 / 25.4 / 26.3 / 26.3 / **27.8** |
| `h4` | 4 / 8 / 16 | 27.8 / 27.1 / 27.1 |
| `h8` | 8 | 27.7 |
| **`h1` SG=8** | 32 | **14.1**（恰为一半）|
| **`h1` SG=32** | 8 | 1.6（溢出崩）|
| `f32` | 4…32 | **14.5–14.7**（40 以上崩）|

要点：
- **与深度无关**（h1 从 4 到 64 都 ~26–28）→ 是**吞吐**受限，不是延迟（R10「depth=8 = 27.7」
  其实是延迟假象，但那不重要，吞吐恰好也在同一量级）。
- **吞吐随子组宽度线性变化**（SG8 恰为 SG16 的一半）→ **EU 每周期只发射 1 条向量指令**，
  SIMD 宽度直接决定 FPU 吞吐。
- `f16 : f32 ≈ 1.9–2`。

→ 这台机器 **FP16 峰 = 32 ops/EU/cyc = 3.328 TFLOP/s**（16 packed FMA），FP32 = 16 ops。
`kernel_bench` / `ClRuntime` 之前用 `EU × clk × 16` 归一（打印 “peak 1664 GFLOP/s”、“% of 16”），
**少算了一半**——和 `kernel.md` 第 20 行自己写的 3.328 TFLOP/s 自相矛盾。**已修正为 32**。

> **修正读法**：R1–R12 里所有 “% of 16” 都要 **×2** 读；`ops/EU/cyc` 绝对值不变。
> 因此 **GEMM 的 13.7 只是真实峰值的 ~43%**（不是 86%），compute-only ~17.7 ≈ 55%。

### 13.2 RF 读带宽被排除

`h1`（每指令/lane 读 1 个 half）与 `h8`（8 个 half）吞吐**完全相同**（~27），只有子组宽度能
让吞吐翻倍/腰斩。→ 稳态 FMA 由 **lane 发射吞吐**决定，**寄存器读端口/字节带宽不是瓶颈**。

### 13.3 IGC 的 SIMD 选择 = 寄存器**大小**压力（离线 ISA 实测）

`SG=0`（让 IGC 自己选）下扫配置，记录它选的 SIMD + GRF 峰值 + spill：

| 配置 (SG=0) | IGC 选 | GRF | scratch |
|---|---|---|---|
| `DBUF0 BK16 full` | **S8** | 127 | 0 |
| `DBUF0 BK16 skipstage` | **S8** | 126 | 23 |
| `PF BK32` | **S8** | 117 | 0 |
| `TN8` | **S8** | 126 | 12 |
| `DBUF0 BK32 full` | S16 | 119 | 0 |
| `DBUF1 BK16 full` | S16 | 112 | 0 |
| `DBUF0 BK64 / BK8 / GN` | S16 | 115–127 | 0 |

规律很干净：**只要 SIMD16 的分配会顶到 ~126–128 GRF，IGC 就退回 SIMD8**（每 GRF 少装一半
lane，逻辑值所需 GRF 减半）。这不是物理不可能——同一配置加 `-DSG=16` 后能塞进 ≤120 GRF、
零 spill（`DBUF0 BK16`：6.1 → 13.1）。所以它是**寄存器分配器的保守启发式**：它宁可降 SIMD
也不愿多花力气把分配压进 128 GRF。

### 13.4 「寄存器大小 → 并发 → 可达 tile」耦合（128 GRF 是硬墙）

想靠「加大寄存器 tile」降低 load/FLOP（提高算术强度）时，全部撞墙（均强制 SG=16）：

| 配置 | SIMD | GRF | scratch | ops/EU/cyc |
|---|---|---|---|---|
| **TM8 TN4 BK32** | S16 | 112 | **0** | **13.7** |
| TM8 TN8 BK32 | S16 | 126 | 6 | 8.4 |
| TM16 TN4 BK32 | S16 | 127 | **513** | 4.2 |
| TM16 TN4 BK16 | S16 | 120 | **477** | 0.53 |

- 用量过 ~120 GRF 开始 spill，过 128 彻底崩。
- `f32` 深度扫描同现象：depth 32（acc[32]）→ 14.5，depth 64（acc[64]）→ 7.1。

→ **物理模型**：RF = `128 GRF × 32 B = 4 KB`/线程；`28 KB/EU` 恰好 = **7 个线程上下文 × 4 KB**
（与 Gen11/Xe 的 7 threads/EU 对上）。**每 EU 每周期发射 1 条向量指令**，所以
`work-items/EU = 7 × SIMD宽度`。SIMD8 把 lane 数砍半 → FPU 饿死（实测正好一半）。
**容量（RF 大小）↔ 并发（线程/lane 数）↔ 可达 tile（算术强度）是同一个约束**，和 R10 的
SLM 容量↔带宽↔并发是这台机器上的孪生墙。

### 13.5 对 GEMM 物理极限的结论

| 层级 | ops/EU/cyc | 占真实峰（32）|
|---|---|---|
| 纯 FP16 FMA（h1/h4/h8, SG16）| ~27.8 | 87% |
| GEMM compute-only（skipstage）| ~17.7 | 55% |
| **GEMM 整核（BK32 SG16）** | **13.7** | **43%** |
| 旧默认（SG=0 掉 SIMD8）| 11.9 | 37% |

- GEMM **不是 FPU 受限**（43%）；限制来自内循环的 **load/issue 喂给 + occupancy/barrier**。
- 而唯一能把「feed」降下来的手段——更大寄存器 tile（更高算术强度）——被 **128 GRF 硬墙**
  挡住（TM16/TN8 全部 spill）。
- 所以在「这台机器 + 这套 tile 算法」下，**13.7 就是寄存器墙下的现实上限**。要再往上只能改
  算法层：让数据复用不依赖大 GRF tile（例如用 sub-group 内寄存器交换/广播减少 SLM+GRF
  压力），或消灭冗余计算（conv 融合、小 M 专用 tile）。

### 13.6 复现

```bash
# 纯 FMA 吞吐 vs 子组宽度（h1 与 h8 同量级；SG8 腰斩）
for s in 8 16 32; do ./build/kernel_bench --op fmalat --width h1 --depth 32 --sg $s; done
./build/kernel_bench --op fmalat --width h8 --depth 8  --sg 16
./build/kernel_bench --op fmalat --width f32 --depth 32 --sg 16   # f16:f32 ≈ 2
# 寄存器墙：TM16 直接 spill
./build/kernel_bench --op gemm --shape 4096,512,512 --tiles 128,64,32,16,4,0,0,0,4,0,0,0,0,0,0,0,16
# 离线看 IGC 自己选什么（SG=0）
ocloc compile -file kernels/gemm.cl -device tgl -options "-DBM=128 -DBN=64 -DBK=16 -DTM=8 -DTN=4 -DDBUF=0 -cl-mad-enable" -output k && \
ocloc disasm -file k_tgllp.bin -device tgl -dump ./isa && grep -m1 'mad (' isa/.text.gemm_f16.asm
```

## Round 14 —— 冲顶尝试与封堵证明：13.7 是「寄存器 / SLM / 延迟」三耦合下的实测上限

R13 标定后剩下的问题：13.7 = 43% of 32，离纯 FMA 的 27.8（87%）还有 2× 空间，能不能冲上去？
本轮先做**决策分解**定位每一块损失，再逐个针对性尝试，最后给出约束推导证明空间被封死。

### 14.1 决策分解（全部 SG=16，@4096×512×512）

新增 `-DNOLOAD`（保留完全相同的 mad 结构与 BK/tile，只把内循环的 SLM 操作数读换成寄存器常量）。

| build | ops/EU/cyc | 相对 |
|---|---|---|
| `normal`（staging + 内循环 loads）| **13.79** | 1.00 |
| `SKIP_STAGE`（无每 k-tile staging，保留 loads）| **18.12** | **staging 抹掉 31%** |
| `NOLOAD`（无内循环 loads，staging 变死代码）| **21.23** | **loads 再抹 17%** |
| `SKIP_STAGE + NOLOAD` | 21.61 | |
| 纯 FMA（h1/h8 微基准）| 27.8 | 结构+寄存器占用再抹 22% |
| `SKIP_COMPUTE`（只 staging）| 0.448 ms | staging 绝对量 |

对照 DBUF1 BK16 SG16：full **13.43** / SKIP_STAGE **17.65** / NOLOAD **26.24**
→ **staging 24%、内循环 loads 33%**（DBUF1 的 B 重读 16×、32 个 k-tile，SLM 流量翻倍）。

三块损失量级相当（~25–33%），来自三个不同来源，必须分别解决。

### 14.2 冲顶尝试（全部失败，保留为负结果）

| 尝试 | 目标 | 结果 |
|---|---|---|
| `PIPE`（kk 级软件流水，预取下一 kk）| load 延迟 | **11.46**（寄存器搬移 +37% 指令）|
| staging 内点 fast-path（去边界检查）| staging 指令 | **13.81**（无变化 → 是延迟/BW 不是指令）|
| TM4 TN4（更小 tile 换 occupancy）| occupancy | **9.79** |
| TM8 TN2 / BM256 | 其他形状 | 9.65 / 11.02 |
| TM16 TN4 / TN8（更大 tile 换算术强度）| 复用 | **spill**（grf=127 + 513 行 scratch）→ 0.5–8.4 |
| GN / PF / AT / ASYNC | — | R12 已证伪 |

### 14.3 严格约束推导

**(a) 寄存器文件 —— 直接决定可用的 tile 与 occupancy**
- 每线程 128 GRF × 32 B = **4 KB**；实测 RF `28 KB/EU` → **7 线程/EU**。
- 数据寄存器下界 = `TM·TN/2`(acc, half) + `TM` + `TN`；TM8/TN4 只有 ~26 个。
  但实测 grf：`8/4 → 112`、`8/8 → 126`、`16/4 → 127(+513 spill)`。
  → **~90 个寄存器是寻址/调度/中间值开销**，真正的墙不是数据而是这套开销。
  可容纳的算术强度上限就是 `TM·TN ≈ 32`；再大必 spill。
- 同时 7 线程/EU 也把 occupancy 钉死：`work-items/EU = 7 × 16 = 112`。

**(b) SLM 容量 —— 直接决定缓冲深度与 k-tile 摊销**
- `SLM/WG = nbuf · BK · (BM+BN) · 2 B`；占用预算 ~12 KB → `nbuf·BK·(BM+BN) ≤ 3072`。
- barrier 数 = `K/BK`。在预算内取值：`nbuf=2, BK=16`（DBUF1，32 barrier）或 `nbuf=1, BK=32`
  （DBUF0，16 barrier）——**两条刚好打平（实测都 ~13.7）**：DBUF1 藏住 staging 但 barrier/loads ×2；
  DBUF0 barrier 少一半但 staging 暴露。这就是 R12 那条经验规律的解析来源。

**(c) SLM 带宽 —— 满速时刚好被卡**
- 本 tile 的 SLM 读 = `2·(TM + BN) / (lanes · TM · TN)` = `2(8+64)/(16·32)` = **0.281 B/FMA**。
- 满速 `16 FMA/EU/cyc × 80 EU × 1.3 GHz = 1.66e12 FMA/s` → 需要 **467 GB/s**。
- 12 KB/WG 下实测 SLM 上限 ~300–450 GB/s（R10/R11）。即**即便完美重叠，也会卡在 SLM 带宽**；
  DBUF1 实测跑到 ~384 GB/s（接近饱和），DBUF0 ~197 GB/s。
- 降 0.281 的唯一办法是加大 TM/TN → 撞 (a)。

**(d) 全局 staging 延迟 —— 单缓冲无法藏**
- A=4 MB（LLC 边界）重读 `N/BN=8` 次 ≈ 32 MB，B ≈ 16 MB；实测 staging 0.36 ms → 有效 ~133 GB/s，
  是**延迟主导**（非指令）。要藏只能 DBUF 双缓冲 → 撞 (b)。

### 14.4 为什么四条路互相抵消、彻底堵死

- 降 **SLM 读**（减 bytes/FMA）→ 加大 TM/TN → 撞 128 GRF 墙（且寻址开销占 ~90，放不下）。
- 降 **staging 暴露** → DBUF 双缓冲 → SLM 翻倍 → BK 减半 → barrier 与 loads 翻倍，净持平（14.1 实测）。
- 提 **occupancy** → 减数据寄存器 → tile 变小 → 算术强度与复用变差（14.2 实测更慢）。
- 提 **BK**（减 barrier）→ SLM 超预算 → occupancy 崩（`DBUF0 BK64`/`DBUF1 BK32` = 5–11）。

三条改进方向各自都被 (a) 或 (b) 挡住，且它们彼此是对偶的（省一样就费另一样），
所以绕不开。**在「SLM-tile 算法 + 128 GRF 寄存器文件」下，13.7 是可达最优点。**

### 14.5 仅剩的另类数据通路及其判定

唯一没试的是**用 `intel_sub_group_block_read` / sub-group 寄存器交换**替代逐 lane SLM 读：
- 它能用一条指令完成一个 sub-group 的 SLM 读，并**省掉 ≥90 的寻址寄存器开销**（(a) 的元凶），
  有可能放下更大的 tile。
- 但：(i) 它读的是**同样的 SLM 字节**，所以 (c) 的 SLM 带宽约束**不变**（467 GB/s 需求 vs ~450 上限）；
  (ii) 交换本身占 GRF 与 issue，在 7 线程/EU 下未必能补回来。
- **判定：即便实现，天花板也只是从「寄存器墙」换成「SLM 带宽墙」，两者都在 ~450–467 GB/s 附近，
  不会突破 27.8 的结构上限，更不会突破 13.7 太多。** 故本轮判定空间已被严格封死，不再继续调这个 kernel。

> 结论量化：`13.7` = 结构/寄存器上限 `21.2` × (loads 折损 0.85) × (staging 折损 0.76)，
> 而 `21.2` 本身 = FMA 峰 `27.8` × (寄存器占用折损 0.76)。每一层折损都对应一堵已证实的硬墙。

### 14.6 复现

```bash
P="--tiles 128,64,32,8,4,0,0,0,4"   # 后接 9 个字段 ASYNC,SKIP_STAGE,SKIP_COMPUTE,AT,PF,GN,SB,SG,NOLOAD
./build/kernel_bench --op gemm --shape 4096,512,512 --tiles 128,64,32,8,4,0,0,0,4,0,0,0,0,0,0,0,16,0  # normal 13.8
./build/kernel_bench --op gemm --shape 4096,512,512 --tiles 128,64,32,8,4,0,0,0,4,0,1,0,0,0,0,0,16,0  # skipstage 18.1
./build/kernel_bench --op gemm --shape 4096,512,512 --tiles 128,64,32,8,4,0,0,0,4,0,0,0,0,0,0,0,16,1  # noload 21.2
./build/kernel_bench --op gemm --shape 4096,512,512 --tiles 128,64,32,8,4,0,0,0,4,0,0,0,0,0,0,0,16,0,1 # PIPE 11.5
```

## Round 15 —— 算法层（一）：原生 3×3 direct conv 的自适应 tile（yolov8n 端到端 −15%）

GEMM 封顶后转向算法层。先看整网账单：yolov8n-pose 里 **conv3x3 占 ~73%**（18.97/26.0 ms），
gemm 仅 ~14%；yolo11n 类似。所以 conv 才是瓶颈。仓库的 3×3 groups=1 早已走**原生 direct conv**
（`kernels/conv.cl`，无 im2col，R2 设计），本轮就是把它调对。

### 15.1 问题：固定 tile 与空间尺寸严重不匹配

旧默认对所有 stride-1 的 3×3 都用 `TX=64, TY=8, CB=32`。但输出空间是 320→160→80→40→20：
- Wout=80：TX=64 → 2 个列块，第 2 块只有 16/64 有效（**浪费 37%**）；
- Wout=40：TX=64 → 1 块，24/64 空转（浪费 37%）；
- Wout=20：TX=64 → 1 块，**浪费 69%**，且 WG 数极少（80×80 只有 40 个 WG，EÜ=80）。

实测（`--op conv3x3`，`ops/EU/cyc`，now % of 32）：

| shape | 旧默认 64,8 | 40,8 | 40,10 | 20,10 |
|---|---|---|---|---|
| 64→64 80×80 | 6.49 | **9.86** | 9.37 | 6.41 |
| 32→32 80×80 | — | 7.63 | 7.31 | 4.88 |
| 64→64 40×40 | — | 6.13 | 5.88 | 4.97 |
| 128→64 40×40 | — | 6.93 | 6.43 | 5.48 |
| 128→128 20×20 | 3.03 | 2.59 | 3.29 | **3.30** |
| 16→16 160×160 | — | 4.61 | 5.13 | 3.09 |

stride-2（输入 320/160/80/40 → 输出 160/80/40/20）：同样 TX=40 最好（5.6–6.5），Wout=20 用 TX=20（3.20）。

### 15.2 规则与实现

`PlanModel` 的 conv3x3 分支改为按输出尺寸选自适应 tile（stride-1/2 通用）：
```
cfg.SG = 16;                       // 与 gemm 同源：钉 SIMD16
cfg.TX = Wout >= 40 ? 40 : (Wout >= 20 ? 20 : 16);
if (cfg.TX > Wout) cfg.TX = Wout;
cfg.TY = 8;                        // CB/CINC/UNROLL_CI 沿用原值
```
（`Conv3x3Cfg` 新增 `SG` 字段；`conv.cl` 加了 `intel_reqd_sub_group_size`。）

### 15.3 结果（`kernel_run --report`，5 iters，与 R12–R14 同口径）

| 模型 | 之前总时间 | **之后** | conv3x3 |
|---|---|---|---|
| yolov8n-pose | 26.16 ms | **22.17 ms (−15%)** | 18.97 → **15.05 ms** |
| yolo11n-pose | 27.30 ms | **24.89 ms (−9%)** | 14.98 → **12.49 ms** |
| mobilenetv3-small | 6.29 ms | 6.36 ms (~持平) | 0.15 ms（占比极小）|

`numerical_check.py` 三模型 **ALL PASS**。

### 15.4 为什么 direct conv 仍然只到 ~30%

复用了 R13/R14 的方法：
- **它本来就是 SIMD16**（反汇编 `mad (16|…)`，grf=112），所以 `SG` 不像 gemm 那样有 cliff 可救。
- **`TM>1`（列方向寄存器复用）会掉 SIMD8 或代码爆炸**：`TM=2` 编译成 `S8x1732`（吞吐腰斩），
  `TM=4` grf=127、指令行数 ×3.5 且性能崩。强制 `SG=16` 也救不回来。
- 于是 kernels 只能用 `TM=1`：**每个权重向量加载一次、只服务一次 mad（1:1 的 load:mad）**。
  内循环每 ci 是「9 strip 载入 + 144 权重载入(half2×CB/2) + 144 mad」——**载入数 ≈ mad 数**，
  这直接把上限压到 ~50%，再叠加寻址/延迟就到 ~30%。
- 加大 `CB` 能提高输入复用，但权重载入比不变；减小 `CB` 增加 WG 数但降低复用。

→ 与 gemm 同源的三堵墙：**128 GRF 限制寄存器 tile 大小 → 无法让权重在一次加载后被多次复用**。
本轮先拿到了「tile 匹配」这块确定的收益；真正的 2× 需要一个新的 direct-conv 结构（例如把权重
在寄存器里按输出通道常驻并沿空间滑动），属于下一轮。

### 15.5 复现

```bash
./build/kernel_bench --op conv3x3 --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,0
./build/kernel_bench --op conv3x3 --stride 2 --conv-shape 64,128,80,80 --conv 40,8,1,32,8,2
./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 5
```

## Round 16 —— 算法层（二）：寄存器分块 3×3（负结果）与原生 depthwise

R15 拿下了 conv3×3 的 tile 匹配（端到端 −15%），并指出它仍卡在「TM=1 → 权重加载 : mad = 1:1」。
本轮按这个方向做两件事：(1) 写寄存器分块版 3×3 `conv3x3_rt`（每线程 TM×TN，权重被 TM 次复用），
(2) 用原生 depthwise 替换朴素 `conv_general`。

### 16.1 寄存器分块 3×3 `conv3x3_rt`：更慢（负结果）

新 kernel：线程算 TM 列 × TN 通道（工作组的通道按 CB/TN 分给不同 ly），权重向量一次加载服务
TM 次 mad。实测（`--op conv3x3rt`，ops/EU/cyc）：

| 配置 | 64→64 80×80 | 128→128 40×40 | 256→256 20×20 |
|---|---|---|---|
| 旧 kernel（R15 最优 TX40,TY8,TM1）| **9.86** | **8.35** | 6.02 |
| RT TM4 TN8 CB32 | 7.20 | 7.04 | 3.10 |
| RT TM4 TN8 TX40 | 8.12 | 7.81 | 3.17 |
| RT TM4 TN16 | 7.20 | 6.89 | 2.51 |
| RT TM8 TN4 | 5.43 | 5.47 | 2.50 |

**RT 全面更慢**。反汇编：RT 已是 SIMD16（868 mad），但 **grf=127**（旧 kernel 112）——多出来的
`w[8]`/`strip[]` 存活值把寄存器顶到墙，occupancy 掉下来，省下的权重加载指令被吃掉。
结论（与 R14 同源）：**原 `TM=1` 设计其实是对的**——权重是 sub-group 内广播（每次加载很便宜），
strip 被 CB=32 个通道复用，真正的限制不是指令数而是 128 GRF。`conv3x3_rt` 保留为负结果（默认不走）。

### 16.2 原生 depthwise（替换 `conv_general`）

mobilevert 的 3×3/5×5 depthwise、yolo11 的 grouped conv 原本都走 `conv_general`：**每线程一个
输出像素、fp32 累加、K/S/ACT 全是运行时参数**（反汇编显示编译器把 silu/hardswish/relu/hardsigmoid
**四条激活路径全编译**，32 条 `math.exp`）。

新 `depthwise_f16`（`conv_general.cl`）：
- 1 work-item = 1 输出，**1D grid 完全 coalesced**；
- **K/S/P/ACT 编译期特化**（每节点一个 options，按 `source|options` 缓存），tap 循环全展开、
  只发射一条激活；
- fp16 累加。

实测（`kernel_run --report`，5 iters）：

| | `conv_general` | **`depthwise`** |
|---|---|---|
| mobilenetv3-small | 0.510 ms ×11 | **0.504 ms** ×11 |
| yolo11n-pose | 0.831 ms ×7 | **0.762 ms** ×7（−8%）|

（先试过一版沿 W 向量化 `vload8` 的 depthwise，反而慢 2–3×：窄 W 下 sub-group 跨行 → 不 coalesced，
且边界谓词爆炸；已改成上面的 coalesced 标量版。）

### 16.3 端到端

| 模型 | R14 | R15 | **R16** |
|---|---|---|---|
| yolov8n-pose | 26.16 | 22.17 | **22.23 ms** |
| yolo11n-pose | 27.30 | 24.89 | **24.74 ms** |
| mobilenetv3-small | 6.29 | 6.36 | **6.29 ms** |

`numerical_check.py` 三模型 **ALL PASS**。

### 16.4 结论

- conv3×3 和 depthwise 都已到**局部最优**：三堵墙（128 GRF 寄存器大小 / SLM 容量 / 延迟）里，
  conv 主要撞的是**寄存器大小**（放不下能复用权重的大寄存器 tile）。
- 本轮确定收益来自 R15 的 tile 匹配；R16 的 depthwise 是边际改善（yolo11 −8%、mobilenet 持平），
  寄存器分块 3×3 明确为负。
- 若要把 conv 再往上推，只能换**不靠大 GRF tile 的数据通路**（sub-group block-read / 寄存器交换），
  但那和 R14 的结论一样：会把「寄存器墙」换成「SLM 带宽墙」，天花板一致。

### 16.5 复现

```bash
./build/kernel_bench --op conv3x3rt --conv-shape 64,64,80,80 --conv 40,8,4,32,16,1,1,0,3,1,16,8,1
./build/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 5
```

## Round 17 —— 算法层（三）：走向 OpenVINO 的输出通道向量化（两个负结果）

GEMM 封顶后，单 kernel 横评（vs OpenVINO 2025.2，per-node profiling）显示：
**GEMM 已赢 OV（13.9 vs 10.4），但 conv3x3 全输（9.9 vs 16.0）**，而 conv 占整网 ~68%。
本轮按 OV 的 `convolution_gpu_bfyx_os_iyx_osv32` 思路做两个新 kernel，结论：**都没打赢现有
`conv3x3_f16`（9.9 ops/EU/cyc）**，但把 conv 的真实瓶颈从「权重加载」上挪开了。

### 17.1 动机：把 SIMD lane 换成输出通道

现有 `conv3x3_f16` 是 **lane=空间、输出通道在寄存器里展开**（CB=32），所以：
- 输入 strip 只载一次就被 32 个通道复用（这是它的最大优势）；
- 但每个 (ci,kh,kw) 要载 `CB/2` 个 half2 权重、做 `CB/2` 个 half2 mad —— TM=1 时
  **权重加载 : mad = 1 : 1**（R15.4 的判断）。

OV 的 `os_iyx_osv32` 相反：**lane=输出通道（SG=16，每 lane 2 个通道）**，每个 (ci,kh,kw)
只载 1 个权重向量、做 `OBW*OBH` 个 mad，权重复用度高；输入用 `sub_group_broadcast`
在 lane 间广播。于是做了两个复刻。

### 17.2 `conv3x3_osv`（lane=通道，仍用 SLM）——最好 7.2，负结果

设计（`kernels/conv_osv.cl`）：WG 覆盖 TX×TY 空间 × CB 通道；`local(0)=RX*SG`（rx=空间块、
lane=通道），`local(1)=TY*CG`；每 lane 拥有 `VECO` 个连续通道 × TM 列；权重 `Ws[ci][kk][CB]`
按通道连续存，lane 读自己那段。**权重:mad = 1:TM**，累加器只有 `TM*VECO/2` 个 half2
（通道摊到 lane 上，不再随 CB 膨胀）。

实测（64→64 @80×80 s1，ops/EU/cyc）：

| 配置 | TM | VECO | CB | ops/EU/cyc |
|---|---|---|---|---|
| `conv3x3_f16` 基线 | 1 | – | 32 | **9.88** |
| osv 标量 | 4 | 1 | 16 | 3.81 |
| osv 标量 | 8 | 1 | 16 | 5.84 |
| osv | 8 | 2 | 32 | **7.19** |
| osv | 4 | 4 | 64 | 5.95 |
| osv | 8 | 4 | 64 | 6.81 |
| osv | 16 | 1 | 32 | 6.05 |

**为什么输**：lane=通道把「输入被 CB 个通道复用」这条最优路径拆掉了——每个 work-item
只服务 `VECO` 个通道，strip 的载入次数按 `CB/VECO` 放大；同时 work-group 数翻数倍，
staging/launch 开销上升。**减少权重加载并没有换来加速 → 说明权重加载不是主瓶颈。**

### 17.3 `conv3x3_sg`（lane=通道 + 无 SLM + `sub_group_broadcast`）——1.5，负结果

进一步照搬 OV 的数据通路（`kernels/conv_sg.cl`）：**完全不用 `__local`、不用 barrier**。
输入块按 lane 分布载进寄存器数组 `in[]`（lane 持有 lane, lane+SG, …），用
`sub_group_broadcast(in[e/SG], e%SG)` 取标量；权重每 lane 直接 coalesced 全局读。
结构上就是 OV 的 `out = mad(w, broadcast(in,...), out)`。

实测（同 shape）：VECO2/OBW4/OBH4 **1.24**、VECO4 **1.50**、VECO4 OBW8/OBH2 1.40、
VECO8 0.48。ISA 反汇编：288 条 `mad :hf`、407 条 `mov`（broadcast）、**scratch=0**，
指令数 ~2.4/mad 并不离谱，但**跑不起来**。

**为什么输**：`ci` 循环无法展开（`Cin` 运行时），每个 `ci` 开头要从**全局**载入 `in[]`
（延迟 ~200+ cyc），紧跟着 144 个依赖它的 mad；全局延迟在单个 rolled 循环里完全暴露。
GEMM 的 `GN=1`（R12）已经证伪过「无 SLM 直读全局」，这里换 conv 复现了同一堵墙：
**SLM 是延迟隐藏的载体，不是可绕开的开销**。（当时以为 OV 的 16 ops 是「1 broadcast + 1 mad
的发射上限」，此说法已在 17.5 用探针更正：OV 快是因为**不做 staging**，不是发射宽度。）

### 17.4 顺手证伪：SLM 权重宽载（half4/half8）无效

给 `conv3x3_f16` 加了 `WVEC`（权重一次载 half2/half4/half8 再拆成多个 half2 mad）：

| WVEC | ops/EU/cyc |
|---|---|
| 2（原）| 9.77 |
| 4 | 9.76 |
| 8 | 9.93 |

**无变化** —— IGC 本来就把相邻 half2 载入合并了。再次说明扫描指令数不是方向。

### 17.5 物理极限探针（**更正** 17.1–17.4 的推断）

给 `conv3x3_f16` 加 `PROBE`（`-DPROBE=k`）：bit0=权重读换运行时寄存器常量、bit1=strip 同理、
bit2=跳过 staging、bit3=跳过 barrier。**关键：必须反汇编确认执行宽度**——当操作数变成编译期
常量时，IGC 会把整核从 **SIMD16 静默降成标量 `mad (1|M0)`**，此时数字完全失真（下表标 ✗）。

| PROBE | 含义 | ops/EU/cyc | SIMD | 有效 |
|---|---|---|---|---|
| 0 | 基线 | 9.83 | 16 | ✓ |
| 1 | 权重读→寄存器常量 | 12.31 | 16 | ✓ |
| 2 | strip→常量 | 12.29 | **1** | ✗ |
| 3 | 两者 | 18.03 | **1** | ✗ |
| 4 | **跳过 staging** | **16.44** | 16 | ✓ |
| 5 | 跳过 staging + 权重常量 | 16.75 | 16 | ✓ |
| 12 | 跳过 staging + barrier | 17.16 | 16 | ✓ |
| 13 | 跳过 staging + barrier + 权重常量 | 20.10 | 16 | ✓ |
| — | 纯寄存器 half2 FMA（`fmalat h2`）| **27.4** | 16 | 参考 |

**结论（推翻了 17.1–17.4 的「发射宽度」说法）**：

1. **FPU 能被填满**：纯寄存器 packed-half2 FMA 达 **27.4 / 32（86%）**。所以「用最宽指令也
   填不满 FPU」是不成立的——只要操作数在寄存器里，`mad` 就能逼近峰值。
2. conv 基线的 9.83（31%）**不是发射宽度**造成的：主循环是 SIMD16、864 条 `mad`，非 FMA
   指令只有 126 条 `mov: hf`，指令数远不是 2:1。
3. 真正的大头是**存储层级的数据供给**：
   - **staging（global→SLM + barrier）≈ 1.67×**：9.83 → 16.44（跳过 staging）；
   - barrier 本身很小：16.44 → 17.16；
   - **SLM 权重读 ≈ 15%**：16.44 →（跳过 staging+权重常量）20.10；
   - 再往上到 27.4 是纯 mad 结构上限。
4. 所以这台机器上 conv 的物理极限和 GEMM 是**同一个家族**：**staging 延迟 + SLM 操作数带宽**
   （R10/R11 的「容量↔带宽↔并发」，R14 的 staging 31% + loads 17%），**不是 FMA 发射宽度、
   也不是寄存器读带宽**（`fmalat` 证明 RF 可以在 3 操作数 `mad` 下喂到 86%）。OV 的 16.0 高，
   是因为它用 `sub_group_broadcast` + 块读**根本不做 SLM staging**。

### 17.6 下一步

1. **`conv3x3_f16` 的 `CINC` 分块做双缓冲软件流水**（R9 在 GEMM 上验证过 +75%），把
   global→SLM staging 与 compute 重叠掉——这是探针指出的最大单项（1.67×），优先做。
2. 减少 staging 冗余：权重被每个 (gx,gy) work-group 重复从 global 载入 SLM；
   可考虑用更大 `TX/TY` 或把权重常驻。
3. 若复刻 OV：必须 `_sub_group_block_read` + 权重预重排 + **不做 staging**，而不是我们那种
   逐 lane `broadcast`（17.3 的 sg kernel 因全局延迟暴露而失败）。

### 17.7 复现

```bash
# 基线 vs osv vs sg（64→64 @80×80 s1）
./build/kernel_bench --op conv3x3    --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2
./build/kernel_bench --op conv3x3osv --conv-shape 64,64,80,80 --conv 16,8,8,32,16,1,1,0,3,1,16,8,0,0,2
./build/kernel_bench --op conv3x3sg  --conv-shape 64,64,80,80 --conv 4,4,1,64,16,1,1,0,3,1,16,8,0,0,4
# 权重宽载对照
./build/kernel_bench --op conv3x3    --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,8
# 物理极限探针（0基线 / 4跳staging / 12+barrier / 13+权重常量）
./build/kernel_bench --op conv3x3    --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,4
./build/kernel_bench --op conv3x3    --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,13
# 纯寄存器 FMA 参考
./build/kernel_bench --op fmalat --width h2 --depth 8 --sg 16
```

> 对照数据：OpenVINO 2025.2 单节点 GPU 时间（per-node profiling）conv3x3 = **16.0**
> （`convolution_gpu_bfyx_os_iyx_osv32`，direct conv，非 Winograd），
> gemm = 10.4；infvino gemm 13.9（见上一轮横评，脚本 `/tmp/opencode/ov_ops.py`）。

## Round 18 —— ISA 视角收口：staging 指令化 + 网格饥饿，以及 ops/EU/cyc 的理论/物理极限

### 18.1 一句话

conv（占整网 68–73%）的 ~30% 效率由**两堵墙**构成：
**(1) global→SLM staging 的非 mad 指令 + 延迟暴露**——本机 ISA 没有通用 L1
（[`xe-lp-isa.md`](xe-lp-isa.md)），全局复用只能显式进 SLM/GRF；
**(2) 小空间层（20×20/40×40）的 work-group 网格饥饿**——80 EU 只喂到 6–12 个 WG。
本轮用**权重 staging 合并访存（WCOAL）**与**自适应输出通道块（adaptive CB）**把
conv3x3 从 15.0→12.5 ms（yolov8，−17%），整网 yolov8 **22.1→19.6 ms（−11%）**、
yolo11 **24.8→22.3 ms（−10%）**，数值全 PASS。同时用新探针把「理论 32 / 纯 FMA 27–30 /
staging-free 16.4 / GEMM 13.7 / conv 整核 10.3」这条标尺钉死。

### 18.2 瓶颈再分解：staging 到底是「延迟」还是「指令」？

R17.5 用 `PROBE` 得到「跳过 staging = 16.44」，但分不清 staging 里是**全局读延迟**还是
**索引/边界/SLM 写指令**。R18 给 `conv.cl` 加了 `PROBE & 16`：保留 staging 的 div/mod、
边界判断与 SLM 写，只把 `X[…]`/`Wt[…]` 的全局读换成寄存器常量（完全不碰全局内存）。
@64→64 80×80 s1（WC=0）：

| build | ops/EU/cyc | 含义 |
|---|---|---|
| normal | 9.81 | 全量 |
| `PROBE=16`（staging 只写 SLM，无全局读）| 11.85 | 去掉全局读 **+21%** |
| `PROBE=4`（完全跳过 staging）| 16.30 | 再去掉 div/mod/边界/SLM 写 **+38%** |
| 纯寄存器 half2 FMA（R13/R17）| 27.4 | 结构上限 |

→ staging 的损失里**全局读约 21%**，**索引/边界/SLM 写占更大头（相对再 +38%）**；
两段不是简单相加。这直接解释了 18.6 的负结果：**双缓冲藏延迟对「指令那部分」无效**。

### 18.3 优化一：WCOAL —— 权重 staging 合并访存（默认开启）

权重是 `Wt[Cout][Cin][3][3]`。原 staging 让 `t`（输出通道）做最快下标，于是同一 sub-group
内相邻 lane 读 `Wt[gout][gc][kk]` 的**步长 = Cin·KHKW·2 B**——每个 lane 命中不同 cache
line（完全非合并），且每个 (gx,gy) 空间 WG 都重来一遍（20×）。改成 `(ci,kk)` 做最快下标后，
一个 lane 群读的是**单输出通道连续的 `Cin·KHKW` 块**，合并访存；纯循环重排、逐位不变。

| shape | WC=0 | **WC=1** | |
|---|---|---|---|
| 64→64@80 s1 | 9.84 | **10.31** | +5% |
| 128→128@40 s1 | 9.77 | **10.39** | +6% |
| 32→32@80 s1 | 7.50 | **8.37** | +12% |
| 64→128@80 s2 | 6.55 | 6.65 | +1.5% |

已设为默认（`Conv3x3Cfg::WC=1`）。

### 18.4 优化二：自适应输出通道块 —— 治网格饥饿

把 profile tag 细化为 `conv3x3@Wout×Hout s…_Cin…_Cout…` 后可见：**20×20 层占 conv 的
38%**，而它们只有 1.4–3.1 ops。原因不是指令而是 **work-group 太少**：`CB=32` 时
128→128@20 只有 `1×3×4 = 12` 个 WG，80 EU 严重饥饿。把 `CB` 降到 16 让 gz 翻倍：

| shape | CB=32 | **CB=16** | 增益 |
|---|---|---|---|
| 16→16@160 s1 | 4.81 | **6.96** | +45% |
| 256→64@20 s1 | 1.62 | **2.50** | +54% |
| 64→64@20 s1 | 1.41 | **2.20** | +56% |
| 128→128@20 s1 | 3.07 | **3.68** | +20% |
| 64→64@40 s1 | 6.18 | **7.77** | +26% |
| 128→64@40 s1 | 6.97 | **8.17** | +17% |
| 3→16@320 s2 | 1.23 | **1.59** | +29% |
| 64→64@80 s1 | **10.32** | 8.72 | CB=32 保持 |
| 128→128@40 s1 | **10.38** | 8.52 | CB=32 保持 |

落地规则（`PlanModel` 的 conv3x3 分支；`CB=32` 为默认）：

```
spatial = ceil(Wout/TX) * ceil(Hout/TY);
wgs32   = spatial * ceil(Cout/32);
if (Cout <= 16 || wgs32 < 16) cfg.CB = 16;   // 否则保持 32
```

`Cout<=16` 治「半个通道块被浪费」（3→16@320、16→16@160）；`wgs32<16` 治网格饥饿。

### 18.5 端到端

| 模型 | R17 | **R18** | conv3x3 分项 |
|---|---|---|---|
| yolov8n-pose | 22.12 ms | **19.64 ms（−11.2%）** | 15.00 → **12.51 ms（−17%）** |
| yolo11n-pose | 24.76 ms | **22.33 ms（−9.8%）** | — |
| mobilenetv3-small | 6.29 ms | 6.27 ms（持平）| conv 占比小 |

`numerical_check.py` 三模型 **ALL PASS**（yolov8 mean_rel=5.37e-4 / max_rel=1.87e-2；
yolo11 9.91e-4 / 3.12e-2；mobilenet 1.20e-2 / 1.12e-2）。

### 18.6 负结果（已实测，代码保留为记录）

| 尝试 | 结果 | 原因 |
|---|---|---|
| **CINC 双缓冲软件流水**（新 `conv3x3_db`：chunk 循环 2× 展开、编译期缓冲下标、预取下一 chunk）| CINC8 **8.64**、CINC16 **5.91**（均 < 9.86）| IGC 在**单缓冲**里已把 global→SLM 与 compute 交织（反汇编里 global send 穿插在 mad 之间），所以双缓冲没有额外收益；而它把 SLM 翻倍 → occupancy 掉，净负。与 R12 GEMM「`SG=16` 后双缓冲不再必要」同源 |
| **XGN**（输入 halo 不进 SLM，strip 直读全局）| **6.85** | 全局读延迟暴露在 mad 依赖链上；再次坐实 R12 `GN` 与 R17.3 |
| CINC=32 | build-fail（SLM）| — |
| CINC=8 / `UNROLL_CI`=1/5 | 9.82 / 9.95 / 9.76 | 不如 CINC=16 / U=3（10.31）|
| TY=4/5/16、TX=80 | 5.8 / 3.7 / build-fail | 行 halo / 寄存器 / SLM |

### 18.7 ops/EU/cyc 的理论极限与物理限制（本轮收口）

**模型**：本机（Iris Xe 80 EU / 1.3 GHz）EU **每周期只发射 1 条向量指令**，FP16 靠
packed half2 提供 **16 FMA/EU/cyc**。于是

```
ops/EU/cyc = 32 × (MAD 指令数 / 总发射指令数) × (实际 SIMD 宽度 / 16)
```

这一条式子把三类损失全部量化：**非 mad 指令**（load/地址/控制/barrier 争同一个发射槽）、
**IGC 把 SIMD 降到 8（×0.5）**、以及**槽位停顿**。conv/GEMM 的优化本质就是
「让每个发射槽都变成 SIMD16 的 mad」。R18.2 的探针正是该式子的直接证据。

**层级标尺（本机实测）：**

| 层级 | ops/EU/cyc | % of 32 | 卡在哪 |
|---|---|---|---|
| 理论峰值 | **32** | 100% | 16 packed FMA/EU/cyc |
| 纯寄存器 FMA（h1/h2/h8，SG16）| 27.4–29.6 | 86–92% | 指令槽 + 循环开销（**结构上限**）|
| GEMM compute-only（skipstage）| 17.7 | 55% | SLM 操作数 feed / 依赖链 |
| **GEMM 整核（BK32 SG16）** | **13.7** | 43% | staging 31% + loads 17% + 寄存器占用 24% |
| conv3x3 staging-free | 16.4 | 51% | SLM 权重块读延迟 |
| **conv3x3 整核（R18，大层）** | **10.3** | 32% | staging + 网格饥饿（小层 2–8）|
| OpenVINO 2025.2 conv3x3 | 16.0 | 50% | OV 不做 SLM staging |

**物理限制（全部有 ISA 证据，见 [`xe-lp-isa.md`](xe-lp-isa.md)）：**

1. **SIMD 宽度 = 寄存器大小压力**：只有 `intel_reqd_sub_group_size(16)` 才能在 staging
   的寄存器压力下阻止 IGC 退到 SIMD8（R12/R13）。128 GRF/线程 × 32 B × 7 线程 =
   **28 KB/EU 是硬墙**；寻址/调度/中间值就吃掉约 90 个寄存器，留给累加器 `TM×TN` 的只够
   算术强度 ≈ 32，`TM=16/TN=8` 必 spill（R14）。
2. **SLM 容量 ↔ 带宽 ↔ occupancy 耦合**：64 KB/WG、128 KB/DSS；带宽随常驻 WG 数上升
   （4 KB/WG 时 558 GB/s，16 KB/WG 只 300 GB/s，R11）。双缓冲/大 tile 会同时推高容量、
   降低并发、腰斩带宽——三者是同一约束。
3. **没有通用 L1**：全局数据只在 **3.75 MB L3**（~300 GB/s）或 **DRAM ~19 GB/s** 之间；
   「读一次用一次」（AI≈1）可用 L3，大 K 必须分块在 L3 内。**任何全局复用都必须显式进
   SLM/寄存器**，这是 conv staging 存在的根本原因。
4. **延迟**：FP16 FMA 延迟 ~29 cyc（FP32 ~54），需 ~29 条独立链填满；SEND/block-read
   延迟数十 cyc。7 线程/EU 的 SMT 决定可并行 sub-group 数上限。
5. **发射**：每周期 1 条向量指令 → 非 mad 指令与 mad 争槽，这是 staging 指令直接折损
   吞吐的机制，也是「双缓冲无效」的根因（它藏延迟、不省指令）。

**结论**：在不改算法（direct conv + SLM tile）的前提下，conv 的现实上限就是
**~16 ops/EU/cyc（staging-free，≈OV）**；本轮把整核从 9.9 推到 10.3（大层），网格
饥饿层 +20–56%。要再往上只能换数据通路（sub-group block-read / 寄存器交换）或换算法
（Winograd 减少乘数）；而按 R14 的推导，前者只是把「寄存器墙」换成「SLM 带宽墙」，
天花板同量级。

### 18.8 复现

```bash
# staging 探针：normal / 无全局读(PROBE16) / 完全跳过(PROBE4)
./build/kernel_bench --op conv3x3 --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,0,0,0,0  --iters 30
./build/kernel_bench --op conv3x3 --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,16,0,0  --iters 30
./build/kernel_bench --op conv3x3 --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,4,0,0   --iters 30
# WCOAL 对照（末三位 = WC,DB,XG）
./build/kernel_bench --op conv3x3 --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,0,0,0,0 --iters 30
./build/kernel_bench --op conv3x3 --conv-shape 64,64,80,80 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,0,0,1,0 --iters 30
# 自适应 CB：同一 shape 扫 CB=32 / 16
./build/kernel_bench --op conv3x3 --conv-shape 64,64,40,40 --conv 40,8,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,0,0,1,0 --iters 30
./build/kernel_bench --op conv3x3 --conv-shape 64,64,40,40 --conv 40,8,1,16,16,1,1,0,3,1,16,8,0,0,2,0,2,0,0,1,0 --iters 30
# 负结果：双缓冲
./build/kernel_bench --op conv3x3db --conv-shape 64,64,80,80 --conv 40,8,1,32,8,1,1,0,3,1,16,8,0,0,2,0,2,0,0,0,1 --iters 20
# 端到端
./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 5
```

## Round 19 —— 「用 L2 带宽减轻 SLM 压力」的验证（负结果）+ 命名澄清

**问题**：能不能调整算法，用（记得带宽挺高的）L2 去减轻 SLM 压力？

**命名澄清（重要）**：这台机器 compute-API 里显示的 **"L2" 就是 GPU L3**
（3.75 MiB，270–345 GB/s），不是 L1/L2/L3 三级里的 L2；Xe-LP **通用 L1 = 0**，
HDC L1 只在驱动 MOCS 命中的小窗口启用（kernel 不可控）。真正 650–780 GB/s 的高带宽
存储是 **SLM**（可编程 scratchpad），它**不是 cache**。所以字面意义上没有「高带宽 L2」
可供 kernel 使用——之前笔记里那个「挺高的」要么是 SLM，要么是把 GPU L3（330）记成了 L2。

**实验**：把最值得挪的一部分（**权重**）从 SLM 放到 GPU-L3。权重预重排成
`[Cin][KHKW][Cout]`（输出通道连续 → 可合并/广播块读），跳过 `Ws` 的 SLM staging，
内循环直接读 L3（新开关 `-DWGL`）。@64→64 80×80 s1：

| 方案 | ops/EU/cyc |
|---|---|
| 权重+输入都在 SLM（R18 基线）| **10.26** |
| 权重在 GPU-L3（WGL）| 8.93 |
| WGL + CB16 | 8.28 |
| WGL + CINC8 | 8.87 |
| 输入+权重都在 GPU-L3（XGN+WGL，OV 式零 SLM）| **6.63** |

**为什么带宽够也没用**：这一层的权重+输入总流量约 3.6 MB/层、0.46 ms 内 ≈ 8 GB/s，
L3 的 330 GB/s 绰绰有余——**瓶颈不是带宽，是延迟**。`--op memlat` 实测全局
pointer-chase 延迟在 16 KB–64 MB 全落在 **~140–190 cyc**（没有低延迟档），而 SLM 是
几十 cyc。内循环一个 block read 只喂 16 条 mad，要藏住 ~150 cyc 需要每线程几十条独立
操作，**128 GRF 放不下**。所以 SLM 的价值在「低延迟」，330 GB/s 的 L3 顶不上。

**结论**：L3 带宽其实**已经在被利用**（staging 的全局读本来就命中 L3），但它不能替代
SLM 作为**内循环操作数**的供给。这从数据通路角度再次坐实 R14/R17：本机可编程、低延迟
的存储只有 **GRF + SLM** 两级；想突破 conv 的效率，只能改**数据通路**（sub-group
block-read / 寄存器交换）或改**算法**（Winograd 减少乘数），而不是换缓存层。
`-DWGL` 保留为负结果记录。

## Round 20 —— OpenVINO conv 策略剖析 + 修正 R17.3 的误判

### 20.1 OV 的 conv 用了什么

对 `yolov8n-pose` 这类 3×3 s1 fp16/bfyx 层，OV 2025.2 单节点计时选中的是
`convolution_gpu_bfyx_os_iyx_osv32`（源码 `~/openvino/src/plugins/intel_gpu/src/kernel_selector/
cl_kernels/convolution_gpu_bfyx_os_iyx_osv32.cl`）。它的结构与我们的 `conv3x3_f16` 完全相反：

| 维度 | OV `os_iyx_osv32` | 我们 `conv3x3_f16` |
|---|---|---|
| **lane 对应** | **输出通道**（`feature_idx = fmg*OSV_SIZE + lid`，每 lane 2 通道，`OSV_SIZE=32`）| **空间位置**（lane=列，通道在寄存器里展开 CB=32）|
| 输入 | 每个 kd 把 `OBW×OBH` 的输入块**直接载入寄存器**（lane=空间 x，合并读），再用 `sub_group_broadcast` 把标量发给所有通道 lane | 先 global→**SLM** halo，再逐 (ci,kh) 读 strip |
| 权重 | 预**swizzle** 成块读格式，每个 (kd,kr,kc) 一次 `UNIT_BLOCK_READ` 给每个 lane 自己 2 个通道 | SLM 通道连续块读（WCOAL 后合并）|
| SLM / barrier | **零 SLM、零 barrier** | SLM 双层 + 每 chunk 2 个 barrier |
| 网格 | `gws = (W/OBW, H/OBH, ceil(Cout/32))`，**一个 WG = 一个 sub-group**，WG 数极大（K3S1 默认 OBW=8、OBH=2）| WG=TX/TM×TY=320 work-item，WG 少（80×80 只有 40 个）|

要点：**OV 用「lane=通道 + 输入块驻寄存器 + 广播」换掉了整个 staging 层**，代价是每个 mad
前有一条 `sub_group_broadcast`（把某个空间输入广播给所有通道 lane）。所以它的内循环是
**1 broadcast : 1 mad**，理论上限 ≈ 32/2 = 16——这正好等于它实测的 16.0。

### 20.2 修正 R17.3：我们复刻失败的真因是「权重非合并」，不是「全局延迟」

R17.3 的 `conv3x3_sg`（lane=通道、无 SLM）只有 **1.5**，当时归因于「全局输入延迟暴露」。
R20 反汇编后**证伪**：kernel 里出现了 64 条 `send.dc0 byte scattering write 16b` + 21 条
gather read —— 权重读 `Wt[oc*Cin*9 + kr*3+kc]` 相邻 lane 步长 `Cin*9`，**每个 lane 命中
不同 cache line**，一次 load = 16 个 64 B 事务，整核根本不是算不动而是被权重的
非合并全局读拖死。

修法：把权重预重排成 **`[Cin][KHKW][Cout]`**（输出通道连续），lane 读变成 contiguous /
可广播。同一 kernel、同一 shape：

| conv3x3_sg @64→64 80×80 | ops/EU/cyc |
|---|---|
| R17.3（自然 `[Cout][Cin][9]` 布局）| 1.50 |
| **R20（重排 `[Cin][9][Cout]`）** | **8.20**（OBW8/OBH4/VECO2）|

反汇编确认已是 **SIMD16、`in[]` 在寄存器里（无 spill）**，`mad` 用寄存器标量广播
（`r4.x<0>:hf`）。所以 R17.3 的结论要改：失败点是**访存布局**，不是「无 SLM 必然暴露延迟」。

### 20.3 但 OV 策略在**本机、本模型**上仍然不如我们的 kernel

修好合并访存后，把 OV 式 kernel 与 R18 的 `conv3x3_f16`（WCOAL+自适应 CB）逐 shape 对比：

| shape | 我们（R18）| OV 式（R20，最好 OBW8/OBH4）|
|---|---|---|
| 64→64@80 s1 | **10.31** | 8.20 |
| 128→128@20 s1 | **3.71** | 1.45 |
| 256→64@20 s1 | **2.49** | 0.74 |
| 16→16@160 s1 | **7.00** | 4.32 |

**为什么**：两种结构的 16 上限来源不同——
- **OV 式**：内循环 **1 broadcast : 1 mad**（静态 576 `mad` vs 715 `mov`），发射槽一半给了广播；
  唯一突破办法是每 lane 多放通道（VECO4/8），但马上 spill（实测 0.2–5.1）。大块（OBW≥12）
  也 spill/浪费。所以它稳定在 ~8。
- **我们**：内循环 **输入/权重都是 ~1:16**（strip 被 3 kw×16 通道复用、权重块读喂 16 mad），
  inner loop 的 staging-free 上限有 **16.4**；短板是 staging（R18.2 探针：全局读 +21%、
  索引/边界/SLM 写 +38%）。

换句话说：**OV 是「用广播费换掉 staging」，我们用「用 staging 费换掉广播」**，两者都撞在
~16 的天花板，只是撞的位置不同。而在这台「无 L1、SLM 带宽-occupancy 耦合」的机器上，
**lane=空间 + SLM** 这个角落比 **lane=通道 + 广播** 更划算（10.3 vs 8.2；小空间层差距更大，
因为我们的自适应 CB 还能加网格，OV 的块粒度反而浪费）。

### 20.4 重新分析后的结论与下一步

> **核心现实：两条数据通路（lane=空间+SLM 的我们，lane=通道+广播的 OV）在
> ops/EU/cyc 上的上限都是 ~16**——一条被 staging 卡在 16.4（staging-free），
> 一条被 1 broadcast : 1 mad 卡在 16.0。想真正突破 16，只能减少**乘数**或换数据通路。

1. **R18 的结论不变且被这轮加强**：WCOAL + 自适应 CB 是当前最优；OV 的路径不是更优解。
2. **R17.3/R17.5 关于 OV 的两处判断要修正**：
   - 「OV 快是因为不做 staging」——对；但「我们复刻不了是因为全局延迟」——错，是权重布局。
   - OV 的 16.0 不是遥不可及的下限，而是它自己的 **broadcast 上限**；我们的 staging-free
     上限（16.4）与它同量级。
3. **真正能突破 ~16 的只有两条**：
   - **减少乘数**：Winograd F(2×2,3×3)（MAC ↓2.25×）或 F(4×4,3×3)（↓4×）；OV 自己也有
     `convolution_gpu_winograd_2x3_s1`，但它的 selector 在本机这些形状上没选 Winograd
     （选了 direct），说明在低算术强度 + 8 宽 SIMD 上未必划算——这是下一轮值得实测的点。
   - **换数据通路**：sub-group block-read / 寄存器交换（R14 判定只是把「寄存器墙」换成
     「SLM 带宽墙」）。
4. `conv3x3_sg`（OV 式）与 `-DWGL` 一起保留为**实验/负结果**工具，带正确的 `[Cin][9][Cout]`
   权重布局要求；生产路径仍走 `conv3x3_f16`。

## Round 21 —— Winograd 能否突破 16？（结论：不能，被 128 GRF 的「16 个 m 值」放大）

### 21.1 目标与算法

R18–R20 已确认：direct conv 和 OV 式两条数据通路的 ops/EU/cyc 上限都是 ~16。唯一能真正
突破的算术手段是**减少乘数**。Winograd F(2×2,3×3)（stride 1）把每个 (tile, cin, cout) 的
乘法从 direct 的 `2×2×9 = 36` 降到 `4×4 = 16`（**2.25×**）：

```
V = B^T d B            // 4x4 输入块 -> Winograd 域（32 个加法）
U = G g G^T            // 3x3 权重 -> 4x4（离线预变换）
M = Σ_ci V ⊙ U          // 16 个逐元素乘累加（GEMM 形态）
Y = A^T M A            // 4x4 -> 2x2 输出（24 个加法）
```

算法本身已验证：用 numpy 完整复刻 host 的 `G g G^T` + lane-major 权重 swizzle + kernel 的
读/存索引，结果与 direct conv 逐元素一致（maxerr 2.4e-7）。**数学没问题，问题在寄存器。**

### 21.2 为什么在这台机器上不成立：M 需要「每 (tile, channel) 16 个活值」

direct conv 每个 (像素, 通道) 只需要 **1–4 个累加器**；Winograd 在做逆变换前必须同时持有
**16 个 m 值**。这是一个寄存器乘数：

- 每线程累加器 = `16 × (通道/线程) / 2`（half2）。CB=8 就是 **64 个 half2**，已经贴到
  128-GRF 墙；再叠加输入变换的 `d/t/V[4][4]`（~48 个 half）与权重/地址，必 spill。
- 想靠减少 CB 降寄存器，则输入变换（每 (tile, ci) 32 个加法）只能摊到很少的通道上，
  摊薄不了；lane=通道（OV 式）还会让每个 lane **冗余**做同一块输入变换。
- 想靠「lane=空间 + 通道在寄存器」摊薄变换，则累加器 `16×TM×CB/2` 立刻爆掉。

这正是 R16「寄存器分块 3×3 反而更慢」同一堵墙，只是 Winograd 把墙放大了 16 倍：
**它用乘数换来的收益，被「多出来的活状态」在 128 GRF 上吃回去。**

### 21.3 实测（fused 原型，负结果）

实现了一版**单 kernel 融合** F(2×2,3×3)（lane=通道、VECO=2、一 WG=一 sub-group 处理一个
2×2 tile，权重预变换+swizzle，输入/逆变换都在寄存器）。结果：

- `ocladoc` 反汇编：出现 `byte gathering read 16b` 等 scratch/SLM 流量（即 `acc[16]`/`V[4][4]`
  溢出到私有内存）；
- 实测 **0.38 ops/EU/cyc**（12.1 ms vs direct 的 0.44 ms，≈**27× 慢**）——寄存器墙的直接后果。
  （该原型还有一处 IGC codegen 导致部分通道结果不对；但即便修好，寄存器预算也把上限
  压到远低于 16，故不再投入。）

### 21.4 为什么 OV 也不靠 Winograd 破 16

OV 确实有 `convolution_gpu_winograd_2x3_s1`，但它是**三段式**：`transform 输入 → Winograd 域
GEMM（把 M 写回全局）→ 逆变换`，正是为了**不把 M 留在寄存器**。代价是 16 倍的中间张量
流量（对 64→64@80 约 3.3 MB，能进 L3 但要走一遍 L3 往返），且 Winograd 域的 GEMM 是
`M=Cout=64, N=tiles, K=Cin` 的一批小 GEMM，在本机（M 小、网格小）效率只有 ~5 ops。OV 的
selector 在本模型这些形状上**没有选 Winograd，选的是 direct `os_iyx_osv32`**——与我们的结论一致。

### 21.5 结论

- **Winograd 不能在这台机器上突破 16**：它减少的是乘数，但把「每 (tile,通道) 的活值」从
  1–4 放大到 16，在 128 GRF / 7 线程 per EU 上是决定性的负项；三段式又会引入 16× 中间流量。
- 因此 **R20 的「两条路都撞 16」依然成立且更完整**：direct（staging 墙）与 OV 式（broadcast 墙）
  之外，Winograd（register 墙）也在 16 以下。
- 要真正突破，只剩**换硬件能力**（更大的 GRF/专用矩阵单元/更强的 L1）或**换问题规模**
  （batch 足够大时 Winograd 域 GEMM 的 M 才够大）；在本机单流、小 batch 的现实下，
  **direct conv ~10.3（大层）/ 网格饥饿层 +20–56% 就是当前最优**。

## Round 22 —— 1×1 专用 kernel（融合 + split-K GEMV）、OpenVINO conv3×3 移植、并行 GAP

目标（本轮任务）：(1) 写 1×1 conv 专用 kernel 优化 mobilenetv3-small 端到端；
(2) conv3×3 直接上 OpenVINO 的 kernel（引入代码、不引入依赖，处理 Apache-2.0）；
(3) 用融合提升三个模型端到端。

### 22.1 先给整网账本分维（profile tag 细化）

把 `gemm` 的 profile tag 细化为 `gemm@MxNxK` 后，mobilenetv3-small 的 4.12 ms gemm 里
**N=1（HW==1）的矩阵-向量积占 ~2.3 ms**（classifier 1024×1×576 / 1000×1×1024、SE 的
fc1/fc2 全部 HW=1），而通用 GEMM 在 N=1 时 `BN=64` 浪费 63/64 lane、网格只有
`ceil(M/BM)` 个 work-group（1000×1×1024 只有 8 个）——是纯粹的网格饥饿。

### 22.2 1×1 conv：融合 bias+激活 + N=1 split-K GEMV

新增 `kernels/conv1x1.cl`：

- **`conv1x1_gemv_f16`（N==1）**：一个 work-group = 一个 sub-group（16 lane）= 一个输出
  通道，lane 沿 K 归约，`sub_group_reduce_add` 收口。W 保持自然 `[Cout][Cin]`，相邻 lane
  连续读（coalesced），X 广播；**fp32 累加**（数值反而更好，mean_rel ~1e-8）。
- **N>1**：直接复用调好的 `gemm_f16`，但加了**融合 bias+激活的 epilogue**（`-DEPI=1`），
  取代原来的 `gemm + bias_add + ew_unary` 三连击。
- 另写了一版「lane=空间 + 输出通道在寄存器」的 `conv1x1_f16`（含 `[Cin][Cout]` 转置权重
  宽载），实测在 N>1 上**全面慢于 `gemm_f16`**（gemm 的小 tile 网格饥饿反而不如它的 SLM
  分块），保留为负结果/实验工具。

`onnx2plan.py` 把 1×1 conv 与 `Gemm` 折叠成单节点 `conv1x1`（融合 bias + 尾部激活）。

**效果（mobilenet，逐层）**：N=1 的 15 个层从 gemm 合计 ~2.3 ms → gemv ~0.46 ms
（**5–27×**）；`bias_add`(42) + `ew_unary`(42) 两次 launch 全部消失。

### 22.3 OpenVINO conv3×3（`convolution_gpu_bfyx_os_iyx_osv32`）移植

把 OpenVINO 的 `convolution_gpu_bfyx_os_iyx_osv32.cl` **数据通路**移植为自包含的
`kernels/conv_ov.cl`（Apache-2.0，见 `THIRD_PARTY_NOTICES.md` + `third_party/openvino/LICENSE`）：

- lane=输出通道（OSV_SIZE=32，每 lane 2 通道）；输入块一次载入寄存器 + `sub_group_broadcast`；
- 权重用 `intel_sub_group_block_read_us2` 从 **OSV swizzle 布局**读取（host 端
  `PlanModel::ovWeight` 按 `GET_FILTER_OS_IYX_OSV_INDEX` 重排）；
- **零 SLM、零 barrier**；bias+激活（+可选残差）在输出阶段融合。

**关键修正**：R20 的自写 OV 式 kernel（`conv_sg`）用「逐 lane 合并读」，只到 8.2。本轮
发现 `_sub_group_block_read*` 是**跨步**语义（lane l 取位置 `l, l+16`，不是 `2l,2l+1`），
权重布局是 `o%32` 自然序——按此实现后 kernel **正确且明显更快**：

| shape | native conv3x3（R18） | **OV 移植（本轮）** | 加速 |
|---|---|---|---|
| 64→64@80 s1 | 10.31 (0.440 ms) | **13.61 (0.333 ms)** | **1.32×** |
| 128→128@40 s1 | 10.34 (0.439) | **12.61 (0.360)** | 1.22× |
| 64→64@40 s1 | 6.16 (0.184) | **7.91 (0.143)** | 1.29× |
| 16→16@160 s1 | 4.81 (0.236) | **5.75 (0.197)** | 1.20× |
| 128→128@20 s1 | 3.09 (0.367) | **5.27 (0.215)** | **1.71×** |
| 256→64@20 s1 | 1.63 (0.694) | **2.92 (0.388)** | **1.79×** |
| 64→128@160 s2 | 7.07 (1.283) | **10.73 (0.846)** | **1.52×** |

→ **在所有测试形状上 OV 移植都赢**，于是 `PlanModel` 的 conv3x3 默认走 OV（`ov=0` 可回退
native）。这修正了 R20「OV 路径不如我们」的结论：**不是 OV 数据通路差，是当时的权重
布局/块读语义没复刻对**。OV 的 13.6 也说明「lane=通道 + block-read 权重」这条通路在本机
上能越过 native direct conv 的 staging 天花板（R18 的 staging-free 16.4 仍是更远的上限）。

### 22.4 并行 GAP

`ops.cl` 新增 `gap_r`：一个 work-group（128 lane）一个通道，SLM 树归约。mobilenet 10 个
GAP 从 **0.857 ms → 0.119 ms（7.2×）**（原来每通道 1 个 work-item 串行扫 HW，C=16 时
整机只有 16 个 work-item）。

### 22.5 融合：已生效与负结果

- **生效**：1×1 conv / Gemm 的 `bias+act` 融合（§22.2）；conv3×3 OV 的 `bias+act` 融合；
  conv3×3 的尾部 HardSwish 折叠（`ACT_CODE_CONV`）。
- **负结果（默认关闭，`INFVINO_FUSE_RESIDUAL=1` 可开）**：把 C2f/mobilenet 的残差 `Add`
  折进 conv epilogue。三个模型**全部变慢**（yolo8 17.30→17.61、yolo11 19.93→20.88、
  mobilenet 3.39→3.72 ms）：融合后的 conv 多了一次全张量残差读 + 一个分支，寄存器压力
  上升的代价大于省下的 `ew_binary`。已在 `onnx2plan.py` 用开关封存。
- **未做**：`concat4`（yolo 里 1.1–1.3 ms）是 DRAM 带宽受限的 C2f concat，真正要省必须把
  concat 融进消费者 conv 的输入 staging（跨多输入 gather），本轮未做。

### 22.6 端到端结果（单流，warm）

kernel busy（`kernel_run --report`，同会话 HEAD 基线对照）：

| 模型 | HEAD 基线 | **R22** | 加速 | 主要来源 |
|---|---|---|---|---|
| yolov8n-pose | 19.60 ms | **17.37 ms** | 1.13× | conv3×3 OV + 1×1 融合 |
| yolo11n-pose | 22.40 ms | **20.27 ms** | 1.10× | conv3×3 OV + 1×1 融合 |
| mobilenetv3-small | 6.28 ms | **3.46 ms** | **1.81×** | N=1 GEMV + 1×1 融合 + GAP |

墙钟（`infvino_bench`，net only，含 launch 开销）：

| 模型 | HEAD 基线 | **R22** | 加速 |
|---|---|---|---|
| yolov8n-pose | 24.09 ms | **22.34 ms** | 1.08× |
| yolo11n-pose | 28.24 ms | **25.58 ms** | 1.10× |
| mobilenetv3-small | 7.29 ms | **4.64 ms** | **1.57×** |

数值：`kernel_check.py` **ALL PASS**（含新增 `conv1x1` / `conv1x1g` / `conv3x3ov`）；
`model_check.py` 三模型 **ALL PASS**（yolov8 5.69e-4、yolo11 9.54e-4、mobilenet 1.31e-2）。

### 22.7 结论

1. mobilenet 的瓶颈是 **N=1 矩阵-向量积的网格饥饿**，split-K GEMV 一次拿下 ~5–27×；
2. conv3×3 用 **OV 的 block-read 权重 + lane=通道** 通路能超过 native direct conv
   （大层 1.2–1.8×），这是本轮最大的单 kernel 收益，且推翻了 R20 的误判；
3. 融合只在「不把 conv kernel 变重」时才有正收益（bias+act 融合赢，残差融合输）；
4. 剩余整网大头：yolo 的 `concat4`（DRAM 带宽）与 `bmm/softmax`（attention），
   mobilenet 的 `depthwise`；都不是「换 kernel tile」能解决的，需要改数据通路/图融合。

## Round 23 —— 小算子/融合收益 + 阻塞式 conv 移植（未成功）+ 现状

### 23.1 端到端账本（先量化差距）

OSV 2025.2（同一块 iGPU，per-node GPU 时间）整网：yolov8-pose 9.33 ms / yolo11-pose
9.61 ms / mobilenet 0.96 ms；端到端（含预处理）11.24 / 11.82 / 1.78 ms。
infvino R22 之后 kernel busy：~19.9 / ~23.0 / ~3.5 ms。**差距主要在 conv3x3**：
yolov8 里 `conv3x3ov@40x40s1_Cin64_Cout64` 单层 1.49 ms ×10 = ~15 ms，占总时间 ~85%。
（**R24 更正**：实测这 10 个层**合计** 1.39 ms（`kernel_run --report`），占 yolov8n kernel busy
~9%；conv3×3 **合计**占 ~64%。R23 这里把「10 层合计」误写成单层再 ×10，放大了 10×。）

### 23.2 已落地的两项**真实、过数值**的收益

| 提交 | 内容 | 效果 | 数值 |
|---|---|---|---|
| `b28a6a4` | `.cl` 源码文本缓存 + PlanModel per-node kernel 句柄缓存（纯缓存）| 去除每次推理上百次磁盘读/编译/建 kernel | 逐位不变 |
| `bef119f` | **concat4 改 3-D 网格**（去掉逐元素 div/mod，纯索引简化）| yolov8n busy 19.9→**17.2**、yolo11n 23.0→**20.1** ms；concat4 1.16→1.02 ms | 精确（copy），三模型 PASS |

（另试了并行 `softmax_axis_r`：0.63→0.155 ms 但它破坏数值（mean_rel 3.7e-2），已回退。）

### 23.3 conv3x3 差距定位：**网格饥饿**，不是单核算术

对 40x40 Cin64 Cout64（占总时 ~85%）（**R24 更正：该 10 层合计 ~9%，conv3×3 合计 ~64%**）做 block 扫描（osv32 的 OBW/OBH）：

| cfg | 8,2 | 5,2 | 6,2 | 4,4 |
|---|---|---|---|---|
| ops/EU/cyc | 7.84 | **8.40** | 7.94 | 6.51 |

各 shape 的最优也只是：80x80 13.6（8,2）、40x40 8.4（5,2）、20x20 5.9（4,2）。
**调参上限 ≤ +7% 且大层回退**，无法弥合 2× 差距 —— 与 R18–R21 的结论一致：
`os_iyx_osv32` 是 **1 broadcast : 1 mad 的发射上限（~16 ops）**，与 block 大小无关。

### 23.4 OpenVINO 阻塞式 conv（`convolution_gpu_bfyx_f16`）移植尝试（**未成功**）

OV 实际选中的是**阻塞式** kernel（`convolution_gpu_bfyx_f16`，527 行 + 大量 JIT/宏），
而非 osv32。其要点：lane=通道（16/块），每 lane 持有 `OUTPUT_X_BLOCK_SIZE` 个**连续**
输出列，输入行 staged 后跨 kx/输出复用，权重用 `block_read`，网格 = `(X_BLOCKS,
feature_blocks)`（远多于 osv32）。R23 写了一个自包含简化版 `kernels/conv_blk.cl`：

- 首版（lane 各自标量读输入行）：**1.3 ops**，远慢于 osv32 的 7.9——16 lane 冗余读同一行、
  权重非合并；
- 改权重 `block_read` + 输入行 `sub_group_broadcast` 分发后：40x40 到 **6.6 ops**，
  仍**低于** osv32（7.9），且输出出现 NaN（越界 lane/leftover 处理未完成）。

**结论**：在当前工时内，简化版阻塞 conv **没有打赢已调优的 osv32**，且正确性未收口。
这与 R18–R21 的判断吻合——两条数据通路都撞 ~16 的天花板，阻塞式并不会自动突破；
OV 的优势更多来自其**全功能 JIT kernel + 逐层 autotune + 布局/融合**的整体工程，
而非单一“阻塞”技巧。**该移植已回退，保留为负结果记录。**
（若继续：需完整实现 leftover/OOB/分组路径并用 `kernel_check` 逐 shape 收口。）

### 23.5 调试教训（bisect 方法）

R23 中途排查「数值 FAIL」时走了弯路：把 `kernels/ops.cl` **单独回退到 HEAD**，却仍用
R23 的 `PlanModel`（它按 **3-D 网格**发起 `concat4`），于是 1-D 的旧 kernel 收到 3-D 的
`gws` —— 人为制造了「kernel/host 不一致」的假 bug，并在这上面浪费了若干轮。**教训**：
bisect 时若改动横跨「kernel 源码 + 其 host 端 launch 约定」，必须**成对回退/成对前进**，
或用同一提交里的完整状态测试；不要把新 host 和旧 kernel 混跑。最终确认：完整一致的
R23（3-D concat + 新 PlanModel）本身数值正确，`concat4` 是纯 copy、精确无误。

### 23.6 现状（R23 收尾）| 模型 | R22 busy | **R23 busy** | 墙钟(R23) | OV infer |
|---|---|---|---|---|
| yolov8n-pose | ~19.9 ms | **~17.2 ms** | ~22 ms | 11.2 ms |
| yolo11n-pose | ~23.0 ms | **~20.1 ms** | ~25 ms | 11.8 ms |
| mobilenetv3-small | ~3.5 ms | ~3.5 ms | ~4.6 ms | 1.8 ms |

剩余差距集中在 conv3x3（占 yolov8 ~64%（R24 更正，见上），已到 osv32 的 ~8–13 ops），以及 yolo 的
`concat4`（已优化到带宽极限）与 attention。下一步若继续，方向应是
「完整移植 OV 阻塞 conv + 逐层 autotune」，属大工程。

## Round 24 —— conv3×3 逐 size 瓶颈定位 + 「中间标准」修正 + 两通路接入调优

> 完整分析见 [`docs/round24-analysis.md`](round24-analysis.md)。本轮不改 kernel 源码
> （一个 ILP 变体试过并回退），核心产出是**用离线 ISA 推翻错误的「~16 上限」**并修正调优目标。

### 24.1 决定性证据（离线 `ocloc` 反汇编 `conv_ov.cl`，无需 GPU）

- 主内循环（2 输入通道）**288 条 packed `mad` / 453 条指令 = 63.6% mad**；
- **`sub_group_broadcast` 被折进 `mad` 标量操作数**，指令流里没有独立广播指令；
- 因此指令配额上限 = `32 × 0.636 ≈ 20.3`，**不是** R20/R23 按「1 broadcast : 1 mad」
  推断的 16（那是 `conv3x3_sg` 的结构，不是本生产路径）；
- 实测 80×80 = 13.77（配额的 68%）、40×40 = 8.46（42%）→ **缺口是延迟/流水/占用**，
  不是指令数。这坐实了「默认软件流水不够好」，并更正了 R18.7/R20 的 `conv3x3 整核 10.3 /
  OV 16.0` 标尺（后者是 OV 整体选核的聚合值，不是本移植 kernel 的天花板）。

### 24.2 逐 size 瓶颈（修正后的中间标准下）

| 类别 | 代表 | ratio=实测/期望 | 短板 |
|---|---|---|---|
| 大空间/大通道 | 80×80 s1 64→64 | 0.69 | 内存/send 延迟 |
| 中空间（重点）| 40×40 s1 64→64 | 0.42 | grid/占用（block 甜点 8×1）|
| 小空间 | 20×20 s1 64→64 | 0.42 | gridFactor < 1 |
| 通道窄/Cin 短 | 112×112 s2 3→16 | **0.08** | lane 半空 + 循环太短 |

40×40 block 扫描（新覆盖 `OBW×1`）：4×1=5.54、**8×1=8.44–8.69**、10×1=7.54、16×1=6.19、
8×2=7.83 → 既非纯 grid 也非纯 ILP，8×1 是甜点。

### 24.3 尝试（负结果，已回退）

给 `conv_ov.cl` 加 `-DUK`（输入通道双累加集，拉长 FMA 依赖距离）：ISA 无 spill，但实测
全面更慢（40×40 8.4→7.9、80×80 13.8→12.9），且变体有 codegen 正确性问题 → **整体回退**。
判读：IGC 单集 codegen 已在 7 线程间提供足够 ILP，瓶颈更可能在 **send/全局读延迟** 与
**每 WG 边界谓词/prologue/epilogue**。

### 24.4 接入自动调优

- `expected_ops` 修正为 `32 × 0.636 × prologue_amort × grid_factor`（上界 20.3），
  不再锚定实测；`tuning_test` 锁住该行为；
- `candidatesConv3x3` 让 **OV 与 native 在全部 shape（含 stride=2）上都是候选**，
  由调优器按 size 选通路；
- 数值：`kernel_check --skip-gemm` **ALL PASS**（OV / native / conv1x1 / gemv）。

### 24.5 split-K 预实验（**否决**）

在写两 kernel 归约前，用两个只改 shape 的 proxy 验证「更多 grid」的前提：
**(a) Cin 扫描**（≈拆 K）：Cin 64→32（S=2）ops 8.74→**6.62（−24%）**；Cin 16（S=4）→4.44。
每 WG 固定/延迟开销 ≈ 27%。
**(b) grid 扫描**（固定 Cin64、8×1）：grid 400=8.77、800≈9.0（+3%）、1600=**11.0（+25%）**。
→ 400 WG 的问题不是「grid 少」，而是**波量化**（并发 ≈ 80×7=560 sub-group；800 WG 仍是
1.43 波，利用率 ~0.71）。split-K 拿确定的 −24% 换 +3%，**S=2/4 都不划算，已放弃**。

### 24.6 内部块 fast path（**负结果，已回退**）

对齐 block read（内部块用 `intel_sub_group_block_read_us`）→ **数值错误**（`mean_rel≈1.1`）：
`PAD=1` 时 `base_x=oc*stride-1` 为奇数，未对齐的 block read 返回错位数据。
去谓词 per-lane 载入 → 数值 PASS 但**更慢**：40×40 8×1 8.77→7.30、8×2 7.83→7.08、
80×80 8×2 13.77→11.84。热循环里运行时的 `interior` 分支比省下的谓词更贵。
**已回退**（kernel 逐字节回到 R23 状态）。

### 24.7 收口与下一步

R24 三个方向（split-K、双累加集、内部块 fast path）全部实测否决，加上 grid/Cin 扫描，
指向同一结论：**`os_iyx_osv32` 移植在本机已接近其数据通路的实际平台**（8×1 渐近 ≈ 11、
8×2 在 80×80 达 13.77；40×40 类受 `≤400 WG < 560` 的波量化限制，拆工作代价大于收益）。
继续微调 kernel 的期望收益很低，要再进一步只剩：
(1) 完整移植 **OV 阻塞式 conv**（唯一可能越过该结构上限的通路，工程量大）；
(2) 更大 **batch** 把 40×40 类的波填满。详见
[`docs/round24-analysis.md`](round24-analysis.md) §3/§5。


## Round 30 —— 剩下的 kernel（非 conv/gemm）的瓶颈与物理模型

> 完整分析见 [`docs/round30-smallops.md`](round30-smallops.md)。用 R1–R29 的流程
> （profile → 离线 ISA → 硬件模型 → 上限标定）分析剩余算子。

### 30.1 为什么要换标尺

剩余算子（`concat4/copy_c/slice/permute/resize/maxpool/ew_*/softmax/gap/depthwise/bmm`）
算术强度只有 ~0.25–0.5 FLOP/byte，碰不到 FMA 发射。它们的上限是**内存 roofline**：
`T_min = launch + bytes/BW(footprint)`。剩余算子合计占整网 GPU busy **17.4% / 24.1% / 24.3%**
（yolov8 / yolo11 / mobilenet）。

### 30.2 本机内存 roofline（R30 实测）

copy(read+write) 带宽-足迹：1 MB **89.7 GB/s**（L3 峰）、2 MB 62.4、4 MB 44.6、8 MB 21.0、
**DRAM 64 MB 19.4 GB/s**；read-only 1 MB **320 GB/s**、DRAM 22.8。launch floor ≈ **3.5 µs**。
换算到 ops/EU/cyc 尺度：搬运算子上限 L3 ≈ **0.43**、DRAM ≈ **0.093**（峰值 32 的 0.3–1.3%）。

### 30.3 结论：哪堵墙

| 算子 | 实测 | 上限 / 墙 |
|---|---|---|
| `concat4`（占整网 ~10%）| 22–24 GB/s | **DRAM copy 19.4 GB/s → 已到物理极限** |
| `copy_c` / `slice` / `permute` | 40–54 GB/s | L3 copy 90 + **launch floor**（单节点 0.1–0.6 MB）|
| `ew_binary` / `ew_unary` | 20–25 GB/s | L3 + launch |
| `ew_binary_bcast`（pose 解码 n=285600）| ~12 GB/s | **整数 div/mod 延迟**（ISA 4×iqot+4×irem，0 mad）|
| `softmax` / `gap` | 27–31 / 7 | 3 趟串行 + 网格饥饿 |
| `maxpool` | 4.9 GB/s* | issue-bound（0 mad，K² 谓词）|
| `depthwise` | 0.3–0.5 ops | **ISA 配额 2.8**（`depthwise_v` 1485 指令/128 mad，mad=8.6%）|
| `bmm` | 2.1 ops | FP32 配额 8.1（K 循环 mad=50.8%）；网格饥饿 |

\* 最小流量口径。**物理极限只有 concat4（DRAM）与搬运类的 launch floor**；其余差 1.5–5×，
卡在指令数/网格/延迟，不是带宽。

### 30.4 落地

1. **广播快路径（正结果）**：`b_scalar=1` 走标量核；混合 stride 广播走新增的
   `ew_binary_bcast4`（3-D 网格 + 运行期 stride，去掉每元素 4×整数 div/mod）。
   三模型 `model_check` PASS。同会话 A/B：yolov8 `ew_binary` 0.493→**0.291** ms、
   busy 14.568→**14.416**；yolo11 0.592→**0.379** ms、busy 15.914→**15.633**（−1.8%）。
2. **中间标准真实模型**：小算子占位 `return 8.0` 换成内存 roofline（`kBwCurve` +
   `smallMemCeiling`）+ depthwise/bmm 的 ISA 配额（只影响 ratio/报告，不影响选优）。
   `tuning_test` 新增 6 条断言。
3. 新增 `scripts/analyze_smallops.py`（逐节点流量 vs 内存上限，不需要 GPU）。

## 稳定性事故记录（重要）

- **跨推理缓存陈旧（R-P0b，2026-10 修复）**：`PlanModel::blkInput`（`conv_blk` 的输入重排）
  按**张量名**缓存重排结果且**只执行一次**。生产里每帧图像不同，走 `conv_blk` 的图层
  （y8 有 13 层）在第 2 帧起会用到**上一帧**的输入。数值表现为 y8 第 2 帧
  `mean_rel=1.382e-02`（正常 ~5e-4）。
  **为什么长期漏掉**：所有测试都是**同一输入重复跑**，第 2 帧的陈旧数据恰好等于正确数据。
  **修复**：`blkInput` 每次调用都重跑 reorder（只缓存设备 buffer，不缓存结果）。
  **护栏**：新增 `scripts/reuse_check.py` + `src/tools/reuse_check.cpp`，并把
  `model_check/numerical_check/engine_check` 全部改成**同一进程喂两份不同输入、输出取第二帧**
  （`--input2/--image2`）。教训：**任何「按名字缓存、跨 run 复用」的激活态都必须用多输入测试覆盖**。

- **`softmax` 负 axis 未归一化**：`[1,2,400,400]` 的 `Softmax(axis=-1)` 被算成
  `outer=800, axdim=400, inner=320000` → **2.56 亿工作项 → 假死**（表现为开发板卡死）。
  已修（axis 归一化为非负），并在 `kernel_run` 加 **gws 安全阀**（>3e8 直接报错退出）。
- **带宽测试 OOM**：`kernel_bench --op bandwidth --mb 1024` 分配 2×1 GB buffer + host 1 GB，
  iGPU 共享主存 → 打满 8 GB 机器。已把 `--mb` 上限钳到 256。
- **所有容器/脚本已加 `--memory=3g --memory-swap=3g`**，容器不可能再拖垮宿主。
- **整机硬死机（无日志）与 i915 GPU hang**：本开发板（i5-1135G7 + **PREEMPT_RT** `5.15.179-rt84`，
  7.5 GB 与 iGPU 共享）历史上有频繁重启与 `[drm] GPU HANG ... in kernel_run`；两次硬死机在重启前
  **没有任何 OOM/i915/panic 日志**（硬锁死），`CL_KERNEL_PRIVATE_MEM_SIZE=0` 排除寄存器溢出；
  部分重启是 **宿主上无关脚本的 OOM**（人工重启），与本项目无关。
  **⚠️ 完整、标准化的防护协议见 [`docs/benchmark_protocol.md`](benchmark_protocol.md)**（容器限资源、
  GPU 实验短促化、优先离线 ISA、运行前后查内存、hang 处理等），做任何 GPU 实验前先读。

## 复现

```bash
# 1) 使用含 OpenCL dev 头的镜像（infvino-dev；infvino 不维护 Dockerfile）

# 2) 数值检验（宿主 venv: numpy）
python3 scripts/kernel_check.py --repo $PWD --image infvino-dev:latest

# 3) kernel 基准（容器内 cmake 构建后）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/kernel_bench --op gemm --shape 4096,512,512 --verify
```
