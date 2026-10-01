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

## 稳定性事故记录（重要）

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
