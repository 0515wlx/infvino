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

## 稳定性事故记录（重要）

- **`softmax` 负 axis 未归一化**：`[1,2,400,400]` 的 `Softmax(axis=-1)` 被算成
  `outer=800, axdim=400, inner=320000` → **2.56 亿工作项 → 假死**（表现为开发板卡死）。
  已修（axis 归一化为非负），并在 `kernel_run` 加 **gws 安全阀**（>3e8 直接报错退出）。
- **带宽测试 OOM**：`kernel_bench --op bandwidth --mb 1024` 分配 2×1 GB buffer + host 1 GB，
  iGPU 共享主存 → 打满 8 GB 机器。已把 `--mb` 上限钳到 256。
- **所有容器/脚本已加 `--memory=3g --memory-swap=3g`**，容器不可能再拖垮宿主。
- **整机硬死机（无日志）与 i915 GPU hang**：本开发板（i5-1135G7 + **PREEMPT_RT 内核**
  `5.15.179-rt84`）历史上有频繁重启（06:20/06:31/06:50/09:40/10:01/10:16…），
  `kern.log`/`syslog` 中可见 `[drm] GPU HANG ... in kernel_run` 与引擎复位记录；
  两次「硬死机」在重启前**没有任何 OOM/i915 日志**，属硬锁死而非 OOM（`dmesg` 不可读，
  但 syslog 无 `oom-kill`）。**排查**：实验内核 `CL_KERNEL_PRIVATE_MEM_SIZE=0`（非寄存器
  溢出/scratch 打爆内存）。
  **防护协议（务必遵守）**：
  1. GPU 实验「短促化」：单个 kernel 目标 < 100 ms，**一条命令只跑一个配置**，
     之间留间隔；不要在一次命令里循环编译/运行大量配置（IGC JIT 反复编译期间风险最高）。
  2. 尽量用 `ocloc compile/disasm` **离线**做 ISA 逆向，不需要 GPU。
  3. 容器 `--memory=2~3g --memory-swap` 限死；`--device=/dev/dri/renderD128` 即可（无需 privileged）。
  4. 用仓库自带、已验证的 `kernel_bench`/`kernel_run` 跑短时基准，少用临时 standalone 驱动。
  5. 机器上还有一个第三方 `deploy-docker`（ROS，restart=always 且无内存上限）在
     段错误-重启循环，与本项目无关，但会持续占用资源。

## 复现

```bash
# 1) 使用含 OpenCL dev 头的镜像（infvino-dev；infvino 不维护 Dockerfile）

# 2) 数值检验（宿主 venv: numpy）
python3 scripts/kernel_check.py --repo $PWD --image infvino-dev:latest

# 3) kernel 基准（容器内 cmake 构建后）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/kernel_bench --op gemm --shape 4096,512,512 --verify
```
