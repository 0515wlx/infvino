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

## 稳定性事故记录（重要）

- **`softmax` 负 axis 未归一化**：`[1,2,400,400]` 的 `Softmax(axis=-1)` 被算成
  `outer=800, axdim=400, inner=320000` → **2.56 亿工作项 → 假死**（表现为开发板卡死）。
  已修（axis 归一化为非负），并在 `kernel_run` 加 **gws 安全阀**（>3e8 直接报错退出）。
- **带宽测试 OOM**：`kernel_bench --op bandwidth --mb 1024` 分配 2×1 GB buffer + host 1 GB，
  iGPU 共享主存 → 打满 8 GB 机器。已把 `--mb` 上限钳到 256。
- **所有容器/脚本已加 `--memory=3g --memory-swap=3g`**，容器不可能再拖垮宿主。

## 复现

```bash
# 1) 使用含 OpenCL dev 头的镜像（infvino-dev；infvino 不维护 Dockerfile）

# 2) 数值检验（宿主 venv: numpy）
python3 scripts/kernel_check.py --repo $PWD --image infvino-dev:latest

# 3) kernel 基准（容器内 cmake 构建后）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/kernel_bench --op gemm --shape 4096,512,512 --verify
```
