# 数值检验与性能基准

> 硬件：Intel Core i5-1135G7（Iris Xe Graphics，80 EU，1.3 GHz），
> 容器 `infvino-dev`，驱动栈见 `docs/dependencies.md`。
> 判据统一用**相对误差**（不使用余弦相似度）：`mean_rel = mean|got-ref| / mean|ref|`，
> `max_rel(amax) = max|got-ref| / max|ref|`。

## 1. 数值检验

### 1.1 算子级（`scripts/kernel_check.py`，vs numpy FP32）

阈值：`mean_rel < 1e-2` 且 `max_rel < 5e-2`。全部 PASS。

| 算子 | 代表 shape | mean_rel | max_rel |
|---|---|---|---|
| gemm | 6400×64×64 | 7.31e-04 | 3.20e-03 |
| gemm | 4096×512×512 | 2.61e-03 | 1.22e-02 |
| conv3x3 s1 | 64→64, 80×80 | 3.08e-03 | 8.19e-03 |
| conv3x3 s2 | 16→32, 160×160 | 1.54e-03 | 4.21e-03 |
| conv1x1 | 384→128, 40×40 | 2.53e-03 | 7.46e-03 |

### 1.2 整网（`scripts/model_check.py`，vs onnxruntime FP32；输入 fp16）

阈值：`mean_rel < 2e-2` 且 `max_rel < 5e-2`。全部 PASS。

| 模型 | mean_rel | max_rel(amax) | max_abs | scale_rel(mean\|d\|/max\|ref\|) |
|---|---|---|---|---|
| yolov8n-pose | 5.22e-04 | 1.03e-02 | 7.05e+00 | ~3e-05 |
| yolo11n-pose | 8.19e-04 | 1.92e-02 | 1.28e+01 | ~6e-05 |
| mobilenetv3-small | 1.08e-02 | 1.08e-02 | 6.68e-02 | 1.1e-03 |

> **关于 mobilenet 的 1.1–1.3e-2（不是口径问题，是真误差，但可解释）**：
> `mean_rel = mean|diff| / mean|ref|` 对**小量级输出**会放大，但诚实地说，
> mobilenet 的**绝对误差本身也偏大**：mean|diff| ≈ 1.6e-2，对 logits（满量程 ~6.2）
> 是 relative-to-range ≈ 2.6e-3，比 yolo 的 ~1.6e-4 **大 ~16×**。
>
> **根因（逐层定位，2026-10）**：mobilenet 尾部是
> `GAP → classifier.1 (1024, K=576) → HardSwish → classifier.3 (1000, K=1024)`。
> 实测误差沿尾部逐级放大：GAP 后 mean_abs≈1.0e-3 → classifier.1 后≈1.6e-3 →
> logits≈1.6e-2。**K=1024 的 fp16 GEMM 头是主因**：即使 fp32 累加，输入激活/权重
> 都是 fp16（~1e-3 相对），1024 项随机游走 ⇒ 输出相对误差 ~√K·1e-3 ≈ 3e-2 量级。
>
> **影响**：分类结果正确（argmax 一致、top-5 仅第 4/5 名互换）。**不是 bug。**
> **P1 候选**：最后一层分类 GEMM 用 fp32 权重/更精确累加，可把 logits 误差压回 ~1e-3。
>
> **判据改进**：`model_check.py` 已输出 `scale_rel` / `quant_rel` / `ulp_frac`
> （尺度无关），比单一 `mean_rel` 更能反映真实质量；后续应把这些纳入 PASS 判据。

### 1.3 库后端（`scripts/numerical_check.py`，`ClBackend` vs onnxruntime FP32）

| 模型 | mean_rel | max_rel(amax) | max_abs |
|---|---|---|---|
| yolov8n-pose | 5.33e-04 | 1.62e-02 | 1.11e+01 |
| yolo11n-pose | 7.86e-04 | 3.08e-02 | 2.04e+01 |
| mobilenetv3-small | 9.37e-03 | 7.70e-03 | 4.78e-02 |

### 1.4 引擎级（`scripts/engine_check.py`，预处理 + `ClBackend` vs onnxruntime，喂图片）

| 模型 | mean_rel | max_rel(amax) | max_abs |
|---|---|---|---|
| yolov8n-pose | 5.18e-04 | 1.96e-02 | 1.43e+01 |
| yolo11n-pose | 8.86e-04 | 4.20e-02 | 2.91e+01 |
| mobilenetv3-small | 1.39e-02 | 1.14e-02 | 6.14e-02 |

## 2. 性能

### 2.1 kernel 自身时间（`kernel_run --report`，GPU busy，单流，warm）

`ops/EU/cyc` 以 conv+gemm FLOPs 计（EU=80，1.3 GHz；硬件 FP16 上限 = **32**，见 `docs/kernel.md`）。

> ⚠️ **统计口径修正（Round 6）**：旧 `kernel_run` 的 “total kernel time” 被 `iters` 整除，
> 但 `PlanModel::run()` 每轮清空了耗时表 → 实际是把**单轮时间**又除以 `iters`，数值随
> `iters` 线性变小。已修（`run()` 不再清空，改用 `clearProfile()` + warmup）。
> 下表旧列为**按 ×iters 还原后的真实值**，与 `docs/kernel.md` Round 5 一致。

| 模型 | conv+gemm GFLOPs | R6 kernel total (ms) | R6 ops/EU/cyc | **R23 kernel total (ms)** | **R23 ops/EU/cyc** | 加速 | % of 32 |
|---|---|---|---|---|---|---|---|
| yolov8n-pose | 9.178 | 26.38 | 3.34 | **17.2** | **5.13** | **1.53×** | 16.0% |
| yolo11n-pose | 7.406 | 27.91 | 2.55 | **20.1** | **3.55** | **1.39×** | 11.1% |
| mobilenetv3-small | 0.110 | 6.10 | 0.17 | **3.5** | **0.31** | **1.74×** | 1.0% |

> **R22**：conv3×3 走 OpenVINO `os_iyx_osv32` 移植（大层 +20–80%，`kernels/conv_ov.cl`）、
> 1×1 conv 融合 bias+act 且 N=1 走 split-K GEMV、GAP 改并行树归约。
> **R23**：`concat4` 改 3-D 网格（纯索引简化，`bef119f`）+ 主机侧 `.cl`/kernel 句柄缓存
> （`b28a6a4`）——yolov8 19.9→**17.2**、yolo11 23.0→**20.1** ms，mobilenet 持平。
> 数值三级检验全部 PASS。详见 `docs/kernel.md` Round 22 / 23。

主要来自：把 K=3、groups=1 的 **stride-2** 层从朴素 `conv_general` 改走调优的
`conv3x3_f16`，并把 s1 tile 改为 `TX40 TY8 TM1 CB32 CINC16`（Round 6/15）；
GEMM staging 改为**行主序 B + half4 向量化**（Round 8）、`SG=16` 钉 SIMD16（Round 12）；
conv3x3 的 **WCOAL 合并权重 staging**（+5–12%，Round 18）与**自适应输出通道块**
（小空间/低通道层 CB=32→16，+20–56%，Round 18）——conv3x3 分项 yolov8 15.0→12.5 ms。
三模型数值检验仍全部 PASS。详见 `docs/kernel.md` Round 18。

#### 与 OpenVINO 2025.2 的整网对照（同一 iGPU）

OV per-node GPU 时间（`enable_profiling`）与端到端（含预处理）：

| 模型 | OV GPU 合计 | OV infer | OV e2e | infvino busy (R23) | 差距 |
|---|---|---|---|---|---|
| yolov8n-pose | 9.33 ms | 11.2 ms | 13.9 ms | 17.2 ms | ~1.5× |
| yolo11n-pose | 9.61 ms | 11.8 ms | 14.5 ms | 20.1 ms | ~1.7× |
| mobilenetv3-small | 0.96 ms | 1.8 ms | 2.2 ms | 3.5 ms | ~1.9× |

> OV 的整网差距主要在 **conv3×3**（yolov8 里 `40x40 Cin64` 单层 ~85% 时间）。
> OV 实际选用的是**阻塞式** `convolution_gpu_bfyx_f16`（非 osv32）。infvino 的简化移植
> 未能打赢已调优的 osv32（见 `docs/kernel.md` Round 23），因此 OV 仍作为对照基线。

### 2.2 整机墙钟（`infvino_bench`，mean，含 launch 开销）

| 模型 | infer (ms) | pipeline (ms) | 说明 |
|---|---|---|---|
| yolov8n-pose | **22.3** | **24.1** | kernel busy ~17.2 ms，其余 ~5.1 ms 为 launch/同步 |
| yolo11n-pose | **25.6** | **27.4** | kernel busy ~20.1 ms |
| mobilenetv3-small | 4.6 | 5.0 | kernel busy ~3.5 ms |

> R23 墙钟（net only，含 launch 开销）。非 profiling 模式下墙钟仍含可观 **kernel launch
> 开销**（yolov8：busy 17.2 vs 墙钟 22.3），这是下一步优化重点（算子融合、减少 kernel 数、
> 批处理/持久化 kernel），见 `docs/kernel.md`。

### 2.3 纯 GEMM 算子（`kernel_bench`，`f16`）

| shape (M×N×K) | 旧 ops/EU/cyc | 新 BK=8 | 新 BK=32 |
|---|---|---|---|
| 4096×512×512 | 8.37 | 10.02 | **11.09** |
| 1024×1024×1024 | 8.05 | 9.79 | **10.91** |
| 6400×64×64 | 6.04 | **6.93** | 5.25 |
| 128×1600×384 | 5.54 | **6.65** | 5.23 |
| 400×256×256 | ~4.0 | 4.34 | 3.84 |

行主序 B（消除转置 staging 的 bank conflict）+ half4 向量化 staging + 按 K 选 BK
（`K>=256` 用 32，否则 8）。详见 `docs/kernel.md` Round 8。

### 2.4 `wall − busy` 的构成（P2 host 分段，`kernel_run --report`）

P0 发现「内存池对墙钟的收益 > GPU busy」，于是 P2 先量化墙钟里 busy 之外的开销。
`PlanModel::run()` 在 profiling 下对每个节点分别累计 **host 入队提交**
（`clEnqueueNDRangeKernel` 的调用耗时）与 **同步等待**（`clWaitForEvents`），`kernel_run`
打印如下分段（同会话、`--iters 30`、单流、warm）：

| 模型 | wall | busy (%wall) | enqueue | sync | host_total≈wall−sync | setarg_est≈host_total−enqueue |
|---|---:|---:|---:|---:|---:|---:|
| yolov8n-pose | 18.202 | 13.469 (74%) | 1.738 | 14.462 | 3.741 | 2.002 |
| yolo11n-pose | 20.695 | 14.514 (70%) | 2.365 | 16.039 | 4.657 | 2.292 |
| mobilenet | 4.912 | 2.541 (52%) | 1.139 | 3.471 | 1.441 | 0.302 |

口径：in-order 单队列 + profiling，逐节点 `clWaitForEvents`，故
`wall = enqueue_host + sync + setarg/其它`；`sync` 含 GPU 执行，故
`host_total = wall − sync`，`setarg_est` 是扣掉入队后的**其余 host**（主要是
`clSetKernelArg` + 循环/视图簿记）。

**结论（供 P2 决策）**：

- 墙钟里 busy 只占 **52%（mobilenet）–74%（yolov8）**；mobilenet 的 wall 几乎是 busy 的 2×。
- host 开销由**两块相当**构成：`clEnqueueNDRangeKernel` **入队提交**（1.1–2.4 ms，
  即 dispatch/launch floor）与**其余 host（含 `setArg`）**（0.30–2.29 ms）。
- 因此纯 per-dispatch 优化（如参数缓存）能砍掉 y8/y11 约 2 ms 的一部分，但要实质收窄
  gap 还需**减少 dispatch 数**（融合/持久化 kernel）——与 R-P0「常驻 `cl_mem` 数影响
  墙钟」的观察一致。P3 的在线调优与本分段共用同一套统计。
- **P2 下一步候选**（按风险，尚未做）：① 去掉 profiling 下逐节点 `clWaitForEvents`
  （部署 `profiling_=false` 本就不等，但可查 host 侧每节点开销）；② **kernel 参数缓存**
  （记录 `(kernel, 参数签名, 内存指针)`，内存/配置不变则跳过 `setArg`，学 OV
  `MEMORY_CHANGED`）；③ out-of-order 队列 + event 依赖（需先确认 Intel 驱动行为）；
  ④ `createSession()` 多 Session 并行缓冲（后端当前用 mutex 串行）。

## 3. 复现

```bash
# 生成模型与计划
python3 scripts/export_models.py --out models
python3 scripts/onnx2plan.py --onnx models/yolov8n-pose.onnx --out-dir models/yolov8n-pose

# 算子级 / 整网 / 库后端 数值检验
python3 scripts/kernel_check.py    --repo $PWD --image infvino-dev:latest
python3 scripts/model_check.py     --model yolov8n-pose --repo $PWD --image infvino-dev:latest
python3 scripts/numerical_check.py --repo $PWD --image infvino-dev:latest
python3 scripts/engine_check.py    --repo $PWD --image infvino-dev:latest

# 墙钟
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/infvino_bench --config config/models.yaml --key yolov8n-pose --iters 50
```

## 4. 历史基线（OpenVINO，已移除）

早期版本用 OpenVINO 2023.0.2 作为数值/性能对照基线（yolov8n-pose 整网 GPU 约 7.66 ops/EU/cyc）。
infvino 已移除 OpenVINO 依赖，本文件不再维护该基线，仅作历史参考；kernel 优化过程见 `docs/kernel.md`。
