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

| 模型 | mean_rel | max_rel(amax) | max_abs |
|---|---|---|---|
| yolov8n-pose | 5.22e-04 | 1.03e-02 | 7.05e+00 |
| yolo11n-pose | 8.19e-04 | 1.92e-02 | 1.28e+01 |
| mobilenetv3-small | 1.08e-02 | 1.08e-02 | 6.68e-02 |

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

| 模型 | conv+gemm GFLOPs | R6 kernel total (ms) | R6 ops/EU/cyc | **R18 kernel total (ms)** | **R18 ops/EU/cyc** | 加速 | % of 32 |
|---|---|---|---|---|---|---|---|
| yolov8n-pose | 9.178 | 26.38 | 3.34 | **19.55** | **4.51** | **2.85×** | 14.1% |
| yolo11n-pose | 7.406 | 27.91 | 2.55 | **22.33** | **3.19** | **2.51×** | 10.0% |
| mobilenetv3-small | 0.110 | 6.10 | 0.17 | **6.33** | **0.17** | 1.04× | 0.5% |

主要来自：把 K=3、groups=1 的 **stride-2** 层从朴素 `conv_general` 改走调优的
`conv3x3_f16`，并把 s1 tile 改为 `TX40 TY8 TM1 CB32 CINC16`（Round 6/15）；
GEMM staging 改为**行主序 B + half4 向量化**（Round 8）、`SG=16` 钉 SIMD16（Round 12）；
conv3x3 的 **WCOAL 合并权重 staging**（+5–12%，Round 18）与**自适应输出通道块**
（小空间/低通道层 CB=32→16，+20–56%，Round 18）——conv3x3 分项 yolov8 15.0→12.5 ms。
三模型数值检验仍全部 PASS。详见 `docs/kernel.md` Round 18。

### 2.2 整机墙钟（`infvino_bench`，mean，含 launch 开销）

| 模型 | infer (ms) | pipeline (ms) | 说明 |
|---|---|---|---|
| yolov8n-pose | **26.0** | **28.1** | kernel busy ~19.6 ms，其余 ~6.4 ms 为 launch/同步 |
| yolo11n-pose | **29.6** | **31.6** | kernel busy ~22.3 ms |
| mobilenetv3-small | 8.0 | 8.5 | kernel busy ~6.3 ms |

> 旧墙钟（Round 8）：yolov8 ~33.3 / yolo11 ~35.0 / mobilenet ~7.9 ms。
> 非 profiling 模式下墙钟仍含可观 **kernel launch 开销**（yolov8：busy 19.6 vs 墙钟 26.0），
> 这是下一步优化重点（算子融合、减少 kernel 数、批处理/持久化 kernel），见 `docs/kernel.md`。

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
