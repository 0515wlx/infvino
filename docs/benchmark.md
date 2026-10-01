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

### 2.1 kernel 自身时间（`model_check.py --report`，GPU busy，单流）

`ops/EU/cyc` 以 conv+gemm FLOPs 计（EU=80，1.3 GHz；硬件上限讨论见 `docs/kernel.md`）。

| 模型 | conv+gemm GFLOPs | kernel total (ms) | 等效 fps | ops/EU/cyc | % of 16 |
|---|---|---|---|---|---|
| yolov8n-pose | 9.178 | 18.61 | 53.7 | 4.74 | 29.6% |
| yolo11n-pose | 7.406 | 20.41 | 49.0 | 3.49 | 21.8% |
| mobilenetv3-small | 0.110 | 2.18 | 459.1 | 0.48 | 3.0% |

### 2.2 整机墙钟（`infvino_bench`，mean，含 launch 开销）

| 模型 | infer (ms) | pipeline (ms) | 说明 |
|---|---|---|---|
| yolov8n-pose | ~62.6 | ~67.1 | 非 profiling；kernel busy 仅 ~18.6 ms，其余为 launch/同步开销 |
| yolo11n-pose | ~69.4 | ~74.1 | |
| mobilenetv3-small | ~8.4 | ~8.9 | |

> 注意：非 profiling 模式下整网墙钟由 **kernel launch 开销**主导（GPU busy 时间远低于墙钟）。
> 这是后续优化重点（算子融合、减少 kernel 数、批处理/持久化 kernel），见 `docs/kernel.md`。

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
