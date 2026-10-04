# infvino 架构设计

> 独立推理后端：`cv::Mat → 预处理 → 自研 OpenCL kernel 整网执行 → 解码 → InferResult`。
> 不依赖 ROS / OpenVINO。

## 1. 定位与依赖方向

```
使用方（任意 C++ 程序）  ──▶  infvino  (纯库: OpenCV + OpenCL + yaml-cpp)
```

- 引擎**不依赖 ROS**：调用方只做「图像 → `InferResult`」的适配。
- 计算全部由自研 OpenCL kernel 完成；ONNX 只在离线阶段用于生成执行计划。

## 2. 分层结构

```
include/infvino/
├── Types.hpp             # Detection / InferResult / Task（后端无关）
├── Tensor.hpp            # 后端无关张量：shape + f32 主机数据
├── ModelInfo.hpp         # 模型元数据 + yaml 解析（含 plan 路径）
├── Device.hpp            # OpenCL 设备信息
├── Preprocess.hpp        # letterbox / normalize / NCHW blob
├── Nms.hpp               # IoU + 单类 NMS
├── Decoder.hpp           # 解码抽象 + 工厂
├── Decoders.hpp          # Detect / Pose / End2End / Classify 实现
├── ClRuntime.hpp         # OpenCL 上下文/队列/kernel 构建缓存/计时
├── Tiles.hpp / Half.hpp  # tile 配置 / fp16<->fp32
├── PlanModel.hpp         # 计划驱动的整网执行器（自研 kernel）
├── ClBackend.hpp         # 后端：PlanModel + 输入输出类型转换
└── InferenceEngine.hpp   # 门面 + Session（每线程一个）
```

数据流：

```
cv::Mat(BGR) --Preprocessor--> cv::Mat blob(f32 NCHW)
             --ClBackend--> (f32→f16) --PlanModel.run()--> (f16→f32) std::vector<Tensor>
             --Decoder--> InferResult（坐标已反映射回原图）
```

## 3. 执行计划（plan）

ONNX 在离线阶段由 `scripts/onnx2plan.py` 转成**执行计划**：纯文本 + 每个 initializer 的 fp16 `.bin`。
计划与权重放在同一目录，`init` 路径用 basename（相对 plan 目录），因此整目录可搬迁。

```
input  <name> <dims...>              # 运行时输入（fp16, NCHW 行主序）
init   <name> <file.bin> <dims...>   # 权重（fp16）
tensor <name> <dims...>              # 中间激活
node   <op> <in_csv|-> <out_csv> [k=v ...]
output <name>                        # 可多行
```

- 所有张量为 **fp16**、行主序；`reshape`/`flatten` 为视图（零拷贝别名）。
- 已实现算子：`conv3x3`、`conv_general`（任意 K/stride/groups）、`gemm`、`bias_add`、
  `ew_binary`(广播)、`ew_unary`、`copy_c`、`slice_axis`、`concat4`、`maxpool`、`resize_nn`、
  `softmax_axis`、`permute_0213`、`bmm`、`gap`。
- 融合：`Conv + Sigmoid + Mul`（SiLU）在转换时合并为一次 conv（`act=1`）；1×1 conv 走 GEMM。

`PlanModel`（`src/PlanModel.cpp`）负责解析计划、分配设备缓冲、构建 kernel、按序执行、读回输出。

## 4. 后端与张量抽象

- `ClBackend`：`infer(const float* nchw, size_t numel) -> std::vector<Tensor>`。
  - 输入 f32 → fp16、`PlanModel::setInput`、`run`、逐输出 fp16→f32。
  - 权重常驻设备；设备缓冲由 `std::mutex` 串行保护，多个 Session 可安全复用同一后端。
- `Tensor`：只含 `shape` + `std::vector<float>`，解码层与后端解耦（便于将来接入其他后端）。

## 5. 设备（OpenCL）

`infvino::ClRuntime` 枚举平台/设备，默认**优先含 GPU 的平台**，无 GPU 时退回任一可用设备；
`DeviceInfo` 记录 `available / requested / resolved / fell_back_to_cpu`。

> Xe-LP(Gen12) 的实际 cache 层级与 EU 寄存器（128 GRF/线程、ARF 专用寄存器、无通用 L1、
> 3.75 MB L3、SLM 64 KB/WG）见 [`xe-lp-isa.md`](xe-lp-isa.md)——kernel 分块与寄存器
> 预算的硬件事实基准。

> 与旧版不同：不再有「CPU/GPU/NPU」多后端选择，`device` 仅作日志；`AUTO` 即自动优选 GPU。

## 6. 关键设计决策

### 6.1 每线程/每相机一个 Session

`InferenceEngine` 持有共享的 `ClBackend`（权重常驻）+ `Preprocessor` + `Decoder`；
`createSession()` 返回轻量 Session。当前后端内部串行化设备访问，Session 主要用于 API 语义与
后续「多套激活缓冲并行」的扩展点。

### 6.2 Decoder 策略模式

不同任务/不同 ultralytics 版本的输出布局不同，全部收进各自 Decoder：

| 任务 | 输出 | 说明 |
|---|---|---|
| detect (v8/v11/v26) | `[1, 4+nc, N]` | 兼容转置 `[1, N, 4+nc]` |
| pose | `[1, 4+1+3K, N]` | `K=kpt_shape[0]` |
| end2end | `[1, N, 6]` | 已含 NMS |
| classify | `[1, nc]` | softmax + top-k |
| seg / obb | — | 未实现，`Decoder::create` 返回明确错误 |

### 6.3 ModelInfo：必须显式 yaml

不同模型的预处理/解码参数（imgsz、kpt_shape、mean/std、conf/iou）在 `config/models.yaml` 显式声明；
`plan` 指向离线生成的执行计划。

### 6.4 预处理一致性

`Preprocessor` 复刻 ultralytics 默认：letterbox（`scale=min`，pad 114）+ BGR→RGB + `/255` + NCHW；
分类模型可用 `letterbox:false` + `mean`/`std`（ImageNet）。`LetterboxInfo` 保存几何参数用于反映射。

### 6.5 跨推理缓存安全（必须遵守）

`PlanModel` 为性能缓存设备侧结构，分为两类：

| 类别 | 例子 | 跨 run 是否安全 |
|---|---|---|
| **常量**（权重/几何）| `ovWeight` / `blkWeight`（重排权重）、`bcastDims`、`small_buf_` | 安全：内容不随帧变 |
| **激活态** | `blkInput`（conv_blk 输入重排）、`dwPadInput`（depthwise 零边输入）| **每 run 必须重算**；只允许缓存 `cl_mem`，不能缓存结果 |

**历史教训**：`blkInput` 按张量名缓存了重排结果且只算一次 → 第 2 帧起用上一帧输入
（y8 第 2 帧 `mean_rel≈1.4e-2`，正常 ~5e-4）。因为所有测试都**同一输入重复跑**，
陈旧值恰好等于正确值，长期未被发现。
**规则**：任何「按名字缓存、跨 run 复用」的激活态，都必须有**多输入**数值测试覆盖
（`scripts/reuse_check.py`；`model_check`/`numerical_check`/`engine_check` 均跑两份不同输入、
取第二帧）。详见 `docs/kernel.md`「稳定性事故记录」。

## 7. 扩展指引

| 需求 | 做法 |
|---|---|
| 新增模型 | 导出 ONNX → `onnx2plan.py` 生成 plan → `config/models.yaml` 登记 |
| 新增算子 | 在 `kernels/` 加 kernel，`PlanModel::dispatch` 加分支，`onnx2plan.py` 加映射 |
| 新增任务 (seg/obb) | 新增 `XxxDecoder`，在 `Decoder::create` 注册 |
| 新增后端 | 仿 `ClBackend` 实现 `infer -> std::vector<Tensor>` 即可，解码层不变 |
| 性能 | 算子融合、内存复用、减少 kernel launch（见 `docs/kernel.md`） |

## 8. 数值检验

判据统一用**相对误差**（不使用余弦相似度，余弦尺度不变会掩盖幅度误差）：
`mean_rel = mean|got-ref| / mean|ref|`，`max_rel = max|got-ref| / max|ref|`。

- 算子级：`scripts/kernel_check.py`（vs numpy FP32）。
- 整网：`scripts/model_check.py`（vs onnxruntime，FP16 阈值 `mean<2e-2, max<5e-2`）。
- 库后端：`scripts/numerical_check.py`（`ClBackend` vs onnxruntime）。
- 引擎级：`scripts/engine_check.py`（**预处理 + ClBackend**，喂图片 vs onnxruntime）。

三个目标模型全部 PASS，见 `docs/benchmark.md`。

## 9. 已知限制

- 仅 fp16 计划 / f32 输出；解码只支持 f32。
- seg / obb 未实现。
- 整网墙钟受 host 开销主导：GPU busy 只占墙钟 52%（mobilenet）–74%（yolov8），其余为
  **入队提交 + `setArg`/同步**。P2 host 分段实测见 `docs/benchmark.md` §2.4。
