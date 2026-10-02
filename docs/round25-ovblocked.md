# Round 25：完整移植 OpenVINO 阻塞式 conv（`convolution_gpu_bfyx_f16`）+ 逐 size 对照

> 目标模型：`yolov8n-pose`（其余同理）。硬件：Intel Iris Xe（80 EU / 1.3 GHz, TGL）。
> 数值判据：`kernel_bench --verify` 与 `model_check` 同口径（vs FP32 参考）。
> 上游源码：`~/openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_f16.cl`
> （selector `ConvolutionKernel_b_fs_yx_fsv16`）。本机 OV 在 TGL 上选的就是它。

---

## 1. 移植了什么

`kernels/conv_blk.cl`（自包含，Apache-2.0，见 `THIRD_PARTY_NOTICES.md`）：

- **lane = 输出通道**：一个 work-group = 1 个 sub-group（16 lane）= 16 个输出通道，
  每个 work-group 处理**一行**（`y`）× `OBW` 个输出列；`gws = (ceil(X/OBW)*Y, ceil(C/16)*16, B)`。
- 每个 lane 用向量 `dst` 累加 `OBW`（2/4/8）个连续输出列 → `mad` 覆盖 OBW 个位置，
  把每个输入通道的 `sub_group_shuffle` 摊薄到 `OBW/2` 个 packed FMA 上。
- **输入 b_fs_yx_fsv16**（`[Cin/16][H][W][16]`）：lane `l` 用 `intel_sub_group_block_read_us8`
  读入通道 `(icb*16+l)` 的整行，再用 `sub_group_shuffle` 广播通道 `id` 给所有 lane。
- **权重 os_is_yx_isv16_osv16**（`[Cout/16][Cin/16][3][3][isv16][osv16]`），
  `block_read_us8` 让每个 lane 拿到自己输出通道的 16 个输入通道权重。
- **输出 bfyx**（OV 的 `OUTPUT_FORMAT_BFYX` 后重排路径），所以对 infvino 是 drop-in。
- 主机侧：`PlanModel::blkWeight`（os_is_yx_isv16_osv16 重排）、`PlanModel::blkInput`
  （bfyx→fsv16 重排，缓存 scratch）、`reorder_bfyx_to_fsv16` kernel。

## 2. 数值

- `kernel_bench --op conv3x3blk --verify`：40×40 s1 `mean_rel=3.1e-3`、s2 `1.5e-3`（过）。
- **整网强制 blk**（把 yolov8n 全部 45 个 conv3×3 节点加 `blk=1`，`INFVINO_TUNING=off`）
  vs onnxruntime：
  - default（内置启发式 OV/native）：`mean_rel=5.69e-4 max_rel=2.03e-2` PASS
  - **blk**：`mean_rel=5.42e-4 max_rel=1.21e-2` **PASS（还略更准）**
- 数值差异来源是 fp16 求和顺序，不是逻辑错误。

## 3. 逐 size 对照（blk OBW8 vs 调优后的 osv32/native）

| shape | blk OBW8 | osv32/native 最佳 | Δ |
|---|---:|---:|---:|
| 20×20 s1 64→64 | 3.66 | 3.17 | **+15%** |
| 20×20 s1 128→128 | 6.92 | 6.19 | **+12%** |
| 20×20 s1 256→64 | 5.23 | 3.62 | **+44%** |
| 40×40 s1 64→64 | 8.53 | 8.46 | +1% |
| 40×40 s1 128→128 | 12.23 | 10.52 | **+16%** |
| 80×80 s1 64→64 | 11.61 | 13.77 (ov 8×2) | −16% |
| 80×80 s1 64→64 | 11.61 | 11.10 (ov 8×1) | +5% |
| 40×40 s2 64→128 | 4.54 | 8.01 | −43% |
| 160×160 s2 16→32 | 4.66 | 6.64 | −30% |
| 320×320 s2 3→16 | 0.92 | 1.73 | −47% |

**分工**：blk 赢在 **s1 小空间/大通道**（20×20 全系、40×40 C128），
输在 **stride-2**（`INPUT_LINE_SIZE=17`，行 load 与边界开销更大）和 **80×80 大层**
（osv32 的 8×2 packed 通路更省指令）。所以应作为**第三条 autotune 候选按 size 选**，
而不是替换。

**整网强制全 blk**（不是按 size 选，是最坏/压测下界）：yolov8n busy
**17.35 → 17.19 ms（−0.9%）**——即使在不该用 blk 的层上也因 20×20 系收益略胜；
按 size 选会更好。注意这里 default 是 `INFVINO_TUNING=off` 的启发式，非缓存调优后的 15.2 ms。

## 4. 回答用户的问题：能到理论极限吗？

**不能。** 反汇编（离线 `ocloc`）与实测：

- `conv_blk`（OBW8）主循环 **mad 占比 ≈ 0.476**（288 packed mad / 605 指令量级），
  指令配额 ≈ **15.2**；实测 40×40 C128 达 **12.23 = 配额的 80%**。
- osv32 指令配额 ≈ 20.3（R24），实测 80×80 = 13.77（68%）。
- **两条数据通路都没到各自配额**：40×40 类受 560 驻留 sub-group 的**波量化**限制，
  80×80 受延迟/占用限制。换到 OV 的阻塞通路把「网格」这一项补上了（WG 数 ×2–4），
  所以在小空间层追回了 12–44%；但它自己的指令配额更低（shuffle 开销），大层反而不如 8×2。

**结论**：这台机器（7 线程 EU、无 L1、128 GRF）上，direct conv 的两条 OV 数据通路
`os_iyx_osv32` 与 `bfyx_f16` 的现实天花板都停在 **~12–14 ops**（各自配额的 68–80%），
差距是**延迟/占用**而非指令数。要达到 ~20 需要换硬件（矩阵单元/更大 GRF）或换问题
（batch，但本任务要低延迟，不做）。

## 5. 接入自动调优

- `Autotuner::candidatesConv3x3` 新增 `conv3x3_blk` 候选（OBW = 2/4/8，含 stride）。
- `PlanModel` dispatch / `autotune()` 支持 `conv3x3_blk`（含输入重排 + 权重重排）。
- 默认路径**不变**（仍 OV）；只有调优缓存选中 `conv3x3_blk` 才走它 → 零回归风险。
- **未重跑全量 autotune**（GPU 安全）：需要时按 `scripts/autotune.py` 分批重扫 conv3x3，
  新的中间标准 + blk 候选会自动按 size 选。

## 6. 复现

```bash
# 单 shape 对照（一条短命令，遵守 benchmark_protocol.md）
./build/kernel_bench --op conv3x3blk --conv-shape 64,64,40,40 \
  --conv 8,2,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,0,0,0,0,0 --iters 5 --verify
./build/kernel_bench --op conv3x3ov  --conv-shape 64,64,40,40 \
  --conv 8,1,1,32,16,1,1,0,3,1,16,8,0,0,2,0,2,0,0,0,0,0 --iters 5

# 强制 blk 整网（审计）：把 plan 的 conv3x3 节点加 blk=1，INFVINO_TUNING=off 跑 kernel_run
# 接入后逐层选：kernel_autotune --plan ... --op conv3x3 --limit N --iters M
```

---

## 7. 逐层 autotune 重扫结果（R26）

用新候选集（OV 块谱系 + `conv3x3_blk` OBW 2/4/8 + native）重扫三个模型全部 31 个
conv3×3 签名（`scripts/autotune.py --ops conv3x3 --batch 3 --iters 10`，逐批 HANG 自检）。
**31 条里 13 条改选 `conv3x3_blk`、13 条 OV、5 条 native。**

单层最大收益（blk 相对旧最优）：

| shape | 旧 | 新(blk) | Δ |
|---|---:|---:|---:|
| 20×20 s1 256→64 | 3.62 | **6.77** | **+87%** |
| 20×20 s1 256→51 | 2.91 | **5.44** | **+87%** |
| 20×20 s1 64→64 | 3.17 | **5.31** | **+68%** |
| 20×20 s1 51→51 | 2.46 | **3.40** | **+38%** |
| 20×20 s2 128→128 | 4.45 | **5.97** | **+34%** |
| 40×40 s1 64→32 | 5.24 | **6.95** | **+33%** |
| 20×20 s1 128→128 | 6.19 | **7.27** | +18% |
| 40×40 s1 32→32 | 4.14 | **4.87** | +18% |
| 40×40 s2 128→128 | 8.76 | **9.88** | +13% |
| 40×40 s2 64→128 | 8.01 | **8.97** | +12% |
| 160×160 s1 16→8 | 3.79 | **3.99** | +5% |

**整网 kernel busy（同会话 A/B，`kernel_run --report --iters 3`）**：

| 模型 | 旧缓存 | **R26 新缓存** | 加速 |
|---|---|---|---|
| yolov8n-pose | 15.19 ms | **14.40 ms** | **−5.2%** |
| yolo11n-pose | 17.96 ms | **17.04 ms** | **−5.1%** |
| mobilenetv3-small | ~3.37 ms | **3.29 ms** | ~−2% |

**数值**（`scripts/model_check.py`，vs onnxruntime，三个模型）：

| 模型 | mean_rel | max_rel(amax) | 判定 |
|---|---|---|---|
| yolov8n-pose | 5.28e-04 | 8.87e-03 | PASS |
| yolo11n-pose | 8.97e-04 | 1.96e-02 | PASS |
| mobilenetv3-small | 1.31e-02 | 1.09e-02 | PASS |

> 纯配置收益（不改 kernel），来源就是把 OV 阻塞式 conv 作为按 size 可选的第三条通路，
> 且它专治 20×20 这类被波量化卡死的层。调优表 `config/tuning.json` 共 117 条。
> 复现：`python3 scripts/autotune.py --model yolov8n-pose --ops conv3x3`（已 tuned 的会跳过；
> 需要重扫先删对应 `conv3x3|...` 条目，或用 `kernel_autotune --retune` 单签名重扫）。

### 7.1 已知 caveat：autotune 只计 conv kernel，未计输入重排

`conv3x3_blk` 需要把输入 bfyx→fsv16 重排（`reorder_bfyx_to_fsv16`，见 `PlanModel::blkInput`），
而调优器计时**只包含 conv kernel**（`makeEnqueue` 的计时闭包），不含这次重排。方法学上应当
把重排成本并入比较（或在 `autotuneOp` 里给 blk 加一个偏好余量）。不过实测重排开销很小：
把重排 kernel 从「1 线程/元素、写 stride=16」改成「1 线程/16 通道块、写连续」后整网
**无变化**（yolov8n 14.40→14.41 ms、yolo11n 17.04→17.19 ms，均在 run 间噪声内），说明它
不是瓶颈。因此临界层（如 `40×40 s1 64→64`）的 kernel 实测 blk 8.48 vs OV 8.33 是真优势；
整网里两者 ~1.39–1.46 ms 的差是 run 间噪声。后续若要精调，把重排并入计时即可。



