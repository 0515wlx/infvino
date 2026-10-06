# R56：算子集收尾（契约/分配/N=1） + 小算子 launch 融合尝试

> 依据用户决策：**先把「算子集的注册表化 / 出口 / 契约 / 分配 / 其他缺口」补齐**，再做
> 布局 min-cut 的「回退局部化 + 目标函数」两步。本文件记录算子层的收尾结果。
>
> 结论先给：**算子层其实已基本补齐**（R51 已做 cat4 注册表 + 非 16 通道分配）；本轮只找到
> 一个 latent 契约 bug，补了 N=1 的「多输出」候选（有隔离收益），并验证 depthwise 向量 store
> 与 SPPF launch 融合为**中性/负**（opt-in 保留）。`config/tuning.json` 未改动。

---

## 0. 结论先给

| 项 | 结果 |
|---|---|
| **契约审计** | 发现并修 `gemm_sk_f16.layout.inIndex` 缺声明（默认 0，但它服务 conv1x1 → 激活在槽 1）。行为中性（该族 NCHW-only）。 |
| **契约不变量测试** | `test_kernel_families` 新增通用不变量：凡服务「槽 0=权重」的 op（gemm/conv1x1/conv1x1_cat4）激活必须在槽 1，其余 0（499 passed）。 |
| **N=1 多输出**（`conv1x1_gemv` 的 `GEMV_TM`） | 隔离 **1.3–1.5×**（TM=2，大 shape）；逐位一致；候选 1→3/签名，autotune 自动按 shape 选。**整网噪声内**（N=1 仅占 mb ~5%），故 `config/tuning.json` 未改。 |
| **depthwise 向量 store**（`-DVEC_STORE`） | 子组 `block_write` 写 fsv16；逐位一致；mb 小 depthwise（贴 launch 地板）**中性/略负** → 默认关。 |
| **小算子 launch 融合**（SPPF：3×MaxPool5 + concat → 1 kernel） | **负结果**（y8 busy **+1.9%**）：3.7× 冗余 fmax + 13×13 窗口重读 > 省下的 3 次 launch，且挤掉原本 CAT4 的免费 concat 融合 → 默认关。 |
| **数值** | y8 `4.114e-04` / y11 `7.908e-04` / mb `1.215e-02`，均与 R51/R54 基线**逐位相同**；`check` 6/6 PASS。 |
| `config/tuning.json` | **未改动**（都是候选/opt-in 级改动）。 |

> **审计更正**：`direct conv1x1` 并非「缺失」——`kernels/conv1x1.cl::conv1x1_f16`
> 早已存在，R22 实测 **N>1 上全面慢于 `gemm_f16`**（负结果），故未注册为候选；
> `gemm` 的 SLM/双缓冲 K 候选（`DBUF=1`）也早已在 `gemmSpectrum` 里。即「新 kernel 候选」
> 清单里多数已被证否或已具备，真正的缺口只有 N=1 的「多输出」。

---

## 1. 契约审计（零 kernel）

### 1.1 现状

对所有 plan 跑 `kernel_autotune --candidates`（零 GPU 计时）：**每个 op 都有族**，无空候选；
`conv1x1_cat4` = `{gemm_cat4_f16=7}`（R51 已收口）；非 16 通道 fsv16 分配 R51 已完成。布局契约
被消费的字段只有 `layout.in` / `layout.inIndex` / `layout.canOutFsv16`（`needsWeightRepack`、
`layout.out` 当前未被读取）。

### 1.2 发现的问题

`gemm_sk_f16`（服务 `conv1x1` N>1）的 `layout.inIndex` 未声明 → 落到默认 **0**，但 conv1x1 的
激活在**槽 1**（槽 0=权重），与同族 `gemm_f16`/`conv1x1_blk`/`gemm_cat4_f16` 不一致。

影响面：`opCanReadFsv16` / `planBlockedLayout`（`slot = cf->layout.inIndex`）会读它；但
`gemm_sk_f16` 是 NCHW-only（`layout.in != FSV16`），两条判据本就返回「不能读」，故**行为中性**。
修复是「契约正确性」层（防止未来 blocked 化时静默漂移）。

### 1.3 修复 + 不变量

- `src/KernelFamilies.cpp`：`gemm_sk_f16.layout.inIndex = 1`。
- `tests/test_kernel_families.cpp`：新增通用不变量（遍历所有族）：
  `op ∈ {gemm, conv1x1, conv1x1_cat4} → inIndex==1`，其余 `==0`。

---

## 2. N=1 多输出（R48 §4bis 的真实缺口）

### 2.1 缺口

`conv1x1_gemv_f16` 此前**每签名只有 1 个候选**（R48："N=1 无调优空间 / 无多输出"）。原实现一个
子组只算一个输出通道 m，X 向量被 Cout 个子组各读一遍（X 读流量 = Cout·Cin）。

### 2.2 实现

- `kernels/conv1x1.cl`：新增 `-DGEMV_TM=T`（每子组算 T 个输出通道；X[k] 只读一次供 T 个权重行
  复用，X 读流量 /T；T 条独立归约链交错提 ILP）。`T=1` 与旧行为**逐位一致**（累加顺序、
  `sub_group_reduce_add` 不变）。网格 `dim0 = ceil(Cout/T)·16`。
- `src/KernelFamilies.cpp`：候选 `TM ∈ {1,2,4}`（`tm>Cout` 跳过）。
- `src/PlanModel.cpp`：dispatch 与 autotune 的网格从 `-DGEMV_TM=` 解析（旧缓存无此宏 → 默认 1，
  向后兼容）。
- 工具：`kernel_numtest --op conv1x1g --opts`、`kernel_bench --conv1x1 ...,GEMV_TM`。

### 2.3 验证

逐位（`kernel_numtest`，TM∈{1,2,4}，含非 16 对齐 Cout）：

| shape | TM=1/2/4 vs fp32 参考 |
|---|---|
| 1000×1024, 576×576, 120×120, 51×64 | **mean_abs = 0（逐位）** |

隔离性能（`kernel_bench --op conv1x1g`，锁频非全程）：

| shape (Cout×Cin) | TM=1 | **TM=2** | TM=4 |
|---|---:|---:|---:|
| 576×576 | 0.029 ms | **0.021** | 0.034 |
| 1024×576 | 0.047 | **0.032** | 0.056 |
| 1000×1024 | 0.060 | **0.047** | 0.055 |
| 96×576 | 0.009 | 0.009 | 0.013 |

→ **TM=2 稳定 1.3–1.5×**；TM=4 网格过粗/寄存器压力反而差。autotune 按 shape 正确选择
（大 shape 选 2、小 shape 选 1）。

整网（mb，`/tmp` 临时缓存，只改 N=1 条目为 `-DGEMV_TM=2`）：交错 A/B **±1% 噪声内**——
N=1 只占 mb ~5%，理论 ~1.4% 被噪声吞掉；4/5 rep 略负、1/5 略正。**判定：隔离有收益、整网中性**，
故不写 `config/tuning.json`（候选可用，`--retune` 后生效）。`model_check`（含 TM 缓存）PASS 9.11e-3。

---

## 3. depthwise 向量 store（中性/负，opt-in）

R48/D6 点名 depthwise「缺向量 store」。移植 OV
`convolution_gpu_bfyx_f16_depthwise.cl` 的 `DT_OUTPUT_BLOCK_WRITE8`：
- `kernels/depthwise_blk.cl`：OUT_FSV16 时，若「整 16 通道块 + 整 X 列块」则用
  `intel_sub_group_block_write_usN`（lane l、分量 i → `base + i*SG + l`，正是 fsv16 布局）；
  否则退回标量。`-DVEC_STORE`（默认 0）。
- 逐位：4 组 shape 含 C=120/96/32、K=3/5、含非 16 对齐 → **逐位一致**。

隔离（`kernel_bench --op depthwiseblk ... fsv16`）：120²(17×17) 0.015→0.014；240²(14×14)
0.016→**0.019**；144² 0.015→**0.017**；576²(7×7) 持平。**中性到略负** → **默认关**。

> 注：`depthwise_blk` 已具备 R50 的 `Y_BLOCK` + fsv16 出口 + autotune 候选；本项是最后
> 一个点名的轴，验证为负。

---

## 4. 小算子 launch 融合（SPPF，**负结果**，opt-in）

### 4.1 动机与发现

小算子占 y8 busy ~7%（ew_binary 0.214 / concat4 0.163 / maxpool 0.154 / resize 0.083 / …）。
审计后发现**易做的融合早已在**：所有 C2f/ELAN/SPPF 的 `concat4` 都在运行时被
`fuseConcatConv1x1` 折进下游 `conv1x1_cat4`（CAT4 B-staging，零 concat 物化）；`copy_c`
（Split）由 R-P0b 别名；`reshape` 别名。因此真正"孤立"的只有 decode 头（模型相关后处理）
与 SPPF 的 3 个 maxpool。

### 4.2 实现（SPPF：3×MaxPool5s1p2 + concat → 1 kernel）

- `kernels/ops.cl::sppf_concat4`：`y = concat(x, mp(x), mp²(x), mp³(x))`，链式 S=1 maxpool
  等价于对 x 的单次 `[-(src·P), src·P]` 窗口 max（max 可结合、与顺序无关）→ **逐位一致**。
- `src/PlanModel.cpp::fuseSppfConcat()`（`INFVINO_FUSE_SPPF=1`，默认关）：识别模式、删除 3 个
  maxpool、把 concat 改写为 `sppf_concat4`。必须在 `fuseConcatConv1x1` **之前**运行。

### 4.3 结果（y8，`kernel_run --report --iters 15`）

| 方案 | maxpool | sppf | total kernel |
|---|---:|---:|---:|
| 默认（CAT4 融合） | 0.154 ms ×3 | — | **10.526 ms** |
| `INFVINO_FUSE_SPPF=1` | — | 0.382 ms ×1 | 10.727 ms（**+1.9%**） |

**负结果根因**：融合 kernel 对 src=3 要算 13×13=169 个 fmax/输出（总 275 vs 原 75，3.7×），
且窗口重读多；同时把 SPPF concat 从「CAT4 的免费融合」拉回「物化」。**判决：默认关，保留机制**。

---

## 5. 复现

```bash
# 契约审计（零 GPU 计时，需 GPU 枚举设备）
LD_LIBRARY_PATH=$PWD/build ./build/kernel_autotune --plan models/mobilenetv3-small/model.plan --candidates | grep conv1x1

# N=1 多输出：候选与隔离
./build/kernel_bench --op conv1x1g --conv-shape 1000,1024,1,1 --conv1x1 "4,4,0,0,16,4,2" --iters 50 --verify
./build/kernel_numtest --op conv1x1g --cin 1024 --cout 1000 --input-w W.bin --input-x X.bin \
  --dump out.bin --opts "-DGEMV_TM=2 -DACT=0 -DRES=0 -DSG=16 -cl-mad-enable -cl-fast-relaxed-math"

# depthwise 向量 store
./build/kernel_bench --op depthwiseblk --conv-shape 240,240,14,14 --depthwiseblk "8,3,1,1,1,1,1,1" --iters 60 --verify

# SPPF launch 融合（默认关；A/B）
INFVINO_FUSE_SPPF=1 ./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 15

# 回归
cmake --build build -j --target check
python3 scripts/model_check.py --model yolov8n-pose --repo "$PWD"
```

---

## 6. 下一步（用户指定顺序的剩余）

算子层收尾完成，进入布局求解器的两步：
1. **min-cut 回退局部化**：`resolveLayoutMinCut` 的「非 submodular / `mayMarkFsv16` 失败 → 整份
   `restore(base)`」改为**局部化**（未覆盖/不可行节点 pin 默认，其余照解；非 submodular 表退化/
   QPBO，不 return false）。
2. **目标函数改为 reorder 级**：提案生成与接受分离；一级目标用「reorder 条数 + 元素量」
   （学 OV `reorder_inputs`），隔离 ms 只排序，L3 只做整网预筛。
