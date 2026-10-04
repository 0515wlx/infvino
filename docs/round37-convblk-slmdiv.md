# Round 37：conv3×3 单 kernel 效率（vs OV 的 1.5×）——OOB/特化负结果 + SLM_DIV 正结果

> 背景：本轮的端到端对照（同机、OV 2026.4.1）显示 **infvino 的差距 80–110% 在
> 卷积单 kernel 效率**，不是引擎结构（host/调度/reorder 合计 <10%）。
> `docs/ooo-queue.md` 已证多队列/OoO 在本机不产出重叠收益，故本轮正面攻 conv。
>
> 结论先给：**OV 用的是我们已经移植的同一个 blocked kernel
> （`convolution_gpu_bfyx_f16`，见 exec graph），但在同一 shape 上比我们的移植快
> ~1.5×；根因不是数据通路，而是移植时省掉的 `SLM_DIV_FACTOR`。补上后 40×40 类
> shape +10–26%。**

## 1. 定位：OV 到底用什么 kernel

用 `benchmark_app -exec_graph_path` 导出 OV 的执行图（y8 73 个 conv）：

| primitiveType | 数量 |
|---|---:|
| `convolution_gpu_bfyx_f16`（blocked 3×3，lane=输出通道） | 56 |
| `convolution_gpu_bfyx_f16_1x1` | 12 |
| `convolution_gpu_bfyx_os_iyx_osv32` | 4 |
| `convolution_gpu_bfyx_to_bfyx_f16`（stem Cin=3） | 1 |

**没有 Winograd。** OV 主力就是 R25 移植的 `conv_blk`。但逐 shape 对比（同机）：

| shape | infvino blk | OV | 比 |
|---|---:|---:|---:|
| 40×40 Cin64 Cout64 | 0.126–0.135 ms (9.0) | ~0.081 ms (14.0) | 1.6× |
| 40×40 Cin128 Cout64 | 0.211 ms (10.8) | — | |
| 20×20 Cin128 Cout128 | 0.150 ms (7.6) | ~0.060 ms (13+) | ~2× |
| 80×80 Cin64 Cout64 | 0.352 ms (12.9) | ~0.29 ms | 1.2× |

即：**同一数据通路，OV 的调度更满。** 40×40/20×20 的 `ops/EU/cyc` 只有 8–10，
低于 80×80 的 13——典型的占用/波量化受限（R24 已定位）。

## 2. 尝试 A：OOB-guard 行载入（OV `ENABLE_OOB_GUARD`）——**负结果，已回退**

把「边界 x-block 退化成逐标量谓词载入」改成 OV 式「块读可达到区间 + 边界置零」。

- 40×40 C64 OBW8：**0.132 → 0.172 ms（−23%）**。
- 原因：OV 版循环上界是运行期量（`reachable`），`#pragma unroll` 失效；原版
  fast path 是**编译期常量的 8+tail**，IGC 全展开。逐标量谓词载入其实被编译成廉价
  predicated load，不值得替换。**已回退**（与 R24 的「无谓词 fast path 更慢」同源）。

## 3. 尝试 B：编译期通道特化（`SPEC_CIN`/`SPEC_COUT`）——**中性**

把 `ic_blocks`/leftover 谓词从运行期变编译期常量（模拟 OV JIT 的 `FILTER_IFM_NUM`）。

- 40×40 C64：0.132 → 0.132（无变化）。IGC 对 4 次循环本就能展开。**未保留。**

## 4. 正结果：`SLM_DIV`（上游 `SLM_DIV_FACTOR`）

R25 移植时**省掉了 `SLM_DIV_FACTOR`**——把一个 work-group 的输入通道循环切给
`SLM_DIV` 个 sub-group（WG=16×SLM_DIV lane），各自算一部分输入通道，最后在 SLM 归约。
这直接增加每 WG 的在飞工作量，掩盖全局读延迟，正是 40×40/20×20 缺的东西。

**实测（`kernel_bench`，FIT 特化与生产一致）：**

| shape | SLM=1 | SLM=2 | SLM=4 | 最好增益 |
|---|---:|---:|---:|---:|
| 40×40 C64 8×8 | 9.07 | 8.04 | 8.52 | — |
| 40×40 C128→64 | 10.77 | 12.44 | **12.92** | **+20%** |
| 40×40 C128→128 | 12.60 | 14.46 | **14.91** | **+18%** |
| 20×20 C128→128 | 7.56 | 8.61 | **9.26**（SD8 9.56） | **+22%** |
| 80×80 s2 C64→128 | 9.10 | 9.96 | **10.99** | **+21%** |
| 80×80 C64→64 | 12.87 | **13.41** | — | **+4%** |
| 80×80 C32→32 | 8.54 | **9.36** | 7.12 | +10% |

- **数值 PASS**（`kernel_bench --verify` mean_rel ~3e-3；`model_check` 三模型 ALL PASS）。
- 归约读 `partial_summ[(lid1%SG)+i*SG]`，仅 sub-block 0 写输出；SLM_DIV=1 时行为与旧 kernel
  逐位等价（已回归三模型）。

**同时顺带**：输出端加**合并向量 store**（`vstore8/4/2`，OV 的 `VSTORE` 路径），
40×40 C64 +6%，且与标量路径逐位一致（`--verify`）。

## 5. 接入

- `KernelFamilies.cpp`：`conv3x3_blk` 候选枚举 `OBW∈{2,4,8} × SLM∈divisors(Cin/16)∩[1,8]`。
- `PlanModel`：autotune / run() 的 blk 分支解析 `-DSLM_DIV=`，`lws[1]=16*SLM`、
  `gws[1]=(Cout/16)*16*SLM`。
- `kernel_bench`：`--conv` 第 2 字段（TY）作为 SLM_DIV，并镜像生产 FIT 特化。

## 6. 根因：autotune「选 SLM=1」是一个静默 bug（已修）

`kernel_autotune --only 40x40 --retune` 对 40×40 C128→64 选 `SLM=1`，而
`kernel_bench` 显示 `SLM=4` 快 20%。加了 `INFVINO_AUTOTUNE_DEBUG=1` 后真相是：

```
[autotune] conv3x3|W40H40s1p1_Cin128_Cout64: 24 candidates
  [cand] conv3x3_blk -DOBW=8 -DSLM_DIV=1 ... 0.2146 ms
  [skip] conv3x3_blk OBW=8,SLM=2,... bench failed
  [skip] conv3x3_blk OBW=8,SLM=4,... bench failed
```

**每个 `SLM_DIV>1` 候选都 `bench failed` 被静默丢弃**，而 `autotuneOp` 对
build/enqueue 异常是 `continue`——所以调优器永远只看到 `SLM=1`。根因是
`PlanModel` 里解析选项的**差一错误**：

```cpp
// len("-DSLM_DIV=") == 10，原来写成 +9 → atoi("=2") == 0 → 几何退回 SLM=1，
// 与按 -DSLM_DIV=2 编译的 kernel 的 reqd_work_group_size 不匹配 → enqueue 失败。
if (ps != npos) slm = std::atoi(opts.c_str() + ps + 10);
```

修好后 SLM 候选正常参与、`SLM>1` 按 shape 胜出。**这是「一连串静默 bug」的第一环**：
真正的破坏力来自 `autotuneOp` 静默吞掉 enqueue 失败。已加 `INFVINO_AUTOTUNE_DEBUG`
打印每个候选的 ms / skip 原因。

其余手写偏移（`-DOBW=`=6、`-DOBH=`=6、`-DDW_TW=`=8）经核对**正确**；只有
`-DSLM_DIV=` 错了。

## 7. conv_ov 的 SLM_DIV：实现了但**数值错误**（负结果，已禁用）

给 `conv_ov.cl` 补了同样的 `SLM_DIV`（WG=(1,SLM_DIV,SG) + SLM 归约）。逐配置
`kernel_bench` 显示有收益（80×80 C32 +10%），但 **`model_check` FAIL**
（mean_rel 5.2e-2，`SLM_DIV>1` 时出现 NaN/inf），强制 `SLM_DIV=1` 立刻 PASS。
归约逻辑经复核正确，根因未定位（可能与本 kernel 的 `sub_group_broadcast`/寄存器
数组的交互有关）。**已在候选枚举里禁用 `SLM_DIV>1`**（内核代码保留待查）。

## 8. reorder-aware 计费：**负结果，已回退**

为了治「blk 单核快、但强制 fsv16 输入重排」的问题，给 conv3×3 也加了
conv1x1/depthwise 那样的**保守重排计费**（`blk.ms + 一次 reorder`）。实测：

| 配置 | y8 net | y11 net |
|---|---:|---:|
| baseline | 15.58 | 16.65 |
| r37clean（blk SLM，无计费） | **14.88** | 16.81 |
| reorder-aware 计费 | 15.70 | 16.87 |

保守计费把「本可持久化、重排为 0」的 blk 也罚了，反而更差（y8 15.70 vs 14.88）。
**已回退**。正确做法是「先按布局算持久化、再对未持久化的输入精确计费」的联合不动点，
但注意 **tuning 缓存是按 signature 存的、跨模型共享**，同一 shape 在 y8/y11 的
persistence 不同时无法各存一份——这是下一步要解决的设计问题。

## 9. 本轮最终 e2e（三模型，`config/tuning.json`=r37clean）

interleaved A/B（3 rep，同会话）：

| 模型 | baseline net | r37clean net | Δ |
|---|---:|---:|---:|
| yolov8n-pose | 15.58 | **14.88** | **−4.5%** |
| yolo11n-pose | 16.65 | 16.81 | +1.0%（busy 其实 −1.9%，重排/提交 +）|
| mobilenetv3-small | ~2.43 | ~2.44 | ≈0 |

三模型 `model_check` PASS、`reuse_check` PASS。对 OV（infer 11.37/12.02/1.74）：
y8 **1.31×**、y11 **1.40×**、mb **1.40×**。**仍未超过 OV。**

## 10. 待办 / 下一步

1. **联合 (族, 布局) 选择不动点**：按 persistence 精确计费；并解决 sig-cache 跨模型
   共享导致无法按模型区分的问题（例如把 layout 相关选择从 sig-cache 里拆出来，
   或按 model/plan 维度加重键）。
2. **`conv_ov` SLM 归约的数值 bug**（根因未定位）。
3. **`conv1x1` 仍是最大相对差距**（OV y8 1×1 家族 ~0.9 ms vs infvino ~3.2 ms），
   尚未系统攻。
4. autotune 单次测量对 ±10–20% 的本机噪声敏感；应对 top-2 候选复测或加 warmup。
