# R67：把容量项加进布局目标函数——**负结果**（默认关）

> 承接 R66：`expected` 只是诊断标尺；要让 L3/占用**真正影响选择**，必须进入**目标函数**。
> 本轮把「容量感知内存惩罚」加进布局 min-cut 的**节点代价**，并做整网 A/B。
>
> 结论先给：**项能改变选择**（y11 fsv16 9→4→1、mb 23→21→8），但**整网一致回归**
> （y11 **+1.65%**、mb **+1.66%**，默认 5/5 胜）。根因：它在**实测 ms 之上又加了一遍**
> 已含于实测的内存时间，同时**无视 blocked 链/局部性收益**。→ **默认关**，保留为 opt-in + 证据。

---

## 1. 实现（目标函数级、逐节点、可加、可开关）

新函数（`Tuning`）：

```cpp
singlePassBytes(sig)                 // 输入+输出+权重（与 expectedOps 足迹口径一致）
capacityMemPenaltyMs(e, sig) = (1/g(R) − 1) · bytes / (BW_mlp(C)·1e9) · 1e3   // ms, ≥0
//   R = occupancyPressure(e,sig)（在飞足迹）  g = l3CapacityFactor(R)（R55 曲线）
//   C = occupancyThreads(e,sig)              BW_mlp = l3MlpFactor(C)
```

接入 `resolveLayoutMinCut` 的 `inCurveMs`：

```cpp
v = e.ms
  + (useL3 ? rho_i · occupancyPressure : 0)          // R55 定价（默认关）
  + (useCap ? α · capacityMemPenaltyMs(e,s) : 0);    // R67 容量项（默认关）
```

开关：`INFVINO_LAYOUT_CAP=1`；`INFVINO_LAYOUT_CAP_W=<α>`（默认 1）缩放。报告行加 `cap=0/1`。

## 2. 选择确实变了（模型自评反而更差）

| 模型 | arm | fsv16 | E | model total |
|---|---|---:|---:|---:|
| yolo11n-pose | default | 9 | 2.4048 | 12.0384 |
| yolo11n-pose | CAP=1 | **4** | 2.8108 | 12.0704 |
| yolo11n-pose | CAP=1 α=5 | **1** | 3.2428 | 12.0826 |
| mobilenetv3-small | default | 23 | 0.9376 | 1.6201 |
| mobilenetv3-small | CAP=1 | **21** | 1.0649 | 1.6274 |
| mobilenetv3-small | CAP=1 α=5 | **8** | 1.3523 | 1.6675 |

→ CAP **减少**持久 fsv16（更少 blocked/链），使**模型自己的总分变差**。即：它把选择推离了
当前模型的**最优点**。

## 3. 整网 A/B（锁频、交错、5 rep、`--iters 15`）

| 模型 | default median | CAP=1 median | Δ | 胜场 |
|---|---:|---:|---:|---:|
| yolo11n-pose | 11.376 | 11.564 | **+1.65%** | 默认 5/5 |
| mobilenetv3-small | 1.382 | 1.405 | **+1.66%** | 默认 5/5 |

→ **一致回归**（不是噪声，5/5）。默认关。

## 4. 根因（为什么必然回归）

1. **重复计费**：`e.ms` 是**实测**单算子时间，**已包含**该候选真实的内存行为；`capacityMemPenaltyMs`
   在其上再加一遍「容量不足的额外内存时间」，量纲/语义重复（R49 §10 的同型错误）。
2. **无视链/局部性**：CAP 惩罚高占用候选 → 偏好 `non`（gemm）而非 `blk`，**拆掉了 blocked 链**
   （少 `fsv16`、多 reorder）。R54 已验证链布局在 mb 上 **−11%**；CAP 的建模内存「收益」远小于
   拆链的实测损失。
3. **占用 ≠ miss 字节**：`occupancyPressure` 是**在飞足迹**，把它经 `g(R)` 折成时间再线性加到
   节点上，仍是 R55 §7.6 批评过的「单点线性化过乐观」。

**结论**：布局目标应保持 `Σ 实测 ms + Σ reorder + 全局 spill`（R59）——**实测项精确、可加**，
**唯一的非可加项（跨算子溢出）用全局模拟/门**处理；在其上叠加**逐节点建模内存惩罚**会
系统性地把选择带偏（R49/R55/R67 三次同型负结果）。

## 5. 保留与复现

* 默认关（`INFVINO_LAYOUT_CAP` 不设 = 旧行为，三模型 `model_check` 逐位不变）。
* 保留 opt-in + `CAP_W` 缩放，作为后续「若引入**链感知**的内存定价」的对照基线。

```bash
scripts/gpu_clocks.sh lock
for rep in 1 2 3 4 5; do
  ./build/kernel_run --plan models/yolo11n-pose/model.plan --report --iters 15 | grep "total kernel"
  INFVINO_LAYOUT_CAP=1 ./build/kernel_run --plan models/yolo11n-pose/model.plan --report --iters 15 | grep "total kernel"
done
scripts/gpu_clocks.sh unlock
```

回归：`check` 6/6；`test_rulers` 104 → **108**（新增 R67：惩罚 ≥0、小算子=0、单遍字节>0）。

---

### 三次同型负结果的统一判读（R49/R55/R67）

| 轮次 | 做法 | 结果 |
|---|---|---|
| R49 §10 | 把 `occupancyPressure`（足迹字节）**当 miss 字节**线性计入 | 破坏可加性与量级 |
| R55 §7.5 | `rho × occupancyPressure` 定价（collective/pernode） | 中性/pernode 回归 |
| R67 | `(1/g(R)−1)·bytes/BW_mlp` 加进节点代价 | **一致 +1.65% 回归** |

→ 重复出现的是同一条：**跨算子溢出的非线性不能用「逐节点占用 × 常数」线性化**。正确形态是
R55/R59 的「**可加主问题（实测项）+ 非可加全局 spill（模拟/门）**」；任何把占用**再次**折进
逐节点代价的尝试都会因**重复计费 + 拆链**而回归。
