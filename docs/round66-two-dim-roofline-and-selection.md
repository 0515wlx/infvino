# R66：把二维有效带宽接进内核级内存 roofline（+ 为何整网仍无变化）

> 承接 R59 §6 / R55 §6.1 的搁置项：把 `effectiveBwGbps(bytes, threads)`
> （并发/MLP 因子 × 足迹容量因子）接进**逐节点内存 roofline**，让 L3/占用进入内核级期望。
>
> 结论先给：**接好了**（`expectedOps` 现在带候选 `TuningEntry`，roofline 用二维带宽）；
> 隔离效应温和（mb 缓存 25/513 项、−15…−24%）。但**整网仍无变化**——因为
> **`expected` 只是诊断标尺（ratio/报告），不参与任何选择**。选择由「实测 ms（autotune）+
> kernel ms/reorder/spill（布局）」决定。要让 L3 影响选择，必须有**目标函数级**的容量项。

---

## 1. 实现

`expectedOps(sig, dev, const TuningEntry * e = nullptr)`（新增可选候选）：

```cpp
auto bwAt = [&](double bytes) -> double {
  if (e && !getenv("INFVINO_NO_L3_ROOFLINE_2D")) {
    const double th = occupancyThreads(*e, s);      // R55：并发线程数 C
    if (th > 0.0) return effectiveBwGbps(bytes, th); // copyBw(bytes) × l3MlpFactor(C)
  }
  return copyBwGbps(bytes);                          // 旧行为（无候选 / 关闭 / 小算子）
};
```

* conv3x3 / gemm 的内存 roofline 改用 `bwAt(bytes)`（`memRooflineOps(..., bwGbps)`）。
* 传入候选的调用点：`Autotuner::applyStandard`、`Autotuner` best、`PlanModel::refreshExpected`。
* 无候选的调用点（如 `KernelFamilies` ceiling、单测）→ **旧行为不变**。
* 开关：`INFVINO_NO_L3_ROOFLINE_2D=1` 关闭（A/B、回归）。

## 2. 隔离效应（mb，`--refresh-expected`，2D-on vs 2D-off）

把缓存里 `expected/ratio` 用当前公式重算两遍对比：

* **25/513 项**变化；幅度 **−15%…−24%**，集中在**低并发** `conv1x1`（小 spatial）。
* 例：`conv1x1|Cout24_N784_Cin72` 1.47→1.23（−17%）；`Cout48_N196_Cin144` 0.59→0.50。
* （初看 `−96%` 是**缓存陈旧** vs 现公式的差异，不是 2D；隔离后 2D 只动 25 项。）

方向正确：低并发候选的内存墙更低、`expected` 更低、`ratio`（ops/expected）更高——即「小/欠占用层
离内存墙更近」。这是 R55「并发 C 决定可达上限」的落地。

## 3. 整网 A/B（锁频、交错、4 rep、`--iters 15`）

| 模型 / 臂 | median (ms) | vs default |
|---|---:|---:|
| yolo11n-pose default(2D) | 11.3635 | — |
| yolo11n-pose 2D-off | 11.3365 | −0.24% |
| yolo11n-pose `INFVINO_LAYOUT_L3=1` | 11.3695 | +0.05% |
| mobilenetv3-small default(2D) | 1.3990 | — |
| mobilenetv3-small 2D-off | 1.3865 | −0.89% |
| mobilenetv3-small `INFVINO_LAYOUT_L3=1` | 1.3965 | −0.18% |

→ **全在噪声内、符号不一致**。选择也**逐位未变**：mincut `fsv16 9/23`、`E=2.4048/0.9376`、
`model score 12.0384/1.6201` 在 2D-on/off 下**完全相同**。

## 4. 为什么整网不变（根因）

`expected` 在代码里只被用于：
1. `ratio = ops/expected`、`hard_ratio`（**报告/热点定位**）；
2. `KernelFamilies` 的软 `ceiling`（喂回 `expected` 的 `min`）；
3. `kernel_autotune` 的打印表。

**没有任何门/求解器用 `expected` 做选择**：
* autotune 按**实测 ms** 选（`all[0].e.ms`）；
* 布局求解/分量门用 **`e.ms`**（`layoutModelScore = Σ kernel ms + Σ reorder + spill`）；
* L3 定价（`INFVINO_LAYOUT_L3`）是唯一把「占用」注入布局节点代价的项，但 R55/R66 实测**中性**。

所以 2D roofline 是**标尺正确性**的改进，不是**选择**的改进。

## 5. 要真正影响选择，只有目标函数级

| 备选 | 状态 | 说明 |
|---|---|---|
| L3 定价 `rho×occupancyPressure` | 已实现、默认关 | R55 中性/pernode 回归；R66（修正容量后）仍中性 |
| **把容量项加进布局节点代价** | 未做 | 例如 `node_cost += α·bytes_candidate / BW_eff(C, footprint)`，使**每个候选的 tiling** 改变内存代价 → 影响 min-cut |
| 改 autotune 选择准则为内存感知成本 | 未做、有风险 | 会偏离「实测 ms 为准」的纪律 |
| 候选生成按占用/容量过滤 | 未做 | 在 KernelFamilies 侧排除注定内存墙的 tiling |

**建议下一步**：做「把容量项加进布局节点代价」——用**同一** `effectiveBwGbps`/`l3CapacityFactor`
给每个候选算一个**内存时间增量**，加进 `resolveLayoutMinCut` 的节点代价 `e.ms` 之上，让
layout 选择**显式看见**候选的占用/足迹。默认关 + 整网 A/B 验收，避免重演 R49 §10 的负结果。

## 6. 回归 / 复现

```text
check 6/6；test_rulers 101 → 104（新增 R66 用例：2D ≤ 1D、2D-off == 1D、正值）
默认行为不变（无候选 → 旧 roofline）
```

```bash
# 标尺隔离效应
cp config/tuning.json a.json && cp config/tuning.json b.json
docker ... './build/kernel_autotune --plan models/mobilenetv3-small/model.plan --cache /work/a.json --refresh-expected'
INFVINO_NO_L3_ROOFLINE_2D=1 docker ... '... --cache /work/b.json --refresh-expected'
# 对比 a.json vs b.json 的 expected

# 整网 A/B（默认 vs 2D-off vs L3 定价）
scripts/gpu_clocks.sh lock
for rep in 1 2 3 4; do
  ./build/kernel_run --plan models/yolo11n-pose/model.plan --report --iters 15 | grep "total kernel"
  INFVINO_NO_L3_ROOFLINE_2D=1 ./build/kernel_run --plan models/yolo11n-pose/model.plan --report --iters 15 | grep "total kernel"
  INFVINO_LAYOUT_L3=1 ./build/kernel_run --plan models/yolo11n-pose/model.plan --report --iters 15 | grep "total kernel"
done
scripts/gpu_clocks.sh unlock
```
