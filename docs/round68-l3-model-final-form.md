# R68：L3 成本建模的**最终形态**（收敛总结）

> 用户决策：把 R47–R67 的 L3 建模收敛成一份「最终形态」，说明**当前形态即最优形态**并附证据。
>
> 一句话：**可加主问题（实测 kernel ms + 可加 reorder）+ 非可加项（跨算子 L3 溢出用全局
> 序列模拟求值，不进逐节点代价）**；容量用**两级和**（GPU 私有 L3 3.75 MiB + 共享 LLC 8 MiB
> ≈ 12.3 MB），替换策略默认严格 LRU（NRU 等价模型可选）；地址映射只用到**周期 512 行**这一层。

---

## 1. 最终成本函数

```
Cost(π) = Σ_i kernel_ms_i          # 实测、逐节点、可加（autotune 按实测 ms 选）
        + Σ_t reorder_ms_t          # 实测 #reorder 优先；缺失用可加结构式
        + spill(π)                  # 唯一非可加项：跨算子 L3 溢出的全局序列模拟
```

* `spill(π)` = `evaluateL3(buildL3Access())`（`L3Model`）：按拓扑序跑一遍全局缓存模拟
  （占用压缩有效容量 → 逐出 → 读 miss/写），返回 `spill_ms = spill_bytes × (1/20 − 1/145) ns/B`。
* `reorder_ms` = 实测 `#reorder`（单趟张量级）；缺失时 `reorderCostMs(read,write,resident)` =
  `launch(3.5µs) + bytes/BW(state)`（state = 输入是否驻留 L3）。
* **布局目标**（`resolveLayoutMinCut` 的 min-cut 节点代价）只用三项：`e.ms`（实测）+ reorder +
  `spill`（经全局模拟/分量门，非逐节点线性化）。`layoutModelScore = kernel + reorder + spill`
  是分量验收门的唯一口径。

**可加 / 非可加的分界**：`kernel` 与 `reorder` 是**实测、可加**的（占总分 ~90%）；`spill` 是
唯一的**非可加、顺序相关**项（~10%），由全局模拟一致处理，**不**折叠进逐节点代价。

---

## 2. 常量（来源与确定性）

| 量 | 值 | 来源 | 确定性 |
|---|---:|---|---|
| GPU 私有 L3 | **3.75 MiB** = 512×120×64 = 8×480 KiB | 公开 PRM + 实测周期 512 行 + 膝点 ~4MB | 高 |
| 共享 LLC | **8 MiB** | `lscpu`/sysfs index3（shared 0–7） | 高 |
| **跨算子有效容量**（`spill` 用） | **≈12.3 MB** = L3+LLC | 别名实测阈值 ~12 MB + R55 双租户 8–12 MB | 中–高 |
| 私有 tile 锚点 | 2 MB | R55 §2.3 膝点 | 中 |
| DRAM / L3 带宽 | 20 / 145 GB/s | R55 copy 实测 | 高 |
| `spill` 折算 | (1/20 − 1/145) ms/B | 上式 | 高 |
| reorder launch / 流式 BW / dispatch 间隔 | 3.5 µs / 60 GB/s / 12.4 µs | R60 `kernel_bench --op reorder[_seq]` | 中–高 |
| 地址索引**周期** | **512 行 = 2¹⁵ B** | R62/R65 步进冲突（锐利） | 高 |

实现：`l3PhysicalBytes()` / `l3LlcBytes()` / `l3WarmCapBytes()` / `l3PrivateCapBytes()` /
`l3DefaultCapBytes()`（= 两级和）/ `kL3SpillPerByteMs` / `kReorder*`。

---

## 3. 建模选择与理由

| 选择 | 理由 |
|---|---|
| L3 溢出用**全局序列模拟**（非逐节点） | 逐出是顺序相关 + 近期重用加权；逐节点线性化三次负结果（§5） |
| 容量用**两级和**而非单级 | 实测：L3 逐出后仍命中 LLC；有效容量 = L3+LLC（R63/R64） |
| 替换策略默认**严格 LRU** | 公开 PRM 是 1-bit NRU；但默认 LRU 与实测排序一致且数值稳定；NRU 作为等价模型可选 |
| 张量粒度（非 cache-line） | 行粒度精确模拟可行（§6）但需先解决地址哈希/几何（R65：不可逆），且成本模型用不到 |
| reorder 实测优先 + 结构分解 | 实测精确；分解解释「可加性差了点」（dispatch 间隔 + 工作集，R60） |

---

## 4. 接受的近似（正式）

1. **地址→(bank,set) 只到周期 512**：完整 XOR 哈希**不可逆**——OA 计数器不可用
   （`gpu_metrics … Activate failed`），且 L3 在 LLC 之后，计时探针**原理上**无法隔离私有 L3。
   接受理由：周期 + 容量（512×120×3.75MiB）已由公开 PRM 完全确定；哈希只影响逐出曲线**形状**。
2. **张量粒度**：不区分 cache-line/set 冲突；与 §1 的可加主问题口径一致。
3. **两级塌缩为单一有效容量**（≈12.3 MB）：不显式建模「L3 miss 命中 LLC」的二级延迟。
4. 以上都**只影响离线评分/排序**，不改任何**实测**项；均有 A/B 开关（§7）。

---

## 5. 负结果与统一判读（**不要**把占用线性化进逐节点代价）

| 轮次 | 做法 | 结果 |
|---|---|---|
| R49 §10 | 把 `occupancyPressure`（足迹字节）当 **miss 字节**线性计入 | 破坏可加性与量级 |
| R55 §7.5 | `rho × occupancyPressure` 定价（collective/pernode） | 中性 / pernode 回归 |
| R66 | 二维 `effectiveBwGbps` 接**诊断** roofline | 中性（`expected` 不参与选择） |
| R67 | `(1/g(R)−1)·bytes/BW_mlp` 加进**布局节点代价** | **一致 +1.65%（y11）/ +1.66%（mb）回归** |

**统一结论**：跨算子溢出**不可**用「逐节点占用 × 常数」线性化——必然**重复计费**（实测 ms 已含
内存）且**拆掉 blocked 链**（R54：链布局 mb −11%）。正确形态即 §1：**实测可加项 + 全局非可加模拟**。

---

## 6. 复杂度（离线求值）

* 单次 `spill`（`compute_prices=false`）：现状 **O(A·D)**（`vector` 线性扫描 + `erase`）；
  最优（哈希 + 双向链表）**O(A)**。
* 有限差分定价：×(P+1) 次模拟（**O(P·A·D)** 现状）。
* `profileL3Stack`（复用/栈距离 CDF）：一次 O(A·D)（最优 O(A log D)）；之后任意容量 O(1)——
  消掉容量维度 K。
* 行粒度（若将来接入）：A_L≈3×10⁶、常驻 1.3×10⁵ 行；O(A) 实现 ~10–100 ms/次（实测
  `l3linesim` 252 ns/访问、线性），**可行**；naive O(A·D)~10¹³ 不可行。
* 张量粒度现状（A、D ~10²）**完全可忽略**。

---

## 7. 开关 / A/B 清单

| 开关 | 作用 | 默认 |
|---|---|---|
| `INFVINO_L3_GEOM=1` | `spill` 容量用仅 GPU 私有 3.75 MiB（而非两级和） | 关 |
| `INFVINO_L3_POLICY=nru` | 替换策略用 NRU 等价模型（1b-NRU） | LRU |
| `INFVINO_LAYOUT_L3=1` (+`INFVINO_L3_PRICE`/`_SCALE`) | 逐节点 L3 定价（R55，负结果） | 关 |
| `INFVINO_LAYOUT_CAP=1` (+`CAP_W`) | 目标函数级容量项（R67，负结果） | 关 |
| `INFVINO_NO_L3_ROOFLINE_2D=1` | 关闭二维有效带宽进 `expectedOps`（R66） | 2D 开 |
| `INFVINO_LAYOUT_REORDER_GAP=1` | 整网补回 reorder dispatch 间隔（R60） | 关 |
| `INFVINO_NO_REORDER_STATE` / `INFVINO_NO_REORDER_STRUCT` | reorder 结构式消融 | 结构式开 |
| `INFVINO_LAYOUT_REPORT=1` | 打印 `model score` / `mincut` / `spill` | 关 |

---

## 8. 回归基线（默认路径）

```text
model_check（逐位不变，选择不受 L3 建模改动影响）:
  yolov8n-pose     mean_rel = 4.114e-04
  yolo11n-pose     mean_rel = 7.908e-04
  mobilenetv3-small mean_rel = 1.215e-02

默认 model score（cap=两级和 12.3MB）:
  y8  kernel=10.1851 reorder=0.0583 spill=0.9239 total=11.1672
  y11 kernel=10.7935 reorder=0.2082 spill=1.0367 total=12.0384
  mb  kernel=1.3323  reorder=0.0561 spill=0.2317 total=1.6201

check 6/6;  test_rulers 108;  test_l3_model 41
```

---

## 9. 证据索引

| 文档 | 内容 |
|---|---|
| `round47-l3-model.md` | 把内存层级做进标尺（起点） |
| `round55-l3-coupling-calibration.md` | occupancy↔miss 耦合；方案 A；负结果 |
| `round59-l3-spill-calibration.md` | 重标定；「近似为何成立」 |
| `round60-l3-exact-spill-plru-and-reorder.md` | 复用/栈距离；逐出归因；reorder 可加性；`NRU` |
| `round61-l3-line-granular-model-and-complexity.md` | 行粒度 1b-NRU（公开 PRM）+ 复杂度 |
| `round62-l3-fidelity-geometry-correction.md` | 几何测量（512 行周期 / ~4MB 膝点） |
| `round63-address-mapping-reverse-engineering.md` | 两级层次；别名探针 |
| `round65-l3-isolation-and-geometry-conclusion.md` | 隔离不可行；接受近似；整网 A/B |
| `round66-two-dim-roofline-and-selection.md` | 二维 roofline（诊断） |
| `round67-capacity-term-in-layout-objective-negative.md` | 目标函数容量项负结果 |
| `third_party/intel-prm/NOTICE.md` | 公开 PRM 许可合规 |

---

## 10. 边界与未来（不做的事 & 何时再做）

* **不做**：继续逆 XOR 哈希（不可逆、无收益）；把占用再折进逐节点代价（三次负结果）。
* **可做（若收益证据出现）**：把**两级容量接进内层 roofline**（R55 §6.1，`effectiveBwGbps`
  已实现、仅诊断）；或**链感知**的内存定价（在保持可加性前提下，分别给 blocked 链整体定价）。
* **触发条件**：只有在**候选集/内核**层面出现新的内存受限候选、或拿到 OA 计数器时，才值得
  升级到行粒度 / 显式两级。当前三模型的增量在**候选/内核**，不在把已有候选选得更准。
