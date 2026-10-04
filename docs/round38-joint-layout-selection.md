# Round 38：联合 (族, 布局) 选择不动点 + sig-cache 解耦

> 承接 R37：`conv3x3_blk` 的 SLM_DIV 让单 kernel 快 10–26%，但端到端只拿到一部分——
> 因为 blk 读 `b_fs_yx_fsv16`，输入不持久化时要付一次重排；而「是否持久化」又取决于
> 选核。R37 试过**保守重排计费**（每个 blk 都罚一次 reorder），是**负结果**：它把
> 「本可持久化、重排为 0」的层也罚了（y8 15.70 vs 14.88）。本文给出正确解法。

## 0. 一句话结论

把「选哪族」和「张量用什么布局」做成**计划期的联合不动点**：
autotune 为每个 sig 存**两个备选**（`#blk` / `#non`）与输入的**实测重排成本**（`#reorder`），
运行时按**该 plan 的实际持久化**逐节点选优。因为不动点按 plan 图跑，
同一 shape 在 y8/y11 得到不同选择——同时解决了「sig-cache 跨模型共享无法区分布局」的问题。

**实测**：y8 net **−6.8%**、y11 net **−6.2%**、mb ≈0；y11 reorder 调用 **28→24、0.79→0.41 ms**；
三模型 `model_check` + `reuse_check` PASS。

## 1. 问题

1. **选核只看 kernel ms**：tuner 选 `conv3x3_blk` 时不知道它的输入会不会被持久化为 fsv16。
2. **持久化是自指**：`planBlockedLayout` 的规则是「生产者是 blocked 且**所有**消费者都吃
   fsv16 → 该张量持久化」，而这又取决于每个消费者选了什么族。
3. **保守计费是负结果**（R37 §8）：对**所有** blk 候选加一次 reorder，把持久化（重排为 0）
   的层也罚了，反而更差。
4. **sig-cache 跨模型共享**：同一 `conv3x3|W40H40...` 在 y8/y11 的 persistence 不同，
   一份「选谁」的缓存无法两全。

## 2. 设计

### 2.1 autotune 存「备选矩阵」（`PlanModel::autotune` conv3×3 分支）

对每个 conv3×3 签名，把候选分成 blk / 非 blk 两组分别 bench：

- `sig`            → 总体最优（保持向后兼容，未做不动点时用它）
- `sig#blk`        → blk 组最优
- `sig#non`        → 非 blk 组最优（ov / native / cin3）
- `sig#reorder`    → 该输入做一次 `bfyx→fsv16` 的**实测 ms**

（`OpSignature::custom(sig.str()+"#blk", {})` 作为扁平 map 的键，无需改缓存结构。）

### 2.2 运行时联合不动点（`PlanModel::resolveLayoutChoices`）

在 `parse()` 末尾（tuning 载入之后、首次 run 之前）跑，`kIters=4`：

```
for it in 0..3:
    planBlockedLayout()                       # 按当前 per-node 选择标记 fsv16
    changed = false
    for each conv3x3 node ni with artifacts:
        in_fsv16 = T_[node.ins[0]].fsv16
        cost(blk) = #blk.ms + (in_fsv16 ? 0 : #reorder.ms)
        cost(non) = #non.ms
        choose smaller -> node_choice_[ni]
    if !changed: break
planBlockedLayout()                           # 最终布局与选择一致
```

结果写 `node_choice_[ni]`，由 `choiceEntry(ni, sig)` 统一消费：`run()` 的 conv3×3 分支、
`convWillUseBlk`、`planBlockedLayout` 的 `nodeFamily`。**缓存里没有 `#blk/#non` 时退化为
一次 `planBlockedLayout()`（R36 行为）**，所以旧缓存零风险。

### 2.3 解耦

- 备选矩阵（kernel ms by shape）是**可移植**的，跨模型共享没问题。
- **决策**（blk vs non）在**每个 plan 的图**上重算 → y8/y11 各得其所，无需 per-model 文件。

## 3. 实测（interleaved A/B，3 rep，同会话）

| 模型 | baseline net | R38 net | Δ | busy |
|---|---:|---:|---:|---:|
| yolov8n-pose | 15.53 | **14.48** | **−6.8%** | 12.02→**11.65** |
| yolo11n-pose | 16.66 | **15.63** | **−6.2%** | 13.50→**12.52** |
| mobilenetv3-small | 2.46 | 2.43 | ≈0（无 conv3×3 blk） | — |

y11 `reorder`：28 次 / 0.79 ms → **24 次 / 0.41 ms**（不动点把非持久化输入的 blk 换成 ov）。
三模型 `model_check` PASS、`reuse_check` PASS。

对 OV（infer 11.37/12.02/1.74）：y8 **1.27×**、y11 **1.30×**、mb **1.40×**（仍未超过）。

## 4. 测量噪声（本轮排查）

- **autotune 选择已可复现**：两次独立 `--retune` 对 40×40 选出完全相同 kernel/config。
  R37 之前的「不可靠」其实是 `-DSLM_DIV=` 差一导致 SLM 候选被静默丢弃。
- 残余绝对时间方差来自 **GPU DVFS**（`gt_act_freq` 100↔1300）与 **CPU 频率波动**
  （governor 已是 performance，但 `scaling_cur_freq` 在 1.4–1.9 GHz）。GPU 频率文件 root-only，
  无法无 sudo 固定。
- 对策：`timeMs` 已返回 **median**；A/B 用 **p50/min + 交错**；稳态 median 稳定
  （2000 iters 下 0.123 ms）。**未发现遗留测量 bug。**

## 5. 待办

1. `conv_ov` 的 SLM 归约数值 bug（R37 §7）——仍未定位。
2. `conv1x1` 仍是最大相对差距（OV y8 1×1 家族 ~0.9 ms vs infvino ~3.2 ms），未系统攻。
3. autotune 现对 conv3×3 bench 两组（约 +1× 时间）；可只对 top shape 存备选。
