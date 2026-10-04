# Round 46：整网调优全流程实测 —— 暴露的基础设施 bug 与设计缺口

> 按 R45 的落地，跑「隔离重扫 + 整网 busy 回验」全流程（三模型，锁频，分批），
> 目的是**用真实流程反查基础设施**。结论先说：
>
> **整网回验能跑通、无 HANG，但（在不加安全门时）会把整网改慢**（mb 实测 +5.8~7.4%）。
> 加了最终验收门后恢复到中性（mb −0.2%、y8 +0.5%，均在噪声内），但**也拿不到可靠增益**。
> 因此 **blocked chain 暂不上**——底座还不稳。本轮的产出是 3 个 bug 修复 + 1 个安全门 +
> 对「为什么整网回验难」的实证。

---

## 0. 一句话

整网回验当前**没有可靠的正收益**，而且会引入回归；根因不是单一 bug，而是三层叠加：
**(1) 逐 op/批次看不到跨 op 布局交互；(2) in-situ 的 `min` 估计器与稳态口径系统性错配；
(3) 整网 busy 本身有 ±5% 噪声**。先修底座（见 §4），再谈 blocked chain。

---

## 1. 实验设置

- 锁频：`scripts/gpu_clocks.sh lock`（min=max=RP0=1300），全程。
- 生产 cache 不动；实验用临时 cache（`models/tuning_r45*.json`，从 `config/tuning.json` 拷贝）。
- 命令：`scripts/autotune.py --model <m> --batch 2 --iters 4 --global --global-topk 2
  --global-iters 4 --global-rounds 2 --global-budget 120`
- A/B：交错 3 rep 取 min busy（`kernel_run --report --iters 15`），同会话。
- 基线（锁频，生产 cache）：y8 10.50 ms、y11 11.52 ms、mb 1.63 ms。

---

## 2. 实测结果

### 2.1 无最终验收门（R45 P1+P2 状态）

| 模型 | 基线 busy | 整网回验后 | Δ |
|---|---:|---:|---:|
| mobilenetv3-small | 1.630 | **1.725**（3 rep：1.725/1.754/1.778）| **+5.8%** |

逐节点定位（基线 → r45）：

| 节点 | Δ | 说明 |
|---|---:|---|
| `depthwise`（聚合）| **+0.042** | 多族从 `depthwise_v` 翻成 `depthwise_blk` |
| `reorder(blk)` | **+0.036** | blk 族变多 → 运行时要重排、邻居没持久化 |
| `conv1x1g@1000x1024` | +0.017 | N==1 GEMV（见 §3.2 的 bug）|
| `conv1x1blk@576x49x96` 等 | +0.010~0.02 | blk 配置变化 |

### 2.2 加最终验收门后

| 模型 | 基线 busy | 整网回验后 | Δ |
|---|---:|---:|---:|
| mobilenetv3-small | 1.630 | 1.627 | −0.2%（噪声内）|
| yolov8n-pose | 10.501 | 10.552 | +0.5%（噪声内，**0 批次接受改动**）|
| yolo11n-pose | 11.523 | 11.489 | −0.3%（噪声内，1 批次接受改动）|

即：门把回归挡住了，但**回验本身也几乎选不出净收益**（三模型全部落回噪声带内）。

### 2.3 验收门日志（mb depthwise 子集，带 report）

```
[global-retune] net 1.6951 -> 1.5946 ms (-5.9%)   # 但 0 signature reselected → 净差其实是噪声
[global-retune] net 1.7069 -> 1.6388 ms (-4.0%)
[global-retune] net 1.6566 -> 1.6888 ms (+1.9%)  REVERT (final worse than baseline)
```

**关键观察**：同一份 assignment，`measureWhole` 前后差可达 ±5%（1.695→1.594 却不含任何改动）
→ 整网 busy 的测量噪声与「候选改善」同量级，门和选择都被噪声主导。

---

## 3. 本轮发现并修复的基础设施 bug

### 3.1 `--global` 在有条目缓存时是空操作 + `--retune` 与分批无法推进

- `globalRetune` 依赖**同进程**的隔离短名单（`cand_short_`）；而 `autotune.py` 正常模式下
  「缓存里已有 `source=tuned` 就跳过」→ 短名单为空 → 全局静默空操作。
- 直接用 `--retune` 又会**每批从第一个签名重来**（`sig_seen` 是进程内的），批次永不推进。

**修复**：整网 campaign 用 `INFVINO_GLOBAL_PROGRESS=1`，把 **per-plan 工件当作分批进度标记**
（本图已写入 `plan_overrides_` 的节点跳过）。`autotune.py --global` 自动：`--retune` +
清空旧 per-plan 工件 + 设该环境变量。实测工件 2→4→7→9→11 递增、正常收尾。

### 3.2 P0#6 漏掉 `conv1x1` 的 N==1（GEMV）分支

`choiceEntry` 统一时只改了 N>1 分支；N==1 的 `conv1x1_gemv_f16` 仍 `tuning_.lookup`
（`src/PlanModel.cpp` N==1 分支）。后果：整网回验**测不到 GEMV 候选**（所有候选测同一件事），
却仍写 per-plan 覆盖 → 运行时 GEMV 用了未经回验的配置。**已修**为 `choiceEntry(ni, sig)`。

### 3.3 缺「最终验收门」：逐签名贪心改坏整体也不回退

`globalRetune` 逐签名贪心，从不校验**最终组合**。**已加**：先量基线 `baseNet`，坐标下降后量
`finalNet`；若 `finalNet > baseNet*(1+1%)` 则整轮回退（不写共享缓存；per-plan 工件写回基线
以继续分批推进）。修复后 mb 回归消失（§2.2）。**落盘也改为只写真改变的 target**。

---

## 4. 尚未修复的设计缺口（blocked chain 的前置）

1. **跨 op/批次的布局交互不可见**：`autotune.py` 按 `--op` 分批，每批的 `globalRetune` 只
   `candidates` 是当前 op 的签名；但布局（fsv16 持久化 / reorder）由**整图**决定。逐 op 独立
   选出的 `blk` 组合在一起，可能比任何单独测量都更差（本轮 `reorder(blk)` +0.036 即此）。
   → 需要**单进程 all-op 联合不动点**，或把「布局耦合的族」作为一个 move。
2. **估计器口径错配**：`benchCandidate`/`busyFor` 用 `min`（R42 为估计**内禀**成本、抗外部
   干扰）。但当方差是**内禀**的（占用/调度），`min` 偏乐观；稳态 `kernel_run` 是典型值之和。
   实测同一 depthwise 改动：in-situ 显示 **−3~5%**，稳态却 **+12%**。
   → 需要 in-situ「min + median」双口径，且**改善必须在典型口径也成立**；或提高 reps。
3. **整网 busy 噪声 ±5%**：与候选改善同量级。少量 reps 下，`baseNet`/`finalNet` 差不可信。
   → 需要更多 reps / 交错 / 每候选独立重复组的中位数。
4. **per-plan 工件覆盖交互**：工件按节点输出名冻结 per-node 选择，绕过运行时的联合不动点；
   若工件是在「别的布局上下文」选出的，运行时可能付额外 reorder。本轮门已缓解，但非根治。

---

## 5. 结论与建议

- **结论**：整网回验的**框架可用**（能跑、可安全中止、可回退、per-plan 融合生效），但
  **选择质量不可靠**——当前不应作为生产默认，更不应在其上叠加 blocked chain。
- **上 blocked chain 的前置**（按顺序）：
  1. in-situ 双口径（min+median）+ reps 提升，使「改善」在典型口径成立；
  2. 单进程 **all-op 联合**坐标下降（或布局耦合族分组 move）；
  3. 门升级为「典型口径 + 多次交错」验收；
  4. 用外部稳态 A/B（`kernel_run --report`）做最终裁决。
- **已落地**（本轮）：3 个 bug 修复（§3.1–3.3）+ 最终验收门。默认仍 `--global` 关闭，
  生产 `config/tuning.json` 未被改动。

> 这也正面回答了 R44 的问题：R44 修对了「目标函数测错了对象」，但**测量条件与估计器**
> 还不足以支撑可靠的全局选择；R45 的保底短名单/噪声地板是对的方向，但需要上面的前置。

---

## 6. 复现

```bash
scripts/gpu_clocks.sh lock

# 全流程（安全驱动：分批 + 进度标记 + HANG 自检）
cp config/tuning.json /tmp/t.json
python3 scripts/autotune.py --model mobilenetv3-small --batch 2 --iters 4 \
  --global --global-topk 2 --global-iters 4 --global-rounds 2 --global-budget 120 \
  --cache /tmp/t.json

# 交错 A/B（3 rep 取 min）
for i in 1 2 3; do
  INFVINO_PLAN_TUNING=none INFVINO_TUNING_CACHE=config/tuning.json \
    ./build-ct/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 15 | grep busy
  INFVINO_TUNING_CACHE=/tmp/t.json \
    ./build-ct/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 15 | grep busy
done

scripts/gpu_clocks.sh unlock
```
