# R57：min-cut 回退局部化 + reorder 级目标（用户决策的 1 / 2）

> 承接 R56（算子集收尾）。用户决策：算子层完成后做布局求解器的两步——
> **1) 精确 min-cut 的「回退局部化」**（不再整份作废）、**2) 目标函数改为 reorder 级**。
>
> 结论先给：两步都已落地，且**在当前默认缓存上行为中性（数值逐位不变）**——因为
> (a) `resolveLayoutMinCut` 的「整份 return false」在当前三个模型上本就**不触发**；
> (b) 相关签名的 `#reorder` 条目都已测量。因此这是**硬化/契约修复**，不是性能改动。
> 真正的「回退」发生在**验收门**（R49/R55 的整网 median gate），本轮的局部化移除的是
> **求解器自身**的全有全无出口。

---

## 1. 回退局部化（`resolveLayoutMinCut`）

### 1.1 问题

R52 起，`resolveLayoutMinCut` 有三处「整份作废」出口，任一触发都让
`resolveLayoutChoices` 执行 `restore(base)`（丢弃**整个**布局提案）：

| 出口 | 条件 | 现状 |
|---|---|---|
| `addPairwiseTable` 非 submodular | `K=(f00+f11-f01-f10)/2 > 0` | 按构造恒 `K≤0`，实际不触发 |
| `!sol.optimal` | 不可行 | `feasible()` 恒真，实际不触发 |
| `mayMarkFsv16` 失败 | 被标 fsv16 的缓冲放不下补齐布局 | 分配能力集是标记集的超集，实际不触发 |

即：**三个出口当前都不触发**，所以「频繁回退」的观感来自**验收门**（整网 median 判否 →
`restore(base)`），而非求解器。但求解器的全有全无结构是隐患（新增族/契约漂移时会突然整份
丢弃），本轮把它去掉。

### 1.2 实现

1. **`BinaryEnergy::addPairwiseTableRelaxed`**（`LayoutSolver`）：`addPairwiseTable` 的
   **永不失败**版本——非 submodular 时把耦合项 `K` clamp 到 0（丢掉落 repulsive 项），得到
   一个合法**松弛**能量；解仍是确定的最优解。`resolveLayoutMinCut` 改用它。
   （严格版 `addPairwiseTable` 保留 + 单测不变。）
2. **fix-and-resolve 循环**：解出后，若某变量被标 fsv16 但 `mayMarkFsv16` 失败，**只把该
   变量钉死 NCHW 并重解**（最多 4 轮），而不是整份 `return false`；末尾兜底把残余违例置 0。
3. 移除 `if (!sol.optimal) return false;`（Dinic 对可行实例必最优；不可行由上面的循环兜底）。

```cpp
// 生成代价表（永不失败）
en.addPairwiseTableRelaxed(a, b, f00, f01, f10, f11);
BinarySolution sol = solveBinaryMinCut(en);
// 局部化：越界变量钉死 NCHW 重解，而非整份作废
for (int it = 0; it < 4; ++it) {
  bool violated = false;
  for (v ...) if (sol.labels[v] == 1 && !mayMarkFsv16(varName[v])) { en.fix(v, 0); violated = true; }
  if (!violated) break;
  sol = solveBinaryMinCut(en);
}
```

### 1.3 验证

- `test_layout_solver` 新增用例（relaxed 表对非 submodular 输入**永不失败**、解有限/最优、
  越界/自环安全）：**375 passed**（原 372）。
- 三模型 `INFVINO_LAYOUT_REPORT=1` 的 `mincut: … fsv16` 与改前**完全一致**
  （y8 0 / y11 9 / mb 23）——即局部化在当前缓存上不改任何选择。

---

## 2. reorder 级目标（结构价 + OV 两级）

### 2.1 问题

min-cut 的成对代价表用 `alt[ni].reorder.ms`（实测 `#reorder`）给「异布局 → 插 reorder」定价。
若某签名**没有** `#reorder` 条目，`reorder.ms` 落到默认 **0** → **reorder 被当免费** →
求解器系统性偏向「过度持久化」（R49 缺口 A 的机制之一）。OV 的 `reorder_inputs` 用的是
**两级结构目标**：一级 = reorder **条数**（固定 launch floor），二级 = 搬运**元素量**。

### 2.2 实现

`resolveLayoutMinCut` 的 `reorderCost(ni)`：

```
r = alt[ni].reorder.ms                     # 有实测以实测为准
if (r <= 0 and struct_fallback):
    r = kReorderFixedMs + kReorderMsPerElem * numel   # 结构性两级
    #  kReorderFixedMs  = 0.0035 ms   （一次 dispatch 的 launch floor）
    #  kReorderMsPerElem = 6.7e-8 ms  （≈60 GB/s，fp16 读+写 4 B/elem）
r *= INFVINO_LAYOUT_REORDER_W                # 缩放（标定/消融）
```

- `INFVINO_NO_REORDER_STRUCT=1`：关闭兜底（= R56 前行为，做 A/B）。
- 成对表与落地 `useBlk` 判定共用同一 `reorderCost`，杜绝口径漂移。

### 2.3 验证

当前缓存下，相关签名（conv1x1/depthwise blk）都有实测 `#reorder` → 兜底不生效 →
`mincut fsv16` 计数、三模型数值**逐位不变**：

| 模型 | mean_rel（默认 / `NO_REORDER_STRUCT`） | 判定 |
|---|---|---|
| yolov8n-pose | 4.114e-04 / 同 | 逐位相同 |
| yolo11n-pose | 7.908e-04 / 同 | 逐位相同 |
| mobilenetv3-small | 1.215e-02 / 同 | 逐位相同 |

即：本项是**契约保真**（消除「免费 reorder」的系统性偏差），在缺 `#reorder` 的新签名/
重扫缓存上才会改变选择。默认开启；`NO_REORDER_STRUCT` 保留 A/B。

---

## 3. 复现

```bash
# 局部化单测
cmake --build build -j --target test_layout_solver && ./build/test_layout_solver

# min-cut 报告（两步都不改当前选择）
INFVINO_LAYOUT_REPORT=1 ./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 1
INFVINO_LAYOUT_REPORT=1 INFVINO_NO_REORDER_STRUCT=1 ./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 1

# 回归
cmake --build build -j --target check
python3 scripts/model_check.py --model mobilenetv3-small --repo "$PWD"
```

---

## 4. 诚实边界 / 下一步

- 本轮**没有整网性能改动**（数值逐位不变）：求解器的全有全无出口当前不触发、结构价兜底
  当前不生效。这是「把隐患去掉」的硬化，符合「先补契约、再谈收益」的排序。
- **真正的回退在验收门**：R49 的整网 median gate 与 R55 的 L3 离线评分门仍是「整图判否 →
  `restore(base)`」。若要继续收窄「频繁回退」，下一步是把门**按连通分量局部化**
  （只回退被判否的分量，保留其余 min-cut 提案），或让门基于**更可信的 spill 敏感度**。
- L3 定价（R55）维持默认关：其根因（单点线性化过乐观）未解决，本轮未动。
