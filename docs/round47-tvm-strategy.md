# Round 47：TVM meta_schedule 策略学习 + 对整网搜索的借鉴（附落地顺序）

> 承接 R44–R46：R44 把目标函数从「隔离 min」改成「整网 busy」，R45 落地 per-plan 覆盖 /
> 保底短名单 / 噪声与安全，R46 全流程实测后得出「整网回验框架可用、但选择质量不可靠，
> blocked chain 前置未满足」。
>
> 本轮不再自造下一版搜索，而是**系统学习 TVM（Apache TVM master，`s_tir/meta_schedule`，
> Ansor 的继任者）的整网调优策略**，逐项对照我们的 `globalRetune`，回答「哪些能借、哪些
> 是它绕开而我们绕不开的」，并给出**尊重 R46 前置**的落地顺序。

---

## 0. 一句话

**TVM 的整网策略本质是「不做整网搜索」**：它把整网目标近似为
`Σ (task_weight × 单算子最优延迟)`，每个算子独立搜（进化搜索 + XGBoost 代理），
跨算子唯一的全局决策是「预算怎么在算子之间分配」。它**从不测端到端**——恰好绕开了
R46 撞上的三堵墙：整网 busy ±5% 噪声、in-situ `min` ≠ 稳态、跨 op 布局耦合不可见。

因此可借用的不是某套算法，而是它的**三层结构**：可加目标（代理整网） + 代理模型剪枝
（少测） + 梯度预算分配（测哪里）。其中「可加目标」是最高价值的一条。

---

## 1. TVM `s_tir/meta_schedule` 分层与代码

| 层 | 职责 | 关键实现 |
|---|---|---|
| 任务抽取 | 把 Relax 图按 **结构哈希去重**成 per-op `ExtractedTask`；**weight = 该 PrimFunc 被 `call_tir` 的次数** | `src/relax/backend/task_extraction.cc`（`TaskExtractor::VisitExpr_`，weight 语义见文件头注释） |
| 设计空间 | trace 生成 + 后处理/mutator，把「调度决策」表示成可重放的 trace | `space_generator/*`、`mutator/*` |
| 搜索策略 | **进化搜索**：从已测 top + 随机种群出发，按代价模型采样 + 变异，`eps_greedy` 保探索 | `search_strategy/evolutionary_search.cc` |
| 代理代价模型 | XGBoost/MLP，预测**归一化 cost ratio**（`min_cost/cost`，按 workload 分组），特征 = per-store（算术强度/访存/形状） | `cost_model/xgb_model.py`、`feature_extractor/per_store_feature.py` |
| 测量 | `time_evaluator(number, repeat, min_repeat_ms)`，**取 median 入库**、跨 trial 取 min；可选 cache flush、cooldown、`alloc_repeat` 随机重填 | `runner/local_runner.py`、`runner/utils.py`、`utils.h:GetRunMsMedian` |
| 数据库 | 全部实测入库；`GetTopK`；`UnionDatabase` / `OrderedUnionDatabase`（有序并集） | `database/*.cc` |
| 全局调度 | `GradientBased`：先轮询每个 task 建立基线，之后每轮选 **gradient 最大**的 task（`α·近期改善率 + (1−α)·best/n`，再乘 weight）；另有 round-robin | `task_scheduler/gradient_based.cc`、`task_scheduler.cc` |
| 整网入口 | `tune_relax` → 抽任务 → `tune_tasks`；结束各算子独立取 top-1，**不联合回验** | `relax_integration.py`、`tune.py` |
| 事后优化 | `PostOpt` + `Droplet`：对离散设计空间做**统计显著性**驱动的局部开发（pvalue） | `post_optimization/{post_opt,droplet}.py` |

两个关键点：

1. **可加目标是刻意设计**。`TaskRecord` 里 `flop = EstimateTIRFlops(mod)`，打印的
   `Weighted Latency = latency × weight`，整网被默认为这些加权项之和。这样每个算子可以
   **完全独立**地选最优，不需要任何端到端测量。
2. **代理只用来排序、不用来定胜负**。XGBoost 预测归一化比值，只决定「哪几个候选值得真测」
   （`num_warmup_samples` 之前返回随机分）；最终胜者永远是**测出来的**。

---

## 2. 与 infvino 整网搜索逐项对照

| 维度 | 我们（R44–R46） | TVM | 借鉴 |
|---|---|---|---|
| 联合目标 | 真·端到端 `busy_ms`（±5% 噪声） | `Σ wᵢ·单算子 ms`（可加代理） | **高价值** |
| 搜索算法 | 全签名坐标下降 | 每算子独立 + 预算调度 | 中 |
| 候选剪枝 | 隔离 top-K ∪ 每族代表 | 代理模型按归一化 ratio 排序 | 中（需轻量化） |
| 耦合处理 | 想用整网回验硬吃 | 靠 fused op + anchor-block 等价回避 | 需自研 |
| 测量口径 | in-situ `min` | median 入库 + min 取最优 | **可直接抄** |
| 两级缓存 | `config/tuning.json` + `<plan>.tuning.json` | `OrderedUnionDatabase` | 已对齐 |
| 预算分配 | 每轮遍历所有签名 | gradient 选最值得的 task | 可借 |

> **结构性差异**：TVM 的融合算子内部自带布局决策，跨 op 布局持久化不是它的强项（只有
> layout rewrite pass）；我们的 `b_fs_yx_fsv16` 持久 blocked 链反而更强。所以**不能纯可加**
> ——残差项正是我们相对 TVM 的增量价值，必须自研。

---

## 3. 可借用点（按 ROI，落到我们的代码）

### P0 — 用「加权可加目标」重构 `globalRetune` 的主目标

- 定义 `net̂ = Σ wᵢ · ms(node_choiceᵢ)`，`wᵢ` = 该签名出现次数 × 隔离 busy 占比
  （对应 TVM 的 weight=调用次数）。**零 GPU**、低噪声、可复现。
- 关键推论：**若目标可加，per-node 独立 argmin 即全局最优**，无需端到端。我们目标里
  唯一不可加的部分只剩**布局耦合**（blk 的输入重排 / fsv16 持久化 / `#reorder`）。
- 于是把 `globalRetune` 从「遍历所有签名各测整网」收缩为：
  1. 用 `net̂` 直接决定**无耦合**签名的选择（不测）；
  2. 只对**布局耦合连通分量**做联合 move 与端到端回验；
  3. 门只做一次最终验收。
- 落地：`src/PlanModel.cpp` 的 `globalRetune`；新增 `predictNet()`（纯主机计算）。

### P0 — 布局耦合分组 move（**前置未满足前不做 blocked chain**）

- 依赖 R46 §4.1 的观察：逐 op 独立选的 blk 组合在一起可能更差，根因是 reorder/持久化
  由整图决定。分组 move = 把「共享 reorder 输入 / 同一条 blk 候选链」的节点作为一个
  坐标一起改。
- 当前可用的耦合信号：`resolveLayoutChoices` 的 `sig#blk/#non/#reorder` 不动点。
  **先**用这张图做分组 move；**真正的持久 fsv16 链（blocked chain）等 R46 前置完成后另开**。

### P1 — 残差预测器替代「top-K ∪ 每族代表」

- TVM 预测的是归一化比值（可跨 workload 迁移），不是绝对 ms。我们让它学
  **iso→net 残差**：特征 = `(op, shape, Cin/Cout, stride, 候选族, 布局上下文)`；
  标签 = 实测 `net_delta`（或 iso/net 名次差）。
- 候选是小离散集（族 × 几个 config），**不要上 XGBoost**；MVP 用「按
  (签名, 族, 布局上下文) 累积残差表 + 线性/秩」即可。
- 这直接治「隔离名次 ≠ 整网名次」，并让短名单从「保底」升级为「有依据地剪枝」。
- **状态**：本轮已落地**零 GPU 的可加代理** `predictNet`（作为 `globalRetune` 的排序/分组依据）；
  学习型残差模型留待下一轮——它需要先把每次回验的 `(iso_pred, net_measured)` 数据落盘（§7 待办）。

### P1 — 测量口径改成 min+median 双口径 + 交错

- R46 §4.2 已确认 in-situ `min` 与稳态错配；TVM 的答案是 **median 入库、min 取最优**。
- 我们改：每个 assignment 采一组 reps，同时记录 `min`（内禀地板）与 `median`（典型）；
  **接受改善必须在 median 口径成立**；基线与候选**交错测量**（ABAB）以抵消热漂移。

### P2 — gradient 式预算分配

- 把「每轮遍历所有签名」改成优先投给：iso→net 残差大、邻居多、工作集跨 L3 的
  签名/分组，并用改善速率动态调整（对应 R45 §4.3 的 priority，公式化自 TVM gradient）。

### 已对齐、建议明确命名/证明

`config/tuning.json`（共享隔离默认） + `<plan>.tuning.json`（per-plan 全局，`choiceEntry`
最高优先）**就是** `OrderedUnionDatabase`（有序并集，per-plan > 共享）。文档中统一用这个
术语，并说明「删工件=回退」等价于有序并集 miss。

---

## 4. 不能照搬的边界

1. **TVM 不对跨融合布局持久化建模**；我们的 blocked chain 更强，纯可加会漏掉残差。
2. **TVM 代理是重模型**（搜索空间连续巨大）；我们候选小、离散，重模型是负收益。
3. **代理只是过滤器**，仍要真测短名单；收益是「更少的端到端噪声测量」，不是免测。
4. TVM 的整网目标靠 weight=调用次数近似 FLOPs 占比；我们应改用**隔离 busy 占比**，
   才反映真实耗时（一次 conv 远重于一次 permute）。

---

## 5. 落地顺序（尊重 R46 §5 前置）

R46 明确：blocked chain 之前必须先满足 4 条前置。本轮的实现严格按此排序：

| 顺序 | 项 | 对应 R46 前置 | 状态 |
|---|---|---|---|
| 1 | in-situ **min+median 双口径** + reps | §5.1 #1 | **已实现**（`NetStat`/`measureAssign`） |
| 2 | 测量**交错** + 接受判据用典型口径 | §5.4 #3 | **已实现**（`measurePair`，median 接受 + min 地板守卫） |
| 3 | **可加目标 `net̂`** 决定无耦合签名（少测） | §4.1 #2 的降维 | **已实现**（`predictNet` + 优先级排序） |
| 4 | 布局耦合**分组 move**（用现有 blk/non 耦合图） | §4.1 #2 | **已实现**（`comp` union-find + grouped move） |
| 5 | 残差预测器（MVP：累积残差表） | §4.2 #3 | **部分**：可加代理 `predictNet` 已落地（零 GPU）；**学习型残差模型下一轮** |
| 6 | 外部稳态 A/B 作最终裁决（`kernel_run --report`） | §5.4 #4 | 复现步骤 |
| — | 持久 fsv16 **blocked chain** | §4.1 #1 | **暂缓**（前置就位后另开） |

### 5.1 本轮实现映射（`src/PlanModel.cpp::globalRetune`）

| 机制 | 代码 | 说明 |
|---|---|---|
| 双口径 | `struct NetStat{mn,med,spread}` + `statsOf` | min=内禀地板、median=典型口径 |
| 交错测量 | `measurePair(A,B,reps)` | 同 rep 内先 A 后 B，各自 capture+采样；抵消热漂移 |
| 可加目标 | `predictNet(asg)` | `Σ 节点 kernel ms + 未持久化 blk 输入的 #reorder.ms`（零 GPU） |
| 耦合分量 | `comp`（union-find：共享输入 + 生产者→消费者） | 只有同分量的签名才需联合端到端回验 |
| 优先级 | `tmeta.headroom = predictNet(base) − min_c predictNet(base,c)` | 降序（TVM gradient 的静态近似） |
| 分组 move | grouped pass（`groupedOn`） | 分量内联合选候选 → 整体端到端回验一次 |
| 门 | 交错 `baseline` vs `final`，**median** 裁决 | R45 门用单一 min；R47 改典型口径 |

- 预算语义：`--global-budget` 现在是**整网执行次数**（含 capture），更贴近 GPU HANG 风险口径。
  关闭分组 move：`INFVINO_NO_GLOBAL_GROUPED=1`。
- 默认：`--global` 仍 opt-in；组内新机制随 `--global` 生效，可被上述环境变量关掉。

### 5.2 机制自检（锁频，小范围，非性能结论）

```
[global-retune] 1 signatures, topK=2 reps=2 rounds=2, metric=min+median, accept=median,
                interleave=on, additive-priority=on, grouped=on, base(med)=1.7575
[global-retune] conv1x1|Cout16_N3136_Cin16_act0_f16  gemm_f16 -> conv1x1_blk  med 1.7044 -> 1.6667 ms (2.2%)  [coupled]
[global-retune] net(median) 1.6744 -> 1.6539 ms (-1.2%)
```

双口径/交错/可加优先级/耦合标记/分组 move/per-plan 工件写回均按预期工作。**不构成性能结论**
（单签名、锁频、机制验证）；正式评估需三模型全流程 + `model_check`/`reuse_check`。

### 5.3 三模型全流程实测结论（详见 `round47-full-flow-findings.md`）

跑通 + 数值/复用回归 PASS + 0 HANG，但**全局步本身仍无可靠正收益**：

| 模型 | 稳态 A/B（full vs base）| reorder 占比 | fsv16 | 判定 |
|---|---:|---:|---:|---|
| mobilenetv3-small | **−2.2%**（收益全来自 blk 族；reorder 反升）| 8.3%→8.9% | 3→5 | 布局耦合是主矛盾 |
| yolov8n-pose | +0.2%（噪声）| 0.6% | 0 | 无耦合残差，瓶颈在 micro-kernel |
| yolo11n-pose | +0.3%（噪声）| 1.8% | 0→1 | 同上 |

并且 `full ≈ iso`（仅隔离重扫）→ **mb 的 −2% 来自缓存刷新，而非整网搜索**。据此判定：
**blocked chain 是 mb 的主矛盾（机会上限≈reorder 的 8%），但不是 yolo 的主矛盾**。倒查出的
8 项 bug/设计缺陷与后续 ROI 见 findings 文档 §3/§5。

> 说明：第 3–5 步是「用 TVM 的可加+代理思路**替代**全签名端到端坐标下降」，正是为了让
> 整网研究**不再依赖**它最不可靠的那部分（端到端噪声测量），而不是在噪声上再堆机制。

---

## 6. 复现（锁频 + 安全分批）

```bash
scripts/gpu_clocks.sh lock

# 隔离扫描（短名单来源） + 整网回验（新目标函数）
INFVINO_GLOBAL_RETUNE_REPORT=1 ./build-ct/kernel_autotune \
  --plan models/mobilenetv3-small/model.plan --cache /tmp/r47.json \
  --op conv1x1 --limit 4 --iters 6 --retune \
  --global --global-topk 3 --global-iters 4 --global-rounds 3

# 外部稳态 A/B 裁决（交错 3 rep 取 min）
for i in 1 2 3; do
  INFVINO_PLAN_TUNING=none ./build-ct/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 15
  INFVINO_PLAN_TUNING=/tmp/r47.pt.json ./build-ct/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 15
done

scripts/gpu_clocks.sh unlock

# 数值/复用回归
python3 scripts/model_check.py  --model mobilenetv3-small --repo $PWD --image infvino-dev:latest
python3 scripts/reuse_check.py --model mobilenetv3-small --repo $PWD --image infvino-dev:latest
```

---

## 7. 待办

1. **落盘残差数据**：每次整网回验记录 `(signature, family, iso_pred, net_measured)`，作为学习型
   残差模型的训练集（P1 的下一半）。
2. 分组 move 就位后，评估是否重新打开「持久 fsv16 blocked chain」（R46 §4.1 #1）。
3. 残差表积累够真实数据后，评估是否升级为轻量秩模型。
4. gradient 式预算分配（P2）进 `autotune.py`（现状是静态 headroom 排序）。
5. 三模型正式 retune + `model_check`/`reuse_check` 全回归（锁频、分批）。
