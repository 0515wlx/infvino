# Round 44：autotune 目标函数修正（隔离 min → 整网 busy 坐标下降回验）+ 体系缺陷倒查

> 承接 R43 §5.1/§7.1（本轮最重要的 open item）：
> autotune 按**单节点、隔离、空 cache** 的 `min(ms)` 选候选，会把「隔离快、流水线慢」的
> 配置选进来（R43 实测 9 个 conv1x1 配置如此，mb busy +1.6%）。本轮把目标函数换成
> **端到端 busy**，并据此系统倒查调优体系自身的缺陷。
>
> 结论：新增 `PlanModel::globalRetune()` + `kernel_autotune --global`，在真实 plan 上对
> 隔离 top-K 做**坐标下降**，只有真正降低整网 busy 的候选才被保留。轻量 GPU 自验通过
> （见 §3），默认关闭（安全/可复现），隔离扫描路径与既有缓存**零改动**。

---

## 0. 一句话

**「局部最优」之所以不反映「全局最优」，是因为目标函数测错了对象**：它优化的是
`min(隔离 latency)`，而系统的真实目标是 `end-to-end busy`。二者在共享 L3/DRAM 与在飞
占用（occupancy）耦合时并不单调相关。修正 = 把目标函数换成整网 busy，并在真实流水线里
评估候选；同时保留隔离扫描作为「候选生成 + 短名单」。

---

## 1. 问题：目标函数的两个错位

| 维度 | 现状（R43 及以前） | 应有的 |
|---|---|---|
| **优化对象** | 单节点、单 kernel 的 `min(ms)` | 整网 `busy_ms`（逐 kernel event 时间之和） |
| **观测条件** | 冷/空 cache、隔离 | 真实 plan、相邻 kernel 之后、布局驻留状态 |
| **耦合** | 候选之间假设独立 | 候选经 L3/DRAM/占用**互相影响**（非独立） |

R43 §5.1 的实证：`conv1x1` 配置 `Cout576_N49_Cin96` 隔离 0.0352→0.0284 ms（更快），
但 in-pipeline 0.0916→0.0994 ms（更慢）。典型机制：`XB4→XB2`（更多 WG/更少复用）、
`SLM1→SLM4`（更多在飞 → L3 压力更大）。**隔离 bench 冷/空 cache，看不到这些外部性。**

---

## 2. 设计

### 2.1 保留隔离 top-K 作为短名单

`Autotuner::autotuneOp` 增加可选输出 `std::vector<TuningEntry>* measured`：记录**所有成功
测到的候选**（kernel/config/options/ms/ops，按 ms 升序），不再只留胜者。
`PlanModel::autotune()` 为每个签名把各分支（blk/非 blk、pad 等）的 measured 合并进成员
`cand_short_`（key = 签名串，值含 `OpSignature` 以便精确写回）。**隔离扫描的语义与结果不变。**

### 2.2 整网 busy 坐标下降

`PlanModel::globalRetune(ops, iters, topK, rounds, limit)`：

```
1. 目标 = 短名单里能在本 plan 找到节点的签名（按隔离 ms 取 top-K）；
2. 初始 per-node 赋值 = 构造期（联合不动点）的实际选择；
3. 对每个目标：把候选赋给该签名的所有节点 → planBlockedLayout() → 重录 dispatch
   → 跑整网 reps 次，取 min(busy_ms)（`profile().busy_ms`）；保留最小者；
   只有相对现状改善 > 0.5% 才接受；
4. 多轮直到无签名改变；结果写回 tuning_。
```

关键实现选择：

- **直接控制 per-node 选择**（`node_choice_ = assign` + `planBlockedLayout()`），**不调用**
  `resolveLayoutChoices()`——否则联合不动点会用隔离 `#blk/#non.ms` 把我们正在回验的候选
  覆盖掉。回验期间布局始终由当前赋值驱动。
- 测量用 `clearProfile() + run()`；第 1 次是 capture/热身，之后取 reps 次的 **min busy**
  （与 R42「干扰只加时间、min 是无偏估计」的哲学一致）。
- 候选若 `run()` 抛异常（几何/资源）→ 记为极大值拒绝，不中断整轮回验。
- 落盘时 base 签名写回 winner，并**同步同族**的 `#blk/#non` 备选为 winner，使运行时的
  联合布局不动点在本 plan 上复现同一选择。

### 2.3 使用（opt-in）

```bash
# 在同一进程内先跑隔离扫描（短名单来源），再做整网回验：
./build/kernel_autotune --plan models/mobilenetv3-small/model.plan \
  --cache config/tuning.json --op conv1x1 --limit 4 --iters 6 --retune \
  --global --global-topk 3 --global-iters 2 --global-rounds 3
```

| 参数 | 作用 |
|---|---|
| `--global` | 开启整网 busy 坐标下降回验 |
| `--global-topk K` | 每个签名参与回验的候选上限（隔离 top-K，默认 3）|
| `--global-iters N` | 每个 assignment 的整网测量次数（取 min busy，默认 3）|
| `--global-rounds R` | 坐标下降轮数上限（默认 3）|
| `--global-limit N` | 最多回验 N 个签名（0=不限；安全分批）|
| `INFVINO_GLOBAL_RETUNE_REPORT=1` | 打印每次改写（`kernel -> kernel busy a -> b`）|

> **注意**：`--global` **必须先跑隔离扫描**（同进程的 `cand_short_` 才有内容）。若全部命中
> 缓存而没有 `--retune`，短名单为空、回验为空操作——这是当前的已知易误用点（见 §5）。

### 2.4 当前覆盖范围

只有 `conv3x3 / conv1x1 / depthwise` 三族在 `run()` 里消费 per-node 覆盖（`choiceEntry`）；
`gemm / conv1x1_cat4 / 小算子` 直接读 `tuning_.lookup`，**全局回验暂不覆盖**（见 §5 缺陷 #7）。
本轮 R43 的回归发生在 conv1x1，正好在覆盖范围内。

---

## 3. 轻量 GPU 自验（未锁频，仅验证机制）

构建：容器内 `cmake --build build-ct --target kernel_autotune tuning_test` 通过；
`tuning_test` 34 项 **PASS**。轻量 GPU 用例（单签名、`--global-topk 2–3`）：

| 模型 | 签名 | 结果 |
|---|---|---|
| y8 | `conv3x3\|W320H320s2p1_Cin3_Cout16` | `conv3x3_cin3 -> conv3x3_f16`，busy 16.81→16.64 ms（1.0%）|
| mb | `conv1x1\|Cout16_N3136_Cin16` | `gemm_f16 -> conv1x1_blk`，busy 3.01→2.92 ms（3.3%）；第 2 轮同族 config 再 −1.0% |

机制成立：**全局目标会选中隔离名次未必最优、但端到端更快的候选**。以上为未锁频的机制
验证，不代表最终数值；正式 retune 应 `scripts/gpu_clocks.sh lock` 后分批跑（§6）。

---

## 4. 复现

```bash
# 构建
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake -S /workspace/infvino -B /workspace/infvino/build-ct \
  -DCMAKE_BUILD_TYPE=Release && cmake --build /workspace/infvino/build-ct -j4'

# 离线自检
docker run --rm --memory=2g --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino \
  -w /workspace/infvino infvino-dev:latest ./build-ct/tuning_test

# 单签名整网回验（锁频后；安全分批）
scripts/gpu_clocks.sh lock
INFVINO_GLOBAL_RETUNE_REPORT=1 ./build-ct/kernel_autotune \
  --plan models/mobilenetv3-small/model.plan --cache /tmp/gt.json \
  --op conv1x1 --only 3136 --limit 1 --iters 6 --retune \
  --global --global-topk 3 --global-iters 3 --global-rounds 3 --expected
scripts/gpu_clocks.sh unlock

# 整网数值回归（全局改写后必须跑）
python3 scripts/model_check.py  --model mobilenetv3-small --repo $PWD --image infvino-dev:latest
python3 scripts/reuse_check.py --model mobilenetv3-small --repo $PWD --image infvino-dev:latest
```

---

## 5. 倒查：调优体系自身的缺陷与漏洞

> 目标函数的错位只是**症状**。下面按「目标 / 表示 / 测量 / 安全 / 指标」分层列出体系
> 本身的缺口，并标注本轮是否修复。**这是本轮的主要产出之一。**

### A. 目标函数层

1. **[已修] 目标错位**：把「单节点隔离 min」当优化目标。→ `globalRetune` 改为整网 busy。
2. **[部分修] 观测条件错位**：隔离冷/空 cache ≠ 流水线驻留状态。→ in-situ 回验；但回验
   仍是**串行逐 event 等待**，且未锁频时 busy 是时钟相关量。
3. **[根治不彻底] 成本边界不清**：`reorder` 曾被折进 kernel `ms`（R42 修的记账 bug）。
   说明「哪些成本属于节点、哪些属于布局」缺少单一真相源。R44 的 `busyFor` 用整网 busy
   天然包含 reorder，但持久化又交回 `resolveLayoutChoices`，两套决策口径仍可能不一致。

### B. 缓存 / 选择表示层（最深的漏洞）

4. **[未修，根本限制] 签名级共享缓存无法表达 per-plan / per-node 最优**。同一 shape 在
   y8/y11 的持久化与最优族不同（R38 已用 plan 期不动点缓解布局维度）。R44 的全局结果
   仍写回**签名级**缓存（并同步同族 `#blk/#non`）——位置相关的最优会被跨模型共享稀释。
   真正根治需要 **per-plan 选择存储**（或 bake 进 plan），而不是共享 `tuning.json`。
5. **[未修] 两套决策口径并存**：`resolveLayoutChoices()` 用**隔离** `#blk/#non.ms` 决定
   blk/non；`globalRetune()` 用**整网 busy** 决定候选。持久化后运行时仍由前者拍板，二者
   可能不一致（R44 靠「同步同族备选」近似对齐，不是构造性保证）。
6. **[未修] 接入不统一**：`conv3x3/conv1x1/depthwise` 走 `choiceEntry`（支持 per-node 覆盖），
   `gemm/conv1x1_cat4/小算子` 直接 `tuning_.lookup`。导致 (a) 全局回验覆盖不全；
   (b) 「per-plan 决策」无法统一施加。**建议**：把 `choiceEntry` 作为所有可调族的唯一入口。
7. **[未修] 缓存 ABI 版本 ≠ 数值契约**：`kTuningCacheAbi` 只在 `-D` 宏语义变化时 bump，
   但 **kernel 数值变化**（fp16 累加顺序、SLM 分块、覆盖写）不体现在 key。R39 的 SLM bug
   证明「tuning 只改性能不改数值」并非普遍成立（SLM 打开后累加顺序变化，只是误差可接受）。
   候选选择隐含了**数值契约**，却只 versioned 了编译宏——这是正确性风险。

### C. 测量 / 噪声层

8. **[已记录，未做] min 估计器在 in-situ 下前提变弱**：R42 的「干扰只加时间 → min 无偏」
   针对**外部**干扰；坐标下降里候选对邻居是**有序影响**，min over reps 仍会被 L3/DRAM
   断崖的偶发慢尾误导。建议锁频 + top-K 复测 + 交错（R42 §3.3 本就列为待办）。
9. **[前置未满足] 锁频是 ops 口径正确的前提**（R42 §3.3）。R44 的 busy 同样是频率相关量；
   未锁频的 in-situ 结果只能做机制验证（§3 即如此）。

### D. 安全 / 工程层

10. **[新引入风险] 整网回验的计算量 = Σ签名 × (K+1) × rounds × reps 次整网执行**，
    命令流提交次数远超隔离扫描 → 开发板 GPU HANG 风险。必须分批（`--global-limit`）+
    `timeout` + `gpu_guard` + HANG 日志检查（`docs/benchmark_protocol.md`）。
11. **[未修] 静默跳过仍会缩小候选集**：`autotuneOp` 把 build/enqueue 失败的候选吞掉
    （R37 曾因此丢 SLM 候选）。短名单只含**成功**候选，失败候选不可见。建议把「候选跳过率」
    作为 retune 告警（R42 §3.4 已建议，未落地）。
12. **[新引入易误用点] `--global` 依赖同进程短名单**：没有 `--retune`/全命中时静默空操作。
    应在短名单为空时打印显式告警（当前有 stderr 提示，但 CI/脚本易忽略）。
13. **[未修] 全局结果写回共享缓存缺乏可回退性/标记**：没有「这是全局回验结果」的 source
    标记，难以与隔离结果区分、难以 A/B 与回退。

### E. 指标 / 审计层

14. **[部分修] `refreshExpected` 反解 key 覆盖不全**（R39 补了 conv3x3，但 `#blk/#non/
    #reorder`、cat4 等仍可能漏）→ 「改了模型却看不出效果」。
15. **[未决，R39 §5.4] 内存族 roofline ceiling 仍偏低**（ratio>1），影响以 ratio 排名时的
    可信度。hard_ceiling 是唯一 capping 尺子，其 per-family 模型的偏差会直接误导「离极限多远」。

### 优先级建议

| 优先 | 项 | 理由 |
|---|---|---|
| P0 | #4 per-plan 选择存储（bake 到 plan / 计划级 cache） | 共享签名缓存的根本冲突，R38/R44 都只是绕过 |
| P0 | #6 `choiceEntry` 统一所有可调族 | 否则全局目标无法覆盖全部算子 |
| P1 | #7 数值契约纳入缓存 ABI（或收紧「逐位一致」验收） | 正确性风险 |
| P1 | #10/#11/#12 安全与可见性（分批、跳过率告警、空操作告警） | 工程可靠性 |
| P2 | #8/#9 锁频 + 交错 + top-K 复测进默认流程 | 测量质量 |
| P2 | #15 内存族 ceiling 标定 | 指标可信度 |

---

## 6. 待办

1. **把全局回验接进安全驱动**：`scripts/autotune.py --global`（分批 `--global-limit`、
   timeout、HANG 自检、锁频）。
2. **per-plan 选择存储**（缺陷 #4）：给 `TuningEntry` 加 `source="global"` 标记，或把
   plan 级选择 bake 进 plan，避免跨模型稀释。
3. **`choiceEntry` 统一**（缺陷 #6）：gemm/cat4/小算子纳入 per-node 覆盖。
4. **跳过率告警**（缺陷 #11）。
5. 锁频 + 交错 + top-K 复测进默认流程（R42 §3.3 遗留）。
6. 三模型正式 retune + `model_check`/`reuse_check` 回归（本轮只做机制验证）。
