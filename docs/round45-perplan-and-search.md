# Round 45：P0–P2 落地（per-plan 覆盖 / choiceEntry 统一 / 数值契约 / 噪声与安全）+ 整网回验与 offline autotune 探索

> 承接 R44 的体系缺陷倒查：按 P0→P1→P2 顺序落地，随后探索「整网 busy 回验」在
> **组合爆炸 + DRAM/LLC 噪声**下如何收敛，以及把**整网调优离线化**（缩小搜索空间 /
> 加快回退重搜 / 与运行时融合）。
>
> 结果：P0/P1/P2 已落地并提交（3 个 commit）；整网搜索加入 top-K∪每族代表、噪声地板、
> 预算；offline 工件（per-plan 覆盖）与运行时融合。轻量 GPU 自验通过。

---

## 1. P0 —— 表示层（最深的两个漏洞）

### 1.1 #6 `choiceEntry` 统一所有可调族

`run()` 里原先只有 `conv3x3/conv1x1/depthwise` 消费 per-node 覆盖（`choiceEntry`），
`gemm/conv1x1_cat4/小算子` 直接读 `tuning_.lookup`。后果：全局回验覆盖不全、per-plan 决策
无法统一施加。**已改**为全部走 `choiceEntry`（per-node 覆盖优先，回退签名缓存）。

- 行为中性：`resolveLayoutChoices` 只为 `#blk/#non` 签名填 `node_choice_`，其余回退 lookup。
- 验证：mb `kernel_run` busy=1.693ms（与 R43 基线一致）。
- 副作用：`globalRetune` 解除「只回验三族」限制，对全部可调算子生效。

### 1.2 #4 per-plan 选择覆盖（位置相关最优 vs 共享缓存）

同一 shape 在 y8/y11 的全局最优可能不同（邻居/持久化不同），而 `tuning.json` 是签名级、
跨模型共享的——一份「选谁」无法两全（R38 只解决了布局维）。

**已加** `PlanModel::plan_overrides_`（`TuningCache`，key = 节点输出张量名，唯一）：

- 加载：`INFVINO_PLAN_TUNING`，否则 `<plan>.tuning.json`；`INFVINO_TUNING=off` 一并禁用。
- `choiceEntry` **最高优先**消费它；`config/tuning.json` 退化为「可移植的隔离默认」。
- `globalRetune` 选出后把本图 per-node 选择写入，`kernel_autotune --global` 默认落盘
  `<plan>.tuning.json`（可用 `--plan-tuning` 指定）。
- 验证：mb 全局选 `conv1x1_blk` → 烘进 plan 后 runtime 生效（`INFVINO_DEBUG_BLK` 计数 1 vs 0）。

> 这让「全局最优」成为**计划级工件**：换模型/回退只需换/删工件，不会互相覆盖。

---

## 2. P1 —— 数值契约与可见性

### 2.1 #7 kernel 源指纹守卫

`cache_abi` 只在 `-D` 宏语义变化时手动 bump；但 **kernel 源码的数值/结构变化**（fp16 累加
顺序、SLM 分块）不体现在 key——R39 的 SLM 数值 bug 就是「缓存不知道 kernel 已改」。

**已加**：`ClRuntime::sourcesHash()`（对缓存涉及的 `.cl` 源做 FNV-1a）；`TuningCache`
记录 `kernel_src_hash`。加载时：

- 默认：源指纹变化 → 告警 `options may be stale; retune recommended`。
- `INFVINO_TUNING_STRICT=1`：整份作废（回退启发式，强制 retune）。

验证：伪造 `kernel_src_hash=deadbeef…` 的缓存 → 正确告警 `deadbeefdeadbeef -> 099f…`。

### 2.2 #11 候选静默跳过告警

R37 曾因 `makeEnqueue` 异常被静默吞掉而丢掉 SLM 候选。**已加**：`autotuneOp` 统计跳过数，
`skipped*5 >= 候选数`（>20%）时告警，提示「候选集可能被静默缩小」。

### 2.3 #10 安全驱动

`scripts/autotune.py` 新增 `--global`（透传 `--global-*`）、`--lock`（跑前
`gpu_clocks.sh lock`、跑后 unlock）；每批仍是独立进程 + `timeout` + HANG 日志自检。
整网回验的命令流远多于隔离扫描，故这批安全措施必需。

> 未做（记录）：#12 `--global` 空操作显式告警仍只是 stderr 提示，CI 需自行 grep。

---

## 3. P2 —— 噪声治理与搜索预算

### 3.1 #8 隔离扫描 top-K 复测（抗热漂移）

顺序扫描会因热漂移系统性偏向先测的（R42 §3.3）。**已改**：`autotuneOp` 在主扫描后对
隔离 **top-3 复测一次**（取两次 min），并保留胜者的离散度告警。

### 3.2 整网回验的搜索改进

`globalRetune` 新增/修正：

| 机制 | 说明 |
|---|---|
| **短名单 = 隔离 top-K ∪ 每族最优** | 隔离 top-K 可能整体漏掉某个族（R43 的 `conv1x1_blk` 因 reorder 在隔离期落后）；每族保一个代表，保证「整网翻盘」的族始终在搜索空间里。额外族数封顶 4 |
| **噪声地板** | 接受改善需 > 1%（`noise_check` 典型 ~1.5%）；采样极差**只告警**。⚠️ 用极差当阈值会把真实 2–3% 改善也拒掉（实测），故用固定地板 |
| **margin 剪枝** | `--global-margin F`：剪掉 `iso_ms > best*(1+F)`。**默认 0（关）**——实测 F=0.03 就剪掉了流水线更快的 blk，重蹈局部最优 |
| **总预算** | `--global-budget N`：整网测量次数上限，耗尽即停（抗组合爆炸 / GPU 风险） |

> 关键教训：**任何基于隔离名次的剪枝都可能剪掉「隔离慢、流水线快」的候选**——这正是 R44
> 要修的目标函数错位。因此 margin 默认关，改为「top-K ∪ 每族代表」这种**保底**式剪枝。

### 3.3 #15 内存族 ceiling（未做，需实测）

内存族 roofline `expected` 仍偏低（ratio>1，R39 §5.4）。操作上 `analyze_budget.py` 已按
ratio≥1 → 0 headroom 处理；**真正的重标定需要 OA/MD API 的字节计数器**（本机
`CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS` 未开，不可用）。记为待办，不做无数据猜测。

---

## 4. 探索 A：整网回验为什么难收敛，怎么改

### 4.1 三个困难

1. **组合爆炸**：N 个可调签名 × K 候选 × rounds × reps 次整网执行。per-signature 坐标
   下降只是近似，联合最优可能在非坐标方向。
2. **噪声**：候选对邻居是**有序影响**（不是 R42 的「外部干扰只加时间」），min over 少量
   reps 仍会被 L3/DRAM 断崖的慢尾误导。
3. **非独立/自指**：`blk` 的代价依赖输入是否持久化，持久化又依赖邻居选什么（R38 的不动点）
   ——单点移动会「付 reorder、失败、回退」，坐标下降会卡住。

4. **隔离名次与整网名次不一致**（R43/R45 实测）：所以任何隔离剪枝都要保底（每族代表）。

### 4.2 已实现（本轮）

- top-K ∪ 每族代表（§3.2）
- 噪声地板 + 离散度告警
- 总预算 + 分批 + 安全驱动
- min-of-reps（R42 既有）+ top-K 复测（§3.1）

### 4.3 尚未实现（按 ROI 排序，附设计）

1. **分组移动（blocked chain）**：把「同布局链」的若干签名作为一个 move 一起改
   （例如一条 `conv1x1_blk→depthwise_blk→conv1x1_blk` 链）。只有整链一起切 blk，才会
   让 fsv16 持久化、reorder 归零；单独切一个必失败。**实现**：从 `planBlockedLayout`
   的消费者图提取「候选含布局族的连通分量」，坐标下降时按分量移动。
2. **拓扑感知的优先级**：先回验「邻居多、工作集跨 L3 边界、族争议大」的签名（用固定
   预算优先投给最可能翻盘者）。**实现**：`priority = (iso_margin小) × (邻居数) × (工作集/3.75MB近)`。
3. **代理/干扰模型**：用候选的 `-DPROBE` 剖面（noIn/noW/noStore）估计它对邻居的外溢，
   只在 top-2 里用真实整网回验。降低测量次数。
4. **确认轮（confirmation round）**：坐标下降收敛后，对最终 assignment 整体再测若干次，
   若与过程中记录的最优不一致则告警（说明噪声主导）。
5. **记忆化**：对「单个签名换候选」的整网 busy 做 key=(sig,candidate,其余assignment摘要)
   的缓存。因为耦合，严格有效需要完整 assignment key，退而求其次缓存 baseline busy。

---

## 5. 探索 B：offline 整网 autotune（缩小搜索 / 加快回退 / 融合）

### 5.1 分层与产物

```
离线（开发机，可长跑）                         部署（开发板，短/安全）
┌───────────────────────────────┐            ┌───────────────────────────────┐
│ 1) 隔离扫描 autotuneOp          │            │ kernel_run 直接加载：          │
│    → config/tuning.json         │  产物      │  - config/tuning.json（隔离默认）│
│      + 每签名隔离 top-K∪每族     │ ───────▶   │  - <plan>.tuning.json（本图全局）│
│ 2) 整网回验 globalRetune        │  per-plan  │   choiceEntry: per-plan > fixpoint│
│    → <plan>.tuning.json          │            │ 灵活：删工件=回退到隔离默认      │
│ 3) 校验 model_check/reuse_check  │            └───────────────────────────────┘
└───────────────────────────────┘
```

- **缩小搜索空间**：隔离扫描先给出每签名候选；整网只回验「top-K ∪ 每族代表」。
  可再加 margin（默认关）与 `--global-budget`。
- **加快回退重搜**：per-plan 工件就是「搜索结论」。回退 = 删/换工件，运行时立刻退回
  隔离默认，**无需重搜**；要重搜时也只需重跑**变化过的 op**（`--op`/`--only`）。
- **二者融合**：`config/tuning.json`（可移植隔离默认）+ `<plan>.tuning.json`（本图全局）
  = 两级缓存；`choiceEntry` 的优先级把「全局」叠加在「默认」之上，运行时零感知。

### 5.2 落地用法

```bash
# 离线整网调优（安全驱动：分批 + 锁频 + HANG 自检）
python3 scripts/autotune.py --model mobilenetv3-small --ops conv1x1,depthwise \
    --batch 3 --iters 6 --global --global-topk 3 --global-iters 3 --global-rounds 2 \
    --global-budget 60 --lock

# 运行时自动加载 <plan>.tuning.json（或显式指定 / 禁用）
INFVINO_PLAN_TUNING_REPORT=1 ./build/kernel_run --plan models/... --report
INFVINO_PLAN_TUNING=none    ./build/kernel_run --plan models/... --report   # 回退到隔离默认
```

### 5.3 残留限制（诚实记录）

- per-plan 工件目前只覆盖**被整网回验过的签名**；未覆盖的仍走共享默认。
- 工件按**节点输出名**键控：plan 重新生成若改了张量名则失效（安全回退）。
- 组合最优仍非保证：坐标下降 + 保底短名单是**更好**的近似，不是全局最优证明。
- 锁频是 busy/ops 口径正确的前置；未锁频的整网结论只能做定性。

---

## 6. 复现

```bash
# 构建
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake -S /workspace/infvino -B /workspace/infvino/build-ct \
  -DCMAKE_BUILD_TYPE=Release && cmake --build /workspace/infvino/build-ct -j4'

# 离线自检（34 项）
./build-ct/tuning_test

# 单签名整网回验 + per-plan 工件
scripts/gpu_clocks.sh lock
INFVINO_GLOBAL_RETUNE_REPORT=1 ./build-ct/kernel_autotune \
  --plan models/mobilenetv3-small/model.plan --cache /tmp/c.json \
  --op conv1x1 --only 3136 --limit 1 --iters 3 --retune \
  --global --global-topk 2 --global-iters 3 --global-rounds 2 \
  --plan-tuning /tmp/mb.pt.json
INFVINO_PLAN_TUNING_REPORT=1 INFVINO_PLAN_TUNING=/tmp/mb.pt.json \
  ./build-ct/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 5
scripts/gpu_clocks.sh unlock

# 源指纹守卫
# （伪造 kernel_src_hash 的缓存 → 应见 [tuning] ... kernel_src_hash changed ...）

# 数值/复用回归
python3 scripts/model_check.py  --model mobilenetv3-small --repo $PWD --image infvino-dev:latest
python3 scripts/reuse_check.py --model mobilenetv3-small --repo $PWD --image infvino-dev:latest
```

---

## 7. R44 缺陷清单的更新状态

| # | 缺陷 | 状态 |
|---|---|---|
| 4 | 签名级共享缓存无法表达 per-plan 最优 | **已修**：per-plan 选择覆盖（§1.2） |
| 6 | `choiceEntry` 接入不统一 | **已修**（§1.1） |
| 7 | 数值契约未进缓存 ABI | **部分修**：源指纹守卫（§2.1）；`cache_abi` 仍手动 |
| 10 | 整网回验 GPU HANG 风险 | **部分修**：分批 + `--global-budget` + `gpu_clocks.sh --lock` |
| 11 | 候选静默跳过 | **已修**：跳过率告警（§2.2） |
| 12 | `--global` 空操作易误用 | **部分修**：stderr 提示；CI 仍需自行检测 |
| 8 | min 估计器在 in-situ 下前提变弱 | **部分修**：top-K 复测 + 噪声地板（§3.1/§3.2） |
| 9 | 锁频是口径前提 | **部分修**：`autotune.py --lock` |
| 15 | 内存族 ceiling 偏低 | **未修**：需 OA/MD 字节计数器（本机内核不支持） |

## 8. 下一步

1. 分组移动（blocked chain）与拓扑优先级（§4.3）。
2. 三模型正式整网 retune（锁频、分批）+ `model_check`/`reuse_check` 全回归。
3. per-plan 工件覆盖全部可调节点（当前只覆盖回验过的）。
4. 有 OA 的内核上重标定内存族 ceiling（#15）。
