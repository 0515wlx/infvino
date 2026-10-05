# R49：布局一等公民 —— 精确最小割试点（conv1x1）

> 背景：R44–R48 反复撞到「逐节点改善不可加 / 整网噪声主导全局选择」的根本问题。判读是
> **软件耦合（上帝对象）与硬件耦合（L3/GRF/SLM/带宽）应分开收容**：前者靠分面（决策/会计/
> 执行），后者靠一个把「共享资源」显式化的可加成本模型。本轮是统一模型的**第一个垂直
> 试点**：把**布局**提升为系统层的一等公民，用**精确最小割**做全局布局标注。
>
> 本文件先给首轮试点（§0–§8），再给按用户决策续做的**步骤 1–3**（§9）。

---

## 0. 结论先给

- **求解器正确性有保证**：`LayoutSolver`（Dinic s-t 最小割）对二元布局能量给出**精确最优**，
  并与穷举暴力解逐例一致（`tuning_test` 新增 7 项，含一般二元代价表分解的**无损性**验证、
  非 submodular 表拒绝、链耦合最优）。
- **试点命中预期收益**：mobilenetv3-small（布局耦合最强）**busy −8~9%（1.60→1.47 ms），
  交错 6/6 全胜**；数值 `model_check` **PASS**（mean_rel 与基线**逐位相同**）、`reuse_check`
  **PASS**。持久 fsv16 张量 **9→30**。
- **但也暴露了模型/目标的真实缺口（本轮最重要的产出）**：
  1. yolo 回归 **+2~3%**——因为 mincut 的目标仍是**隔离 ms**，而 yolo 的大 spatial 1×1
     最优族是 NCHW GEMM（R48 §6.2）；模型「看不到」这一点，过度持久化。
  2. 模型里必须显式包含**生产者直写能力**（D4 的 `ew_binary_ch -DEWCH_OUT_FSV16`），否则会
     把可持久化的张量误判为需 reorder——第一版就在此翻车（详见 §4）。
- **落点**：opt-in（`INFVINO_LAYOUT_MINCUT=1`，默认关）。`config/tuning.json` 未改。

---

## 1. 为什么是「布局作为一等公民」

按用户定的分层原则：

| 层级 | 标尺 | layout 的地位 |
|---|---|---|
| kernel 层 | `ops/EU/cyc`（离 hardCeiling 多远） | **不是**一等公民；kernel 优化才是 |
| 算子族 / 整图系统层 | `ms`（推理目标） | **是一等公民**；决定全局 |

只有当族的覆盖足够广、kernel 在任意 layout 下都接近最优时，layout 才配决定性地位。本轮
**假设 layout 已成为一等公民**（先解耦，代价用 warning 溯源，不静默、不惩罚），在 conv1x1
一族上验证。

---

## 2. 问题形式（为什么是 min-cut）

每个激活张量一个二元变量 `x_t ∈ {NCHW, FSV16}`（约定 0=NCHW，1=FSV16）。总代价：

```
E(x) = Σ_t unary[t][x_t]  +  Σ_edges w·[x_prod ≠ x_cons]
```

- 节点私有：某节点走哪个 kernel 由**输入张量布局**决定（输入 NCHW→non，输入 FSV16→blk），
  并入其输入变量的 unary。
- 张量共享：一张量取 NCHW 时，若被 blk 消费者读则付**一趟** reorder（整张量一次，非每消费者）；
  取 FSV16 时付**生产侧代价**（能直写=0，否则=∞）。
- 成对项 `w ≥ 0`（同布局免费、异布局付代价）→ **吸引式 → submodular → s-t 最小割可精确求解**。

这就是 OV `layout_optimizer` 的能量函数；区别是 **OV 用局部贪心近似，infvino 用精确最小割**
（`include/infvino/LayoutSolver.hpp` + `src/LayoutSolver.cpp`，Dinic 最大流 + 二元代价表分解）。

---

## 3. 借鉴 OV（源码级）

读 OV `select_preferred_formats.cpp` / `reorder_inputs.cpp` / `remove_redundant_reorders.cpp`：

| OV 阶段 | infvino 对应 |
|---|---|
| ① `ImplementationManager::query_formats()` 声明 per-port 偏好 format（可 `any`） | `KernelFamily` 的布局契约（后续应升级为 per-port） |
| ② `propagate_formats` 带门传播（刚性偏好/不支持/可融合） | 最小割的成对项 + pin |
| ③ `minimize_local_reorders`：代价 = `(#reorder, Σbytes)`，先比个数 | 本轮目标函数采纳同一顺序（见 §6） |
| ④ `insert_reorders` + `remove_redundant_reorders`：插 reorder 并**融进生产者** | 对应 D4「生产者直写 fsv16」；本轮把它建成生产侧代价 |

**判断**：OV 的 ③ 是局部贪心（能量同布局 0 / 异布局 w，submodular）；我们可做精确解。

---

## 4. 实现与两次真实的翻车（倒查）

落点：
- `include/infvino/LayoutSolver.hpp` / `src/LayoutSolver.cpp`：二元能量 + 精确最小割。
- `PlanModel::resolveLayoutMinCut(alt)`：conv1x1 试点；`resolveLayoutChoices()` 里
  `INFVINO_LAYOUT_MINCUT=1` 时接管（成功后 `mincut_active_` 让 `planBlockedLayout` 不再覆盖）。
- `tuning_test`：最小割 vs 穷举。

### 4.1 翻车 1：reorder 被**每消费者重复计费** → 永远不选 blk

第一版把节点代价表写成 `f(x_in,x_out)` 并把一趟 reorder 加进每个 blk 消费者（`f10=f11=eB+r`）。
结果：几乎全选 NCHW，busy **1.60→2.45（更差）**，fsv16 只剩 4。
**根因**：reorder 是**张量级、一趟**的，不是每消费者一趟。修正为「张量取 NCHW 时记一趟 r
（unary on tensor）」。

### 4.2 翻车 2：漏掉**生产者直写能力** → 可持久化的张量被误判

修正 reorder 计费后仍选 NCHW：`Cout16_N3136_Cin16` 的输入 `Mul_output_0` 由 SE `Mul`
（`ew_binary_ch`）产生，D4 早就能 `-DEWCH_OUT_FSV16` **直接写 fsv16**（reorder=0），但第一版
的 pin 逻辑把所有「生产者非 conv1x1」的张量钉死 NCHW → reorder 白付。
**修正**：新增**生产侧代价** `producerCap(t)`：本族 blk 生产者或 `ew_binary_ch`（通道广播）→ 0；
NCHW-only 生产者（conv3x3 native/ov、gemm、pool…）→ ∞。加入后 fsv16 **4→30**，busy **1.60→1.47**。

> 这两次翻车正是 R48 §10.5 记的教训：**布局契约是系统最关键的隐式接口，缺一个字段就静默
> 全局损失**。本轮把它显式建成「生产侧代价」，并纳入最小割。

---

## 5. 实测

锁频非全程（机制验证口径）；`kernel_run --report --iters 15`，交错。

| 模型 | base busy (median) | mincut busy (median) | Δ | fsv16 | 数值 |
|---|---:|---:|---:|---:|---|
| mobilenetv3-small | 1.60 | **1.47** | **−8~9%（6/6 全胜）** | 9→30 | PASS（逐位一致）|
| yolov8n-pose | 10.54 | 10.85 | **+3%** | — | — |
| yolo11n-pose | 11.48 | 11.87 | **+3%** | — | — |

`model_check`（mb）mean_rel=1.069e-2 **与基线逐位相同**；`reuse_check` PASS；
`tuning_test` PASS（含 7 项最小割自检）。

---

## 6. 倒查：本轮暴露的设计缺口

| # | 缺口 | 证据 | 方向 |
|---|---|---|---|
| **A** | 目标仍是**隔离 ms**（`#blk/#non.ms`），不反映 in-curve 真实代价 | yolo 大 spatial 1×1 最优是 NCHW GEMM（R48 §6.2），模型却因隔离 blk 更快而过持久化 → +3% | 引入 in-curve 校准项；或用 R47 的 L3/占用模型预筛 |
| **B** | **生产侧能力是硬编码的**（本族 blk / ew_binary_ch），不是注册表声明 | 本轮为 D4 手工加分支 | 把「能否直写 FSV16 + 代价」做成 `KernelFamily` 的生产契约（对齐 §3 的 `PortLayout.canProduce`） |
| **C** | pin 规则仍是启发式（生产者非本族即钉死），未用声明 | 4.2 的直接原因 | 布局契约声明化后，pin 由「不能产 fsv16」自然导出 |
| **D** | 只覆盖 conv1x1 一族；conv3x3/depthwise 未纳入 | 本轮范围 | 家族扩展需先补 B/C（否则契约不全）|
| **E** | **可加性只到线性**：未含 L3/占用非线性 | 用户定的「先线性、保可加」 | 时机与收益明确后再上 |
| **F** | 隔离口径 vs 稳态口径仍错配 | yolo 回归 | 这是 R44–R48 的老问题；本轮的贡献是**把布局那一半从噪声里拿出来**，剩下 L3 那一半 |

---

## 7. 判据与下一步（供决策）

**试点达到了目的**：求解器**精确且可离线自检**；在强布局耦合模型上**拿到 −9%**；并暴露出
缺口 A/B/C（模型层）——这正是「先解耦、再统一」的价值。

建议顺序：
1. **B/C（布局契约声明化）**：把 per-port `pref/firm/canProduce/fusable` + 生产侧代价搬进
   `KernelFamily`，让 §4.2 的硬编码消失。这是「加族只加声明」的兑现。
2. **A（目标校准）**：先只把**隔离口径**用于候选生成，接受/拒绝用**外部稳态 A/B**；或让
   目标纳一个从 R47 L3 模型来的 in-curve 项。⚠️ 优先保住「不在噪声上下结论」。
3. **D（族扩展）**：契约齐了再把 depthwise（也是布局耦合族）纳入，验证「多族联合标注」。
4. **E**：occupancy 能精确标定后才上非线性。

**明确的负结果**：在 yolo 上，当前隔离目标会过度持久化 → 默认关；**不能用它替换现有
conv3x3/depthwise 决策**，除非先解决 A。

---

## 8. 复现

```bash
# 离线自检（无 GPU）
docker run --rm --memory=2g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake --build build-ct -j4 --target tuning_test && \
  LD_LIBRARY_PATH=$PWD/build-ct ./build-ct/tuning_test'   # 见 R49 七项

# 整网 A/B（mb，mincut opt-in）
export INFVINO_PROGRAM_CACHE=/tmp/pc
./build-ct/kernel_run              --plan models/mobilenetv3-small/model.plan --report --iters 15
INFVINO_LAYOUT_MINCUT=1 INFVINO_LAYOUT_REPORT=1 ./build-ct/kernel_run \
  --plan models/mobilenetv3-small/model.plan --report --iters 15   # [layout] mincut: ...

# 数值/复用
INFVINO_LAYOUT_MINCUT=1 python3 scripts/model_check.py --model mobilenetv3-small --repo "$PWD"
INFVINO_LAYOUT_MINCUT=1 python3 scripts/reuse_check.py --model mobilenetv3-small --repo "$PWD"
```

---

## 9. 步骤 1–3（用户决策后本轮续做）

> 用户决策：**精确 min-cut**；布局契约表达力够用（不必更细）；证据链粒度 = **(族,shape,layout)**；
> **先一族试点**，看收益 + 回头分析设计。随后按 findings §7 的顺序做 1→3。

### 9.1 步骤 1 —— 布局契约声明化（去硬编码）

首轮 `resolveLayoutMinCut` 里有两处硬编码：`producerCap` 用 `p.op=="conv1x1"` / `p.op=="ew_binary"`
判生产者能否直写 fsv16；pin 规则也没查**消费者**能力（潜在正确性风险）。步骤 1 全部改为
**注册表驱动**：

- `Candidate.canOutFsv16`（新字段）：候选能否**直接产出** fsv16。`candidatesFromRegistry` 从
  族契约 `KernelFamily.layout.canOutFsv16` 填充；`smallCandidates` 给 `ew_binary_ch`（D4）单独置位。
- `resolveLayoutMinCut` 用 `familyByName(kernel)` / `candidatesFromRegistry(sig)` 判：
  - `producerCanFsv16(t)`：生产者是 mincut 节点→看其 blk 族 `canOutFsv16`；否则看**实际选中**
    kernel 的 `canOutFsv16`。
  - `consumerCanReadFsv16(t)`：每个消费者要么是 mincut 节点（读激活槽），要么其**选中 kernel**
    的族 `layout.in==FSV16` 且 `inIndex` 匹配；否则 pin NCHW。
  - 激活槽号 `actSlot` 取自 blk 族的 `layout.inIndex`（conv1x1=1、conv3x3/depthwise=0），
    不再写死。
- **修正代价表**：首轮「input-only unary + 每张量 reorder」会重复计费/强制 fsv16；改为正确的
  二元表 `f(x_in,x_out)`：`f00=min(eN,eB+r), f01=eB+r, f10=eB, f11=eB`（submodular，见 §2）。

### 9.2 步骤 2 —— 目标校准：隔离提案 + 稳态验收门

隔离 ms **只用于生成布局提案**；接受/拒绝以**整网交错 median** 为准：

- baseline（联合不动点）先算好；mincut 只提出 proposal。
- `INFVINO_LAYOUT_MINCUT_GATE=1` 且 profiling 可用时：交错测 baseline vs proposal 各 3 组、
  每组 median，只有 **proposal median < baseline median × 0.99** 才接受，否则**回退**。
- 这是 R47「整赋值验收」用在布局维度：**不在噪声上下结论**（R44–R48 的老坑）。

### 9.3 步骤 3 —— 族扩展（depthwise）

mincut 节点集从 `op=="conv1x1"` 放宽到**任何声明了布局契约的族**（有 `#blk/#non` 且 blk 族
`canOutFsv16`）：现覆盖 **conv1x1 + depthwise**（mobilenet 的 `1x1→dw→1x1` 布局耦合链）。
`conv3x3` 暂排除（`conv3x3_blk` 结构性弱于 `ov`，缺口 C），其张量由 producer/consumer 能力门
自动 pin NCHW。

### 9.4 步骤 1–3 实测（锁频非全程；交错）

| 模型 | base busy | mincut(无门) | mincut(有门) | 门裁决 |
|---|---:|---:|---:|---|
| mobilenetv3-small | 1.58 | 1.38 | **1.35** | **ACCEPT**（median 1.60→1.40）|
| yolo11n-pose | 11.49 | 11.37 | **11.35** | **ACCEPT**（median 11.57→11.33）|
| yolov8n-pose | 10.52 | 10.75 | 10.52（回退） | **REJECT**（median 10.50→10.52）|

- 数值：mb `mean_rel=1.069e-2`、y11 `7.91e-4`，**PASS**；`tuning_test` PASS。
- **门按设计工作**：mb/y11 接受，y8 因隔离提案更差而被拒、回退到 baseline → **无回归**。
- 结论：布局 mincut 在**布局耦合型模型**上是确定性增益（mb −14%、y11 −1.5%）；对 yolo 的
  大 spatial 1×1（最优族是 NCHW GEMM）由门兜底。

### 9.5 剩余缺口（下一步）

1. **目标仍是隔离 ms**（缺口 A）：门能防回归，但「提案质量」仍受隔离口径限制（y8 提案为
   全 NCHW）。根治需把 R47 的 L3/占用**会计**接进节点代价（线性、可加）——见 §10（本轮
   尝试后的结论：**逐节点 L3 是负结果**，正确形态是整网模拟/评分器）。
2. **生产侧/布局契约仍是族级**：`Candidate.canOutFsv16` 是候选级声明，但 per-port
   `pref/firm/fusable` 尚未全部进 `KernelFamily`（当前够用，见用户决策 2）。
3. **conv3x3 未纳入**（缺口 C）：需先让 `conv3x3_blk` 在契约上不弱于 ov，或明确放弃其 blocked 链。
4. **门需要 profiling + 构造期多次整网执行**：生产应离线跑一次（kernel_autotune）后 bake 成
   per-plan 工件，而非运行时每次自检。

---

## 10. 缺口 1：把 R47 的 L3/占用会计接进节点代价（**负结果**）

> 目标：让 mincut 的节点代价「看见」整网 L3 占用，而非只用隔离 ms。

**已做（可复用，保留）**：
- 把 `globalRetune` 里的局部 `occupancyPressure` 抽成**单一真相源** `infvino::occupancyPressure`
  （`Tuning.cpp`）+ `kL3SpillPerByteMs`；`globalRetune` 的 L3 模拟与 mincut 共用，杜绝口径漂移。
- mincut 节点代价加一项 `occupancyPressure(e,s) × kL3SpillPerByteMs`，**线性、可加**。

**实测（锁频非全程，交错）**：

| 模型 | base | mincut(默认) | mincut + 逐节点 L3 | 门裁决 |
|---|---:|---:|---:|---|
| mobilenet | 1.59 | **1.34** | 1.55（回退） | L3 提案 median 1.57→**2.46** → REJECT |
| yolo11n | 11.48 | **11.33** | 11.51（回退） | REJECT |
| yolov8n | 10.52 | 10.55 | 10.61（回退） | REJECT |

**结论（缺口 1 的答案）**：**逐节点 L3 项破坏可加性/量级校准，是负结果**。
- `occupancyPressure` 是**总并发足迹**，不是 **L3 miss 字节**；把它当节点成本会压过隔离 ms、
  把选择推向**过度持久化**（mb 提案 1.57→2.46 ms，比 baseline 差 57%）。
- R47 的正确形态是**整网 LRU 模拟（顺序相关）**，它**不能**塞进成对 min-cut 的节点项（会二次
  计费、破坏可加性——R47 §5.2 已警告过这一点）。
- 因此该**默认关**（`INFVINO_LAYOUT_L3=1` 仅作消融）；mincut 维持「隔离提案 + 稳态门」。
- **正确的下一步**：把 L3 会计做成**整网评分器**（对 baseline/proposal 各跑一次 R47 `predictNet`
  的 LRU 模拟，零 GPU），作为**提案预筛**，而不是节点代价；或对逐节点项做**独立标定**（用
  受控微基准拟合系数），再评估是否值得进 mincut。

> 附带修复一个真 bug：验收门回退（REJECT）后**没有 `invalidateCapture()`** → 回退后仍重放
> mincut 的录制 dispatch，导致「门已拒绝但运行仍是坏布局」（L3 实验时 mb 显示 2.44 ms）。
> 已在 `restore()` 里补 `invalidateCapture()`。
