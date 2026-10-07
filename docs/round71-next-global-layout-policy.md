# R71 下一次任务：**全局布局策略**（从"逐节点 NCHW 默认"改为"fsv16 贯穿"）

> 本文是**任务交接/待办**，不是本轮已完成的结论。前序证据见
> [`round71-1x1-depthwise-layout.md`](round71-1x1-depthwise-layout.md)（§2.4 blocked-cat4 负结果、
> §3 链感知定价、§3.4 部分链 fsv16）与 [`round71-ov-comparison.md`](round71-ov-comparison.md)。
>
> **一句话**：OV 在 1×1（1.5–2.4×）/ depthwise（2–3×）上系统性领先，根因不是 kernel 算术，
> 而是**布局决策口径**——OV 以 `b_fs_yx_fsv16` 为全局默认、只在边界转换；我们逐节点保守、
> 以 NCHW 为默认，于是**每一处"局部换成 blocked kernel"都会被 reorder 税抵消**（两次独立实测：§2.4、
> §3.4）。要让项目**系统性**胜过 OV，必须做全局布局策略，而不是继续搬 kernel。

---

## 0. 已被实验排除的路线（不要再走）

| 路线 | 结果 | 证据 |
|---|---|---|
| 单独把 cat4 换 blocked 1×1 | 逐层 2×，整网 **+7%/ +6%** | `round71-...layout.md` §2.4；`f30b33e` |
| unfuse concat→blocked（物化 concat） | **wash**（concat 0.86 vs OV 0.42） | §2.4 |
| 给 `concat4`/`copy_c`/`ew_binary` 加 fsv16 输出 | 链只部分形成，reorder 仍 29–33 趟（~1.1ms） | §3.4；`fd6e8ac` |
| 全量 retune / 布局 fixpoint | 无可靠正收益 | R46/R47/R70 |

**共同失败原因**：只要链上**任一节点**或**任一分叉消费者**仍是 NCHW，整条链就退回 reorder；
且默认 **min-cut 求解器只把有 `#blk/#non` 的计算节点当变量**，小算子/别名不在变量集里，
其在 `planBlockedLayout` 里的标记会被求解器结果覆盖。

---

## 1. 目标（验收口径）

* **主指标**：`INFVINO_CAT4_BLK=1`（blocked-cat4）在 y8/y11 上 **由负转正**：
  busy **< gemm 路径**，且 `reorder_calls_per_frame` 从 ~30 降到 **≤ 5**。
* **副指标**：`depthwise` 的 blocked 采用率上升、y11 depthwise 分项下降（当前 0.473ms vs OV ~0.15）。
* **硬门**：三模型 `model_check` PASS（阈值内）、`reuse_check` PASS、`ctest` 6/6；
  默认路径不得回归；锁频 + 安全容器 + 跑后 `gpu_guard` 无 HANG。
* **期望**：y8/y11 conv1x1 **各 −1~2 ms**（把 gemm 的 2.96/3.86ms 压到 ~1.3ms），
  即端到端 net −5~10%；mb 中性或小正。

---

## 2. 要做的事（三步，建议按序；每步独立 A/B + 数值门）

### 步骤 1 — 把**布局求解器变量集**扩到有布局自由度的小算子（**先做这个**）
**现状**：`PlanModel::resolveLayoutMinCut`（`src/PlanModel.cpp` ~1899）的 `nodes` 只收
`alt[ni].has`（即缓存里有 `#blk/#non`）的节点；生产 loop 又要求 `blkFamilyOf(ni)->layout.canOutFsv16`。
`copy_c/ew_binary/concat4/resize/maxpool/slice` 没有 `#blk`，所以不在变量集，标记被覆盖。

**改法**：
1. 给这些 ops 的在 `KernelFamilies.cpp` 里声明的布局契约（`KernelFamily::layout`）补 `canOutFsv16`
   （已就位：`copy_c_fsv16`、`ew_binary_fsv16`、`small_concat4`；待补 `resize_nn/maxpool/slice_axis`）。
2. 在 `resolveLayoutMinCut` 里让变量集包含这些「布局可传播的 pass-through 节点」：
   它们**没有 blk/non 两个 kernel**，只有「传播 fsv16 或不传播」这一自由度——
   建模成**变量之间/变量与节点代价的边**（unary/attraction），而不是节点自身的 unary。
   具体：为 `copy_c_fsv16/ew_binary_fsv16/concat4(outer==1)` 增加
   * 输入变量 = 输出变量 的**无穷吸引项**（别名/同布局约束）；
   * 但 `ew_binary_fsv16` 是二输入，需**两个输入都=f才能输出=f**（三阶项 → 用辅助变量或
     保守地只在两输入都已 f 时允许，按 `planBlockedLayout` 的现有守护处理）。
3. 保留 `planBlockedLayout` 作为 fallback（`INFVINO_NO_LAYOUT_MINCUT=1`）。

**锚点**：
* 变量/边构造：`resolveLayoutMinCut`（~1899–2100，代价表 `f(x_in,x_out)` ~2023）。
* `BinaryEnergy`（`LayoutSolver`，见 `tests/test_layout_solver.cpp`）已有 unary/吸引项 API。
* 现有 pass-through 特判：`planBlockedLayout` 里 `copy_c_fsv16/ew_binary_fsv16` 的「所有输入已 fsv16」守护。

### 步骤 2 — 多消费者 + **反向 reorder（FSV16→NCHW）** + 去硬 pin（**本项的核心**）
**现状**：`resolveLayoutMinCut` 的 `consumerCanReadFsv16`（~1972）与 `planBlockedLayout` 的
`all_blk` 检查：只要有**一个**消费者不能读 fsv16，就把张量 **pin 死 NCHW**（`en.fix(v,0)`，~2019）。
这是分叉张量永远进不了 fsv16 的根因。

**改法**：
1. **去掉硬 pin**：改为「该张量若 fsv16，则每个不能读 fsv16 的消费者要付一趟**反向 reorder**」。
   在二元 min-cut 里这是对变量 `x_t` 的 **unary 代价** `r_rev(t)·x_t`（恒 submodular）。
   需要 `r_rev`：先加 kernel/NCHW 双向 reorder 的实测（近似 NCHW→FSV16 同价，或补标定）。
2. **补反向 reorder kernel**：现只有 `reorder_bfyx_to_fsv16(_slm)`；需 `reorder_fsv16_to_bfyx`
   （index 逆变，~20 行，加到 `kernels/conv_blk.cl`，复用 `ReorderPlan` 的 gws/lws 约定）。
3. **运行时插入**：`run()` 里当某张量 `T_.fsv16==true` 且某消费者需要 NCHW 时，
   为它建一个 fsv16→NCHW scratch（与现有 `blkInput` 前向 reorder 对称；一次/帧，按张量缓存），
   消费者读 scratch。注意：消费者可能很多小算子 → 需要统一「输入访问器」而不是逐 op 改。
4. 计费进 `layoutModelScore`/min-cut 节点代价，保证 A/B 可判。

**这是 1×1/depthwise 能否翻正的关键**；不做这条，步骤 1、3 都会被分叉处的 reorder 抵消。

### 步骤 3 — **fsv16 默认策略** + 边界转换
**改法**：把「默认 NCHW、按需 fsv16」反过来：
* 计算图内部默认 fsv16；只在 (a) 网络输入/输出、(b) 无法 fsv16 的算子（如某些 small op、
  强制 NCHW 的 resize 边界）、(c) 无法/不值得转换处，插入前向/反向 reorder。
* 用步骤 1+2 的求解器**全局**选「哪些边界转换」，而不是逐节点。
* 目标：`reorder_calls_per_frame` 趋近 OV（y8 ~3 个 f32 reorder，见 `round71-ov-comparison.md` §3）。

---

## 3. 已就位的积木（下一轮直接复用，勿重造）

| 积木 | 位置 | 状态 |
|---|---|---|
| 链感知定价（选 blk 令单消费者生产者直写 fsv16） | `PlanModel::resolveLayoutChoices` R71 段 | 已落地（默认中性） |
| `concat4` 直写 fsv16（`-DOUT_FSV16`，`outer==1`） | `kernels/ops.cl` + `KernelFamilies` | 已落地 |
| `copy_c` fsv16 视图（字节区间不变，零改动） | `copy_c_fsv16` 契约 + `nodeFamily` | 已落地 |
| `ew_binary` fsv16（同布局逐元素线性） | `ew_binary_fsv16` 契约 | 已落地 |
| 融合 blocked-cat4（`-DCAT4=1` + 4 源 + 多槽） | `kernels/conv1x1_blk.cl` + `conv1x1_cat4_blk` 族 | 已落地（opt-in，负结果，见 §2.4） |
| 布局契约/求解器/不动点 | `KernelFamily::layout`、`resolveLayoutMinCut`、`LayoutAlt` | 既有 |
| OV 对照工具（逐层 profiling + shape join） | `scripts/ov_layer_profile.py`、`scripts/ov_compare.py` | 已落地 |

---

## 4. 关键洞察备忘（下轮省时间）

* `copy_c`（`dst_off=0`）连续通道切片的字节区间在 **NCHW 与 FSV16 下相同**（`(c0/16)·HW·16 == c0·HW`），
  只要 `c0%16==0 && cnt%16==0`。→ C2f `Split` 别名天然可做 fsv16 视图，**不需要改别名 offset/length**。
* 逐元素 `ew_binary`（同 shape、非广播）在**所有操作数同布局**时与布局无关（线性映射相同）→ 无需 kernel 改动。
* cat4 的源构成（y8）：`Split`(copy_c) + `add`(ew_binary) + **`Resize`**（NCHW-only，是残留 reorder 大户）。
* 默认 min-cut 只把「有 `#blk/#non`」的计算节点当变量；小算子必须在求解器里被**显式**建成变量/边，
  否则 `planBlockedLayout` 的标记会被覆盖（本轮踩过）。

---

## 5. 风险与安全

* 触及 dispatch 数值路径与布局分配 → **必须**每步 `model_check` + `reuse_check`（R51/R52 的越界
  bug 就是这类改动），并保留 `INFVINO_NO_*` 回退开关。
* GPU 侧全程遵守 `docs/benchmark_protocol.md`（锁频、限内存、逐命令 timeout、跑后 `gpu_guard`）。
* 反向 reorder 的 scratch 生命周期并入激活池（生存期并集），别漏（R-P0 首版数值 FAIL 的根因）。
