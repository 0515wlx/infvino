# infvino 开放项登记表（遗留项 / 未决 / 暂缓）

> **本表的定位**：系统「未完成 / 未决 / 明确暂缓 / 已归档负结果」的**单一权威清单**。
> R1–R69 里每一项 `下一步 / 待办 / 缺口` 都散落在各自轮次文档中，反复被重开或误以为
> 「还没做」。此后**新增/关闭**开放项都应更新本表（轮次文档保留过程证据）。
>
> **状态图例**
>
> | 标记 | 含义 |
> |---|---|
> | 🔴 未做 | 已识别、可做，但尚未落地（含需要新 kernel / 引擎改动） |
> | 🟡 暂缓 | 有明确**前置条件**未满足（收益证据 / 硬件计数器 / 另一项先落地） |
> | ⚪ 负结果 | 已实测为负/中性，**保持关闭**；仅当证据变化才重开 |
> | ✅ 已完成 | 本轮（R70）或此前已闭环，仅留一行指路 |
>
> 维护约定：关闭一项时把它移入「已完成/负结果」并注明轮次；重开一项必须给出**新证据**
> （新硬件计数器 / 新候选 / 端到端 A/B），不接受「再试一次」。

---

## A. 契约 / 注册表（单一真相源）

| # | 项 | 状态 | 证据 / 说明 | 解除条件 |
|---|---|---|---|---|
| A1 | **逐节点 spill 归因**暴露到生产路径 | ✅ R70 | `L3Model` 早有 `node_evict_ms`，但只在测试可见。R70 在 `INFVINO_LAYOUT_REPORT=1` + `INFVINO_LAYOUT_SPILL=1`（或 `INFVINO_LAYOUT_DEBUG=1`）下打印 top-N 逐出归因（`PlanModel.cpp`）。诊断项，**不参与选择**。 | — |
| A2 | `KernelFamily::launch` **绑定声明化** | 🔴 | R48 §7.3-D / §10.6-B：设计稿有 `launch` 字段，实现无；新 kernel 仍需改 `PlanModel::run` / `makeEnqueue` 的参数绑定。注：`familyByName` + `KernelFamily.source` 已覆盖 kernel→源文件，`mem` 契约（R69）覆盖占用；缺的是**按族绑定 clSetKernelArg + gws/lws**（各族几何/缓冲槽差异大）。 | 需要一次专门的「dispatch 声明化」重构（中高风险、会影响三模型数值），不宜塞进收尾轮 |
| A3 | per-port `pref / firm / fusable` 进注册表 | 🔴 | R49 §9.5-2：当前够用（族级 `LayoutReq` + 候选级 `canOutFsv16`）；多容器/多后端时才需要 per-port。 | 出现「同一张量不同消费者要求不同布局」的真实需求 |
| A4 | **布局不变量 / 一致性断言**（planner 声明 ↔ dispatch 实际；小图穷举对照） | 🔴 | R48 §10.6-A/C：`inIndex` 缺失曾让 conv1x1 链永不持久化且**无告警**。已有 `poolAliasProbe` / `mayMarkFsv16` 起步（「分配补齐集合 ⊇ 标记集合」）。 | 补：每个被选 blk 节点「输入为 fsv16 **或**已计费一趟 reorder」的断言 + 小图穷举 |

---

## B. 布局 / 内存建模

| # | 项 | 状态 | 证据 / 说明 | 解除条件 |
|---|---|---|---|---|
| B1 | **完整 blocked chain（S4）**：生产者直接写 fsv16、整链传播、reorder 归零 | 🟡 | R47 §4.2：mb 的 reorder 占 busy ~8%（机会上限）。前置 = S3 决策级 chain move 在**锁频稳态 A/B** 上给出可靠正收益——R47 §4.3 实测 **chain move 从未被接受**（逐节点改善不组合 + ±5% 噪声），未满足。 | S3 复现可靠正收益；优先做「单进程 all-op 联合不动点」 |
| B2 | 整网搜索：**分组移动 / 拓扑优先级 / 代理模型 / 确认轮 / 记忆化** | 🟡 | R45 §4.3（设计已给）。S3 已做「整分量联合赋值」；其余未落地。 | 先解决「per-node in-situ 改善不可组合」这一更底层瓶颈（R47 §4.3） |
| B3 | conv3x3 的**布局链**（输出 fsv16 epilogue / 消费者 prologue 直读） | 🔴 | R50 §7、R51 §6-1、R52 §5-2：`conv3x3_blk` 已声明 `canOutFsv16` 且 `conv_blk.cl` 支持 `-DOUT_FSV16`，但缺 `#blkfsv16` 成本测量，且 R49 判其结构性弱于 `conv3x3_ov`。R41 已证 kernel 侧无空间，价值全在布局。 | 需 kernel/epilogue 改动 + 离线证明不弱于 ov；高风险 |
| B4 | 逐节点 L3 定价 / 目标函数容量项 | ⚪ | R49 §10、R55 §7.5、R67：三次负结果。正确形态是「实测可加项 + 全局非可加模拟」（R68 §1）。`INFVINO_LAYOUT_L3` / `INFVINO_LAYOUT_CAP` 保留 opt-in 作对照。 | 新证据（如候选形态变化使逐节点可加性成立） |
| B5 | 两级容量接进**内层 roofline** | 🟡 | R68 §10：`effectiveBwGbps` 已实现、**仅诊断**（R66）。并入 `expected` 不改变选择（R66 中性）。 | 出现新的内存受限候选，或拿到 OA 计数器 |
| B6 | 定量 `reuse^0.30` 接入有效容量 | 🟡 | R60 §6：NRU 已用该标定曲线；把它**定量**并进有效容量需端到端 A/B。 | 有稳态 A/B 能分辨 ~1% 级差异 |
| B7 | 候选生成按**占用 / 容量**过滤 | 🔴 | R66 §：在注册表侧排除注定内存墙的 tiling，未做。当前候选预算（R48 §3.1）只按数量截断。 | 与 A2/A3 一起做更自然 |
| B8 | mincut **per-plan 最优 bake** 成工件 | 🟡 | R49 §9.5-4 / R50 §7：门需 profiling + 构造期多次整网执行；生产应离线跑一次后 bake，而非运行时自检。 | 与 C3（per-plan 工件覆盖）合并推进 |
| B9 | 非 16 通道的 fsv16 持久化（激活按 `ceil(C/16)*16` 分配） | ✅ R51 | mb 默认 −2.7%；见 `round51-cat4-registry-and-non16-fsv16.md`。 | — |
| B10 | **blocked 链布局**：不动点需「链感知定价」——选 blk 会令单消费者生产者直写 fsv16（reorder 实为 0） | 🟡 R71 | R71 定位：不动点按 `blk+reorder` 逐节点定价、且只按**当前**是否 fsv16，导致跨族链（concat→1×1、1×1→dw）断掉、卡 NCHW 局部最优。已落地单消费者版链感知定价（`PlanModel` R71 段）；**默认路径中性**，但使 unfused cat4 链 fsv16 张量 3→21。 | 多消费者/整分量赋值（B1）、融合 blocked-cat4（B11） |
| B11 | **融合 blocked-cat4**：cat4 保留融合但走 blocked 1×1 直读 4 路 fsv16（不物化 concat） | ⚪ R71 | **已实现 + 负结果**：逐层更快（`128→128@20×20` 2.0×），但整网 y8 +7% / y11 +6%——因 4 路源钉不进持久 fsv16，每帧多 32 趟 reorder（+1.1ms）。`INFVINO_CAT4_BLK=1` 可开（默认关，见 `round71-1x1-depthwise-layout.md` §2.4）。 | 先做 B10 的多消费者/整分量链（源能持久 fsv16）后再验证 |

---

## C. 自动调优基础设施

| # | 项 | 状态 | 证据 / 说明 | 解除条件 |
|---|---|---|---|---|
| C1 | `--global` **空操作显式告警**（CI 可判） | ✅ R70 | `globalRetune` 未执行时返回 `-1`；`kernel_autotune --global` 打印 `global-retune: NO-OP` 并**退出码 3**（`INFVINO_GLOBAL_ALLOW_NOOP=1` 可容忍）；`scripts/autotune.py` 显式检测并停止（exit 4）。R44 #12 / R45 §67。 | — |
| C2 | **整网回验残差数据集**落盘 | ✅ R70 | 每次候选评估记录 `(sig, kernel, iso_pred, net_measured, accepted)`；`--residual <f>` / `INFVINO_GLOBAL_RESIDUAL=<f>` 写 JSONL（R47-tvm §7.1，学习型残差模型的训练集）。 | — |
| C3 | **per-plan 工件覆盖全部可调节点** | 🔴 | R45 §5.3：当前只覆盖被整网回验过的签名；未覆盖的走共享默认。 | 与 B8 合并：一次完整全局 retune 后 bake |
| C4 | 全量 retune 把 `#blk / #non / #reorder` 写进生产缓存以激活布局 fixpoint | ⚪ | R47 §4.3「残留」：生产缓存无 `#` 键 → fixpoint 退化。**R70 做了决定性实验**：mb retune 后确实生成 278 条 `#` 键，但交错 A/B **无可靠收益**（base 中位 1.378 vs retuned 1.390 ms），与 R46/R47 一致 → **生产缓存保持不动**。见 `round70-status.md` §3。 | 需先证明全量 retune 在三模型上**稳定正收益**（当前证据为负） |
| C5 | `globalRetune` 联合 move（S2 契约门 / S3 chain move） | ✅ R47 | S2/S3 已落地；S4 见 B1。 | — |
| C6 | 锁频 + 交错 + top-K 复测进默认流程 | 🟡 | R42 §3.3：`autotune.py --lock` 部分；「交错 + 多 rep」在 `--global` 内已用（R47）。默认流程仍非强制。 | 把 `--lock` 设为 `--global` 的默认前置 |
| C7 | 内存族 **ceiling 标定**（缺陷 #15） | 🟡 | R44/R45 §3.3、R39 §5.4：非计算族 roofline ceiling 偏低（ratio>1）。 | 需要 OA / MD 字节计数器——本机内核 `CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS` 未开（R43） |
| C8 | `choiceEntry` 统一所有可调族 | ✅ R45/R46 | R46 §3.2 补上 conv1x1 N==1（GEMV）分支。 | — |
| C9 | kernel 源指纹守卫（`cache_abi` 手动项） | 🟡 | R45 §2.1：源指纹已加；`cache_abi` 仍人工维护（R44 #7）。 | 把数值契约纳入 ABI 自动派生 |

---

## D. 算子 / 内核候选

| # | 项 | 状态 | 证据 / 说明 | 解除条件 |
|---|---|---|---|---|
| D1 | **小算子 launch 融合**（D5）：gap+fc、resize→conv 等 | 🔴 | R48 §4bis P2、R51 §6-2：SE Mul→conv1x1 prologue（`MUL_SCALE`）已试，kernel 侧逐位正确、dispatch↓，但**整网中性**（与 D4 布局转换相抵），opt-in。下一步做**不与布局转换冲突**的融合（gap+fc / resize→conv）。 | 选择不会与 fsv16 转换互相抵消的融合对 |
| D2 | D4+：gap 直读 fsv16 | 🔴 | R51 §6-2 / R52：曾给 mb −2%/−6.9%，但暴露 correctness bug（mincut/旧缓存下 `mean_rel≈0.5`），**已回退**。 | 先给 gap 的 fsv16 读做**单元级数值验证**，再重启 |
| D3 | **direct conv1x1**（窄通道 / 小 N） | 🔴 | R48 D6 / R53 §8：只在窄通道/小 N 能超过 gemm/blk 才有意义，需先离线证明。 | 离线 `ocloc` 证明存在胜出区间 |
| D4 | D2 split-K conv3x3 | 🟡 | R48 M2：R41 判 conv3x3 大层是「无 L1 + 128-GRF ILP」结构墙；gemm_sk 已有。 | 推翻 R41 结构墙结论 |
| D5 | conv3x3 `Cin=3` 全展开 27-tap（去 chunk 循环） | 🔴 | R40 §：唯一结构方向；R34 已把小 Cin 的 CINC 泛化（`conv3x3_cin3`）。 | 离线 ISA 显示指令数有可观下降 |
| D6 | epilogue **可组合化**（bias+act+residual+简单 broadcast）接 autotune | 🔴 | openvino-gap P1-1：当前融合是硬编码模式；应升级为可组合 kernel 宏 + 计划属性。 | 与 E6（view/skip 泛化）协同 |
| D7 | `cat4` 的 **blocked 变体**进 mincut 布局决策 | 🔴 | R51 §：契约层已收口（`gemm_cat4_f16` 族），但要进 (族,布局) 决策需它能输出/吃 fsv16（blocked 变体）。 | 新 kernel（P3） |
| D8 | 权重 blocked **计划化**（连续 conv 共享重排权重） | 🔴 | openvino-gap P1-3。 | — |
| D9 | depthwise padded（A 方向） | ⚪ | R31：指令 −41%、逐位一致，但 pad 带宽相抵 → 整网负结果，默认关（`INFVINO_DW_PAD=1` 才启用）。 | 更便宜的 pad 路径 |
| D10 | SPPF **launch 融合** | ⚪ | R56 D5：重读 > 省下的 3 次 launch，且挤掉 CAT4 免费融合（y8 busy +1.2%），opt-in（`INFVINO_FUSE_SPPF=1`）。 | — |
| D11 | D1 conv3x3 **输出 tiling**（`Y_BLOCK`） | ⚪/🔴 | R48 M1：`conv1x1_blk` YB 已做（整网近零）；conv3x3 YB 未做，ROI 存疑（§7.3-E）。 | 离线证明改变布局图或有非受限替代 |
| D12 | Winograd / CINC 泛化 | 🟡 | R48 M4：负预期（R41）；维持「离线可证伪」定位。 | 新的离线证据 |
| D13 | 非 blk depthwise 的输出 fsv16 / 向量 store | 🔴 | R50 §4bis：depthwise Memory/Launch 轴；小 spatial 已近上限。 | — |

---

## E. 引擎 / 运行时

| # | 项 | 状态 | 证据 / 说明 | 解除条件 |
|---|---|---|---|---|
| E1 | **降低 launch 开销**：参数缓存 + 非 profiling 去逐节点同步 | 🔴 | P2（`benchmark.md` §2.4 / R30）：busy 只占墙钟 52–74%，缺口 = 入队提交 + `setArg`/簿记。已有 `launch_cache_`（R36）覆盖部分 dispatch 录制。 | 先做最安全的「非 profiling 不逐节点同步 + 参数变化才 setArg」并量墙钟 |
| E2 | out-of-order 队列 + 事件依赖 | 🟡 | P2-3：若不打算做 OOO，至少把 event 依赖用起来，让 host 不停顿。 | 与 E1 一起评估 |
| E3 | **多 Session 并行缓冲** | 🔴 | README TODO / P2-4：`createSession()` 现为语义占位；结合 per-Session 激活池可真正并行。 | 引擎调度重构 |
| E4 | `cl_khr_command_buffer` | ⚪ | R35：本机（dev 23.17 与最新 26.35）均不暴露；上游录制入口未实现。见 `command-buffer.md`。 | 上游/驱动可用 |
| E5 | JIT **源码级** + 落盘缓存 | 🟡 | autotuning P3：磁盘 program 二进制缓存已做（冷启动 8.5–12s → ~0.1s）；源码级 JIT（生成完整 `.cl`）低优先。 | 需要更彻底的特化时 |
| E6 | 泛化 view/skip **alias** 到 `slice`/`concat`/`permute` | 🔴 | openvino-gap P0-2：`copy_c` 连续别名已做（R-P0b）；其余需逐消费者加**独立数值用例**（吸取 R30 §7.7 教训）。 | 与 D6/D2 协同 |
| E7 | 在线调优 `onlineTuneMissing` 生产接线 | 🔴 | `PlanModel::onlineTuneMissing` 已实现但未在生产默认路径启用（离线查表优先，`benchmark_protocol` 的安全取舍）。 | 部署需要冷启动自调优时 |

---

## F. 模型 / 功能

| # | 项 | 状态 | 证据 / 说明 | 解除条件 |
|---|---|---|---|---|
| F1 | **seg / obb 解码**（已预留） | 🔴 | README：detect/classify/pose 已具备，seg/obb 预留。 | 需要对应 ONNX 模型 + `PlanModel`/`onnx2plan.py` 扩展 |
| F2 | 支持更多模型（detect 系列、其他 backbone） | 🔴 | README：`onnx2plan.py` + 算子 kernel 可继续扩展。 | 广度按模型需求增长（不追 kernel 数量） |
| F3 | detect / classify / pose 三任务 | ✅ | 三个目标模型端到端数值对齐 onnxruntime。 | — |
| F4 | 动态 shape / i8u8 / USM | 🟡 | P3：按需。当前静态 shape 够用。 | 出现量化/动态需求 |

---

## G. 测量 / 工具

| # | 项 | 状态 | 证据 / 说明 | 解除条件 |
|---|---|---|---|---|
| G1 | 多墙 ceiling：`W_dram`（唯一字节）+ `W_lat`（经验 derate）+ `binding` 输出 | 🔴 | R42 §6-1。当前 hard ceiling 只放 ISA 配额/roofline 下界。 | 先定义 W_lat 的可标定形式 |
| G2 | `scripts/kernel_diag.sh`（feed/store 隔离 + hard_ratio + 墙判决） | ✅ | R42 §6-2 已做成一条命令。 | — |
| G3 | 接 `unitrace` / MD API 做 L3/DRAM 字节与 EU stall 交叉验证 | 🟡 | R42 §6-5：OA 采样受本机内核 `CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS` 未开限制（R43）。 | 内核 tracepoint 可用 |
| G4 | host 分段 / 墙钟进一步归因 | 🔴 | `benchmark.md` §2.4：入队提交 / 同步 / 其余 host 三分。 | 与 E1 联动 |
| G5 | **绑定墙归因框架**（带宽三级 + 计算多墙） | ✅ R71 | 新增 `memTierTime`/`computeWall`/`attributeWall` + `l3TwoLevelSplit`；`kernel_autotune --wall-report` + `scripts/analyze_walls.py`（按调用加权）。诊断层，**不参与选择**；三模型数值逐位不变。见 `round71-binding-wall-model.md`。 | — |
| G6 | 绑定墙框架**应用于三模型 + 与 OpenVINO 逐层对照** | 🟡 | 框架已提交；把框架跑在 OV 的同一批模型上、定位 OV 优势来源的结论尚未落库（待 OV 构建/对照）。目的是判断本项目技术是否**系统性**强于 OV，而非单点。 | 完成 OV 逐层对照并把结论写入文档 |

---

## H. 文档

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| H1 | README 文档表补齐 round49–68；状态清单去重/纠正 | ✅ R70 | 见 `README.md`。 |
| H2 | 本登记表（开放项单一权威清单） | ✅ R70 | 本文件。各轮新增/关闭开放项应同步更新。 |
| H3 | `round64` 缺号（63 → 65） | ✅ | 编号跳过，无独立文档；R64 的「两级层次澄清」内容并入 `round63`/`round65`。 |

---

## 本轮（R70）收尾小结

* **新增落地**：A1（逐节点 spill 归因）、C1（`--global` NO-OP 硬信号）、C2（残差数据集落盘）、
  H1/H2（文档一致性 + 本表）；三模型 `model_check`/`reuse_check` 全回归 PASS。
* **实验关闭**：C4（全量 retune 激活布局 fixpoint）——mb 上生成了 278 条 `#` 键但 A/B **无可靠
  收益**，生产缓存保持不动；见 `round70-status.md` §3。
* **未做（有意）**：A2（`KernelFamily::launch` 声明化）——**数百行、跨三模型数值风险**的重构，
  不适合收尾轮；已在 A2 给出范围与前置。B1/B3 等高风险 kernel/布局项同样保持暂缓。
* **负结果保持关闭**：B4（逐节点 L3 定价 / 容量项）、C4（全量 retune）、D9（depthwise pad）、
  D10（SPPF 融合）、D11（conv3x3 YB）、E4（command buffer）。
* 详见本轮状态文档 `round70-status.md`。
