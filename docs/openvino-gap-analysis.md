# infvino vs OpenVINO GPU：差距分析、优势盘点与可借鉴的设计

> 对照基线：**OpenVINO 最新稳定版**，由 [`scripts/openvino_baseline.py`](../scripts/openvino_baseline.py)
> 在运行时解析（查 PyPI、排除预发布，目前解析为 2026.4.x），**不硬编码版本号**；
> 复现历史数据时用 `--version` 显式指定。本文中的历史数值标注了当时测量所用的版本。
> `src/plugins/intel_gpu`（cldnn）。
> infvino：当前 `main`（R30c，合入 Route A concat→conv1x1）。
> 硬件：Intel Iris Xe（TGL, 80 EU / 1.3 GHz, LPDDR 单通道）。
>
> 本文目标：回答「我们差在哪里、好在哪里、该学什么」。**不改任何 kernel 源码**，
> 是一份对标与路线图。

---

## 0. 一句话结论

**差距不是 kernel 微优化不够，而是「引擎结构性能力」的代差**：

- OpenVINO 赢在**四件结构性的事**（都不在单个 kernel 里）：
  1. **布局是全局优化变量**（blocked format + reorder 插入/消除），不是「全 NCHW」；
  2. **融合是一种通用能力**（post-op 折进 producer kernel + 运行时零 launch 的 view 节点）；
  3. **内存是编译期分析后的复用**（dependency → memory pool → reinterpret alias），不是每个 tensor 一块 buffer；
  4. **调度是「拓扑序 + 隐式 event + 队列/多 stream」**，host 开启命令后不再逐节点同步。
- infvino 赢在**方法论与可解释性**：ISA 级的物理上限标定、按 shape 的自动调优，
  以及**离线调优 + 运行时查表**的产品化取舍（无长 GPU 任务、可复现、可审计）。
- 两条曲线：**kernel busy 上 infvino 已到 OV 的 ~60–70%**（R24/R25 证明两条 OV 通路
  的现实天花板就在 ~12–14 ops，我们到 8–13）；**但整网墙钟只到 OV 的 ~55–65%**。
  后面这段差距几乎**全在 launch 数 / 布局物化 / 内存复用**，不在 kernel 里。

---

## 1. 账本：差距到底在哪一段

同一 iGPU，infvino R30c 与 **OpenVINO 最新稳定版**（下表为历史测量值，当时为 2025.2；
新测请用 `scripts/openvino_baseline.py run` 动态解析的版本）的公开对照（README / benchmark.md）：

| 模型 | OV GPU 合计 | OV infer | OV e2e | infvino busy | infvino 墙钟 | busy 比 | 墙钟比 |
|---|---|---|---|---|---|---|---|
| yolov8n-pose | 9.33 ms | 11.2 | 13.9 | ~13.6 | 22.3 | 0.69× | 0.62× |
| yolo11n-pose | 9.61 ms | 11.8 | 14.5 | ~14.7 | 25.6 | 0.65× | 0.58× |
| mobilenetv3-small | 0.96 ms | 1.8 | 2.2 | 2.8 | 4.6 | 0.34× | 0.48× |

拆成三段看：

```
OV:        [ kernel 9.3 ]  [ gap 1.9 ]  [ e2e 2.7 ]      → 总计 13.9
infvino:   [ kernel 13.6 ] [ launch/布局 ~3 ] [ ... ]     → busy 13.6, 墙钟 22.3
              ↑ +4.3ms（数据通路/布局）      ↑ +8.7ms（引擎结构）
```

**关键判读**：

1. **kernel busy 的 4 ms 差**：主要来自 OV 用**阻塞式 layout（fsv16/fsv32）+ 权重预重排**，
   我们的 OV 移植（osv32/blk）已经在指令配额上追平（R24/R25 证明 ~20 / ~17 的配额，
   实测到 8–13），剩余受限于**布局不是最优**（NCHW 导致 conv/gemm 的 staging 与
   gather 更贵）+ 延迟/占用。
2. **墙钟比 busy 多出的 ~8.7 ms**：这是**引擎结构**造成的，与 kernel 无关。yolov8 有
   **149 个节点 + 149 个辅助 tensor**，每个节点一次 dispatch、每个中间 tensor 一块
   `cl_mem`（默认 pool 关、无 alias）：
   - `launch floor ≈ 3.5 µs/dispatch`（本机实测）→ 149 × 3.5 µs ≈ 0.5 ms，**但**每个
     dispatch 还叠加 host 侧 kernel 查找/参数设置/事件处理，实测 gap 远大于 0.5 ms；
   - OV 把 reshape/permute/concat/crop 等大量节点**零 launch 化**，我们的 `reshape`(13)
     已 alias，但 `copy_c`(16)、`slice_axis`(4)、`concat4` 等仍在跑。
3. **mobilenet 的 busy 比只有 0.34×**：因为它几乎没有大 conv，**全是被 launch/内存支配的
   小算子**（0.68 ms 小算子 / 2.8 ms busy = 24%，且 gap 主导）。这印证了第 2 点。

> **结论：下一步的主战场是「引擎结构」，不是 kernel。** 继续单 kernel 抠 5–10% 的边际
> 收益递减（R24 已证明四个方向全部负结果、R30 已到内存墙），结构性改动才有 1.3–1.6× 空间。

---

## 2. 逐维对标

| 维度 | OpenVINO intel_gpu | infvino | 差距性质 |
|---|---|---|---|
| **布局** | 全局布局优化：`layout_optimizer` 选 `b_fs_yx_fsv16/32`、`byxf`、`bs_fs_yx_bsv16_fsv16`；`reorder_inputs`/`remove_redundant_reorders`/`add_required_reorders` 插入/消除 reorder | **全 fp16 NCHW**，无 blocked，无 reorder 概念 | **大**（结构） |
| **算子数** | 236 个 `.cl` 模板 × 运行期 JIT 特化 = 数千变体；每个 op 多实现（ocl/ocl_v2/onednn/cpu） | 12 个 `.cl`，约 30 个 kernel 变体 | **大**（广度） |
| **融合** | 通用 post-op 融合（activation/eltwise/quantize/bias/swiglu 折进 producer），**+ 运行时零 launch 的 view/skip 节点** | 硬编码少数融合：SiLU（Conv+Sigmoid+Mul）、1×1 融合 bias+act、残差（实测负）、concat→conv1x1（Route A） | **大**（结构） |
| **内存** | `memory_pool`：dependency 分析 → 尺寸桶复用 → `reinterpret_buffer` alias；生命周期 = 拓扑区间 | 每个 `tensor` 一行 = 一块独立 `cl_mem`，**无复用、无 alias**（只 reshape 别名） | **大**（结构） |
| **调度** | 拓扑序 + **隐式 event**，host 连续 enqueue，惰性同步；支持 **out-of-order 队列**、多 stream | 单队列（in-order），`profiling_` 时**每个节点 `clWaitForEvents`**（即使非 profiling 也有 host 侧逐节点开销） | **中–大** |
| **JIT** | `jitter`：宏注入 → 编译期特化 shape/对齐/leftover；program 二进制缓存（`.cl_cache`） | **命令行式 `-D` 宏**，无源码级 JIT；有 `programs_` 内存缓存（无磁盘二进制缓存） | **中** |
| **调优** | `kernel_selector` + `ParamsKey` 筛选 + **离线 `auto_tuner` cache.json** | `OpSignature` + `TuningCache` + 候选枚举 + `expected_ops` 中间标准 | **小**（设计对齐甚至更强，见 §3） |
| **ISA/物理模型** | 无公开的逐 kernel 指令配额文档 | **有**：`ocloc` 反汇编、`mad` 占比、寄存器模型、内存 roofline、`expected_ops` | **infvino 领先** |
| **设备/后端** | OCL / **Level-Zero / SYCL** 三后端，oneDNN/CM 多实现，USM、remote tensor、DMA-BUF/VA | 仅 OpenCL（buffer 对象，无 USM） | 中（当前不需要） |
| **数值** | 多精度（f32/f16/i8/u8/i4），动态 shape | 仅 f16 计划 / f32 输出，静态 shape | 中（当前够用） |

---

## 3. infvino 好在哪（应保留的强项）

1. **「物理上限 → 中间标准 → 实测」的三层标尺**（`docs/autotuning.md` §0）。
   OV 的 `cache.json` 只存 `(kernel_name, index)`，**不含任何实测指标**；infvino 的
   `TuningEntry` 额外记录 `ops`、`expected`、`ratio`，缓存本身就是「离物理极限多远」的数据库。
   这让我们能**自动指出最差的层**（如 `320×320 s2 3→16` ratio 0.11），而不是靠猜。
2. **ISA 级证据链**（`docs/xe-lp-isa.md`、`round24-analysis.md`、`register-model.md`）。
   用离线 `ocloc` 反汇编得到 `conv_ov` 主循环 288 mad / 453 指令 = 63.6% → 配额 20.3，
   推翻了此前「上限 16」的误判。**这是 OV 也没有公开的深度**，是团队最强的能力。
3. **负结果被系统性记录**（R24 的 split-K / 双累加集 / 内部块 fast path、R30 的 Route B、
   多消费者 alias）。避免重复踩坑，且让「为什么不再优化」有据可查。
4. **离线调优 + 运行时查表**的安全取舍（`docs/benchmark_protocol.md`：防止开发板死机）。
   OV 在首次推断时在线跑 `GetKernelsDataForAutoTune`；infvino 默认离线，更稳。
5. **单核纯粹的依赖面**：纯 C++，只依赖 OpenCV/OpenCL/yaml-cpp，无 ROS/OV 链接。
   对嵌入式部署的**冷启动、体积、可审计性**是实打实的优势。

---

## 4. 我们差在哪些具体的东西（可落地的差距清单）

### 4.1 布局物化链（最大且最系统）

yolov8 里 `copy_c` 16 + `slice_axis` 4 + `concat4` 19 + `reshape` 13，且每个中间
张量都**物化到一块独立 `cl_mem`**。OV 的做法是：
- **view/skip 节点零 launch、alias 内存**（`primitive_inst::execute` 的 SKIP 路径）；
- **implicit concat / crop**（`prepare_buffer_fusing.cpp`：`concat_in_place_optimization`）；
- **reorder 融合**（`remove_redundant_reorders.cpp`：把 reorder 折进消费者）。

我们已有 Route A（concat→conv1x1）打开了正确的口子，但只是**单点**，不是**通用机制**。

### 4.2 内存复用

OV：`basic_memory_dependencies` → `memory_pool::get_memory`（尺寸桶 + 冲突集）→
`reinterpret_buffer`。生命周期 = 拓扑序区间，缓冲在节点间**反复复用**。
infvino：`parse()` 里每个 `tensor` 一句 `rt_.alloc(...)`，**永久占用**。
149 个辅助 tensor × 每块数百 KB–数 MB，对 3.75 MB LLC 的 Iris Xe 是灾难性的
——**热数据被冷数据挤出 L3**，直接抬高了 R30 里那些「本该 L3 命中」的算子（如 concat
部署时掉一半带宽）的成本。

### 4.3 调度与 host 开销

- infvino 单 in-order 队列；`timed()` 在 `profiling_` 下**每节点 `clWaitForEvents`**。
  即便部署时 `profiling_=false`，host 仍需逐节点 `getKernel`+设置参数，没有
  「变化才重设参数」的缓存（OV 的 `IMPL_CHANGED`/`MEMORY_CHANGED` 标志位）。
- 没有 out-of-order 队列 / 多 stream / event 依赖图。OV 用事件把依赖表达出来，
  host 可以**持续 enqueue**，GPU 不空转。

### 4.4 融合广度

我们只融合了 SiLU 和 1×1 的 bias+act；OV 是**任意支持的 post-op 折进 producer**，
且**融合是编译期 JIT 生成代码**（`FUSED_OPi_LOAD`/`FUSED_OPi_ACTION`），不是手写组合。
我们每加一种融合就要手写一个 kernel 分支，边际成本高。

### 4.5 kernel 广度与 JIT

12 个 `.cl` vs 236 个模板；且我们是**命令行宏**，OV 是**源码级 JIT 特化**
（shape/对齐/leftover/boundary 全在编译期展开）。R24 的 `FIT_WH`/`FIT_CIN` 是我们
已经在做的方向，但远没有系统化。

---

## 5. 学习清单：按 ROI 排序（建议路线）

> 原则：**先做「引擎结构」，后做「kernel 广度」**。每项都给出 OV 的参考文件与 infvino 的落点。

### P0 — 内存复用 + view/skip alias（预期整网 −15~25%）

**学什么**：`memory_pool` + `reinterpret_buffer` + `can_be_optimized` 节点。
**怎么做**：
1. 在 `PlanModel::parse()` 后加一个**liveness 分析**（拓扑序 + 首次/末次使用），
   给每个 `tensor` 算生命周期区间；
2. 实现一个**尺寸桶 memory pool**（学 `memory_pool::get_from_non_padded_pool`：
   从 `lower_bound(bytes)` 找可复用块，检查冲突集）；
3. 对 `reshape`/`flatten` 已 alias 的，扩展成通用 **base-offset view**（`copy_c`/`slice`
   的输出若消费者支持 offset，直接 alias 父张量）；
4. 冲突集从「直接依赖」推（学 `basic_memory_dependencies`），先不用管 OOO 版本。

**为什么 ROI 最高**：不改任何 kernel，直接减少物化 → 提高 L3 命中 → 同时改善
R30 里所有「贴 DRAM 墙」的算子。Route A 已经证明了这条路（省 1.5 ms concat）。

### P1 — 布局：引入 blocked format（预期 conv/gemm +10~30%）

**学什么**：`layout_optimizer::get_expected_format` + `b_fs_yx_fsv16` 传播 +
`post_optimize_weights`（权重预重排）。
**怎么做**：
1. 先在**权重侧**引入 blocked（`os_is_yx_isv16_osv16`），这已是 R25 `conv_blk` 的一部分，
   把它从「单 kernel 选项」升级成**计划级的权重布局**；
2. 再考虑**激活侧** blocked（`b_fs_yx_fsv16`），让连续 conv 之间**避免 reorder**；
   这需要先有 P0 的 view/alias 机制和 reorder 概念；
3. **不要全盘照搬**：OV 的布局机制极其庞大（format traits、reorder 插入/消除、
   `_optimization_attributes` 全局启发式）。建议**从」只给 conv/gemm 链做 blocked」开始**，
   用 autotune 按 shape 决定是否 blocked。

### P1 — 融合通用化（预期少 10–20 个 launch + 小算子时间）

**学什么**：`prepare_primitive_fusing` 的 `fuse_simple_primitives`（activation/eltwise/
quantize/bias）+ JIT 的 `FUSED_OPS` 代码生成。

**已落地（R-P1a）**：把 `onnx2plan.py` 的激活融合从「只覆盖 1×1/SiLU」扩到
**通用/depthwise conv 的 epilogue**（`conv_general`/`depthwise_*` 内核本就支持
act=1 SiLU / 2 Hardswish / 3 ReLU / 4 Hardsigmoid）。mobilenet 的 11 条
`Conv(depthwise) → ReLU/HardSwish` 全部折进 conv，`ew_unary` **11→0**；
yolo 无此类模式（其 general conv 后是 SiLU，无独立激活节点）。
对折进的 depthwise 签名补跑 autotune（9 条新条目）。
实测 mobilenet busy **2.65→2.59 ms（−2.4%）**，数值**逐位一致**。
`INFVINO_NO_FUSE_GENERAL=1` 可关闭以做 A/B。

**后续**：
1. 把 epilogue 扩成**可组合**（bias+act+residual+简单 broadcast），接 autotune；
2. `onnx2plan.py` 的融合从「硬编码模式」升级成**模式匹配 pass**。
3. 运行时：`reshape/permute/concat` 的**零 launch alias**（部分已在 P0 落地）。

### P1 — 布局：blocked format（**本机天花板 ~5%，需大重构**）

**实测（R-P1 调研）**：`conv_blk` 每层**每帧都要做一次 bfyx→b_fs_yx_fsv16 输入重排**，
y8/y11 各 **24/23 次 dispatch、0.66–0.68 ms（≈5% busy）**。这是 ov-style 持久 blocked
布局能省掉的部分。**但**：
- 重排是**launch floor + 小 dispatch** 主导（~9 GB/s），**不是带宽问题**：
  试过向量化 store 版本（`reorder_bfyx_to_fsv16_v`），实测**更慢**（0.76 vs 0.67 ms）——
  gather 读主导，写放大不是瓶颈。**负结果，已回退。**
- 真正要省掉它需要**持久 blocked 布局**（生产者直接输出 blocked、跨层传播），
  是 multi-hour 重构，且当前 blk/ov/native 按 shape 混选，只有部分相邻 conv 能受益。
- 因此 **P1-layout 在本机的现实收益 ≈5%，风险高，暂缓**；若做，优先「持久 blocked 链」。

> **结论**：P1 的融合项已落地（−2.4% mobilenet）；布局项的余额（reorder ~5%）需要
> 持久 blocked 重构，ROI 与风险不匹配，建议先看 P2（调度/墙钟）或 depthwise 内核
> （R30 记录 5–9× ISA 配额空间，比 reorder 更大且自包含）。

### P2 — 调度：事件 + out-of-order + 参数缓存（预期收窄墙钟 gap）

**学什么**：`network::execute_impl`（拓扑序 + `dep_events`）、`ocl_stream::sync_events`
（barrier 合并）、`primitive_inst::prepare_primitive`（变化才 `set_arguments`）。
**怎么做**：
1. **去掉 profiling 下的逐节点同步**（改为批量 + 最后同步），部署路径尤其；
2. **kernel 参数缓存**：记录每个节点的 `(cl_kernel, 参数签名, 内存指针)`，
   内存/配置不变时跳过 `setArg`（学 OV 的 `MEMORY_CHANGED` 标志）；
3. 试 **out-of-order 队列 + 事件依赖**：如果不打算做 OOO，至少把
   `enqueueNDRangeKernel` 的 event 依赖用起来，让 host 不因 `clWaitForEvents` 停顿。
4. 多 stream：`createSession()` 现在是语义占位；结合 P0 的 pool（每 Session 一套激活池）
   就能真正并行（README 的 TODO）。

### P2 — JIT 化（长期）

**学什么**：`jitter.h` 的 `MakeJitConstant`/`toCodeString` + `kernels_cache` 的二进制缓存。
**怎么做**：把现在命令行 `-D` 的宏，升级成**源码级 JIT**（生成完整 `.cl` 再编译），
并把编译好的 program 二进制**落盘缓存**（按 `source + options + 设备` 哈希），
这样运行期零编译、且特化更彻底。这会让 R24 的 `FIT_*` 从「少数手写」变成「全自动」。

### P3 — 补算子/后端（按需）

- Level-Zero/SYCL 后端（Iris Xe 上 OCL 已够；收益有限）；
- i8/u8（若目标模型量化）；USM（省 buffer 对象开销）；
- 动态 shape（当前静态，够用）。

---

## 6. 「学 OpenVINO」时要警惕的点

1. **不要照搬 `format`/`reorder` 全家桶**。OV 的布局机制有 20+ passes、上百 format，
   是为「支持任意模型 + 多后端 + 量化」累积的。infvino 只需**为 conv/gemm 链做 blocked**，
   用 autotune 兜底，保持简单。
2. **不要把 kernel 数量当目标**。OV 有 236 个模板，是历史与多硬件适配的产物。
   我们的 12 个 kernel + autotune 已经覆盖三个模型，**广度应按模型需求增长**，不是对标数字。
3. **保留「离线调优」的安全取舍**。OV 在线调优在开发板上是风险；我们的离线+查表
   （`benchmark_protocol.md`）是更适合场景的工程决策，不要为了「像 OV」而放弃。
4. **保留中间标准/ISA 方法论的领先**。这是 OV 都没公开的深度，是团队真正的护城河；
   把它**产品化**（比如自动生成「优化优先级报告」）比学 OV 更有价值。

---

## 7. 建议的下一步（可执行）

1. **[P0] 写 `docs/memory-reuse-design.md`**：liveness 分析 + 尺寸桶 pool + alias，
   先在 yolov8 上量「复用率 / L3 命中变化 / busy 变化」，独立验证再改 `PlanModel`。
2. **[P0] 通用 view/skip**：把 `reshape` 的 alias 推广到 `copy_c`/`slice`/`concat`（单消费者），
   每加一个消费者就加**独立数值用例**（吸取 R30 §7.7 多消费者 alias FAIL 的教训）。
3. **[P1] 权重 blocked 计划化**：把 `conv_blk` 的权重重排从 kernel 选项提升为计划属性，
   让连续 conv 共享 blocked 权重。
4. **[P1] epilogue 宏化**：在 `gemm.cl`/`conv_*.cl` 加可组合 `EPI`，接 autotune。
5. **[P2] 参数缓存 + 去同步**：先做最安全的「非 profiling 不逐节点同步 + 参数变化才 setArg」，
   量墙钟变化。

> 预期：P0+P1 落地后，整网墙钟有希望从 22.3 → ~17–18 ms（接近 OV e2e 13.9 的 0.8×），
> 且**大部分是引擎结构收益，不需要再动 kernel 数据通路**。

---

## 8. 参考（OpenVINO 关键文件）

| 主题 | 文件 |
|---|---|
| kernel 选择 | `src/kernel_selector/kernel_selector.cpp`、`kernel_selector_base.h` |
| 参数/适用性 | `src/kernel_selector/kernel_selector_params.{h,cpp}`（`ParamsKey::Support`） |
| 离线调优 | `src/kernel_selector/auto_tuner.{h,cpp}`（`TuningCache`）、`cache/cache.json` |
| JIT | `src/kernel_selector/jitter.{h,cpp}`、`kernel_base_opencl.cpp`（`CreateJit`） |
| 融合 | `src/graph/graph_optimizer/prepare_primitive_fusing.cpp`、`prepare_primitive_fusing_through.cpp` |
| 布局 | `src/graph/layout_optimizer.cpp`、`graph_optimizer/{reorder_inputs,add_required_reorders,remove_redundant_reorders}.cpp` |
| 内存复用 | `src/runtime/memory_pool.{hpp,cpp}`、`graph/graph_optimizer/basic_memory_dependencies.cpp`、`primitive_inst.cpp::allocate_output` |
| 调度/事件 | `src/graph/network.cpp::execute_impl`、`src/runtime/ocl/ocl_stream.cpp::enqueue_kernel/sync_events` |
| view/skip | `src/graph/primitive_inst.cpp::execute`（SKIP 路径）、`prepare_buffer_fusing.cpp` |
| 编译管线 | `src/graph/program.cpp::build_program` |
| 插件入口 | `src/plugin/{plugin,compiled_model,graph,program_builder}.cpp` |
