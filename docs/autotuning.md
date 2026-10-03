# infvino 自动调优体系（含 JIT）设计

> 目标：把「逐轮手写 kernel + 手扫 tile」的优化方式，升级成**可复现的层级化自动调优**，
> 让 `ops/EU/cyc` 从「事后测量指标」变成「可优化的目标函数」，并补齐它与物理极限之间的
> **中间标准**。
>
> 参考对象：OpenVINO GPU plugin 的 `kernel_selector` / `auto_tuner` / `TuningCache` /
> `jitter`（见下文 §1）。infvino 不链接 OpenVINO，只借鉴其**分层调优架构**。

---

## 0. 为什么需要「中间标准」

infvino 已经用 `ops/EU/cyc` 建立了清晰的**物理极限标尺**（`docs/kernel.md` R18.7）：

| 层级 | ops/EU/cyc | 含义 |
|---|---|---|
| 理论峰值 | 32 | 16 packed FP16 FMA/EU/cyc |
| 纯寄存器 FMA | 27.4–29.6 | 结构上限（指令槽 + 循环开销） |
| GEMM compute-only | 17.7 | SLM 操作数 feed |
| conv3x3 OV 指令配额（**R24**）| **20.3** | ISA：主循环 288 packed mad / 453 指令 = 63.6% |
| conv3x3 staging-free（native）| 16.4 | SLM 权重块读延迟 |
| GEMM 整核 | 13.7 | staging + loads + 寄存器占用 |
| conv3x3 整核（大层，R18）| 10.3 | staging + 网格饥饿 |

> **R24 更正**：上表 R18/R20 的 `conv3x3 ≈ 10–16` 是**旧口径**。离线反汇编 `conv_ov.cl`
> 证明 `sub_group_broadcast` 被折进 `mad` 操作数，指令配额上限是 **20.3**（不是 16），
> 实测大层 13.8 = 配额的 68%、40×40 = 42%，缺口在延迟/流水/占用。中间标准已据此改为
> `32×0.636×prologue_amort×grid_factor`（上界 20.3）。见
> [`round24-analysis.md`](round24-analysis.md)。

问题在于：**只有「天花板」和「实测值」，缺少「这一层、这个 shape、在这台机器上应该能到多少」的
中位标尺。** 于是每次调优都要靠人肉 reinterpret：
- 「40×40 只有 8.4 ops，是没调好还是网格饥饿的物理必然？」
- 「`96×49×576` 的 gemm 只有 1.0，该继续调 tile 还是它本来就该走别的通路？」

自动调优要能自己回答这些问题，就必须有一个**期望效率模型（middle standard）**：
`expected_ops(op, shape, device)` —— 由 roofline / 寄存器预算 / 网格占用推导，而非实测。
调优器把「实测 / 期望」作为**归一化目标**（而不是直接比绝对值），这样：
- 不同 shape 之间可比较、可排序；
- 「离极限还有多少」有量化依据；
- 负结果可以自动判定并停止（避免 R17/R20/R21 那样反复手试）。

---

## 1. 参考：OpenVINO 的调优体系（分层）

OpenVINO GPU plugin 的自动调优由四层组成，infvino 逐一对应：

| OV 组件 | 作用 | infvino 对应 |
|---|---|---|
| `ParamsKey` + device-feature key（`GetAllImplementations`）| 按 op 参数与设备能力**筛选可用实现** | `OpSignature` + `ClRuntime::DeviceInfo` 能力位 |
| kernel 实现的 `autoTuneOptions`（`GetAutoTuneOptions` / `GetTunedKernelsDataByIndex`）| 每个实现**枚举候选配置** | `Tiles.hpp` 的 `candidates*()` |
| `TuningCache`（JSON，`computeUnits → kType → params_hash → (kernel, index)`）| **离线调优缓存**，按设备 key | `gk::TuningCache`（§3） |
| `JIT`/`jitter` + kernels cache | 编译期特化 + 运行时 kernel 缓存 | `ClRuntime::buildKernel` 缓存 + `PlanModel::getKernel` |

关键设计差异（保留 infvino 的强项）：
- OV 的 config **不含实测指标**，只有 `(kernel_name, index)`；infvino 的缓存额外记录
  **实测 ops/EU/cyc 与 expected**，这样缓存本身就是「物理极限的数据库」。
- OV 的调优在**首次推断时在线跑**（`GetKernelsDataForAutoTune`）；infvino 默认走
  **离线工具** `kernel_autotune`，运行时只查表，避免开发板上的 GPU 长任务（安全协议）。

---

## 2. 分层调优体系（infvino 版）

```
┌─ Level 0: 物理极限标尺（已有）─────────────────────────────┐
│  ops/EU/cyc 的层级：32 / 27.4 / 16.4 / 13.7 / 10.3        │
└───────────────────────────────────────────────────────────┘
             ▲ 参照
┌─ Level 1: 期望效率模型 expected_ops(op, shape, device) ───┐
│  roofline + 寄存器预算 + 网格占用 → 「应该能到多少」        │
│  = 中间标准（middle standard）                             │
└───────────────────────────────────────────────────────────┘
             ▲ 归一化目标
┌─ Level 2: 候选枚举 candidate configs ├ 实测 benchmark ─────┐
│  Tiles / Conv3x3Cfg / Conv1x1Cfg / depthwise / gap          │
│  每个 (op,shape) 枚举 N 个配置，逐个 GPU 计时取最优         │
└───────────────────────────────────────────────────────────┘
             ▼ 结果
┌─ Level 3: 调优缓存 TuningCache（按 device + op + shape）──┐
│  记录 best config + 实测 ops + expected ops + 余量          │
└───────────────────────────────────────────────────────────┘
             ▼ 消费
┌─ Level 4a: 运行时查表 PlanModel（cache.json）─────────────┐
│  有命中用命中；无命中回退内置启发式（当前 R18/R22 规则）    │
└───────────────────────────────────────────────────────────┘
┌─ Level 4b: 烘培进 plan（--bake）──────────────────────────┐
│  把 best cfg 写成节点属性，部署无需 cache.json              │
└───────────────────────────────────────────────────────────┘
┌─ Level 5: JIT 特化 ───────────────────────────────────────┐
│  除 tile 数值外，把 shape/groups/leftover/对齐 编译期特化   │
└───────────────────────────────────────────────────────────┘
```

---

## 3. TuningCache 格式

文件：`config/tuning.json`（可由 `INFVINO_TUNING_CACHE` 覆盖路径；空路径=禁用）。

```jsonc
{
  "version": 1,
  "device": { "name": "...", "eu": 80, "clock_mhz": 1300, "id": "8086:9a49" },
  "entries": {
    // key = opType + "|" + signature
    "conv3x3|W40H40s1_Cin64_Cout64_act1_dtype_f16": {
      "kernel": "conv3x3_ov",
      "config": "OBW=5,OBH=2,STRIDE=1,PAD=1,ACT=1",
      "ops_per_eu_cyc": 8.40,
      "expected_ops_per_eu_cyc": 12.0,
      "ratio": 0.70,
      "ms": 0.149,
      "device_id": "8086:9a49"
    },
    "gemm|M128N1600K192": { ... }
  }
}
```

- **key 的 signature** 必须包含一切影响 kernel 选择的物理量：op 类型、输入/输出通道、
  空间、stride、groups、融合激活、dtype、batch。见 `OpSignature`。
- **device_id** 来自 `CL_DEVICE_ID` / PCI id（`8086:9a49` 是 TGL iGPU）；不同 EU 数的
  设备缓存分开（同 OV 的 `computeUnitsCount` 键）。
- 缓存**不参与数值**，只影响选哪个 kernel/config；因此格式演进是安全的。

---

## 4. 运行时接入（PlanModel）

`PlanModel` 在 `dispatch` 时按 `OpSignature` 查表：

```
sig = OpSignature::conv3x3(Wout, Hout, stride, Cin, Cout, act)
if (auto e = tuning_.lookup(sig))  build/launch(e->kernel, e->config)
else                               内置启发式（现状 R18/R22 规则）
```

- 查表失败**绝不报错**，只回退；这样没有 cache 也能跑（数值/部署不依赖调优）。
- 表命中时仍做**编译期特化 build**（`getKernel`），所以与手写逻辑等价。
- `INFVINO_TUNING=off` 强制走内置启发式（用于对照实验）。

---

## 5. 离线调优工具（kernel_autotune）

```
kernel_autotune --plan models/yolov8n-pose/model.plan \
                --cache config/tuning.json \
                --op conv3x3 --batch 4 --iters 30 \
                [--bake] [--expected] [--report]
```

流程：
1. 解析 plan，收集所有需要调优的节点及其 `OpSignature`；
2. 对每个节点枚举候选配置（每 op 一个 `candidates*()`）；
3. 逐个 build + 计时（`ClRuntime::timeMs`，warmup 3 / iters N），记录实测 ops；
4. 与 `expected_ops` 比对，输出余量；
5. 取最优写缓存（merge，不丢已有条目）；
6. `--bake`：把最优 cfg 以属性写回 plan（`cfg=...` / `obw=...`）。

**安全**：一次命令只调优一个节点/少量配置，分批调用（见 `docs/benchmark_protocol.md`）。
工具支持 `--only <substr>` 与 `--limit N` 以便分批。

---

## 6. JIT 特化（Level 5）

在「枚举 tile」之外，把**形状常量编译期化**，消除运行时 div/mod/分支。当前 kernel 已有
部分特化（`STRIDE/PAD/ACT`、depthwise 的 `DW_K/S/P/ACT`）。JIT 层的扩展方向：

| 特化项 | 收益来源 | 状态 |
|---|---|---|
| `ACT`（silu/hardswish/…）| 去掉运行时激活分支（R16 已证：depthwise 全展开后只发一条）| **已生效** |
| `STRIDE/PAD` | 去边界谓词 / 索引常量折叠 | **已生效** |
| `groups == Cin`（depthwise）| 已专门 kernel | **已生效** |
| `Wout/Hout` 是 tile 整数倍 | 去尾部 leftover 分支 | 待做（P2）|
| `Cin % CINC == 0` | 去 ci chunk 尾部处理 | 待做（P2）|
| `OSV_SIZE/四分块` | OV 式 kernel 的块常量 | **已生效**（`conv3x3_ov`）|

> **与自动调优的关系**：JIT 特化后的 `options()` 字符串天然成为 `OpSignature` 的一部分，
> 因此每个 JIT 变体都能被 TuningCache 独立记录与选择——`sg`/`epi`/`res` 等宏就是这条机制
> 的现有实例（同一 op 因宏不同而产生不同 kernel 变体，调优器分别计时选优）。
>
> 本轮（P0）落地的是**JIT 变量的枚举与选择**（例如 depthwise 的 K/S/P/ACT 全组合、
> conv3x3 的 OV block 与 native tile），P2 再把「形状整除」这类新的编译期常量加进 kernel。
>
> **P3（未做，低优先）**：当前只有**内存内** program 缓存（`ClRuntime::programs_`，key =
> `source|options`）。下一步是 JIT 生成完整 `.cl` 源码再编译，并把**编译产物落盘缓存**
> （key = `source + options + 设备`），避免每次冷启动重新 JIT。Level-Zero/SYCL 后端、
> USM、i8/u8、动态 shape 当前需求不足，暂缓。

### 6.1 JIT 与 `expected_ops` 的相互作用

同一个 op 的不同 JIT 变体会改变 `expected_ops` 的**分项系数**（例如做了 leftover 特化后
`staging` 折损下降），因此中间标准的系数应在 P2 随特化实测更新——这正是「层级化」：
底层 kernel 能力变了，中间标尺随之抬高，调优目标也随之抬高。

---

## 7. 期望效率模型（middle standard）公式

`expected_ops(op, shape, dev)` 的推导（与 `docs/kernel.md` R18.7 的模型一致）：

```
issue_bound = 32 × (mad_frac) × (SIMD/16)          # 每周期 1 条向量指令
```

其中 `mad_frac = mad 指令 / 总发射指令`，由**每个数据通路的字节/FLOP 与复用度**推导：

- **conv3x3（lane=空间+SLM, native）**：
  `mad_frac ≈ 1 / (1 + staging_frac + weight_load_frac + strip_load_frac)`
  staging_frac ≈ 0.5（R18.2：全局读 +21%、索引/SLM 写 +38% 合计 ~1.67× → 1/(1+0.67)）
  weight/strip 复用 = CB 与 3×kw，得 inner-loop mad_frac ≈ 0.9。
  → expected ≈ 32 × 0.9 × (CB/(CB+overhead)) × SIMD_factor，对 64→64@80 ≈ 13–16。
- **conv3x3（lane=通道 + block-read, OV）**：R20 曾按「1 broadcast : 1 mad」推断
  `expected = 32/2 = 16`；**R24 更正**：ISA 反汇编显示 `sub_group_broadcast` 被折进 `mad`
  操作数，主循环 mad 占 63.6%，所以
  `expected = 32 × 0.636 × prologue_amort × grid_factor`，上界 **20.3**（不是 16）。缺口是
  延迟/流水/占用（实测大层 13.8、40×40 8.5）。
- **conv3x3（lane=空间 + SLM, native）**：staging-free 16.4（R18），整核受 staging + 网格
  饥饿；调优器在 OV 与 native 之间按 size 取实测更优者（R24 起两者对所有 shape 都是候选）。
- **gemm**：`expected = 32 × mad_frac × reg_occupancy`，
  `mad_frac = TM·TN / (TM·TN + TM + TN + addr_overhead)`；寄存器占用修正由 128 GRF 决定。
- **网格饥饿修正**：`grid_factor = min(1, n_wg / (EU × k))`，k≈2（R18：<16 WG 时 +20–56% 可恢复）。

> 该模型不追求精确绝对值，只要**单调、可比**，能把「离极限的距离」量化成 `ratio = measured/expected`。
> 所有系数在 `expected_ops` 内标注出处（Round 编号），便于随实测修正。

---

## 8. 目录与构建

```
include/infvino/
  Tuning.hpp        # OpSignature / TuningCache / ExpectedOps
  Autotuner.hpp     # 候选枚举 + 在线 benchmark
src/
  Tuning.cpp
  Autotuner.cpp
  tools/kernel_autotune.cpp
scripts/
  autotune.py       # 容器安全封装（分批、timeout、gpu_guard）
config/
  tuning.json       # 调优缓存（随仓库携带一份与本机匹配的）
```

### R28 新增/变更（代码）

| 文件 | 内容 |
|---|---|
| `include/infvino/Tuning.hpp` / `src/Tuning.cpp` | `OpSignature::params` + `custom()`；小算子 `expected_ops` 正期望 |
| `include/infvino/Autotuner.hpp` / `src/Autotuner.cpp` | `candidatesSmall()`；`candidatesConv3x3` 自动加 `-DFIT_*`；gemm 小 tile |
| `kernels/ops.cl` | `ew_binary_v/ew_unary_v/concat4_v/copy_c2/slice_axis3/maxpool3/resize_nn3/permute_0213_3d/bmm2/ew_binary_ch` |
| `kernels/conv_ov.cl` / `conv_blk.cl` / `conv.cl` | `-DFIT_WH/FIT_COUT/FIT_CIN/FIT_CB` 编译期形状特化 |
| `src/PlanModel.cpp` | `smallSig()` / `smallLaunch()` / `onlineTuneMissing()`；小算子走调优查表 |
| `src/tools/tuning_test.cpp` | 新增 custom 签名 + 小算子候选自检 |

---

## 9. 落地节奏

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0（已完成）** | TuningCache + OpSignature + ExpectedOps + conv3x3/conv1x1/depthwise 候选枚举 + kernel_autotune + PlanModel 查表 + bake + tuning_test | 三模型调优表生成；yolov8 busy −7.4%；数值逐位不变 |
| **P1（已完成）** | gemm/conv1x1-N>1 独立候选谱系（新增大块 + 小 M/N tile）| 由调优器按 shape 选；本代模型 1×1 已折叠成 conv1x1，gemm 节点少 |
| **P2（已完成）** | JIT 形状特化（`FIT_WH/FIT_CIN/FIT_CB/FIT_COUT`，conv_ov/conv_blk/conv native）| 编译期（非运行时）去谓词；候选自动按签名对齐设置 |
| **P3（已完成）** | 在线 autotune（`INFVINO_TUNING=online`，首推断按需调优未命中子集，带缓存）| 部署端自适应；默认关（开发板安全） |
| **R28 扩展** | 小算子纳入调优（ew/copy/slice/concat/pool/resize/permute/bmm/gap 的数值等价变体）| 见 §9.2 |

### 9.1 P0 实测结果

**三模型 kernel busy（`kernel_run --report --iters 5`，同会话对照）**：

| 模型 | 基线（内置启发式，`INFVINO_TUNING=off`）| **自动调优** | 加速 |
|---|---|---|---|
| yolov8n-pose | 17.224 ms | **15.157 ms** | **−12.0%** |
| yolo11n-pose | 20.316 ms | **18.074 ms** | **−11.0%** |
| mobilenetv3-small | 3.539 ms | **3.373 ms** | **−4.7%** |

**数值**（`scripts/numerical_check.py`，ClBackend vs onnxruntime，**缓存生效**）：

| 模型 | mean_rel | max_rel(amax) | 判定 |
|---|---|---|---|
| yolov8n-pose | 5.60e-04 | 1.85e-02 | PASS |
| yolo11n-pose | 9.38e-04 | 2.48e-02 | PASS |
| mobilenetv3-small | 9.61e-03 | 9.16e-03 | PASS |

> yolov8 的 tuned vs baseline 输出**逐位不变**（`max_abs=0`）——同一 kernel、同一计算，只是
> tile 选择不同，故数值零风险。**不改变任何 kernel 源码**是本轮 −12% 的全部来源。

调优发现（内置启发式漏掉的配置）：
- `conv3x3|W40H40s1_Cin64_Cout64`：内置用 `OBW=8,OBH=2`；调优选 **`OBW=8,OBH=1`** → 8.46 ops。
- `W80H80s1_Cin64_Cout51`：调优选 **`OBW=10,OBH=2`** → 11.27 ops（启发式固定 8）。
- `W20H20s1_*` 系列普遍选 **`OBH=1`**（20 高时 OBH=2 会浪费一半）。

**中间标准（`--expected`）暴露的真实差距**（P0 旧口径）：

- 大层 `80×80 Cin64 Cout64`：实测 13.77 / 期望 12.15 = **ratio 1.13**。
- `40×40` 系列 ratio 0.55–0.70：网格/复用受限。
- `320×320 s2 Cin3 Cout16`：实测 1.73 / 期望 10.32 = **ratio 0.17**——离极限最远的一层。

> **R24 修正口径**：ISA 证明指令配额上限是 20.3（不是 16），于是同一批实测变成：
> `80×80` ratio **0.69**、`40×40` **0.42**、`320×320 s2 3→16` **0.11**——
> 即连大层也只有配额的 ~68%。完整逐 size 表见
> [`round24-analysis.md`](round24-analysis.md) §2。

> 结论：P0 在不改任何 kernel 源码的前提下，仅靠自动选择配置就拿到整网 −5%～−12%，
> 且中间标准能自动定位「离极限最远」的层，这正是之前缺失的「层级化自动调优」。

**调优表规模**：`config/tuning.json` = 117 条（conv1x1 ×71、conv3x3 ×31、depthwise ×15），
设备键 `Intel(R) Iris(R) Xe Graphics_eu80_clk1300`（NEO 驱动未暴露 PCI id，用 name+EU+频率）。

---

## 10. 使用与复现

```bash
# 1) 离线自检（不需要 GPU）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target tuning_test
./build/tuning_test        # 签名/缓存/期望模型 18 项自检

# 2) 列出可调优的唯一签名（不跑 GPU 计时）
./build/kernel_autotune --plan models/yolov8n-pose/model.plan --list

# 3) 生成/更新调优缓存（推荐用安全分批驱动脚本，遵守 benchmark_protocol.md）
python3 scripts/autotune.py --repo $PWD --model yolov8n-pose --ops conv3x3,conv1x1,depthwise \
    --batch 5 --iters 12 --cache config/tuning.json

# 4) 运行时消费：PlanModel 自动加载 config/tuning.json（或用 $INFVINO_TUNING_CACHE 指定）
./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 5
INFVINO_TUNING=off ./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 5  # 对照

# 5) 单节点扫候选（调试用；一条命令少量配置）
./build/kernel_autotune --plan models/yolov8n-pose/model.plan --cache /tmp/t.json \
    --op conv3x3 --limit 1 --iters 10 --report --expected

# 6) 小算子调优（R28）：每个 op 一批
./build/kernel_autotune --plan models/yolov8n-pose/model.plan --cache config/tuning.json \
    --op resize_nn --op permute_0213 --op concat4 --limit 1 --iters 8 --expected

# 7) 在线调优（P3，仅部署端按需；开发板默认关闭）
INFVINO_TUNING=online INFVINO_ONLINE_BUDGET=4 ./build/kernel_run \
    --plan models/yolov8n-pose/model.plan --report --iters 3
```

### 关键环境变量

| 变量 | 作用 |
|---|---|
| `INFVINO_TUNING_CACHE` | 调优缓存路径（默认 `config/tuning.json`，相对 cwd / 源码树）|
| `INFVINO_TUNING=off` | 强制禁用查表（走内置启发式，用于 A/B 对照）|
| `INFVINO_TUNING=online` | **P3** 在线调优：加载模型时对「未命中」的签名按需 benchmark 并 merge（需 `profiling=true`）|
| `INFVINO_ONLINE_BUDGET` | 在线调优的签名数上限（默认 4；开发板请保持小）|
| `INFVINO_ONLINE_ITERS` | 在线调优每候选计时迭代数（默认 5）|
| `INFVINO_ONLINE_OPS` | 在线调优的 op 子集（默认 `conv3x3,conv1x1,depthwise`）|

### 接入点一览（代码）

| 文件 | 内容 |
|---|---|
| `include/infvino/Tuning.hpp` / `src/Tuning.cpp` | `OpSignature` / `TuningCache` / `expected_ops`（中间标准）|
| `include/infvino/Autotuner.hpp` / `src/Autotuner.cpp` | 候选枚举 + `autotuneOp` 计时选优 |
| `src/PlanModel.cpp` `autotune()` / `tuningTargets()` | 复用 PlanModel 自己的缓冲/OSV 权重重排，按签名调优 |
| `src/PlanModel.cpp` `dispatch` 各分支 | 查表命中 → 用缓存 kernel/options；未命中 → 内置启发式 |
| `src/PlanModel.cpp` `smallSig()` / `smallLaunch()` | **R28** 小算子签名/接线的唯一来源，run 与 autotune 共用 |
| `src/tools/kernel_autotune.cpp` | 离线工具（`--list/--limit/--iters/--report/--expected/--cache/--bake`）|
| `src/tools/tuning_test.cpp` | 离线自检（CTest）|
| `scripts/autotune.py` | 容器安全分批驱动（timeout + GPU HANG 自检 + merge 缓存）|
| `config/tuning.json` | 调优缓存（3 个目标模型、本机设备键）|

### 安全（必须遵守 `docs/benchmark_protocol.md`）

- 驱动脚本把大规模扫描拆成**多批独立进程**，每批 `--limit N`、各自 `timeout`，
  每批之后检查 `/var/log/{kern.log,syslog}` 的 `GPU HANG`；一旦命中立即停止。
- 每批结果立即 merge 进缓存 → 一批失败不会前功尽弃。
- 容器 `--memory=3g --pids-limit=256`，只挂 `renderD128`。



P0 的判定标准：**在不动 kernel 源码的前提下**，仅靠自动选择配置，整网 kernel busy
不劣于 R23 基线，且调优缓存能解释每个 shape 的「实测 vs 期望」余量。

---

## 11. Round 28：P1/P2/P3 + 小算子调优

> 目标：把「自动调优基础设施」补全（P1/P2/P3），并把**小算子**（launch/带宽受限的
> copy/slice/concat/ew/pool/resize/permute/bmm/gap）也纳入同一套候选枚举 + 中间标准。
> 数值判据同 `docs/kernel.md`（`model_check` mean_rel<2e-2 & max_rel<5e-2）。

### 11.1 P1 —— gemm / conv1x1-N>1 独立候选谱系

`gemm` 节点在本代三个模型里已被 `onnx2plan` 折叠进 `conv1x1`，但 `conv1x1` 的 N>1 分支
仍复用 `candidatesGemm`。R28 补上**小 M/N tile**（`BM=32/BN=64`、`BM=64/BN=32`），
减少小层（mobilenet 的 96×49×576 类）的尾部浪费。`candidatesGemm` 就是唯一入口，
`gemm` 节点若回归也会自动受益。

### 11.2 P2 —— JIT 形状特化（编译期去谓词）

关键点：**必须是编译期常量**。R24 试过运行时 `interior` 分支，热循环里更贵
（40×40 8.77→7.30）而回退。R28 换成 `-DFIT_*` 宏，由**调用方在 build 时**决定：

| kernel | 宏 | 条件 | 去掉的谓词 |
|---|---|---|---|
| `conv3x3_ov` | `FIT_WH` | `Wout%OBW==0 && Hout%OBH==0` | 输出 oy/ox 边界 |
| `conv3x3_ov` | `FIT_COUT` | `Cout%(2·SG)==0` | 输出通道 leftover |
| `conv3x3_blk` | `FIT_WH` | `Wout%OBW==0` | 输出列边界 |
| `conv3x3_blk` | `FIT_COUT/FIT_CIN` | `Cout%16==0 / Cin%16==0` | 通道 leftover（含输入行快路径）|
| `conv3x3_f16` | `FIT_WH/FIT_CIN/FIT_CB` | `Wout%TX / Hout%TY / Cin%CINC / Cout%CB` | 输出/staging 谓词 |

`candidatesConv3x3` 按签名自动加 `-DFIT_*`（只有真的整除才加，语义不变），调优器逐 shape 实测。
因为不同 FIT 组合产生不同 `options` 字符串，它们天然被 `TuningCache` 作为**独立变体**记录。

**R28 实测**：31 个 conv3×3 签名只有少数 shape 整除，其中这些改选了 `fit` 变体（同会话）：

| shape | 旧 | R28 | Δ |
|---|---|---|---|
| `112×112 s2 3→16` | `OBW=5` | `OBW=8` + fit | 单层 ~+5% |
| `320×320 s2 3→16` | `OBW=4,OBH=4` | 同 + fit | 单层 ~+2% |
| `160×160 s2 16→32` | — | `OBW=8,OBH=2` + fit | 单层 ~+3% |

大层（`80×80 s1 64→64` 等）**不整除**（80 % 8 == 0 但 80×80 的 OBH=2 整除、已受益；
真正的瓶颈仍是 R24 的延迟/波量化），因此 P2 的整网收益小于小算子调优。
P2 的价值在于**基础设施**：调优器现在能对「形状整除」的层自动去掉谓词变体，
且随模型/size 自动生效。

### 11.3 P3 —— 在线调优（opt-in）

`PlanModel` 构造时若 `INFVINO_TUNING=online` 且队列开了 profiling，就调用
`onlineTuneMissing(budget, iters, ops)`：只处理**缓存未命中**的签名，按 `budget` 限量，
结果 merge 进内存缓存。默认**关闭**（开发板长 GPU 任务的 hang 风险，见
`docs/benchmark_protocol.md`），部署端若要自适应再显式打开。

### 11.4 小算子纳入调优

**基础设施**：`OpSignature` 增加通用 `std::vector<int> params` + `OpSignature::custom(op, params)`，
以后加一个小算子不必再改结构体字段。`PlanModel::smallSig(node)` 是 (dispatch / autotune /
tuningTargets) 三者共用的**唯一签名来源**；`smallLaunch(...)` 统一设置参数与网格几何。

**候选**（`candidatesSmall`，全部**数值等价**——逐元素表达式与归约顺序不变）：

| op | 变体 | 依据 |
|---|---|---|
| `ew_binary` / `ew_unary` | scalar vs `_v`(VEC 2/4/8) | 每 work-item 多个连续元素 |
| `ew_binary` (bdims) | 通用 bcast vs `ew_binary_ch`（通道特化）| 通道标量 × 空间（SE Mul）|
| `concat4` | scalar vs `_v`(VEC 2/4/8) | 同上 |
| `copy_c` / `slice_axis` / `maxpool` / `resize_nn` | 1-D vs 2-D/3-D 网格 | 去 per-element div/mod |
| `permute_0213` | 1-D vs `_3d` | 同上 |
| `bmm` | 1-D vs `bmm2`(3-D) | 同上 |
| `gap` | `gap_r` WGS 64/128/256 | 归约粒度 |

**R28 实测**（一次小算子 sweep，196 条缓存；同会话 A/B）：

| 模型 | baseline(off) | **R28 缓存** | 加速 |
|---|---|---|---|
| yolov8n-pose | 17.10 ms | **14.68 ms** | −14.2% |
| yolo11n-pose | 19.92 ms | **17.04 ms** | −14.5% |
| mobilenetv3-small | 3.07 ms | **2.83 ms** | −7.9% |

单算子亮点（yolov8 per-run）：`resize_nn` 0.227→**0.114**（选 `resize_nn3`）、
`permute` 0.063→**0.032**（`_3d`）、`gap`/`copy_c` 全形状改选 2-D/3-D 网格。

**数值**：`model_check.py`（vs onnxruntime，缓存生效）
yolov8 `mean_rel=5.28e-4 / max_rel=8.87e-3`、yolo11 `8.97e-4 / 1.96e-2`、
mobilenet `1.31e-2 / 1.09e-2`，**三模型 PASS**。

> `bmm`/`softmax` 仍偏慢（yolo11 0.80/0.54 ms）：`bmm2` 只是去索引开销，
> 计算仍是「1 WI/输出 + 串行 K」，本质是 R22 记的 attention 网格饥饿，属 kernel 本体待改。

---

## 12. Round 29：串行数据通路的专用 kernel（attention bmm / softmax）

> 用户问题：bmm 这类算子「软件流水没写、串行堵塞」。用 R18.7 的硬件模型
> `ops/EU/cyc = 32 × mad_frac × SIMD/16` 分析并写专用 kernel。

### 12.1 诊断（硬件模型 + ops/EU/cyc）

yolo11 的 attention（R28 缓存）实测：

| 算子 | ms | FLOPs | ops/EU/cyc | 卡在哪 |
|---|---:|---:|---:|---|
| `bmm` 1（400×400×32, B2）| 0.27 | 20.5 M | **0.73** | 每输出 1 WI、串行 K、A/B 每次重载（mad_frac≈0.33）|
| `bmm` 2（64×400×400, B2）| 0.54 | 41.0 M | **0.73** | 同上，且 B 被每个 m-tile 重读 |
| `softmax`（800×400）| ~0.47 | — | — | **1 WI 串行扫 400 三遍**，仅 800 WI → 延迟堵塞 |
| `softmax`（1×16×33600）| 0.07 | — | — | 轴短、inner 大，串行已合适 |

FP32 峰值为 16 ops/EU/cyc（8 FMA/EU/cyc）；`0.73` = **4.6%**。Isa 反汇编确认
`bmm_t`（初版）762 条指令里只有 64 条 `mad`、289 条 `mov`、52 条 `send.dc0`——纯标量、
无向量载入、无流水。

### 12.2 专用 kernel

- **`bmm_t`（寄存器分块）**：一个 WI 算 `BMM_TM×BMM_TN` 个输出，A 复用 TN 次、B 复用
  TM 次（标量版每次都从 L3 重载）。K 按 `BMM_UK` 展开、B 用 `half8` 显式向量载入，
  把「K 次依赖链」变成 `BMM_UK` 路独立链以隐藏 `half→float`/FMA 延迟。fp32 累加，
  K 顺序与旧 `bmm` 一致 → **逐位相同**（kernel_check 实测 `max_abs=0`）。
- **`softmax_axis_r`（并行归约）**：一个 work-group 处理一行，`SM_WGS` 个 lane 分块扫
  `axdim`，SLM 上做 fp32 max/sum 树归约（三趟：max → exp-sum → 写回）。
  轴长时选它，轴短（dfl 16）时串行版更优，由调优器按 shape 选。

### 12.3 R29 实测

| 指标 | R28 | **R29** | 加速 |
|---|---:|---:|---:|
| bmm 合计 | 0.810 ms | **0.305 ms** | **2.66×** |
| softmax 合计 | 0.544 ms | **0.128 ms** | **4.25×** |
| yolo11 kernel busy | 17.04 ms | **16.11 ms** | −5.5% |

调优器选择：`bmm` → `bmm_t`（`TM8x4u4` / `TM4x4u4`）、attention softmax → `softmax_axis_r`
（`WGS=64`）、dfl softmax → 串行 `softmax_axis`。

**数值**（`kernel_check.py`，新增 bmm/softmax 用例）：bmm 三个变体
（`bmm`/`bmm2`/`bmm_t`）**逐位相同**（`max_abs=0`）；`softmax_axis_r` vs numpy FP32
`mean_rel≈8e-8`，与串行版一致。整网 `model_check` 三模型 PASS。

### 12.4 mobilenet logits 的 `mean_rel≈1.3e-2`（口径澄清）

用户质疑该值偏大。查证结论：**既非抖动也非 bug**。

- 3 次运行输出**逐位相同**；R22 起记录一直是 1.31e-2。
- 误差来源：classifier 前 ~50 层 fp16 激活累积，使 `Flatten` 输出 `h` 有 mean rel
  `8.9e-3`；classifier 权重 `sum|W3|≈73` 把它放大成 logits 绝对误差 `1.6e-2`；
  而 `mean_rel = 1.6e-2 / mean|logits|(=1.22) = 1.3e-2`。
  直接验证：用 infvino 的 `h` 与参考 `h` 经 fp64 classifier，得到 logits 误差
  `mean=1.598e-2`，与实测 `1.593e-2` 吻合。
- yolo 的输出是坐标（`|x|≈200`），同样的绝对误差读成 `5e-4`；**两者数值质量一致**，
  差异纯粹来自输出量级。
- 已给 `model_check.py` 增加尺度无关指标：`quant_rel`（逐元素相对误差中位数）、
  `ulp_frac`（落在 1 ULP 内的比例）、`scale_rel`（误差/输出动态范围），避免分类
  logits 这类小量级输出被误读。

### 12.5 同问题的另一个算子：depthwise（滑窗寄存器复用）

`depthwise_f16` 也是「每个输出 K×K 次独立全局载入 + 一次 mad」——和 bmm 同一类
（无复用/无流水，实测 ~0.3 ops/EU/cyc）。R29 增加 `depthwise_v`：
一个 work-item 沿 x 算 `DW_TW` 个连续输出，输入条带 `(DW_TW-1)·DW_S + DW_K` 一
次载入逐 tap 复用，K×K 权重一次进寄存器；累加顺序与标量版一致 → **逐位相同**。

| 模型 | depthwise（标量）| **depthwise_v** | 加速 |
|---|---:|---:|---:|
| yolo11 | 0.776 ms | **0.562 ms** | 1.38× |
| mobilenet | 0.440 ms | **0.391 ms** | 1.13× |

调优器对 15 个签名选 `depthwise_v` 8 个（大层）、标量 7 个（小层）。

### 12.6 R29 收尾（三模型端到端）

`kernel_run --report --iters 5`（同会话 A/B）：

| 模型 | baseline(off) | R28 | **R29** |
|---|---:|---:|---:|
| yolov8n-pose | 17.29 ms | 14.61 | **14.61** |
| yolo11n-pose | 19.80 ms | 17.04 | **15.86** |
| mobilenetv3-small | 3.08 ms | 2.83 | **2.81** |

新增 op 级数值覆盖：`kernel_check.py` 增加 bmm（3 变体）、softmax（2 变体）、
depthwise（3 变体）用例——**ALL PASS**（bmm/depthwise 变体逐位相同）。

### 12.7 R31 —— `depthwise_vp`（padded）候选：opt-in + pad 成本入账

R31 增加了 `depthwise_vp`（零边 `depthwise_pad` + 无边界卷积）候选，并把它从普通候选里
**单独拆出来计时**：vp 的每帧真实成本 = `depthwise_vp` + **单独量到的 pad 一趟**
（`benchCandidate` 量 `depthwise_pad` 后加到 vp 的 ms 上，再与其它候选比）。
这样「额外一趟 pad」不会被漏算（对比 §4「运行时接入」里普通候选只比单 kernel 时间）。

实测（见 `docs/kernel.md` Round 31）：vp 把 depthwise 分项砍 7–11%，但 pad 是带宽受限的
额外一趟，二者相当，**整网持平/略负**（y11 +0.6%、mb 持平）。因此：

- vp 候选**默认不进候选集**；只有 `INFVINO_DW_PAD=1` 时才枚举（便于复验/换硬件再试）。
- 生产 `config/tuning.json` 不含 vp 条目；`kernel_check.py` 仍保留 vp 的算子级用例
  （TW4/8，vs numpy PASS、与 `depthwise_v` 逐位相同）。
- 该 op 的 `expectedOps` 仍按 R30 的 ISA 配额（不随 kernel 变体改变，只用于 ratio 报告）。

### 12.8 R31 —— P2 host 分段统计（供调优之外用）

`PlanModel::run()` profiling 下新增 `hostEnqueueMs()`（入队提交累计）与
`hostWaitMs()`（同步等待累计），`kernel_run --report` 打印
`wall / busy / enqueue / sync / host_total / setarg_est`。这不是调优项，而是 P2 的
「先测什么」：实测 busy 只占墙钟 52–74%，host 开销 = 入队 + setArg/其它两块。
数据与结论见 `docs/benchmark.md` §2.4。


