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

---

## 9. 落地节奏

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0（已完成）** | TuningCache + OpSignature + ExpectedOps + conv3x3/conv1x1/depthwise 候选枚举 + kernel_autotune + PlanModel 查表 + bake + tuning_test | 三模型调优表生成；yolov8 busy −7.4%；数值逐位不变 |
| P1 | gemm 独立候选实测（本代模型 1×1 已折叠成 conv1x1，gemm 节点少）| — |
| P2 | JIT 形状特化（Wout/Hout 整块、Cin 整除）| 大层 +x% |
| P3 | 在线 autotune（首推断按需调优子集，带缓存）| 部署端自适应 |

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
```

### 关键环境变量

| 变量 | 作用 |
|---|---|
| `INFVINO_TUNING_CACHE` | 调优缓存路径（默认 `config/tuning.json`，相对 cwd / 源码树）|
| `INFVINO_TUNING=off` | 强制禁用查表（走内置启发式，用于 A/B 对照）|

### 接入点一览（代码）

| 文件 | 内容 |
|---|---|
| `include/infvino/Tuning.hpp` / `src/Tuning.cpp` | `OpSignature` / `TuningCache` / `expected_ops`（中间标准）|
| `include/infvino/Autotuner.hpp` / `src/Autotuner.cpp` | 候选枚举 + `autotuneOp` 计时选优 |
| `src/PlanModel.cpp` `autotune()` / `tuningTargets()` | 复用 PlanModel 自己的缓冲/OSV 权重重排，按签名调优 |
| `src/PlanModel.cpp` `dispatch` 各分支 | 查表命中 → 用缓存 kernel/options；未命中 → 内置启发式 |
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
