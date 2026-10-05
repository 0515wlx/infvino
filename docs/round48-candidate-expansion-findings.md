# R48：系统性扩充算子候选集 —— 进展（M0 + 首个候选 D1-YB）

> 依据 [`round48-candidate-expansion-plan.md`](round48-candidate-expansion-plan.md) 执行。
> 本文件记录**已完成的基础设施（M0）**与**第一个落地的扩充候选（D1：conv1x1_blk 输出行
> tiling `Y_BLOCK`）**，附隔离实测与复现命令。整网外部稳态 A/B 作为下一步（§4）。

---

## 0. 结论先给

- **M0 基础设施已落地**：候选预算（每签名上限 + 族配额，确定性截断）、数值契约标记
  （`Candidate`/`TuningEntry`/缓存 JSON + `model_check` 按候选放宽）、L3/占用模型解析新
  tiling 几何、`kernel_autotune --candidates` 零 GPU 审计。`tuning_test` PASS。
- **首个扩充候选（D1）落地**：`conv1x1_blk` 增加 `-DY_BLOCK=`（每 WI 处理 Y_BLOCK 个输出
  行，权重跨行复用）。**逐位一致**；大 spatial 层隔离 min 比 XB4/YB1 快 **1.2–1.6×**，
  对最优既有候选仍是 **1.0–1.3×** 边际收益；小 spatial 层变慢——正是计划要的
  **跨瓶颈候选**（大层走 tiling、小层保留原状，由 autotune/per-shape 选择）。
- **默认路径零回归**：`conv1x1_blk` 的 `Y_BLOCK` 默认 1，旧缓存 options 无 `-DY_BLOCK`
  → 行为不变；`model_check` mobilenet PASS（mean_rel 1.07e-2）。

---

## 1. M0：基础设施（计划 §3）

| 计划项 | 落地 | 位置 |
|---|---|---|
| §3.1 每签名候选上限 + 族配额 | 确定性预算：族超配额等距采样（含首尾），跨族轮转交错后取签名上限。默认 `kFamilyCandidateQuota=32`、`kSigCandidateCap=64`（**当前最大 26，不触发**，属安全护栏）；env `INFVINO_FAMILY_QUOTA`/`INFVINO_SIG_CAP` 覆盖，`INFVINO_CAND_STATS=1` 打印截断 | `KernelFamily.hpp`、`KernelFamilies.cpp` |
| §3.2 数值契约标记 | `Candidate.exact/tol` → `TuningEntry.exact/tol` → 缓存 JSON `numeric_exact`/`numeric_tol`；`model_check.py` 读缓存**按被选候选**放宽 mean_rel（max_rel=2.5×tol） | `Autotuner.hpp`、`Tuning.hpp/.cpp`、`Autotuner.cpp`、`scripts/model_check.py` |
| §3.3 L3/占用模型接入候选 | `occupancyPressure` 解析 `conv1x1_blk` 的 `-DX_BLOCK`/`-DY_BLOCK`/`-DSLM_DIV`，按 tile×Cout/16 算并发与每 WG 足迹 | `PlanModel.cpp` |
| §3.4 隔离口径为准 | 新候选一律以隔离 `min` 评估（`benchCandidate` 已用 min）；整网留外部 A/B | `Autotuner.cpp` |
| §3.5 文档/上限模型 | 新候选沿用 `conv1x1_blk` 的 `ceiling`/`hardCeiling=11.5`（族级上界） | `KernelFamilies.cpp` |
| 候选规模审计 | `kernel_autotune --candidates`：逐签名候选总数 + 按族分解 + 超预算告警（零 GPU） | `src/tools/kernel_autotune.cpp` |

**候选规模现状**（`--candidates`，零 GPU）：

| 模型 | 签名数 | 总候选 | 单签名最大 |
|---|---:|---:|---:|
| mobilenetv3-small | 57 | 481 | 49（conv3x3） |
| yolov8n-pose | 79 | 1398 | 49 |
| yolo11n-pose | 111 | 1789 | 49 |

> 默认护栏（cap 64）尚未触发；它约束的是后续 D1–D6 继续加候选时的规模。

---

## 2. 首个扩充候选：D1 输出行 tiling（`conv1x1_blk` 的 `Y_BLOCK`）

### 2.1 动机（与计划的对应）

计划 D1：「对 conv1x1 增加输出 tiling 候选，让工作集/复用落在更优区间」。对 blocked 1x1
（lane=输出通道），**权重读是发射/带宽主项**：每个输出位置组每 K 步都要重读 lane 自己的
16 个输入通道权重，而输入只需读 X_BLOCK 列。新增 `Y_BLOCK`：每 WI 处理 Y_BLOCK 个连续输出
行，`wei[16]` 跨行复用，把权重读摊薄 Y_BLOCK 倍。

### 2.2 实现

- `kernels/conv1x1_blk.cl`：`-DY_BLOCK`（默认 1）；`dst[Y_BLOCK]` 独立累加器，K 循环内权重
  只读一次、逐行读 src 并 mad；行越界（`y>=H`）安全跳过。**数值逐位一致**（每个输出的 K
  累加顺序不变；仅重排/复用权重读）。
- 注册表：`conv1x1_blk` 候选新增 `(YB=2: XB∈{2,4,8})`、`(YB=4: XB∈{2,4})`，约束
  `XB*YB<=16`（寄存器预算），且 **YB>1 不叠 split-K**。族候选 9 → 14，单签名总候选 ≤26。
- dispatch：`run()` 与 `autotune` 的 `makeEnqueue` 按 `-DY_BLOCK` 计算 `gws[0]`；
  `occupancyPressure` 已见其几何（§1）。
- `kernel_bench`：`Conv1x1BlkCfg.YB` + `--conv1x1blk XB,SLM,ACT,OUTFSV16,RES,YB`。

### 2.3 隔离实测（iris Xe / 80EU / 锁频；`--iters 20`；vs FP32 numpy `--verify`）

（表中 `×` 为加速比 = 基线 ms / 该配置 ms。）

| shape (Cin,Cout,H,W) | XB4/YB1 | 最优既有 XB8/YB1 | **YB 胜者** | vs 基线 | vs 最优既有 | 数值 |
|---|---:|---:|---:|---:|---:|---|
| 64,64,80,80 | 0.175 | 0.157 | **0.136 (XB8/YB2)** | **1.29×** | **1.15×** | 逐位一致 |
| 128,96,40,40 | 0.102 | 0.078 | **0.077 (XB8/YB2)** | **1.32×** | 1.01× | 逐位一致 |
| 64,32,80,80 | 0.104 | 0.090 | **0.081 (XB8/YB2)** | **1.28×** | **1.11×** | 逐位一致 |
| 32,64,80,80 | 0.162 | 0.137 | **0.132 (XB8/YB2)** | **1.23×** | 1.04× | 逐位一致 |
| 64,64,40,40 | 0.057 | — | **0.052 (XB8/YB2)** | **1.10×** | — | 逐位一致 |
| 128,64,40,40 | 0.071 | — | 0.070 | 1.01×（噪声内） | — | 逐位一致 |
| 576,96,7,7 | 0.060 | — | 0.098 | **0.61×（小 H，网格饥饿）** | — | 逐位一致 |
| 240,40,14,14 | 0.027 | — | 0.041 | **0.66×** | — | 逐位一致 |

**yolov8n 精确大 spatial 签名**（`--iters 20`，单位 ms；「最优既有」= 原候选集里最快者）：

| Cin,Cout,H,W | XB4/YB1 | 最优既有 | **YB 胜者** | vs 基线 | vs 最优既有 | 数值 |
|---|---:|---:|---:|---:|---:|---|
| 32,32,160,160 | 0.312 | 0.312 | **0.244 (XB8/YB2)** | **1.28×** | **1.28×** | 逐位一致 |
| 64,64,80,80 | 0.182 | 0.182 | **0.143 (XB8/YB2)** | **1.27×** | **1.27×** | 逐位一致 |
| 51,51,80,80 | 0.164 | 0.164 | **0.123 (XB8/YB2)** | **1.33×** | **1.33×** | 逐位一致 |
| 128,128,40,40 | 0.142 | 0.119 | **0.098 (XB4/YB4)** | **1.45×** | **1.21×** | 逐位一致 |
| 256,128,40,40 | 0.186 | 0.148 | **0.116 (XB8/YB2)** | **1.60×** | **1.28×** | 逐位一致 |
| 64,1,80,80 | 0.038 | 0.025 | 0.025 (XB8/YB2) | 1.52× | 1.00× | 逐位一致 |

读法：
- **大 H（≥40）** tiling 胜，**小 H** 因 grid 变小而饥饿 → 这正是「按场景分族」；
  autotune 会按签名（含 H）选择，不会误用。
- YB=4 同向（64,64,80,80：XB4/YB4=0.142）——已作为候选加入（XB≤4）。
- 所有 YB 变体与 YB=1 **mean_rel 完全相同**（如 64,64,80,80 均为 1.039e-03），确认逐位一致。

---

## 3. 复现

```bash
# 构建（容器，CPU）
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake --build build-blk -j4 --target kernel_bench kernel_autotune tuning_test'

# M0 自检（零 GPU）
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'export LD_LIBRARY_PATH=$PWD/build-blk; ./build-blk/tuning_test'

# 候选规模审计（需 GPU 建 PlanModel）
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'export LD_LIBRARY_PATH=$PWD/build-blk; \
  ./build-blk/kernel_autotune --plan models/mobilenetv3-small/model.plan --candidates'

# D1-YB 隔离 A/B + 数值（形状大 H 时 YB2 胜）
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'export LD_LIBRARY_PATH=$PWD/build-blk; \
  ./build-blk/kernel_bench --op conv1x1blk --conv1x1blk 8,1,0,0,0,1 --conv-shape 64,64,80,80 --iters 20 --verify; \
  ./build-blk/kernel_bench --op conv1x1blk --conv1x1blk 8,1,0,0,0,2 --conv-shape 64,64,80,80 --iters 20 --verify'

# 默认路径数值回归
python3 scripts/model_check.py --model mobilenetv3-small --repo "$PWD"
```

---

## 4. 下一步（计划 M1 收尾 → M2）

见 §6/§7 的整网验证结论：**本次候选扩充整网收益未确认**，故 M1 暂缓，
先解决 §7 暴露的布局/融合设计问题。`config/tuning.json` 未改动。

## 5. 风险与边界（本轮）

- `Y_BLOCK` 只在大 H 生效；小 H 变慢属预期，由 autotune 按签名规避。
- 候选数 9→14（单签名 ≤26）在当前护栏内；继续加候选时须关注 IGC JIT 暴露量。
- 数值契约字段对旧缓存向后兼容（缺省=逐位一致），无 ABI 破坏。
- **ABI**：`TuningEntry` 增字段改变了结构体布局，旧的可执行文件与新 `libinfvino.so`
  混用会 segfault（本轮实测 `kernel_run` 旧二进制崩溃）。必须整体重建；缓存格式向后兼容。

---

## 6. 整网外部稳态 A/B（YB 裁决）

协议：同会话、锁频、`kernel_run --iters 3 --report` 取 per-frame `busy`，base/cand **交错**
各 5 次，比较 min 与 median。工作缓存：`config/tuning.json` 的副本经
`kernel_autotune --op conv1x1 --retune --iters 12` 重扫（新候选参与）。

### 6.1 结果（单位 ms，per-frame busy）

| 模型 | base min / median | cand（含 YB）min / median | 结论 |
|---|---|---|---|
| yolov8n-pose | 14.003 / 14.057 | 14.016 / 14.029 | **−0.2%（噪声内）** |
| mobilenetv3-small | 2.195 / 2.227 | 2.208 / 2.232 | **+0.2%（噪声内）** |

- yolo：只有 5 个 **N=400**（小 spatial）签名被重选，且多为噪声级（±5%）；大 spatial
  （N=1600/6400/25600）**仍全部选 gemm**——YB 的 blk 版本虽变快（如 64×6400：0.148→0.133），
  仍慢于 tuned gemm（`#non` = 0.092）。故 YB 无从被选中。
- mobilenet：重选的 YB2 极少（如 `Cout576_N49_Cin96`），多数是 `YB=1` 的等价项；
  YB2 常因 **寄存器压力**（XB8×YB2=16 个半累加器）反而更慢。

### 6.2 关键：大 spatial 1×1 上 GEMM 本就更快（与 reorder 无关）

隔离对比（`kernel_bench`，ms）：

| M=Cout,N,K=Cin | tuned GEMM(NCHW) | BLK XB8/YB2(fsv16) | GEMM 优势 |
|---|---:|---:|---:|
| 64,6400,64 | **0.092** | 0.133 | 1.45× |
| 128,1600,128 | **0.081** | 0.091 | 1.12× |
| 32,25600,32 | **0.154** | 0.240 | 1.56× |

→ 大 spatial pointwise 的最优族是 **NCHW GEMM**，不是 blocked；reorder 只是让 blocked 更差。
**因此"候选集缺 blocked 变体"对这类签名不是瓶颈。**

## 7. 倒查：reorder 税与系统设计

### 7.1 现状事实（结构剖面，`kernel_run --report`）

| 模型 | 节点 | **持久 fsv16 张量** | reorder 次数/帧 | reorder ms/帧 | 占 busy |
|---|---:|---:|---:|---:|---:|
| mobilenetv3-small | 74 | 3 | 19 | 0.133 | **~6%** |
| yolov8n-pose | 128 | 0 | 5 | 0.067 | ~0.5% |

**消融**（把所有 `_blk` 选择替换为对应 `#non`，即强制全 NCHW，消除 reorder）：

| 模型 | 有 blocked（base） | 全 NCHW（no_blk） | 结论 |
|---|---:|---:|---|
| yolov8n-pose | 14.057 | 14.019 | blocked **净负 −0.3%** |
| mobilenetv3-small | 2.227 | 2.286 | blocked **净正 +2.7%**（其 kernel 增益 > reorder 税） |

读数：mobilenet 的 reorder 税 ~6% 很大，但被 blk kernel 增益盖过（净 +2.7% 于全 NCHW；
即若无 reorder，本可再多 ~6%）。

### 7.2 根因链（系统设计）

1. **blocked 族要求输入 `b_fs_yx_fsv16`**（lane=输出通道）；NCHW 生产者必须先
   `reorder_bfyx_to_fsv16` 物化一趟。
2. **布局只有持久化/不持久化二选一**：`planBlockedLayout` 仅在**所有消费者都吃 FSV16** 时
   持久化；一旦有 NCHW 消费者（conv3x3/GEMM），该张量保持 NCHW。
3. **conv3x3 的 blocked 族在本机不占优**：ISA 配额 `conv3x3_blk`=16.9 < `conv3x3_ov`=20.3，
   实测 W80H80/Cin64：blk 0.334 > ov 0.302。→ conv3x3 几乎全选 NCHW。
4. 于是 blocked 只能**孤立**出现在 1×1/depthwise 节点 → 每次进出 persistent 链都付一次
   reorder（mobilenet 19 次/帧）。**fsv16=0（yolo）/仅 3（mb）** 就是证据。

**结论：reorder 税不是"reorder kernel 没写快"，而是"布局图被 NCHW 的 conv3x3 主族切碎"，
转换只能以独立 pass 形式出现。** reorder 本身接近其地板（mb ~7µs/次 ≈ launch 3.5µs + 小张量搬运），
没有可观的微优化空间。

### 7.3 设计层面的不足（反思）

| # | 不足 | 证据 | 方向（对应计划） |
|---|---|---|---|
| A | **布局转换是独立 dispatch**，不是 producer epilogue / consumer prologue | mb 19 reorder/帧，6% busy；reorder 已近地板 | **D4** 内核级融合（producer 直写 fsv16 / consumer 直读 NCHW） |
| B | **布局选择是"保守计费 + 选后规划"，非联合搜索** | `conv1x1` 候选按"一趟 reorder"计费；R47 发现 2-pass 不动点本机无收益，但前提是 reorder 不可融合 | D4 后重估；计划 §15 开放项 |
| C | **blocked conv3x3 结构上弱于 NCHW**，使"持久 blocked 链"架构无法成立 | ISA 16.9<20.3；W80 blk>ov | 要么提升 blk conv3x3，要么放弃 3×3 的 blocked 链 |
| D | **"加族只加声明"并未完全成立**：新 kernel 仍需在 `PlanModel::run`/`makeEnqueue`/
`occupancyPressure` 三处补 launch 绑定 | 本轮 YB 改了 4 处（含 kernel_bench） | 注册表应把 `launch` 也声明进去（`KernelFamily` 设计稿里有 `launch` 字段，未落地） |
| E | **候选扩充方向与真实瓶颈正交** | 大 N：GEMM 已最优；小 N：blk 已最优；YB 无处发挥 | 下一步应是**布局图/融合**而非更多同族候选 |

### 7.4 对计划 D1–D6 的建议（供决策）

- **优先级应重排**：`D4（reorder 融合）` 的期望收益最明确（mb 潜在 ~6%），应提前到
  D1/D6 之前；`D2 split-K conv3x3` 与 `D1 conv3x3 输出 tiling` 需先用 R47 模型证明
  conv3x3 大层不是"无 L1 + 128-GRF ILP"结构墙（R41 已判大概率负结果）。
- **D6（direct conv1x1 / depthwise_v2）** 只有在"窄通道/小 N"区间能超过 gemm/blk 才有意义，
  需先离线证明。
- **候选层面的继续扩充（D1/D3）ROI 存疑**（§6.2）：大 spatial 已由 GEMM 覆盖，blocked
  加变体不改变选择。

## 8. 状态

- 整网收益 **未确认**（yolo −0.2%、mb +0.2%，均噪声内）→ 按用户约定 **暂不 commit**。
- 工作树含：M0 基础设施 + D1-YB 候选 + 数值契约 + 本文档（未提交）。

## 9. 下一步（待用户决定）

1. 是否 **commit** 当前工作（M0 基础设施 + D1-YB + 负结果记录）？
2. 是否按 §7.4 **把 D4 提前**、暂缓 D1/D2/D3/D6？
