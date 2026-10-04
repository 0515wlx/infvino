# Round 33：conv3×3 剩余空间分析（三条数据通路）+ CINC 候选落地

> 目标模型：`yolov8n-pose` / `yolo11n-pose` / `mobilenetv3-small`。
> 硬件：Intel Iris Xe（TGL，80 EU / 1.3 GHz / 128 GRF / 无通用 L1）。
> 数值判据：`kernel_check`（mean_rel<1e-2 & max_rel<5e-2）与 `model_check` 同口径。
> 本文回答：「conv3×3 还有没有优化空间，尤其在我们已经赢 OV 的地方」。**主要贡献是一份
> 口径修正 + 一个已落地的正结果（CINC 候选）**，不改生产行为（候选只在重新 sweep 后生效）。

---

## 0. 一句话结论

1. **"赢 OV" 的两块区域，剩余空间截然不同**：
   - 小空间/大通道（`blk` 赢）：已到各自 ISA 配额的 **0.4–0.6**，剩下是波量化/延迟，属硬件/问题切分。
   - 窄通道（`native` 赢）：真 ratio 只有 **0.23–0.42**，是全部 31 条里最低的一档，**且调优器对它的候选集几乎是空的**——这是唯一还值得做的 kernel 方向。
2. **发现并修正一个度量问题**：`config/tuning.json` 里 blk/native 的 `ratio` 是用 **osv32 的几何**
   （写死 OBW=8/OBH=2）算的，对这两条通路失真。按各自 ISA 配额重算后，**blk 的范围整体上移**，
   `20×20 256→64` 从"看起来 0.89"变成真实的 0.40。
3. **落地两个正结果（都来自同一个"填充浪费"类问题）**：
   - **R33**：把 `CINC` 纳入 native 候选。目标层 `160×160 s1 8→16`（Cin=8 被 CINC=16 半空填充）
     实测 **3.73 → 5.41 ops/EU/cyc（+45%）**，数值**逐位一致**；调优器端到端已选出 `CINC8`。
   - **R34**：把 CINC 泛化到**小 Cin**（`Cin≤16` 用精确 `CINC=Cin`，另加 `CINC=4` 兜底）。stem
     `320×320 s2 3→16`（Cin=3）**2.97 → 3.56 ops（+20%）**，数值逐位一致。
4. **主导层确认触顶**：`40×40 s1 64→64`（y8 ×10 = 1.35 ms）与 `80×80 s1 64→64`（×4 = 1.33 ms）
   经多次重复测量已在各自配额 ~0.5 / ~0.6 的平台；显式 `kd` 循环 unroll×4 **无收益（负结果）**。

---

## 1. 三条数据通路的现状（`config/tuning.json` 31 条 conv3×3 签名）

| 通路 | 文件 | 结构 | 选中的签名数 |
|---|---|---|---:|
| `conv3x3_f16`（自研） | `kernels/conv.cl` | lane=空间，SLM halo + 寄存器展开 CB 通道，TM=1 | **5** |
| `conv3x3_ov` | `kernels/conv_ov.cl` | lane=输出通道（2 ch/lane），无 SLM，`sub_group_broadcast` | **13** |
| `conv3x3_blk` | `kernels/conv_blk.cl` | lane=输出通道（1 ch/lane），blocked 输入，OBW 连续列/WG | **13** |

三条通路按 shape 分工，这一点 R22–R26 已经建立。本文聚焦两个问题：**"赢在哪"** 和 **"还能追多少"**。

---

## 2. 口径修正：expectedOps 对 blk/native 是失真的

`src/Tuning.cpp::expectedOps` 对 `conv3x3` 一律用 **osv32 的几何**：

```cpp
const int obw = (s.stride == 2) ? 5 : 8;
const int obh = (s.stride == 2) ? 4 : 2;   // ← 写死，与选中的 kernel 无关
const long n_wg = ((W+obw-1)/obw) * ((H+obh-1)/obh) * (((Cout+1)/2+15)/16);
```

但 `blk` 的实际 WG 是 `(W/OBW) × H × (Cout/16)`（高 1 行/WG），native 是 `(W/TX) × (H/TY) × (Cout/CB)`
（每 WG 20 个 sub-group）。于是：

- **blk 的 grid 比 osv32 模型大 2–4×** → 它的 `ratio` 被系统性**低估**；
- native 的 WG 口径完全不同 → `ratio` 不可与 ov 直接横向比较。

用 `docs/register-model.md` 里三条通路各自的 ISA 反汇编重算"真配额"：

| 通路 | ISA 配额（`32×mad_frac`） | 出处 |
|---|---:|---|
| `conv3x3_ov` OBW8/OBH2 | **20.3** | R24：288 mad / 453 指令 = 63.6% |
| `conv3x3_ov` OBW8/OBH1 | 16.1 | register-model §4（mad_frac 0.503） |
| `conv3x3_blk` OBW8 | **16.9** | register-model §4（mad_frac 0.529） |
| `conv3x3_blk` OBW4 / OBW2 | 14.0 / 10.9 | register-model §4 |
| `conv3x3_f16` staging-free | **16.4** | R18.7（整核 ~10.3 含 staging） |

**结论**：调优器用来排"哪层离极限最远"的 `ratio`，在 **18/31（blk+native）上是坏的**。
建议（本文未改代码，属下一步）：`expectedOps` 按选中 kernel 家族分别给配额。

---

## 3. "已经赢 OV" 的两块区域

### 3.1 native 窄通道（`Cout≤32` 且 `Cin≥8`）——空间最大

OV 两个移植通路都是 **lane=输出通道**：`osv32` 每 sub-group 覆盖 32 个输出通道（`fid=0/1`），
`blk` 每 WG 16 个。`Cout≤32` 时空闲 lane 被结构性浪费（`osv32` 的 `fid=1` 在 `Cout=16` 时整半作废），
而 native 是 **lane=空间**，Cout 多小都不浪费。5 条 native 胜出签名：

| shape | 实测 ops | / 配额 16.4 | 说明 |
|---|---:|---:|---|
| 160×160 s1 **8→16** | 3.77 | **0.23** | Cin=8 < CINC=16，半 chunk 打在零通道上（本轮的 CINC 修复点） |
| 80×80 s1 16→32 | 6.49 | 0.40 | osv32 lane 半空 + ~0.7 波 |
| 80×80 s1 32→16 | 6.78 | 0.41 | 同上 |
| 160×160 s1 16→16 | 6.66 | 0.41 | Cout=16，OV 主循环 2× 浪费 |
| 40×40 s1 32→64 | 6.87 | 0.42 | 网格 ~0.7 波 |

### 3.2 blk 小空间/大通道——已接近自身上限

`blk` 赢在把 WG 数从 `spatial×Cout/32` 提到 `spatial×Cout/16`（×2–4），治 R18.4/R24 的**波量化**。
按它自己的配额 16.9 重算：

| shape | 实测 ops | /16.9 | 备注 |
|---|---:|---:|---|
| 40×40 s1 128→64 | 9.93 | **0.59** | 还有空间 |
| 40×40 s1 64→64 | 8.40 | 0.50 | y8 主导层（×10，~1.35 ms） |
| 20×20 s1 256→64 | 6.77 | 0.40 | 旧口径显示 0.89（失真） |
| 20×20 s1 64→64 | 5.31 | 0.31 | |
| 40×40 s1 32→32 | 4.87 | 0.29 | |

`blk` 的 OBW8 已是半字向量累加的寄存器极限（`dst half8` + `line_cache` + `wei[16]` ≈ 34 数据 GRF
+ ~93 overhead ≈ 127），OBW16 必 spill。剩余缺口与 ov 同为延迟/占用。

---

## 4. R33 落地：native 的 CINC 候选（正结果，已提交到 `Autotuner.cpp`）

### 4.1 问题

`Autotuner.cpp::candidatesConv3x3` 对 native 只枚举 `TX∈{40,20} × CB∈{32,16}`，**CINC 写死 16**。
当 `Cin` 不是 16 的倍数时，最后一个 chunk 里 `gc≥Cin` 的通道全程按 0 参与 mad——纯发射槽浪费。
目标层 `160×160 s1 8→16`（`Cin=8`）正好踩中：**一半的 mad 是空的**。

### 4.2 改动

`src/Autotuner.cpp`：native 循环把 `CINC` 也纳入候选，集合为 `{16} ∪ (Cin%16!=0 ? {8} : ∅) ∪
(Cin≤16 ? {Cin} : ∅) ∪ (Cin≤8 ? {4} : ∅)`（去重、有界，见 §4.5）。`FIT_CIN` 的条件同步改成
`Cin % CINC == 0`。`PlanModel` 的 tuned-native 回放**无需改动**：CINC 只影响 kernel 编译与 SLM 占用，
不影响 launch 几何（`gws/lws` 由 TX/TY/TM/CB 决定），而 kernel 本就是用 `te->options`（含 `-DCINC=…`）构建的。

### 4.3 A/B 实测（`models/yolo11n-pose`，`W160H160s1p1_Cin8_Cout16`）

`kernel_bench --op conv3x3 --conv-shape 8,16,160,160 --conv <cfg> --iters 10`：

| 配置 | ms | ops/EU/cyc | Δ |
|---|---:|---:|---:|
| **旧：TX40 CB16 CINC16** | 0.152 | 3.73 | — |
| TX40 CB16 **CINC8** | **0.105** | **5.41** | **+45%** |
| TX20 CB16 CINC8 | 0.106 | 5.36 | +44% |
| TX40 CB32 CINC8 | 0.146 | 3.88 | +4%（CB32 对 Cout16 浪费一半） |

调优器端到端（`--only 160x160 --limit 3 --iters 8`）选出：
`conv3x3_f16 TX40 TY8 TM1 TN8 CB16 CINC8` → **0.091 ms / 6.23 ops / ratio 0.34**（旧 0.20）。
同批 `160×160 16→8`（blk）0.131 ms / 4.33 ops，`160×160 s2 16→32`（ov）0.273 ms / 8.32 ops。

### 4.4 数值

对同一输入（`kernel_numtest` dump vs numpy FP32，需偏置 + SiLU）：

| 配置 | mean_rel | max_rel(amax) | 判定 |
|---|---:|---:|---|
| TX40 CB16 CINC16 | 1.061e-03 | 2.298e-03 | PASS |
| TX40 CB16 **CINC8** | 1.061e-03 | 2.298e-03 | PASS |

**两者逐位相同**（`max_abs_diff = 0`）——CINC 只改 chunk 切分，不改算术顺序。

---

## 4.5 R34：CINC 泛化到小 Cin（stem）+ 主导层复核

### 4.5.1 动机

R33 的 gate 是 `Cin%16!=0` 才加 `CINC=8`。但 `CINC=8` 对 `Cin=3` 仍浪费 5/8。把它推广成：
**`Cin≤16` 时枚举精确 `CINC=Cin`**（如 stem 的 `CINC=3`），另加 `CINC=4` 兜底；`Cin>16` 且
`Cin%16!=0` 仍加 `CINC=8`。集合有界（每个 `(TX,CB)` 至多 4 个 CINC）。

### 4.5.2 stem `320×320 s2 3→16`（y8/y11 各 ×1）

`kernel_bench` 同会话：

| 通路/配置 | ops/EU/cyc |
|---|---:|
| ov `OBW=4,OBH=4`（旧缓存） | 1.76 |
| native `CINC=8` | 1.83 |
| native `CINC=4` | 2.89 |
| native **`CINC=3`（精确）** | **3.21** |

调优器端到端（`--only 320x320 --limit 1 --retune --iters 10`）：旧候选集选 ov `2.97`；
新候选集选 **native `TX20 CB16 CINC3` = 3.56（+20%）**。数值：`CINC=3` vs `CINC=16`
**逐位相同**、两者 vs numpy FP32 均 PASS（mean_rel 6.26e-4 / max_rel 2.18e-3）。

### 4.5.3 顺带核对：`Cin=51` 层

`Cin=51`（CINC=16 → 4 chunk、最后 3/16，约 20% 的 mad 打在零通道）实测（`80×80 s1 51→51`，
2 reps 稳定）：

| 配置 | ops/EU/cyc |
|---|---:|
| ov `8,2` | **9.54** |
| native CINC16 | 5.75 |
| native CINC8 | 6.24（+9%） |
| native CB32 CINC8 | 7.33 |

`CINC=8` 对 native 确有 +9%，但 **ov 仍胜**，所以这些层不翻转——候选无害。

### 4.5.4 主导层复核（负结果）

`40×40 s1 64→64` / `80×80 s1 64→64` 是 conv3×3 的两大项。多次重复测量已稳定：

| 层 | 配置 | ops/EU/cyc |
|---|---|---:|
| 40×40 C64 | ov 8×1 / blk OBW8 / ov 10×1 | 8.25–8.32 / 8.56 / 8.00–8.15 |
| 80×80 C64 | ov 8×2 / ov 12×1 | 12.30–12.35 / 12.21 |

- `OBW=10/12,OBH=1`（此前候选只覆盖 `8,1`）在 `40×40` 上**与 `8,1` 在噪声内**，非真收益。
- 给 `conv_ov.cl` 的 `kd` 循环加显式 `#pragma unroll 4`：40×40 `8.19–8.23`（base 8.25–8.32）、
  80×80 持平——**无收益（已回退）**。与 R24「加 ILP 无用」一致：IGC 已自动 ×2 unroll，
  剩余停顿是 send/全局延迟与波量化，不是循环开销。

### 4.5.5 e2e 量级

受影响（`Cin%16!=0` 或 `Cin≤16`）的 conv3×3 分项：y8 **1.39 ms**、y11 **1.52 ms**。但多数是
`Cin=16`（精确 CINC=16，无变化）或 `Cin=51`（ov 仍胜）。**真正被 CINC 改动且翻转**的只有
stem（各 ×1，约 −0.04 ms）与 y11 `160×160 s1 8→16`（×1，约 −0.06 ms）——合计 e2e **~0.3–0.6%**。
即：CINC 类修复是**正确且免费**的，但量级有限；不是整网瓶颈。

---

## 5. 工具/方法学发现（顺带）

- **`kernel_bench --op conv3x3 --verify` 对 native 是坏的**：在已知良好的
  `64→64 80×80`（ops=10.23，与 R18 一致）上 `mean_rel≈0.5`、在 `8→16` 上同样 ~0.53。
  native 的数值请走 `kernel_numtest` / `scripts/kernel_check.py`（`kernel_check` 用 C++ 默认值
  或 `--conv` 覆盖，**不含**这个 `--verify` 路径）。blk 的 `--verify` 是好的（R25 记录 3.1e-3）。
- **`config/tuning.json` 的 `ratio` 对 blk/native 不可比**（§2）。
- **blk 的输入重排成本仍未计入 autotune**（R26 §7.1）：重排 ~28 µs/层 vs blk conv 50–460 µs，
  对 `40×40 s1 64→64` 这种边际层可能改变 blk/ov 的选择。

---

## 6. 下一步（按 ROI）

1. **local sweep 重跑受影响的 conv3×3 签名**（CINC 候选已在，`--retune` 即可）：
   `python3 scripts/autotune.py --model yolo11n-pose --ops conv3x3 --batch 3 --iters 10`。
   预期命中 `Cin%16!=0` 的层（及 stem `Cin=3`）。
2. **`expectedOps` 按 kernel 家族给配额**（ov 20.3 / blk 16.9·14.0·10.9 / native 16.4），
   让 `ratio` 能真正用于跨 shape 排热点。
3. **stem 特化**：exact CINC（§4.5）已拿到一半收益；剩余可做 `Cin=3` 时全展开 27 tap、
   彻底去掉 chunk 循环（预期再 1% 量级）。
4. **把 blk 重排并入计时**（或在 `autotuneOp` 给 blk 一个偏好余量），修选择卫生。
5. 破 `40×40` 波量化只剩 split-K / batch——**R24 已实测否决**，且本项目要求 batch=1，不做。

**不要重试（已记录负结果）**：native `conv3x3_db`（IGC 单缓冲已交织 send）、`XGN` 直读全局、
`WVEC` 宽载、`CINC=32`（SLM build-fail）、`TY=4/16/TX=80`（R18.6）；`conv3x3_rt` 大通道
（GRF=127，R16.1）；OV 的运行时 interior fast path / 双累加集 UK（R24）；残差融合进 conv3×3（R33 前序）；
ov `OBW=10/12,OBH=1`（与 8,1 同噪声）与 `kd` 循环显式 unroll（R34 §4.5.4，无收益）。

---

## 7. 复现

```bash
# CINC A/B（性能；一条命令一个配置，遵守 benchmark_protocol.md）
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc '
    timeout 40 ./build/kernel_bench --op conv3x3 --conv-shape 8,16,160,160 \
      --conv 40,8,1,16,16,1,1,1,3,1,16 --iters 10   # 旧 CINC16
    timeout 40 ./build/kernel_bench --op conv3x3 --conv-shape 8,16,160,160 \
      --conv 40,8,1,16,8,1,1,1,3,1,16  --iters 10   # 新 CINC8'

# 调优器端到端（单批 3 签名；只在重扫时写入 cache）
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'timeout 220 ./build/kernel_autotune \
    --plan models/yolo11n-pose/model.plan --cache /tmp/one.json --op conv3x3 \
    --only 160x160 --limit 3 --iters 8 --retune --expected --report'

# 数值（CINC8 vs CINC16 逐位一致）：用 kernel_numtest dump + numpy 参考
# 见本文 §4.4；`kernel_check.py --conv <cfg>` 可覆盖其余 shape。

# R34 stem：调优器单签名（旧候选 ov 2.97 → 新候选 native CINC3 3.56）
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'timeout 220 ./build/kernel_autotune \
    --plan models/yolo11n-pose/model.plan --cache /tmp/stem.json --op conv3x3 \
    --only 320x320 --limit 1 --iters 10 --retune --expected --report'

# 离线候选枚举/回归自检（不需要 GPU）
./build/tuning_test
```
