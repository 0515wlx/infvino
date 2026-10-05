# R51：先补契约覆盖面 —— `conv1x1_cat4` 收口 + 非 16 通道 fsv16 持久化

> 依据 [`round48-candidate-expansion-plan.md`](round48-candidate-expansion-plan.md) §4bis 的
> **结论性排序**执行：**先补契约覆盖面，再扩充算子族**。本轮做前两项（P0 + P1），均为
> 「契约/单一真相源」缺口，不新增 kernel。
>
> - **P0**：把 `concat4→conv1x1` 融合（R30c Route A）从 Autotuner/PlanModel 的 ad-hoc 分支
>   **迁进 `KernelFamily` 注册表**（`gemm_cat4_f16` 族），声明 supports/candidates/布局契约
>   (`inIndex=1`) 与上限模型。
> - **P1**：让激活池按 **`ceil(C/16)*16`** 分配（仅对「可持久」张量），移除布局规划/mincut
>   里「`C%16!=0` → 钉死 NCHW」的 gate，解锁 mb 的 88/120/144 等非对齐张量进入 blocked 链。

---

## 0. 结论先给

| 项 | 结果 |
|---|---|
| **P0 注册表收口** | `conv1x1_cat4` 从 `--candidates`=`{}` 变为 `gemm_cat4_f16=7`；Autotuner/PlanModel 不再手工复制 gemm 谱。**数值/选择逐位不变**（候选 options 完全等价）。 |
| **P1 非 16 通道 fsv16** | 激活池按补齐通道分配；mb 默认路径 **−2%~−3%**，mincut 路径 **−10%**；y8/y11 噪声内。 |
| **数值** | 三模型 `model_check`（默认 + mincut）**PASS**；mb/y11 `reuse_check` **PASS**；`tuning_test` **PASS**。 |
| **踩坑（已修）** | 首版按 `dims[size-3]` 补齐，把 **rank-3** reshape/decode 张量的 batch 维当通道 → y11 `requested 69→140MB`、busy +1.2%。**改为：只补 4-D conv 张量 + 只补「可能被标记 fsv16」的张量**（按族 supports 的能力超集）。 |

---

## 1. P0：`conv1x1_cat4` 收口（契约覆盖面）

### 1.1 缺口

`docs/round48-candidate-expansion-plan.md` §4bis P0：`conv1x1_cat4` 注册表无族
（`--candidates`=`{}`），ad-hoc 路径在 `PlanModel::autotune` 里手工复制 `gemm` 候选谱并追加
`-DCAT4=1`，**完全在「单一真相源 + 布局契约」之外**。

### 1.2 实现

- `src/KernelFamilies.cpp`：把 gemm 候选谱抽成共享 helper `gemmSpectrum(sig, cat4)`（gemm_f16
  与 cat4 共用，杜绝两处漂移）+ `gemmCeiling(sig)`；新增族：

  | 字段 | 值 |
  |---|---|
  | `name` | `gemm_cat4_f16` |
  | `op` | `conv1x1_cat4` |
  | `source` / kernel | `gemm` / `gemm_f16`（编译 `-DCAT4=1 -DEPI=1 -DACT=`） |
  | `layout` | `{NCHW, NCHW, canOutFsv16=false, needsWeightRepack=false}`，**`inIndex=1`**（槽 0=权重，激活源在槽 1..4） |
  | `ceiling`/`hardCeiling` | 同 gemm（13.7 上限） |

- `src/PlanModel.cpp`：`autotune` 的 cat4 分支由「`candidatesGemm()` + 手工追加宏」改为
  `candidatesFromRegistry(sig)`。
- `run()` / `makeEnqueue` 的 cat4 dispatch 早已 `getKernel("gemm","gemm_f16", options)`，不依赖
  候选的 `kernel` 名，因此 **ABI/缓存向后兼容**：旧缓存里的 `kernel=gemm_f16` 仍可消费
  （`familyByName` 回退到 gemm_f16 的上限模型，数值等价）。

### 1.3 验证（零 GPU）

```
$ kernel_autotune --plan models/yolo11n-pose/model.plan --candidates | grep conv1x1_cat4
  conv1x1_cat4|Cout64_N25600_Cin48_cat16,16,16,0_act1_f16   7  {gemm_cat4_f16=7}
  ...（13 个签名，全部 7 候选，此前均为 0 {}）
```

`refreshExpected` / `bestFamilyCeiling` 现在也能看到 cat4 的族上限。

> **未做**：让 cat4 进入 mincut 的 (族,布局) 决策需要它有一个 **blocked 变体**
> （`canOutFsv16`/`layout.in=FSV16`），属「扩充算子族」范畴（后续 P3）。当前已声明布局契约
> `inIndex=1`，mincut 会正确地把它的激活源 pin 成 NCHW。

---

## 2. P1：非 16 通道 fsv16 持久化（契约覆盖面）

### 2.1 缺口

R50 修复的 mincut correctness bug 里，`Cout%16!=0` 的张量被**钉死 NCHW**——因为
`planBlockedLayout`/mincut 一律要求 `C%16==0`（fsv16 按 16 通道补齐，否则池里 NCHW 大小的
缓冲会越界）。代价是 mb 的 88/120/144 链无法进入 blocked 布局（R48 §4bis P1、R50 缺口 E）。

### 2.2 实现

1. **分配补齐**（`src/PlanModel.cpp`）：`paddedChannelNumel(dims)` 对 **4-D NCHW** 张量把
   `C= dims[1]` 补齐到 16 的倍数；`allocateActivations` 对**可能被持久 fsv16** 的张量用它算
   `l.bytes`，`parse()` 的 `tensor` 分配同样按补齐尺寸（`INFVINO_NO_POOL` 诊断路径也不越界）。
2. **能力判据**（`mayBeFsv16`，只补「可持久」张量，避免无谓膨胀）：
   - 4-D conv 张量；生产者 op 有族/候选声明 `canOutFsv16`（conv1x1/depthwise/conv3x3/`ew_binary_ch`）；
   - **每个**消费者都能在其「激活输入槽」读 fsv16（conv1x1/depthwise/conv3x3）。
   - 这是 `planBlockedLayout`/mincut 实际标记条件的**超集**（按族 `supports`，而非当前选中），
     保证任何被标记 fsv16 的张量都已补齐、不越界。
3. **移除 pin**：`planBlockedLayout` 两处 `C%16!=0 → continue`、`resolveLayoutMinCut` 的
   `C%16!=0 → pin`，以及 depthwise `#blkfsv16` 成本测量的 `Cin%16==0` gate（改为
   `Cin%16==0 || fsv16_capable_`，确保 OUT_FSV16 只写补齐缓冲）。
4. **诊断读回**（`readTensor`）：fsv16 读回转 NCHW 时按 `ceil(C/16)*16` 读取并只写有效通道。

内核侧无需改：`conv1x1_blk` / `depthwise_blk` / `conv_blk` / `ew_binary_ch` 早已用
`(C+15)/16` 的 block 数与 `cok/oob` 谓词处理非对齐通道，只写有效 lane，补齐 lane 不被读。

### 2.3 首个「系统性」踩坑与修正

首版无差别地按 `dims[size-3]` 补齐**所有**激活张量：

| 模型 | 指标 | 首版 | 修正后 |
|---|---|---:|---:|
| yolo11n | pool requested | 140.1 MB（基线 69.0） | **69.9 MB** |
| yolo11n | pool allocated | 49.7 MB（基线 23.1） | **23.1 MB** |
| yolo11n | busy（默认） | **+1.2%** | **−0.3%（噪声内）** |

根因：`dims[size-3]` 对 **rank-3** reshape/decode 张量（如 `[1,64,6400]`）是 batch 维
（=1），被当成通道 → 16× 膨胀。修正：**只补 4-D conv 张量** + **只补能力集合**。

### 2.4 外部稳态 A/B（同会话、锁频、交错各 8 次；`busy` median，ms）

A = R50 基线；B = R51（P0+P1）。P0 逐位不变，故 B/A 即 P1 的贡献。

| 模型 | 路径 | A base | B new | Δ |
|---|---|---:|---:|---:|
| mobilenetv3-small | 默认（planBlockedLayout） | 2.223 | **2.163** | **−2.7%** |
| mobilenetv3-small | mincut（`INFVINO_LAYOUT_MINCUT=1`） | 2.125 | **1.907** | **−10.3%** |
| yolo11n-pose | 默认 | 15.35 | 15.30 | −0.3%（噪声内） |
| yolo11n-pose | mincut | 15.03 | 15.06 | +0.2%（噪声内） |
| yolov8n-pose | 默认 | 14.03 | 14.02 | −0.1%（噪声内） |
| yolov8n-pose | mincut | 14.02 | 13.99 | −0.2%（噪声内） |

结构（mb，`kernel_run --report`）：

| 指标 | R50 基线（默认） | R51（默认） | R51（mincut） |
|---|---:|---:|---:|
| 持久 fsv16 张量 | 9（R48 §10）| **12** | **22** |
| reorder 次数/帧 | 13 | 13 | 12 |
| reorder ms/帧 | 0.098 | 0.079 | 0.060 |

> mincut 门（`INFVINO_LAYOUT_MINCUT_GATE=1`，交错 median）：mb
> `ACCEPT 1.5582 → 1.3691 ms`（R49 舞台内 −12%）。

### 2.5 数值与回归门

| 检查 | 结果 |
|---|---|
| `model_check` mb 默认 / mincut | PASS（mean_rel 1.069e-2 / 1.215e-2） |
| `model_check` y8 默认 / mincut | PASS（4.114e-4） |
| `model_check` y11 默认 / mincut | PASS（7.908e-4，与 R50 逐位相同） |
| `reuse_check` mb / y11 | PASS（跨推理一致） |
| `tuning_test` | PASS |
| `config/tuning.json` | **未改动**（P1 是布局/分配级改动，不改选择） |

---

## 3. 复现

```bash
# 构建
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake --build build-blk -j4 \
  --target kernel_run kernel_autotune kernel_bench tuning_test'

# P0 审计（零 GPU）
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'export LD_LIBRARY_PATH=$PWD/build-blk; \
  ./build-blk/kernel_autotune --plan models/yolo11n-pose/model.plan --candidates \
  | grep conv1x1_cat4'

# 数值（默认 / mincut）
python3 scripts/model_check.py --model mobilenetv3-small --repo "$PWD"
INFVINO_LAYOUT_MINCUT=1 python3 scripts/model_check.py --model mobilenetv3-small --repo "$PWD"
```

---

## 4. 追加：M5 收尾 —— `conv1x1_blk` 的输出 fsv16 成本（`#blkfsv16`）

R50 只给 `depthwise_blk` 测了「输出 fsv16」的真实成本；`conv1x1_blk` 一直用 **bfyx 输出成本**
给「输出 fsv16」定价（`alt[].blkFsv16` 缺失 → 回退 `blk.ms`），系统性高估 blocked 族，
使布局规划偏保守。本轮补上：autotune 在测完 `conv1x1_blk` 后，用同一配置 + `-DOUT_FSV16=1`
再测一次，存为 `<sig>#blkfsv16`（仅当输出缓冲被补齐：`Cout%16==0` 或 `fsv16_capable_`，
避免越界）。`resolveLayoutChoices`/mincut 的 `eBf` 从此用它。

**隔离 A/B**（同一 binary，仅缓存差异：`<sig>#blkfsv16` 有/无；`--op conv1x1 --retune` 重扫）：

| 模型 | 路径 | 无 `#blkfsv16` | 有 `#blkfsv16` | Δ |
|---|---|---:|---:|---:|
| mobilenetv3-small | 默认 | 2.151 | 2.156 | 噪声内 |
| mobilenetv3-small | mincut | 1.912 | 1.923 | 噪声内 |
| yolo11n-pose | 默认 | 15.258 | 15.319 | +0.4%（噪声内） |
| yolo11n-pose | mincut | 15.092 | **14.876** | **−1.4%（8/8 全胜）** |

- mb 上布局本就大多选 fsv16 → 定价修正无差异；y11 的 conv1x1 布局成本被高估 → 修正后 mincut
  少付 reorder / 更准。
- **默认缓存 `config/tuning.json` 未改动**：`#blkfsv16` 需 `kernel_autotune --op conv1x1 --retune`
  才会写入；这是一个「成本模型保真」修复（能力增量），不改现有默认选择。
- 数值：mb 默认 `model_check` PASS（1.069e-2 与基线逐位相同）；y11 mincut + 重扫缓存 PASS
  （9.18e-4）。`tuning_test` PASS。

---

## 5. 追加：D5 小算子 launch 融合 —— SE Mul 折进 conv1x1 prologue（**中性，opt-in**）

**动机**（计划 §D5 / §4bis P2）：小算子占 busy ~50% 的 **dispatch 数**，launch 地板 3.5µs；
把相邻逐元素折进 conv 的 prologue/epilogue 是确定性杠杆。

**实现**：识别 `x * scale[c]`（`ew_binary` op=2、`bdims` 通道广播，如 SE）——其输出是某个
`conv1x1` 的**激活**（槽 1）且唯一消费者；把该 conv1x1 改为读**未缩放的 value** + 新增
`ins[4]=scale`，删除 Mul 及其实的名字。kernel 侧新增 `-DMUL_SCALE=1`：
`gemm_f16` / `gemm_sk_f16` 在 B 载入后按行 `k` 乘 `Scale[k]`；`conv1x1_blk` 在 lane 的输入通道
`gc` 上乘 `Scale[gc]`。**乘积舍入到 half**（与 Mul 的 half 输出一致）→ **逐位等价**。
`MUL_SCALE` 的 `Scale` 参数用 `#if MUL_SCALE` 条件声明，未启用时内核签名不变（不破坏
cat4/gemm/kernel_bench 等调用点）。

**验证**：mb `model_check` 融合开/关 **逐位相同**（mean_rel 1.069e-2）；y8/y11 默认不变。

**整网 A/B（同 binary，融合开/关，交错 ×8，busy median）**：

| 模型 | 路径 | off | on | Δ |
|---|---|---:|---:|---:|
| mobilenetv3-small | 默认 | 2.173 | 2.184 | +0.5%（噪声内） |
| mobilenetv3-small | mincut | 1.953 | 1.937 | −0.8%（噪声内） |

**结构性负结果（根因）**：融合确实**减少 dispatch**（mb 83→79，节点 74→65），但同时
`fsv16` 张量 12→7、reorder 10→15（0.079→0.114 ms）——**被去掉的 Mul 同时承担了 D4 的
fsv16 布局转换**（`ew_binary_ch` 的 `-DEWCH_OUT_FSV16`：读 NCHW value、写 fsv16）。
融合后这个转换重新变成独立 `reorder`，两者成本相抵 → 净中性。**仅当 value 生产者本就输出
fsv16 时，融合才是纯收益**（省一趟全激活读+写）。

**决策**：按 R31 `depthwise_pad` 先例，保留实现但 **opt-in**（`INFVINO_FUSE_SCALE=1`），
**默认关** → 默认路径与 R50 逐位一致、零回归。若要转正，需与「value 生产者 fsv16 持久化」
联合决策（即 D4 的延伸：让生产者直写 + prologue 直读/乘 scale 一步完成）。

### 5.1 D4+ 联动（本轮续做 → **未落地，发现 correctness bug，已回退**）

**动机**：让 SE 的 `value` 生产者（`depthwise_blk`）直写 fsv16，使 D5 的 conv prologue
既能免 reorder 又能乘 scale。

**做法（曾实现）**：给 `gap`/`gap_r` 加 `-DGAP_IN_FSV16`（按 `[C/16][H][W][16]` 直读），
注册无候选的 `gap_fsv16` 布局契约族，让 `planBlockedLayout`/mincut 把 `gap` 当作
「可读 fsv16 的消费者」；`run()` 在输入持久 fsv16 时自动加宏。

**实测（mb，`--op depthwise --retune` 临时缓存；同 binary 融合开/关，交错 ×8）**：
默认 2.194→**2.150（−2.0%）**、mincut 1.915→**1.783（−6.9%）**，8/8 —— **但只看了 busy，
未验证 mincut 数值**。

**随后发现 correctness bug**：`INFVINO_LAYOUT_MINCUT=1 INFVINO_FUSE_SCALE=1` 的
`model_check` **FAIL（mean_rel 0.507）**；用 `INFVINO_NO_GAP_FSV16` 隔离后 PASS。
进一步发现**旧缓存 + 默认路径 + 融合**也 FAIL。即 **`gap` 直读 fsv16 与布局规划的组合
会产生错误的激活读取**。

**单元级定位（本轮）**：给 `kernel_numtest` 加了两个用例，逐元素对照 numpy：
- `--op gap`（NCHW→reorder 成 fsv16→`gap_r -DGAP_IN_FSV16=1`）：C∈{16,96,120,144,240,576}
  **全部 PASS**（max_abs ~2.4e-4）。
- `--op conv1x1blk`（fsv16 输入 + `-DMUL_SCALE=1`，osv16 权重 + scale）：4 组形状含
  Cin=120/576 **全部 PASS**。

期间发现并修复 `conv1x1_blk` 的**真实 kernel bug**：`MUL_SCALE` 在 **packed `BLOCK_READ`
路径**下，lane 并不等于「自己的通道」（是 16×8 的 OV tile），直接乘 `Scale[gc]` 会乘错通道。
修复：`-DMUL_SCALE=1` 时强制走**标量逐通道载入**（`input[base + i*FS + sglid]`），使
lane 的 `src` 确为通道 `gc`。

**但整网组合仍 FAIL**（两个 kernel 各自正确 → 问题在**图级**：某条
`depthwise_blk(OUT_FSV16) → gap(直读 fsv16) + fused conv(直读 fsv16)` 链上仍有一致性缺口，
本轮**未定位**）。

**决策（安全优先）**：
- **回退 `gap_fsv16` 的 planner/dispatch 接线**（`gap_fsv16` 族、nodeFamily、mincut、
  run() 宏）；**保留** `ops.cl` 的 `GAP_IN_FSV16`（供单元测试）与两个 kernel_numtest 用例，
  以及上面的 `conv1x1_blk` packed-read 修复。
- D5 保持 **opt-in**；回退后 mb 四种组合（默认/mincut × 融合开/关）**全部 PASS 且逐位相同**
  （默认 1.069e-2、mincut 1.215e-2）。
- **`config/tuning.json` 未改动**（缓存烘焙一并回退）。
- **重启 D4+ 的前置**：用单元用例进一步逼近「生产者写 fsv16 → 消费者读」的真实链
  （例如 depthwise_blk OUT_FSV16 → gap），定位图级缺口。




---

## 6. 下一步（回到 §4bis 排序）

1. **P1（高风险）conv3x3 的布局链**：给 `conv3x3` 一个能输出 fsv16 的 epilogue（或让消费者
   prologue 直读），纳入 mincut；否则维持 R49「明确放弃 3×3 blocked 链」。
2. ◑/⚠ **P2 小算子 launch 融合（D5）已试（本文 §5）**：SE Mul→conv1x1 prologue 融合
   （`MUL_SCALE`）kernel 侧逐位正确、减少 dispatch；整网中性（与 D4 的布局转换相抵），
   opt-in。**D4+ 联动（gap 直读 fsv16）曾给出 mb −2%/−6.9%，但暴露 correctness bug
   （mincut/旧缓存下 mean_rel 0.5），已回退**（见 §5.1）。下一步：先给 gap 的 fsv16 读
   做单元级验证，再重启 D4+；或改做 gap+fc / resize→conv 等不与布局转换冲突的融合。
3. **P3 direct conv1x1（窄通道/小 N）**、conv1x1 N=1 split-K、gemm staging。
4. ✅ **M5 收尾已完成（本文 §4）**：`conv1x1_blk` 的输出 fsv16 成本（`#blkfsv16`）纳入 autotune。
