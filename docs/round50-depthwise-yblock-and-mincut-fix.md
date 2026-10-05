# R50：depthwise Y_BLOCK 候选 + 布局契约成本 + mincut 数值修复

> 依据 [`round48-candidate-expansion-plan.md`](round48-candidate-expansion-plan.md) 的 **D6**
> （`depthwise_v2`：多输出/WI + K=5 专用），在 R49「布局一等公民」的框架上续做。本轮最重要的
> 产出不是新候选本身，而是**用现有系统倒查出 R49 mincut 的一个 correctness bug**（并且发现
> `model_check` 的假 PASS 机制）。

---

## 0. 结论先给

1. **D6 候选已落地**：`depthwise_blk` 新增 `-DY_BLOCK`（每 WI 多输出**行**，输入行滑动窗口
   复用），候选 2→5/签名。**逐位一致**（所有 YB 变体 mean_rel 相同）。大 spatial 有效
   （W80 K3：blk 0.1165→**0.0487**，含 fsv16 输出；见 §1/§2）。
2. **发现的关键事实：`depthwise_blk` 输出 fsv16 时比 bfyx 快 ~2–3×**（lane=通道合并写
   vs 跨通道散写），但注册表候选与 autotune **一直按 bfyx 输出测量**。据此新增
   **`#blkfsv16` 契约成本**：布局规划在「输出 fsv16」时用真实成本定价（不改变 kernel 选择，
   只改成本口径）。y11 上 **−1.3%**。
3. **修复 R49 mincut 的 correctness bug**：`resolveLayoutMinCut` 漏了
   `planBlockedLayout` 一直有的 **`Cout%16==0` 持久化门**，把 88/120/144 等**非 16 对齐**
   张量标成 fsv16 → 池里 NCHW 大小缓冲写越界 → **mb 整网输出错乱（mean_rel 0.44）**。
   修复后 mb mincut 从「1.35 ms 但数值错」变为 **1.52 ms 且数值对**（model_check PASS）。
4. **R49 的 `model_check` 是假 PASS**：`scripts/model_check.py` 的 `docker run` 不带 `-e`，
   `INFVINO_LAYOUT_MINCUT=1` 根本没进容器 → 实际测的是默认路径。已修（透传 `INFVINO_*`）。
5. **净结果**（修复+契约成本后，锁频非全程、交错 median）：
   **mb −3.9%（门 ACCEPT）、y11 −2.1%（门 ACCEPT）、y8 门 REJECT（回退，无回归）**。
   mincut 仍 **opt-in**；`config/tuning.json` 未改动。

---

## 1. D6：`depthwise_blk` 多行（Y_BLOCK）

### 1.1 物理动机（ISA 证据链，不是 ratio）

`depthwise_blk` 每个 WI 为 **1 个输出行**加载 `K` 个输入行；K=5 时每个输出行加载 5 行。
ISA 直方图（`ocloc disasm`，K=5 XB=8）显示非 mad 指令（add/mov/shl/cmp/send）远超 mad —
即**指令/喂数受限**。`Y_BLOCK=T` 让一个 WI 产出 T 个连续输出行，输入行按滑动窗口只加载
`(T-1)·S + K` 行（每行只读一次、服务 K 个输出行），地址计算与 load 摊薄 ~K/T 倍。

实现（`kernels/depthwise_blk.cl`）：外层对 `ir = 0..(T-1)S+K-1` 遍历输入行，每行加载一次
register line，再对每个受影响的输出行 `t`（`kh = ir - t·S ∈ [0,K)`）累加；权重 `wt[K·K]`
每 WI 只读一次。**累加顺序 philosophy 不变（每个输出 kh=0..K-1）→ 逐位一致**；默认
`Y_BLOCK=1` 退化为原实现。约束 `XB·YB<=16`（line/累加器寄存器预算）。

### 1.2 隔离实测（`kernel_bench --op depthwiseblk --verify`，ms）

| 形状（输出 C,H,W / K,s,p） | non（depthwise_v/f16） | blk bfyx（YB 胜者） | blk **fsv16** 输出 |
|---|---:|---:|---:|
| 64,80,80 K3 s1p1 | 0.1283 | 0.1165 (XB8/YB2) | **0.0487** |
| 128,40,40 K3 s1p1 | 0.0989 | 0.0629 (XB4/YB4) | **0.0335** |
| 64,40,40 K3 s1p1 | 0.0550 | — | **0.0242** |
| 256,20,20 K3 s1p1 | 0.0567 | — | **0.0228** |
| 240,14,14 K5 s1p2 | 0.0327 | 0.0281 | **0.0210** |
| 96,14,14 K5 s2p2 | 0.0344 | 0.0250 | **0.0217** |
| 576,7,7 K5 s1p2 | 0.0393 | 0.0251 | **0.0226** |
| 16,56,56 K3 s2p1 | 0.0367 | 0.0278 | **0.0214** |

- **YB 胜者按形状分族**：大 spatial（H≥20）YB2/YB4 胜；小 spatial（14×14）YB1 胜
  （网格饥饿/寄存器压力）——与 R48 D1（conv1x1 YB）同一 regime 分裂，autotune 按签名选。
- **数值**：`--verify` 在 act1/2/3 下 mean_rel ~4–6e-4，且 YB1/2/4 **完全相同**（逐位一致）。
- **顺带修工具 bug**：`benchDepthwiseBlk --verify` 的激活码表把 2/3 写反（2=HardSwish,
  3=ReLU），与 kernel 的规范码（2=ReLU,3=HardSwish）不一致，导致 act3 假 FAIL。已修。

### 1.3 候选规模（`kernel_autotune --candidates`，零 GPU 计时）

`depthwise_blk` 每签名 2 → 5（H≥4）/ 3（小 H）；mobilenet 总候选 566→591，未触预算上限。

---

## 2. 布局契约成本 `#blkfsv16`（发现：fsv16 输出快 ~2–3×）

### 2.1 问题

注册表的 `depthwise_blk` 候选 options **不含 `-DOUT_FSV16`**；`run()` 在运行时按布局
（`out.fsv16`）追加。因此 **autotune 测量的一直是 bfyx 输出的成本**，而布局规划在
「持久 blocked 链」里实际执行的是 `OUT_FSV16=1`。二者对 depthwise 差 **2–3×**（§1.2），
即布局规划的**成本模型系统性高估 blocked 族**——这与 R49 §9.1 的「契约声明化」同源。

### 2.2 修复（不改变 kernel，只改成本口径）

- autotune 在量到 blk 胜者后，用**同一 options + `-DOUT_FSV16=1`** 再量一次，存
  `sig#blkfsv16`（仅 `Cin%16==0`）。
- `LayoutAlt.blkFsv16` + 布局规划：
  - 默认联合不动点：节点输出已规划 fsv16 时用 `blkFsv16.ms`，否则 `blk.ms`（**缺失即回退旧行为**）；
  - mincut 代价表：`f01/f11`（输出 fsv16）用 `eBf`，`f00/f10`（输出 NCHW）用 `eB`，仍 submodular。
- `occupancyPressure` 增加 `depthwise_blk` 几何（XB/YB），与 conv1x1_blk 同口径。

### 2.3 效果

- **y11 −1.3%**（交错 6 组、交替顺序消漂移：nofsv median 11.303 → fsv 11.154）。
- **mb ≈ 中性**（nofsv 1.510 / fsv 1.519，噪声内）——mb 的 depthwise 都在小 spatial，
  且多数通道非 16 对齐（见 §3），可持久化的张量少。

---

## 3. 关键修复：R49 mincut 的 `Cout%16` 门缺失（correctness）

### 3.1 现象

`INFVINO_LAYOUT_MINCUT=1` 跑 mb：**busy 1.35 ms**（看似 −14%），但输出 **mean_rel 0.44**
（vs 参考/默认，量纲 ~1）——**数值完全错乱**。y11 正常（其通道都是 16 对齐，bug 不触发）。

### 3.2 根因

`PlanModel::planBlockedLayout` 标记持久 fsv16 前有一道门：**`Cout % 16 == 0`**
（fsv16 按 16 通道补齐；非对齐时池里 NCHW 大小的缓冲会越界）。R49 新增的
`resolveLayoutMinCut` **漏了这道门**，把 88/120/144… 张量标成 fsv16 → kernel 按 fsv16 写出
`ceil(C/16)*16` 个通道，越过 NCHW 缓冲 → 输出错乱。

### 3.3 修复与前后对比

在 mincut 的 pin 循环补上「通道非 16 对齐 → 钉死 NCHW」。实测（mb，直接对比输出）：

| | fsv16 张量 | mincut busy(ms) | mean_rel vs 默认 |
|---|---:|---:|---:|
| 修复前 | 23 | 1.35 | **4.4e-1（错）** |
| 修复后 | 14 | 1.52 | **3.9e-3（对）** |

> 结论：**R49 报的 mincut −9%/−14% 里，有相当一部分来自非法布局**（非法布局常更快，
> 因为省了 reorder、写得更"整齐"）。修复后 mb 的真实、正确收益是 **−3.9%**（门 ACCEPT）。

### 3.4 为什么此前没被发现：`model_check` 假 PASS

`scripts/model_check.py` 用 `docker run` 跑 kernel_run，但**没有 `-e` 透传环境变量** →
`INFVINO_LAYOUT_MINCUT=1 python3 model_check.py` 实际在默认路径上 PASS。R49 §9.4 的
「mincut model_check PASS」因此是假 PASS。已修：把调用方的 `INFVINO_*` 显式
`-e` 进容器。修复后重跑：

- mb `INFVINO_LAYOUT_MINCUT=1` model_check：**PASS（mean_rel 1.215e-2）**（修复前会 FAIL）。
- mb / y11 `reuse`/默认路径：见 §4。

---

## 4. 整网验收（锁频非全程；`kernel_run --report --iters 15`；交错 median）

| 配置 | mb | y11 | y8 |
|---|---:|---:|---:|
| 默认（base） | 1.583 | 11.488 | 10.49 |
| mincut + 契约成本（修复后） | **1.519 / 门 1.522** | **11.154 / 门 11.149** | 10.53 |
| 门裁决 | **ACCEPT −3.9%** | **ACCEPT −2.1%** | **REJECT（回退）** |

数值门：mb mincut model_check PASS（1.34e-2）；y11 mincut **与默认逐位相同**（7.908e-4）；
`tuning_test` PASS（含 R48/R49 全部自检）。

> mincut 仍 **opt-in**（`INFVINO_LAYOUT_MINCUT=1` + `_GATE=1`）；`config/tuning.json`
> **未改动**（本轮是候选/成本/修复，不改部署选择）。

---

## 5. 复现

```bash
# 构建
docker run --rm --memory=6g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake --build build-ct -j6 --target kernel_bench kernel_autotune kernel_run tuning_test'

# 隔离：YB 变体逐位一致 + 分族
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'export LD_LIBRARY_PATH=$PWD/build-ct; \
  for yb in 1 2 4; do ./build-ct/kernel_bench --op depthwiseblk --depthwiseblk 4,5,1,2,3,0,$yb \
    --conv-shape 120,120,14,14 --iters 20 --verify; done'   # mean_rel 相同

# 生成 #blkfsv16 契约成本（工作缓存副本）
cp config/tuning.json /tmp/t.json
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'export LD_LIBRARY_PATH=$PWD/build-ct; \
  ./build-ct/kernel_autotune --plan models/mobilenetv3-small/model.plan --op depthwise \
    --retune --iters 20 --cache /tmp/t.json'

# 门验收（交错 median；ACCEPT/REJECT 自动裁决）
INFVINO_LAYOUT_MINCUT=1 INFVINO_LAYOUT_MINCUT_GATE=1 INFVINO_TUNING_CACHE=/tmp/t.json \
  ./build-ct/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 15

# 数值（现已能真正透传 mincut）
INFVINO_LAYOUT_MINCUT=1 python3 scripts/model_check.py --model mobilenetv3-small --repo "$PWD"
```

---

## 6. 系统反思（本轮对「用系统找缺口」的回答）

| # | 发现/不足 | 证据 | 方向 |
|---|---|---|---|
| A | **`canOutFsv16` 契约未贯穿成本**：候选按 bfyx 测，运行时按 fsv16 跑 | 隔离差 2–3× | 本轮加 `#blkfsv16`；应推广到 `conv1x1_blk`（同源） |
| B | **布局规划正确性门在两套实现间漂移**：`planBlockedLayout` 有 `C%16` 门，mincut 没有 | mb mean_rel 0.44 | 已修；应把「可持久化」抽成**单一真相源**谓词（planner/mincut 共用） |
| C | **验证工具假 PASS**：`model_check` 不透传 env | R49 mincut 假 PASS | 已修（`-e INFVINO_*`）；应给所有 GPU 验证脚本统一 env 透传 |
| D | **mincut 仍只覆盖 conv1x1/depthwise**，conv3x3 排除（R49 缺口 C） | — | 维持结论：conv3x3_blk 结构性弱 |
| E | **非 16 通道的持久 fsv16 不可用**（池缓冲按 NCHW 大小分配） | 本轮 pin 掉 88/120/144 | 若要把 mb 的窄通道也纳入 blocked 链，需**按 fsv16 补齐大小分配激活缓冲** |
| F | `#blkfsv16` 按「blk bfyx 胜者」再量，未必是 fsv16 最优配置 | W80 fsv16 最优 XB8/YB2 | 可对 blk 候选也枚举 OUT_FSV16 维度（候选×2），但受预算约束 |

> 一句话：本轮再次印证 R48 §10.6-E「候选扩充前先过『是否改变布局图/契约』这一关」——
> 真正带来整网收益的不是多一个 YB 变体，而是**把布局契约从「声明」贯穿到「成本」与
> 「正确性门」**；而一个缺失的正确性门 + 一个不传 env 的验证脚本，足以让两轮结论失真。

---

## 7. 状态与后续

- **已落地**：`depthwise_blk -DY_BLOCK`（默认 1，逐位一致）+ `#blkfsv16` 契约成本 +
  mincut `C%16` correctness 修复 + `model_check` env 透传 + `--verify` 激活码修复。
- **整网收益**：mincut（opt-in）mb −3.9% / y11 −2.1%，y8 门拒。`config/tuning.json` 未改。
- **后续**：将 `#blkfsv16` 推广到 `conv1x1_blk`；把「可持久化」谓词抽成 planner/mincut 共用；
  评估按 fsv16 补齐分配激活缓冲（缺口 E）；再考虑把 mincut 的 per-plan 最优 bake 成工件
  （R49 §9.5-4）。
