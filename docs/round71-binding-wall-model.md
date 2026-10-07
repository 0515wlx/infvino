# R71：绑定墙（binding-wall）分析框架

> 本文件只描述**分析框架本身**（模型 + 接口 + 工具 + 用法），不含任何具体模型的分析结果。
> 把框架用于 yolo/mobilenet（以及后续与 OpenVINO 的逐层对照）的结论另文记录。
>
> 动机：R1–R70 的 roofline 只有两把尺子——计算侧「ISA 指令配额」、内存侧「单条 copy 带宽
> 曲线」。开发中反复出现两个物理维度，此前不在文档也不在模型里：
> **(1) 带宽受限需要区分 GPU 私有 L3 / 共享 LLC / DRAM 三级（含两级 spill）；
> (2) 计算受限需要补上寄存器文件大小/带宽与 SLM（L1）带宽/大小耦合——FPU 根本跑不满。**
> 本框架把两者显式化，用于「逐节点判断撞的是哪堵墙、离它多远」。
>
> **关键纪律：纯诊断/报告层，不参与任何选择**（`expectedOps` / `hard_ceiling` /
> autotune 选择 / mincut 布局 / L3 定价 / spill 计分均不变），因此**不影响数值**，可随时回归。

---

## 1. 模型

### 1.1 带宽侧：三级足迹 + 两级 spill

* `MemTierTime memTierTime(sig)`：把算子**单遍字节** `singlePassBytes(sig)` 按
  **GPU 私有 L3（3.75 MiB）→ 共享 LLC（8 MiB）→ DRAM** 三段拆开，各段用各自实测带宽
  （145 / ~35 / 20 GB/s），返回 `l3_ms / llc_ms / dram_ms` 与主导 `tier`（串行下界）。
* `L3TwoLevelSplit l3TwoLevelSplit(nodes, cfg)`：用**栈距离**（与 `spillAtCapacity` 同源、
  容量无关）把全局读 miss 拆成：
  * `llc_served_bytes`：栈距离 > GPU 私有 L3、但 ≤ 两级和（服务自共享 LLC，便宜）；
  * `dram_miss_bytes`：栈距离 > 两级和（或冷启动，服务自 DRAM，贵）。
  分别按 `(1/BW_LLC − 1/BW_L3)`、`(1/BW_DRAM − 1/BW_L3)` 计价。仅诊断；`evaluateL3` 既有
  语义（单一 `spill_ms`，选择路径）不变。

常量：GPU 私有 L3 = `l3PhysicalBytes()` = 3.93 MB；两级和 = `l3WarmCapBytes()` ≈ 12.3 MB
（R62/R63 实测/公开 PRM；滞留曲线见 R71 结果文）。

### 1.2 计算侧：ISA 配额 × 寄存器 ILP × SLM 带宽/容量

* **寄存器 ILP**：每线程独立累加链 `ACC`（conv3x3 = `OBW·OBH`、gemm = `TM·TN`、
  conv1x1_blk = `X_BLOCK·Y_BLOCK`、GEMV = split-K），EU 驻留线程 `T_res = 7`、
  FP16 FMA 延迟 `L ≈ 30`（`docs/register-model.md`），
  `rf_ilp = min(1, 7·ACC / 30)` + `rf_ops = isa_ops · rf_ilp`。
  注：寄存器读**带宽**不是墙（R13.2：两操作数同寄存器无收益）；约束来自寄存器**大小**。
* **SLM 带宽**：`slm_bw_ops = (FLOP / SLM字节) · BW_SLM / (EU·clk)`。
  conv3x3 lane=输出通道（16 FLOP/输入字节）；gemm `BM·BN/(BM+BN)`；`BW_SLM ≈ 350 GB/s`。
* **SLM 容量**：每 WG 足迹（与 `occupancyStats` 同源）≤ `64 KiB`；另记每 lane 字节
  （Intel「>73 B/lane 掉 SIMD」阈值，仅报告）。
* **指令配额**（depthwise 等地址/边界谓词主导）：沿用现有 `32 × mad_frac`。

### 1.3 判决

`WallVerdict attributeWall(sig, entry, dev, measured_ms)`：

```
fma_ops = min(isa_ops, rf_ops, slm_bw_ops)     # 计算侧硬上限（FMA/SLM 族）
mem_ops = flops / (EU·clk · t_mem)             # 内存侧硬上限（三级足迹 roofline）
hard    = min(fma_ops, mem_ops);  binding = argmin → WallKind
ratio_to_wall = measured_ops / hard
```

`WallKind` = `FmaIssue / RegIlp / SlmBandwidth / SlmCapacity / GpuL3 / SharedLlc / Dram /
Launch / Unknown`。小算子（`flops=0`）归 `launch`；N=1 GEMV 是**归约延迟/网格**，不按内存
roofline 判（归 `launch`）。

### 1.4 常量（全部有实测/公开出处）

| 量 | 值 | 出处 |
|---|---:|---|
| GPU 私有 L3 / 两级和 | 3.75 MiB / ≈12.3 MB | R62/R63 PRM + 实测 |
| L3 / LLC段 / DRAM 带宽 | 145 / ~35 / 20 GB/s | R47/R55 + R71 `membw` |
| 每 EU 驻留线程 `T_res` | 7 | R27 `occ` |
| FP16 FMA 延迟 `L` | ~30 cyc | R10/R18 |
| 128 GRF/线程；SLM 64 KB/WG | — | `xe-lp-isa.md` |
| SLM 流式读带宽 | ~350 GB/s（实测 300–500） | R71 `slmbw` |
| 访存延迟 | ~150 cyc（几乎与工作集无关） | R41/R71 `memlat` |

---

## 2. 接口 / 工具

| 位置 | 内容 |
|---|---|
| `include/infvino/Tuning.hpp` + `src/Tuning.cpp` | `WallKind` / `wallKindName` / `MemTierTime` / `ComputeWall` / `WallVerdict`；`memTierTime` / `computeWall` / `attributeWall` |
| `include/infvino/L3Model.hpp` + `src/L3Model.cpp` | `L3TwoLevelSplit` / `l3TwoLevelSplit`；`L3ModelConfig` 增 `l3_private_bytes` / `l3_bw_gbps` / `llc_bw_gbps` / `dram_bw_gbps` |
| `src/PlanModel.cpp` | `INFVINO_LAYOUT_SPILL=1` 时多打一行两级 spill（`spill tiers: GPU-L3-miss … →LLC … →DRAM …`） |
| `src/tools/kernel_autotune.cpp` | `--wall-report`：零 GPU，逐签名输出墙判决 + 墙预算（用缓存实测 `ms`） |
| `scripts/analyze_walls.py` | 把 `--wall-report` 与 `kernel_run --profile-json` 按 signature **join**，给**按调用次数加权**的墙预算 |
| `tests/test_rulers.cpp` | R71 离线单测（三级拆分、寄存器 ILP 因子、指令配额族、绑定判决、两级 miss 可加性） |

---

## 3. 用法

```bash
# 0) 构建 + 离线门（零 GPU）
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null \
    && cmake --build build -j8 >/dev/null && ctest --test-dir build --output-on-failure'

# 1) 零 GPU：逐签名墙判决（用 config/tuning.json 的实测 ms）
./build/kernel_autotune --plan models/<m>/model.plan --wall-report > /tmp/wall_<m>.txt

# 2) 逐节点实测（锁频、安全容器）
scripts/gpu_clocks.sh lock
timeout 90 ./build/kernel_run --plan models/<m>/model.plan --report --iters 20 \
    --profile-json /tmp/<m>.json

# 3) 按调用加权的墙预算
python3 scripts/analyze_walls.py --model <m> --wall /tmp/wall_<m>.txt --profile /tmp/<m>.json

# 4) 两级 spill 报告
INFVINO_LAYOUT_REPORT=1 INFVINO_LAYOUT_SPILL=1 \
  timeout 60 ./build/kernel_run --plan models/<m>/model.plan --report --iters 1

# 5) 标定探针（一条命令、短核）
./build/kernel_bench --op membw --sizes 512,1024,2048,3072,3584,3840,4096,6144,8192,12288,16384
./build/kernel_bench --op slmbw --slm-kb 16 --width scalar --nwg 64 --iters 50000
./build/kernel_bench --op occ --depth 8 --sg 16
./build/kernel_bench --op l3retain --hot-lines 16384 --hot-gws 4096 --hot-sweep 1,4 \
  --wi 2048,4096,8192 --fp 8,16,24,32,48
scripts/gpu_clocks.sh unlock
```

> 说明：`--wall-report` 与 `analyze_walls.py` 需要 `kernel_run --profile-json` 提供**每节点**
> 数据才能得到按调用加权的真实预算；只做单签名排序时前者已足够。
