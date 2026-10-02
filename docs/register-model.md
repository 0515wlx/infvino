# 7 线程 EU 的寄存器限制：完整模型与推导

> 用途：给定一个 kernel / tile，**预测它的 ops/EU/cyc 天花板**，并判断它撞的是哪一堵墙
> （寄存器 / SIMD / 占用 / SLM / 带宽 / 延迟）。以后遇到「这个 kernel 还能不能调」时，
> 先按本文的清单算一遍，再决定要不要动 kernel，避免凭感觉反复试。
>
> 所有系数尽量给**实测出处**（Round 编号）；纯推导的标 [推导]，实测标的 [实测]。
> 硬件：Intel Iris Xe（TGL，80 EU / 1.3 GHz / 128 GRF / 无通用 L1）。

---

## 1. 硬件事实（基线常数）

| 量 | 值 | 出处 |
|---|---|---|
| EU 数 / 频率 | 80 / 1.3 GHz | `ClRuntime` |
| 每 EU 硬件线程 | **7** | `xe-lp-isa.md` §3.1 `[官方]`；本文 §3.1 实测确认 |
| 每线程寄存器 | **128 GRF × 32 B = 4 KB** | `xe-lp-isa.md` §3.1 `[官方/ISA]` |
| 每 EU 寄存器总量 | 128 × 32 B × 7 = **28 KB**（80 EU ≈ 2.24 MB） | 同上 |
| 发射 | **每 EU 每周期 1 条向量指令** | R13.1 `[实测]`（SG8 吞吐恰为 SG16 一半） |
| FP16 峰值 | **32 FLOP/EU/cyc**（SIMD16 scalar-half FMA 即达峰） | R13.1 `[实测]` |
| FP16 FMA 延迟 L | **≈29–32 cyc** | R10 / R18.7；本文 §3.1 `[实测]` |
| 通用 L1 | **无**（只有 3.75 MB L3 / SLM） | `xe-lp-isa.md` §2 |
| SLM | 64 KB/WG、128 KB/DSS | `xe-lp-isa.md` §2 |

> 关键推论（全文的地基）：EU 的寄存器文件是 **4 KB/线程 × 7 线程**。所以
> **(i)** 单线程可用 128 GRF，超了就 spill 或掉 SIMD；
> **(ii)** EU 的驻留线程数上限恒为 **7**（与寄存器用量无关）→
> `sub-groups/EU ≤ 7`（一个 SIMD16 sub-group = 一个线程）。

---

## 2. 第一性推导

### 2.1 寄存器计费（GRF accounting）[推导]

一个「每 lane 一个值」的逻辑变量，在 SIMD 宽度 `W` 下占用的 GRF：

```
GRF(变量) = 元素位宽×W / 256 b
   half  (scalar, W=16) = 16×16/256 = 1 GRF
   half2 (W=16)         = 2 GRF
   half8 (W=16)         = 8 GRF
   half  (W=8)          = 0.5 → 向上取整 1 GRF（半个浪费）
```

线程的寄存器预算：

```
128 GRF = GRF_data + GRF_overhead
GRF_data  = 累加器 acc + 操作数缓冲（A/B/strip/line_cache/权重）+ 双缓冲
GRF_overhead = 地址、send 描述符、谓词、循环状态、IGC 中间值
```

**实测的 overhead 很大且近乎常数**：R13/R14 对 GEMM 的统计——

| tile | 数据寄存器（估算）| 实测总 GRF | 反推 overhead |
|---|---:|---:|---:|
| TM8 TN4 BK32 | acc 8×4/2=16 + A 8 + B 4 ≈ 28 | **112** | ~84 |
| TM8 TN8 BK32 | acc 32 + A 8 + B 8 ≈ 48 | **126** | ~78 |
| TM16 TN4 BK32 | acc 32 + A 16 + B 4 ≈ 52 | **127 + 513 行 scratch** | 溢出 |

本仓库其它 kernel 同量级：`conv3x3_ov` 127 GRF（acc 16×half2=32 + input 4 ≈ 36 → overhead ~91）、
`conv3x3_blk` 127 GRF（dst half8=8 + line 10 + wei 16 ≈ 34 → overhead ~93）。
→ **这台机器/这套 IGC 上，数据寄存器实际只能用到 ~35–45 个，其余 ~85–95 被寻址/调度吃掉。**

### 2.2 发射配额 [推导]

每周期每 EU 一条向量指令，所以

```
ops/EU/cyc = 32 × (mad 指令数 / 总发射指令数) × (W/16) × occ_factor
```

- `32`：SIMD16 scalar-half mad 的 FLOP 数（= 峰值）。
- `W/16`：SIMD 宽度因子（SG8 → 0.5，实测吞吐恰好腰斩，R13.1）。
- `occ_factor ∈ (0,1]`：因占用不足/停顿导致的发射槽浪费（见 §2.4）。
- `mad 占比`：由内循环的**每 FMA 需要多少非 mad 指令**决定（load/地址/广播/branch）。
  可用离线 `ocloc` 反汇编直接数（本仓库 R24/R25 即此法）。

### 2.3 寄存器墙 → SIMD 悬崖 / spill [推导 + 实测]

- 若 `128 GRF` 放不下 SIMD16 的分配：IGC 的保守启发式会**把整个 kernel 降到 SIMD8**
  （每 GRF 只装一半 lane），`W/16` 从 1 → 0.5，吞吐**腰斩**；而不是先 spill。
  实测（R13.3）：GEMM `DBUF0 BK16` SG=0 → IGC 选 S8；强制 `SG=16` 后塞进 ≤120 GRF，
  6.1 → **13.1**（仅靠钉 SIMD16）。
- 硬顶：用量过 ~120 开始 scratch spill，过 128 性能崩溃（TM16/TN8 = 4.2 / 0.5）。
- **因此「加大寄存器 tile 换算术强度」这条路，在数据寄存器 ~35–45 处就撞墙。**

### 2.4 占用 × ILP：7 线程如何决定发射 [推导]

设每线程独立累加器链数 `ACC`（= 每指令依赖距离里的独立链数）。FP16 FMA 延迟 `L`。
一个累加器相邻两次更新之间必须 ≥ `L` 个周期。EU 上最多 7 个线程并发，所以

```
issue_distance = T_res × ACC        (周期；T_res ≤ 7)
ops/EU/cyc = 32 × mad_frac × (W/16) × min(1, T_res·ACC/L)
```

- `T_res·ACC ≥ L` → **发射受限**（issue-bound），可达 `32 × mad_frac × W/16`。
- `T_res·ACC < L` → **延迟受限**，按比例打折。
- 因为 `T_res` 被钉死在 7，**要发射受限，需要 `ACC ≥ L/7 ≈ 4.2` 条独立链**——这很容易；
  真正难的是**内存/依赖停顿**（§2.5），以及 `ACC` 增长被 §2.1 的 GRF 墙挡住。

### 2.5 完整约束链 [推导]

```
数据 tile 大小  →(GRF≤128−overhead≈90)  →  ACC、操作数缓冲
ACC、缓冲      →  mad_frac（每 FMA 的非 mad 指令数）与 ILP
ILP×7          →  能否填满发射槽（vs L）
发出指令       →  send/SLM/全局的字节量 → 是否撞 SLM/L3/DRAM 带宽
```

四条约束互相耦合、且**此消彼长**：加 tile 提复用 → 撞 GRF；双缓冲藏延迟 → SLM 翻倍；
减 tile 提占用 → 算术强度掉。R14.4 已证明 GEMM 上四条路互相抵消。

---

## 3. 关键系数的实测标定

### 3.1 驻留线程 T_res ≈ 7（新，Round 27）

`kernel_bench --op occ`（`kernels/micro.cl:fma_cyc`）：**单条循环依赖链**（ILP=1）
`r[(d+1)&(D-1)] = mad(r[d],a,b)`，扫网格大小。延迟受限下 `ops = 32·T_res/L`，
所以平台值直接给出 `T_res`。DEPTH=2 实测：

| 网格（sub-group 总数）| 256 | **512** | 768 | 1024 |
|---|---:|---:|---:|---:|
| cyc/mad | 32.0 | **32.4** | 42.6 | 43.4 |
| ops/EU/cyc | 3.2 | **6.32** | 7.21 | 9.44 |

- 512 sub-group（= 6.4/EU）仍是一个完整波（cyc/mad 不变）；768 起出现第二个波（cyc/mad 跳到 1.33×）。
- 平台 `ops≈6.3–7` ⇒ `T_res = ops·L/32 ≈ 7·32/32 = 7`。
  → **实测确认：EU 驻留线程 ≈ 7**（与 `xe-lp-isa.md` 一致）。

### 3.2 峰值、SIMD 缩放、RF 读带宽（R13.1，复述）

| 探针 | ops/EU/cyc |
|---|---:|
| `h1` scalar half，SG16 | 25.9–27.8（与 depth 4–64 无关）|
| `h8`（8 half/lane），SG16 | 27.7 |
| **`h1` SG8** | **14.1（恰为一半）** |
| `f32` SG16 | 14.5–14.7 |
| `h8_1op`（两操作数同寄存器）| 与 `h8` 同 → **RF 读带宽不是瓶颈** |

→ 峰值 ~27.8（结构上限），**由 lane 发射吞吐决定**；`f16:f32 ≈ 2`；SG 宽度线性。

### 3.3 寄存器墙标定（R13.3/R13.4）

见 §2.1/§2.3 的表：`8/4→112 GRF, 13.7`；`8/8→126, 8.4`；`16/4→127+513 spill, 4.2`。

---

## 4. 模型 vs 实测对照

用 `ops = 32 × mad_frac × min(1, 7·ACC/L) × (W/16) × stall_factor`，`L≈30`；
`quota = 32 × mad_frac`（内循环逐指令数出的指令配额，R24/R25）：

| kernel | mad_frac [ISA] | quota | ACC | ILP 因子 min(1,7·ACC/30) | 模型上限 | 实测 | stall/占用缺口 |
|---|---:|---:|---:|---:|---:|---:|---|
| `fma h1/h8`（纯 FMA）| ~1.0 | 32 | ≥L | 1.0 | 32 | **27.8** | 结构/循环开销 13% |
| GEMM 8/4 BK32 | ~0.75 | 24 | 16 | 1.0 | ~24 | **13.7** | staging 31% + loads 17%（R14.1）|
| `conv3x3_ov` 8×2 | 0.637 | **20.4** | 16 | 1.0 | 20.4 | 13.6（80×80）| 0.67（延迟/占用）|
| `conv3x3_ov` 8×1 | 0.503 | **16.1** | 8 | 1.0 | 16.1 | 8.5（40×40）| 0.53 |
| `conv3x3_blk` OBW8 | 0.529 | **16.9** | 8 | 1.0 | 16.9 | 12.2（40×40C128）| 0.72 |
| `conv3x3_blk` OBW4 | 0.439 | **14.0** | 4 | 0.93 | 13.0 | 6.2–7.6 | 0.5–0.6 |
| `conv3x3_blk` OBW2 | 0.341 | **10.9** | 2 | 0.47 | 5.1 | 4.1–4.7 | 0.85（已 ILP 受限）|

读法：
- **模型给的是「指令配额天花板」**（mad 占比 × SIMD × ILP 修正），不是最终性能；
  剩下的是 stall_factor（staging/延迟/占用），需要 R14.1 那种探针分解。
- `conv_blk OBW2` 被 ILP 项 `7·2/30=0.47` 直接预测为延迟受限（模型 5.1 vs 实测 4.1–4.7）——
  这解释了 OBW 越大越好、直到寄存器上限。
- `conv_blk OBW4` 模型 13.0 但实测只有 6–7.6 → 不是 ILP，而是**占用/延迟**（40×40 OBW4 的
  WG 数已够，但每 WG 的 send 延迟暴露）。
- `conv_ov 8×1`（ACC=8）接近 ILP 拐点，是它在 40×40 上比 8×2 差、却因 grid 大在别处胜出的原因。


---

## 5. 使用清单（遇到新 kernel/tile 时按顺序算）

1. **数 GRF**：数据 tile（acc + 操作数 + 双缓冲）估算；加上实测 overhead ~85–95；
   若 > 128 → spill；若 SIMD16 分配顶到 ~126 → IGC 可能掉 SIMD8（吞吐 ×0.5，先强制 `SG=16` 试）。
2. **数 mad 占比**：`ocloc` 反编译，数 `mad` vs 总指令。配额 = `32 × (mad/总) × (W/16)`。
3. **数 ILP/ACC**：每指令链有几条独立累加器链；检查 `7·ACC ≥ 30`，否则延迟受限。
4. **数访存**：每 FMA 的 SLM/全局字节；对照 R11 的 SLM 带宽-占用曲线、L3 ~300 GB/s。
5. **对表** §4 找最接近的既有 kernel；若模型上限远高于实测 → 是 stall（延迟/占用），
   优先查：grid/波量化（`spatial×Cout/32 ≤ 7×80`？）、边界谓词、send 延迟；
   若模型上限本身就低 → 是结构（mad 占比/ILP/寄存器），换数据通路或减乘数。

> 本轮 3×3 conv 的两次「换数据通路」就是按此清单做的：
> `osv32`（8×2, ACC16, mad 0.637, 配额 20.4）→ `blk`（OBW8, ACC8, mad 0.529, 配额 16.9）。
> blk 配额更低，但它把 **WG 数**从 `spatial×Cout/32` 提到 `spatial×Cout/16`（×2–4），
> 直接治 §2.4 的**波量化**，所以在 `spatial×Cout/32 < 560` 的小层（20×20 系）反超 +12–87%。

---

## 6. 复现

```bash
# 占用/寄存器压力探针（ILP=1 循环依赖链；平台 ops→T_res）
./build/kernel_bench --op occ --width cyc --depth 2 --sg 16
./build/kernel_bench --op occ --width h8  --sg 16          # 高 ILP，看满速平台

# 峰值 / SIMD 缩放 / latency
./build/kernel_bench --op fmalat --width h1 --depth 32 --sg 16
./build/kernel_bench --op fmalat --width h1 --depth 32 --sg 8

# 数 GRF（离线，不需要 GPU；IGC 的寄存器编号恒到 127，须看 scratch/指令或上层统计）
ocloc compile -file kernels/micro.cl -device tgllp -options "-DDEPTH=8 -DSG=16 -cl-mad-enable" -output /tmp/k
# 数 mad 占比（main loop 的 `while` 回跳体）
ocloc compile -file kernels/conv_ov.cl -device tgllp -options "-DOBW=8 -DOBH=2 -DSTRIDE=1 -DPAD=1 -DACT=1 -DRES=0 -DSG=16 -cl-mad-enable -cl-fast-relaxed-math" -output /tmp/ov
ocloc disasm -file /tmp/ov_tgllp.bin -device tgllp -dump /tmp/ov   # 288 mad / 453 指令 = 0.636
```

---

## 附：已知的模型边界与未决项

- **GRF overhead ~85–95 是 IGC/本机实测的经验值**，不是架构常数；换编译器/更激进的内联
  可能变。它是「数据寄存器只有 ~40」这一反直觉结论的来源，值得警惕。
- **T_res 实测为 ~7**（本文 §3.1），但 §3.1 的探针受波/尾效应影响，只把 T_res 夹在
  [6.4, 9.6]；结合 `xe-lp-isa.md` 取 7。
- `L`（FP16 FMA 延迟）取 29–32；ILP 拐点的精确值对结论不敏感（7·ACC 通常远大于 L）。
- SIMD 悬崖的确切阈值（~126 vs 128）是 IGC 启发式，不是硬件位宽；强制 `SG=16` 可越过。
