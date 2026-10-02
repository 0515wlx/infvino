# Round 24 分析：conv3×3 逐 size 瓶颈定位 + 「中间标准」修正 + 两通路接入调优

> 目标模型：`yolov8n-pose` / `yolo11n-pose` / `mobilenetv3-small`
> 硬件：Intel Iris Xe（80 EU / 1.3 GHz, TGL iGPU），单流，warm。
> 数值判据：`kernel_check` mean_rel<1e-2 & max_rel<5e-2（本轮 OV/native/conv1x1 全 PASS）。
> 本轮不改生产 kernel 数据通路；产出是**瓶颈定位证据 + 中间标准修正 + 调优接入**。

---

## 0. 本轮要回答的三件事

用户判断（本轮起点）：

1. 「有些 size 的 ops/EU/cyc 太低，而且按理来讲也不是 DRAM 带宽受限」；
2. 「我们写的 conv3×3 不太行，但理论极限跟 OpenVINO 的差不多，可以两个 kernel 都接入 autotuning」；
3. 「不能直接把上限定成当前表现（~10 数），要继续按之前分析的理论极限来；在找到决定性证据前
   都默认是软件流水优化的不够好」。

本轮用**离线 ISA 反汇编**（`ocloc`，不需要 GPU，符合 `docs/benchmark_protocol.md` R3）拿到了
决定性证据，答案是：**第三条判断对了**——移植 kernel 的指令配额上限不是 16、也不是 10，而是 ~20.3；
缺口在流水/延迟/占用，不在指令数。下面逐一展开。

---

## 1. 决定性证据：`conv_ov.cl` 的真实指令配额（离线 ISA）

编译并反汇编移植的 `conv_ov.cl`（OBW=8/OBH=2/STRIDE=1/ACT=1，与 80×80 大层一致）：

```bash
ocloc compile -file kernels/conv_ov.cl -device tgllp \
  -options "-DOBW=8 -DOBH=2 -DSTRIDE=1 -DPAD=1 -DACT=1 -DRES=0 -DSG=16 -cl-mad-enable -cl-fast-relaxed-math" \
  -output /tmp/ov
ocloc disasm -file /tmp/ov_tgllp.bin -device tgllp -dump /tmp/ov
# 主循环 = KernelHeap.asm 中 `while` 回跳体（`while` 在同一条指令流里回跳到循环头）
```

主内循环（一次迭代处理 2 个输入通道，IGC 展开 ×2）实测指令直方图：

| 指令 | 条数 | 说明 |
|---|---:|---|
| `mad`（packed `:hf`，即 half2）| **288** | 计算主体 |
| `mov` | 43 | |
| `add` | 30 | 地址 |
| `csel` | 14 | 谓词 |
| `shl` | 13 | 地址 |
| `send.dc0`（全局读）| 13 | 4× 输入 gather + 9× 权重 block read |
| `mul` | 10 | |
| `mach`/`cmp`/`if`/`else`/`sync.nop`/`while` | ~42 | |
| **合计** | **453** | |

- **mad 占比 = 288 / 453 = 63.6%** → 指令配额上限 `32 × 0.636 ≈ 20.3 ops/EU/cyc`。
- **关键更正**：`sub_group_broadcast` **没有**出现在指令流里——IGC 把它折进了 `mad` 的标量操作数
  （`mad ... rX.x<0>:hf`）。因此 R20/R23 的「内循环 1 broadcast : 1 mad ⇒ 上限 = 32/2 = 16」
  **描述的不是我们移植后的代码**。R20 里的「1 broadcast : 1 mad」是 `conv3x3_sg`
  （另一种自写 kernel）的结构，不是本生产路径。
- 寄存器：`maxR=127`（用满 128 GRF，`distinct=112`），无 spill。
- 整核（含 output/激活）：1320 条指令 / 288 mad / 49 send；激活用 32×`math.exp`。

**结论**：这台机器上移植 kernel 的现实上限 ≈ 20.3（指令配额），而不是 16 或 10.3。
实测 80×80 = 13.77（配额的 68%）、40×40 = 8.46（42%）。
**缺掉的那 32–58% 只能是延迟/流水/占用的停顿**——这正是「软件流水不够好」的直接量化形式。

---

## 2. 逐 size 瓶颈定位（按修正后的中间标准）

把 `config/tuning.json` 的 31 条 conv3×3 实测与修正后的
`expected_ops`（`32×0.636×prologue_amort×grid_factor`，上界 20.3）对照，按 ratio 升序：

| shape | kernel | 实测 | 期望 | ratio | 主要短板 |
|---|---|---:|---:|---:|---|
| 112×112 s2 3→16 | ov | 1.34 | 16.07 | **0.08** | Cout<32 → lane 半空 + Cin=3 太短 |
| 320×320 s2 3→16 | ov | 1.73 | 16.07 | **0.11** | 同上 |
| 160×160 s1 16→8 | native | 3.79 | 19.34 | 0.20 | 通道太窄，复用不足 |
| 160×160 s1 8→16 | ov | 3.91 | 18.43 | 0.21 | 同上 |
| 20×20 s1 51→51 | ov | 2.46 | 7.51 | 0.33 | gridFactor 0.375 |
| 40×40 s1 32→32 | ov | 4.14 | 12.40 | 0.33 | 小空间 + 小通道 |
| 40×40 s1 51→51 | ov | 6.71 | 20.02 | 0.34 | **网格/占用** |
| 40×40 s1 32→64 | native | 6.83 | 19.84 | 0.34 | 网格/占用 |
| 160×160 s2 16→32 | ov | 6.64 | 19.38 | 0.34 | grid 大但 Cin 小 |
| 40×40 s2 64→64 | ov | 6.98 | 20.10 | 0.35 | 网格/占用 |
| 20×20 s1 128→128 | ov | 6.19 | 15.17 | 0.41 | gridFactor < 1 |
| 40×40 s1 64→64 | ov | 8.46 | 20.09 | 0.42 | **网格/占用（重点层）** |
| 40×40 s1 128→64 | ov | 10.01 | 20.22 | 0.49 | 同上，Cin 翻倍后改善 |
| 80×80 s2 64→64 | ov | 10.34 | 20.10 | 0.51 | |
| 80×80 s1 51→51 | ov | 10.54 | 20.02 | 0.53 | |
| 80×80 s1 64→51 | ov | 11.27 | 20.09 | 0.56 | |
| 80×80 s1 64→64 | ov | 13.77 | 20.09 | **0.69** | 最接近配额，剩余是延迟 |

> 完整 31 行见 `config/tuning.json` 与下方复现脚本。修正后的标准不再把「10 几」当上限：
> **大层也只有配额的 ~68%，小空间层只有 ~33–42%，stem 层只有 8–11%**。

### 2.1 四类瓶颈（可操作的分类）

1. **大空间/大通道（80×80 s1，ratio 0.53–0.69）**：最接近指令配额；缺口是**内存/send 延迟**
   （每 2 个输入通道 13 条 `send` + 4 条 `sync.nop`）。可压榨空间 ~30%。
2. **中空间（40×40 s1/s2，ratio 0.34–0.49）**：`conv3x3` 合计占 yolov8n kernel busy ~64%，
   其中单签名最大的是 `40×40 s1 64→64`（×10，1.39 ms，~9%）与 `80×80 s1 64→64`
   （×4，1.33 ms，~9%）。block 扫描（本轮实测）：

   | OBW,OBH | 累加器 | grid | ops |
   |---|---:|---:|---:|
   | 4,1 | 4 | 800 | 5.54 |
   | **8,1** | **8** | **400** | **8.44–8.69** |
   | 10,1 | 10 | 320 | 7.54 |
   | 16,1 | 16 | 240 | 6.19 |
   | 8,2 | 16 | 200 | 7.83 |
   | 4,2 | 8 | 400 | 8.03 |
   | 2,2 | 4 | 800 | 4.64 |

   **既不是纯 grid 也不是纯 ILP**：4×1 把 grid 翻倍但累加器掉到 4 → 更差；16×1 累加器翻倍但
   grid 掉到 0.6× → 也更差；**8×1 是 8 累加器 × 400 WG 的甜点**。R23 的 block 扫描
   （8×2/5×2/6×2/4×4）没覆盖 8×1，本轮把它钉住。
3. **小空间（20×20，ratio 0.33–0.48）**：`gridFactor` 已 < 1（20×20 8×1 仅 60 WG），
   是 R18.4 的网格饥饿。
4. **通道极窄 / 极短 Cin（160×160 8→16、112/320 s2 3→16，ratio 0.08–0.21）**：
   - Cout<32 时 OSV=32 的 lane 一半闲置（32 lane 只算 16 通道）；
   - Cin=3 时循环太短，prologue/epilogue/输出阶段占比过大。
   这解释了调研标准自动标出的「离极限最远」层（R24 修正后 ratio 0.08–0.11）。

### 2.2 两条数据通路的逐 size 分工

`config/tuning.json` 显示 6 条 native 胜出的层（其余 OV 胜）：

| shape | 胜者 | ops |
|---|---|---:|
| 40×40 s1 32→64 | native | 6.83 |
| 160×160 s1 16→16 | native | 6.77 |
| 80×80 s1 32→16 | native | 6.74 |
| 80×80 s1 16→32 | native | 6.52 |
| 160×160 s1 16→8 | native | 3.79 |

→ 确认应让调优器在**每个 shape** 上同时看到 OV 与 native（round 24 起 native 对 stride=2 也枚举），
而不是只在非 s2 时给 native 候选。

---

## 3. 尝试：输入通道双累加集（软件流水/ILP）——**负结果，已回退**

按「默认软件流水不够好」的思路，给 `conv_ov.cl` 加了 `-DUK`：一次处理 UK 个输入通道，
每通道一套独立累加器，最后归约（把单累加器依赖距离 `OBW*OBH` 提到 `UK*OBW*OBH`）。
离线 ISA 确认 UK=2 无 spill（`maxR=124`，8×2）。实测：

| shape | UK=1 | UK=2 |
|---|---:|---:|
| 40×40 s1 64→64 OBW8 OBH1 | 8.44 | 7.87 |
| 40×40 s1 64→64 OBW8 OBH2 | 7.83 | 7.15 |
| 80×80 s1 64→64 OBW8 OBH2 | 13.77 | 12.90 |

**UK=2 全面更慢**；且该变体在 `kernel_numtest` vs numpy 上未过数值（`mean_rel≈0.5`），
根因是变体自身的 codegen/正确性问题，**已整体回退**（生产 kernel 逐字节回到 R23 状态）。

**判读**：IGC 的单累加器 codegen 在 7 个硬件线程之间已经提供了足够的 ILP；再堆独立累加器
只是抬高寄存器压力与调度长度，并没有消掉真正的停顿（更可能是 **send/全局读延迟** 与
**每 WG 的边界谓词/prologue/epilogue**）。所以下一轮要压的不是「累加器 ILP」，而是
**内存访问指令**与**每 WG 固定开销**。

---

## 4. 自动调优接入（本轮落地的正向改动）

1. **修正中间标准 `expected_ops`**（`include/infvino/Tuning.hpp` / `src/Tuning.cpp`）：
   - 旧：`conv3x3` 大层期望 ~12.15（`16.4 × slotFraction(0.35)`），把 ~16/10 当上限；
   - 新：`expected = 32 × 0.636 × prologue_amort × grid_factor`，上界 `kConvOvIssueCeiling = 20.3`；
   - 常数带 R24 ISA 出处（`kConvOvMadFraction = 288/453`）。
   - `tuning_test` 相应更新（新增「期望 > staging-free 16.4」的断言，锁住「不再锚定实测」）。
2. **两通路候选**（`src/Autotuner.cpp::candidatesConv3x3`）：
   - OV `os_iyx_osv32` 的 OBW/OBH 谱系（s1/s2）；
   - native `conv3x3_f16` 的 `TX∈{40,20} × CB∈{32,16}`，**现在对 stride=2 也枚举**
     （R23 已证 s2 + TX=40 可 build），使「按 size 选通路」覆盖全部层。
3. **文档**：本文件 + `docs/kernel.md` Round 24 + README 效率结论更新 +
   `docs/autotuning.md` §0/§7 的中间标准表修正。

> 注意：本轮**不改 kernel 源码**；调优缓存里已有条目的 kernel/options 仍有效，
> 新标准只改变 `ratio` 的解释（`measured/expected`），不影响运行时选核与数值。

---

## 4.5 split-K 预实验（两个廉价 proxy 直接否决）

「提高 grid/占用」的第一直觉是 split-K（按输入通道路由到更多 WG）。在动手写两 kernel 的
归约之前，先用两个**只改 shape、不改 kernel** 的 proxy 验证前提：

| proxy | 变化 | 实测 ops | 说明 |
|---|---|---:|---|
| Cin 扫描 @40×40 8×1 C64 | Cin 64→32（= split-K S=2 的每 WG 工作量）| 8.74 → **6.62（−24%）** | 每 WG 固定开销 ≈ 27% |
| | Cin 16（= S=4）| 4.44 | |
| grid 扫描 @Cin64 8×1 | grid 400 | 8.77 | |
| | grid 800 | ~9.0（+3%） | |
| | grid 1600 | **11.0（+25%）** | 只有到 ~1600 WG 才有明显收益 |

判读：**split-K 是拿确定的 −24%（工作/WG 减半）换 +3% 的 grid 收益**，S=4 更是灾难
（Cin=16 → 4.44）。grid 曲线也说明 40×40 的问题是**波量化**：max 并发 ≈ 80 EU × 7 线程
= 560 sub-group，400 WG = 0.71 波、800 = 1.43 波（利用率仍 ~0.71）、1600 = 2.86 波（~0.95）——
即 400→800 几乎无收益、1600 才满，而不是「WG 越多越好」。
→ **split-K 不论 S=2/4 在本机都不划算，已放弃实现。**

---

## 5. 下一步（按预期收益排序）

1. **压每 WG 的固定/延迟开销（真正的短板，~27%）**：ISA 显示输入是逐元素
   `byte gathering read 16b` + `if/else` 谓词、输出是 `byte scattering write 16b`。
   对内部块（80×80 约 72%）走 `intel_sub_group_block_read`/合并写，削掉谓词与散列访存，
   同时让 prologue/epilogue 更短。**这是当前证据指向的第一杠杆。**
2. **波/网格量化**：40×40 类无论怎么调 block 都受限于 `spatial×Cout/32`（≤400 WG < 560），
   靠调 grid 无法补齐；只能靠减少每波里的停顿（同上）或换问题切分（split-K 已否决）。
3. **VECO/OSV64：每 lane 4 个输出通道**，让 1 条广播喂 2 个 `half2` mad（提高每广播算术量）。
   注意：VECO 会在保持工作/WG 的同时把 `Cout/(SG·VECO)` 的通道组数减半，
   **grid 不增**（见 §2.1 的推导），因此它只改善指令配额，不改善占用；风险：128 GRF。
4. **重跑逐层 autotune**（新候选覆盖 stride=2 native；新中间标准可自动指出最远的层）。

---

## 6. 复现

```bash
# 离线 ISA（不需要 GPU）
ocloc compile -file kernels/conv_ov.cl -device tgllp \
  -options "-DOBW=8 -DOBH=2 -DSTRIDE=1 -DPAD=1 -DACT=1 -DRES=0 -DSG=16 -cl-mad-enable -cl-fast-relaxed-math" \
  -output /tmp/ov
ocloc disasm -file /tmp/ov_tgllp.bin -device tgllp -dump /tmp/ov
# 主循环在 /tmp/ov/conv3x3_ov_KernelHeap.asm：288 packed mad / 453 指令 = 63.6%

# block 扫描（每个配置一条短命令，遵守 benchmark_protocol.md）
./build/kernel_bench --op conv3x3ov --conv-shape 64,64,40,40 \
  --conv 8,1,1,32,16,1,1,1,3,1,16,8,0,0,2,0,2,0,0,0,0,0,1 --iters 5

# 数值（跳过较重的 GEMM 套件）
python3 scripts/kernel_check.py --repo $PWD --image infvino-dev:latest --skip-gemm

# 中间标准自检（不需要 GPU）
./build/tuning_test
```
