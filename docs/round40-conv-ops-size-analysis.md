# Round 40：conv kernel 的 ops/EU/cyc × size 全局分析（含被推翻的结论）

> 用 `ops/EU/cyc` 中间标准 + **每族物理上限（`KernelFamily.ceiling`）** 两把尺子，
> 对 `config/tuning.json`（R39 重扫后）的全部 conv3×3 签名按 size 分桶，并做受控
> head-to-head（`kernel_bench`，同会话）。
>
> 重要口径：签名里的 `W/H` 是**输出**空间；`kernel_bench --conv-shape Cin,Cout,H,W`
> 里的 `H/W` 是**输入**空间（`Hout=(Hin+2p-3)/s+1`）。跨会话绝对值有 ±15% 噪声，
> **绝对排名用 tuning.json（部署实测），族间相对比较用同会话 head-to-head。**

---

## 0. 一句话结论

1. **修好 SLM 后，`conv3x3_ov` 在 24/32 个部署签名上胜出**（R37 时 blk 曾赢 13 条）。
   conv3×3 的格局从「blk 打大通道、native 打窄通道、ov 打大空间 s1」变成了
   **ov 通吃 + native/blk 只在极窄/极小 Cout 保留**。
2. **离指令上限（ov ≈ 20.3）最近的**是「大空间 s1、Cin/Cout 对齐」层（80×80 64→64，
   ratio **0.74**，ops 15.0）；**最远的**依次是 stride-2 大层（ratio 0.55）、
   `Cout%32≠0`（0.59）、stem `Cin=3`（0.28）、`Cout≤16`（0.26–0.48）。
3. 剩余缺口**主体是延迟/占用，不是配置缺失**——SLM 是最后一个大杠杆，已被这轮吃掉。
   真正还有**明确配置空间**的只有两类小热点（见 §4）：
   `Cout=8` 加 native `CB=8`（+21%）、stride-2 大层加 `OBW=7`（~+4%）。

---

## 1. 已部署视角：32 个 conv3×3 签名按 size 分桶

`ratio = measured / 该胜出族自己的 ceiling`（ceiling 已含 R39 的 WG/sub-group 修正）。

| 桶 | 代表签名（输出） | 胜出族 | ops | ceiling | ratio | 结构性限制 |
|---|---|---|---:|---:|---:|---|
| 大空间 s1·对齐 | 80×80 64→64 | ov 8×2 SLM4 | **15.0** | 20.1 | **0.74** | 已近指令上限 |
| 中空间 s1·对齐 | 40×40 64→64 | ov 8×2 SLM8 | 11.7 | 20.1 | 0.58 | 网格/延迟 |
| 小空间 s1 | 20×20 64→64 | ov SLM8 | 7.1 | 20.1 | 0.36 | 波量化 |
| 小空间 s1·大 K | 20×20 128→128 | ov 10×2 SLM4 | 11.6 | 20.2 | 0.58 | |
| **stride-2 大层** | 80×80 s2 64→64 | ov 6×2 SLM4 | 11.1 | 20.1 | 0.55 | 输入复用↓ + 占用 |
| stride-2 小通道 | 160×160 s2 16→32 | ov 8×2 SLM2 | 8.8 | 19.4 | 0.45 | 同上 |
| **Cout%32≠0** | 80×80 64→51 | ov 8×2 SLM2 | 11.8 | 20.1 | 0.59 | 输出 lane 浪费 ~20% |
| Cout%32≠0 | 20×20 256→51 | ov SLM8 | 9.2 | 20.3 | 0.45 | |
| **Cout≤16** | 80×80 s1 32→16 | native CB16 | 7.1 | 16.1 | 0.44 | lane=channel 半空 |
| Cout=8 | 160×160 16→8 | blk OBW8 | **4.2** | 15.9 | **0.26** | 8/16 lane 浪费 |
| **Cin%16（51）** | 80×80 51→51 | ov 8×2 SLM2 | 11.1 | 20.0 | 0.56 | 输入 chunk 尾块 |
| Cin=8 | 160×160 8→16 | native CINC8 | 6.3 | 15.1 | 0.41 | 同上 |
| **stem Cin=3 s2** | 320×320 s2 3→16 | native CINC3 | 3.8 | 13.4 | **0.28** | 复用极低 + 固定开销 |

**家族部署汇总**（按 ms）：

| 族 | 签名数 | Σms | mean ops | mean ratio |
|---|---:|---:|---:|---:|
| `conv3x3_ov` | 24 | 3.653 | 10.09 | 0.50 |
| `conv3x3_blk` | 3 | 0.659 | 7.96 | 0.49 |
| `conv3x3_f16` | 3 | 0.395 | 5.73 | 0.38 |
| `conv3x3_cin3` | 2 | 0.089 | 2.33 | 0.19 |
| **合计** | 32 | **4.797** | | |

按 `ms·(1−ratio)` 估的「到指令上限的余量」合计 **2.29 ms**（但如 §5 所述，
其中大部分是延迟，不是可白拿的配置空间）。

---

## 2. 受控 head-to-head（同会话，`kernel_bench --iters 6`，ops/EU/cyc）

### 2.1 stride-2 大层：out 80×80 s2 64→64

| 族/配置 | SLM1 | SLM2 | SLM4 | SLM8 |
|---|---:|---:|---:|---:|
| ov 5×4 | 7.88 | 7.65 | 7.49 | 6.93 |
| ov 5×2 | 9.30 | 9.81 | 10.28 | 10.18 |
| ov 6×2 | 9.26 | 10.70 | **11.08** | 10.84 |
| ov 8×2 | 8.21 | 8.98 | 9.50 | 9.60 |
| ov **7×2** | — | **11.12** | **11.56** | — |
| blk OBW8 | 9.99 | 10.91 | 10.87 | 6.22 |
| native | **build FAIL**（s2 大通道） | | | |

→ ov 6×2/7×2 SLM4 最优（部署用的 6×2 SLM4）。`native` 在 stride-2 大通道上
**编译失败**（GRF/SLM），所以这一档没有第三条通路。

### 2.2 输出通道浪费：out 160×160 16→8（Cout=8）

| 配置 | ops |
|---|---:|
| **native TX40 CB8 CINC16** | **5.03** |
| blk OBW8 SLM1（现部署） | 4.16–4.37 |
| native TX40 CB16 | 3.47 |
| blk OBW4 | 3.19 |
| ov 10×2 | 3.21 |

→ **native `CB=8`（+21%）是这一档明确、可落地的候选缺失**。当前 native 候选只枚举
`CB∈{32,16}`（`KernelFamilies.cpp`），没有 `CB=8`。

### 2.3 stem：out 320×320 s2 3→16（Cin=3）

| 配置 | ops |
|---|---:|
| native TX40 CB16 CINC3（现部署） | 3.1–3.8 |
| native TX20 CB16 CINC3 | 3.0 |
| ov 5×4 / 8×2（任意 SLM） | 1.1–1.7 |
| blk OBW8 | 0.8–0.9 |

→ native `CINC=Cin`（精确 3）仍是唯一可用通路；ov/blk 在 Cin=3 上都不行。
距上限 13.4 还有 ~3.5×，但 Cin=3 每输出只有 27 个 mad、固定开销占主导。
**唯一结构方向**：Cin=3 时全展开 27 tap、去掉 chunk 循环（R34 §6 提过，未做）。

### 2.4 Cin 非对齐：out 80×80 s1 64→51（Cout=51）/ 40×40 51→51

| 配置 | 64→51 | 51→51 |
|---|---:|---:|
| ov 8×2 SLM4 | 10.90 | 6.54 |
| ov 8×1 SLM4 | 8.98 | 6.03 |
| ov 10×2 SLM1 | **10.78** | 5.88 |
| ov 10×2 SLM2 | 9.65 | **7.40** |
| blk OBW8 SLM2 | **11.22** | 6.07 |
| native | — | 5.09 |

→ `Cout=51`（%32=19）两者都只剩 ~80% 输出 lane；blk（16/WG）在这里和 ov 打平。
这是 **OSV=32 的固有浪费**，除非做 OSV=16 / 小 Cout 专用通路。

---

## 3. 被本轮**推翻**的既有结论

| 出处 | 旧结论 | 现状 | 依据 |
|---|---|---|---|
| R33 §3.2 | 「blk 是赢 OV 的小空间/大通道区域」 | **推翻** | 40×40 64→64：ov 11.7 vs blk（旧 8.4）；80×80 64→64：ov 15.0 vs blk 13.5；blk 只剩 3 签名 |
| R33 §3.1 | 「native 窄通道是**唯一**值得做的方向」 | **部分推翻** | ov+SLM 同样把小网格拉起来；native 只在 `Cout≤16` 极窄层保留 |
| R24 / R33 §4.5.4 | 「40×40 波量化受限，加 ILP 无用，只能 split-K/batch」 | **修正** | 根因（全局读延迟/占用）判对，但「无解」错：SLM 正是解药，40×40 64→64 从 8.4 → 11.7（**+39%**） |
| R37 §4 | 「SLM 是 `conv3x3_blk` 的收益」 | **修正** | ov 的 SLM 收益更大更广（ov SLM 后 24/32 签名） |
| R37 §7 | 「conv_ov SLM 数值错误，禁用」 | **已修** | R39；`model_check` 三模型 PASS |
| R33 §2 | 「`expectedOps` 对 blk/native 口径失真」 | **已修** | 注册表 per-family ceiling + R39 WG/sub-group 修正 |
| R33 §4.5.5 | 「CINC 类修复量级有限 ~0.3–0.6%」 | **维持** | 仍成立；但它是「免费的正确性」，不是杠杆 |

> 归纳：**R33–R37 的“剩余空间分析”是在 ov 的 SLM 被禁用的前提下做的**——
> 候选集缺失会把「还能不能压」的结论系统性带偏。这是分析框架的一条元教训。

---

## 4. 离极限远的层 × 优化空间（按可落地性排序）

| # | 热点 | 现状 ratio | 已探明的空间 | 预期 | 成本 |
|---|---|---:|---|---|---|
| 1 | **stem Cin=3**（320×320 s2，0.225 ms） | 0.28 | native 已是最佳；Cin=3 全展开 27 tap、去 chunk 循环 | +1% 量级整网 | 一个专用 `.cl` / 改 `conv_cin3` |
| 2 | **stride-2 大层**（80×80 s2 64→64，0.408 ms） | 0.55 | ov `OBW=7`（+4%）；s2 专用 pass（block-read 输入） | 每层 3–5% | ✅ 已加 `OBW=7` 候选（待重扫） |
| 3 | **Cout=8**（160×160 16→8，0.136 ms） | 0.26 | native `CB=8`（**+21%**，已实测 5.03 vs 4.16） | 该档 +21% | ✅ 已加 `CB=8` 候选（`Cout≤8`，待重扫） |
| 4 | `Cout%32≠0`（64→51，0.307 ms） | 0.59 | OSV=16 / 小 Cout 专用通路 | 该档 +10–20% | 新 kernel 变体 |
| 5 | 小空间 20×20 | 0.36–0.58 | SLM 已吃掉主要空间；余下网格 | 小 | — |

**已落地的最小改动**：#3（native 候选 `CB∈{32,16}` → `Cout≤8` 时追加 `CB=8`）
和 #2（stride-2 ov 候选追加 `{7,2}`）都已在 `KernelFamilies.cpp::candidates` 里加上，
`tuning_test` PASS。它们是纯候选枚举补充，**在下次 `--retune` 重扫前不改变运行时行为**；
重扫后按 shape 生效（`config/tuning.json` 需一并更新）。

---

## 5. 为什么不再有「大杀器」：口径提醒

把 `ratio` 低直接当「可挖 headroom」会系统性高估：

- ov 的指令上限 20.3 需要**每个 mad 都发射且无停顿**；实测最高 15.0（80×80 C64）。
- 剩余缺口的成因是 **global-read 延迟 + 波量化**（R24 已反汇编/定位）。SLM 通过
  「一个 WG 里塞多组在飞工作」把这块吃掉了大半；再往下只剩更激进的 occupancy
  （更多 SLM/更多在飞 WG）或减少访存。
- 因此 `ms·(1−ratio)` 的 2.29 ms **不是**可直接收获的量；按
  `docs/profiling-budget.md` §3.0 的纪律，`ratio`/`headroom` 只用于**同族内排序**，
  真正决定「值不值得写」的是 §4 这类**有明确机制**的条目。

---

## 6. 复现

```bash
# 部署视角（无需 GPU）：从 tuning.json 生成 §1 表
python3 - <<'PY'
import json,re
exec(open('/tmp/an_conv.py').read().split("d=json.load")[0])  # 见 Round 39 的分析脚本
PY

# head-to-head（容器，同会话；每条一条命令）
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc '
    # 2.1 stride-2 大层
    OV_SLM=4 ./build/kernel_bench --op conv3x3ov --conv-shape 64,64,160,160 \
      --conv 7,2,1,32,16,2,1,1,3,1,16 --iters 6
    # 2.2 Cout=8：native CB=8 vs 现部署 blk
    ./build/kernel_bench --op conv3x3 --conv-shape 16,8,160,160 \
      --conv 40,8,1,8,16,1,1,1,3,1,16 --iters 6
    ./build/kernel_bench --op conv3x3blk --conv-shape 16,8,160,160 \
      --conv 8,1,1,1,16,1,1,1,3,1,16 --iters 6
    # 2.3 stem
    ./build/kernel_bench --op conv3x3 --conv-shape 3,16,640,640 \
      --conv 40,8,1,16,3,2,1,1,3,1,16 --iters 6'
```
