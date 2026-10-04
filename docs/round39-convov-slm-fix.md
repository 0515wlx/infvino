# Round 39：conv_ov SLM 数值 bug 修复 + conv3×3 再攻一轮 + 分析框架反查

> 承接 R37 §7/§10（「`conv_ov` SLM 归约数值错误，根因未定位」）。本轮：
> 1. **定位并修复** `conv_ov` 的 SLM_DIV 数值 bug（一句结论：**寄存器工作组几何写错，
>    子组没有沿最快的 local 维铺开**）；
> 2. 借修复重新打开 `conv_ov` 的 `SLM_DIV` 候选，conv3×3 在**小网格**上再拿一大块；
> 3. 用**算子族上限框架**（`KernelFamilies.cpp` 的 `ceiling`）重算 conv3×3 余量，并
>    **反向核对框架**，修掉 3 个口径 bug、记录 1 个未决模型问题。
>
> 结果：三模型 `model_check`/`reuse_check` **PASS**；busy（`kernel_run --report`，iters=20）
> **yolov8n −11.2%、yolo11n −8.1%、mobilenetv3 −3.2%**。

---

## 1. 根因：SLM_DIV 的**工作组几何**错了，不是归约逻辑

R37 把 `SLM_DIV` 加进 `conv_ov.cl` 时，写成：

```cl
__attribute__((reqd_work_group_size(1, SLM_DIV, SG)))   // ← bug
const int sub = get_local_id(1);
```

OpenCL 的**子组按 linear local id 连续 16 个 work-item 组成**，而 linear id 的展开是
**x 最快、其次 y、最后 z**。local size = `(1, SLM_DIV, SG)` 时：

```
linear_id = y + z * SLM_DIV          // x=0, size_x=1, size_y=SLM_DIV
```

所以一个 16-lane 子组**横跨多个 z、并且 y 在子组内交替**：子组内的 lane 一半
`sub=get_local_id(1)=0`、一半 `=1`，但代码却假设「同一子组共享同一个 `sub`」。

后果链：
- 每个 lane 按自己看到的 `sub` 取 `kd ∈ [sub·Cin/SLM, (sub+1)·Cin/SLM)`，同一子组里
  不同 lane 累加了**不同的输入通道区间**；
- 归约 `partial[(sub*SG+lid)*neu + i]` 只由「自己那个 sub 的 lid 奇偶」写入，
  sub==0 的 lane 去读 `partial[(s*SG+lid)*neu+i]` 时，读到**从未被写过**的槽位；
- `__local` 未初始化 → **NaN / inf**（与 R37 观察到的 mean_rel 5.2e-2 / NaN 一致）。

**归约的数学本身是对的**（每个 kd 只属于一个 sub；sub 0 求和所有 sub 的部分和），
错的是「哪些 work-item 属于同一个 sub」这个映射。

`SLM_DIV=1` 时几何退化成 `(1,1,16)`，行为逐位等价——这也解释了为什么 R37 的
「`SLM_DIV=1` 立刻 PASS」。

---

## 2. 修复

**kernel（`kernels/conv_ov.cl`）**：把 `SLM_DIV` 个子组放到**最快的 local 维**（dim2），
dim1 固定为 1：

```cl
__attribute__((reqd_work_group_size(1, 1, SG * SLM_DIV)))
const int fmg = get_group_id(2);          // 输出通道子组
const int lid = get_sub_group_local_id(); // = local_id(2) % SG
const int sub = get_sub_group_id();       // = local_id(2) / SG
```

这样单个子组的 16 个 lane 必然共享同一个 `sub`（`sub_group_broadcast` 的语义才成立），
`SLM_DIV=1` 与旧几何逐位相同。

**launch 几何**（`SLM_DIV` 之前只改了 kernel，没改网格，这是第二处必须同步的地方）：
`lws = {1, 1, 16·SLM}`、`gws[2] = ceil(ceil(Cout/2)/16)·16·SLM`，
`gws[1] = ceil(Hout/OBH)`。同步改了 3 处：
- `src/PlanModel.cpp`：tuned 运行路径 + autotune 的 `makeEnqueue`；
- `src/tools/kernel_bench.cpp`：`OV_SLM` 的 bench 几何。

---

## 3. 数值验证

| 检查 | 结果 |
|---|---|
| `kernel_bench --op conv3x3ov --verify`（128→64/40×40 等 多 shape，SLM=1/2/4） | PASS（mean_rel ~2–4e-3，`max_rel` 随 SLM 下降） |
| `model_check` 三模型（缓存已启用 `SLM>1`） | **ALL PASS**（y8 4.3e-4 / y11 8.3e-4 / mb 1.2e-2） |
| `reuse_check` 三模型（A→B vs 新进程 B） | **ALL PASS** |
| `tuning_test`（离线候选/上限自检） | PASS |

> 附带观察：`SLM>1` 的 `mean_rel` 比 `SLM=1` **略优**（例如 128→64/40×40：
> 4.32e-3 → 2.36e-3），因为 `kd` 区间更短、fp16 累加链更短（不是精度回归）。

---

## 4. 性能：SLM 打开后 conv3×3 再拿一块（主要在**小网格**）

`kernel_bench`，`ops/EU/cyc`（越高越好）：

| shape (Cin,Cout,H,W) | 配置 | SLM1 | SLM2 | SLM4 | SLM8 |
|---|---|---:|---:|---:|---:|
| 64,64,40,40 | 8×1 | 8.48 | 8.42 | 9.55 | **9.75** |
| 128,64,40,40 | 8×1 | 10.12 | 9.93 | 11.02 | **11.08** |
| 64,64,80,80 | 8×2 | 12.73 | 12.94 | **13.46** | 13.27 |
| 64,64,80,80 | 10×2 | **13.59** | 11.78 | 11.95 | 10.51 |
| 256,64,20,20 | 8×1 | 3.62 | 5.91 | 7.88 | **8.17** |
| 128,128,20,20 | 8×1 | 6.31 | 7.62 | 7.27 | **8.89** |

规律与 R37 一致：**网格越小、SLM 收益越大**（20×20 256→64：**+126%**），
大网格（80×80 C64）当 1 个子组就够时反而 SLM 不划算——所以它是**按 shape 的 knob**，
交给 autotune 选。

重扫 conv3×3 后（`config/tuning.json` 已更新），若干原先由 `conv3x3_blk` 胜出的层
翻转给 `conv3x3_ov`+SLM：

| shape | 旧胜出（ms） | 新胜出（ms） | Δ |
|---|---|---:|---:|
| 20×20 s1 128→128 | blk SLM8 0.1512 | ov 10×2 SLM4 **0.0976** | −35% |
| 20×20 s1 256→64 | blk SLM8 0.1209 | ov 10×2 SLM8 **0.0987** | −18% |
| 20×20 s1 64→64 | blk SLM4 0.0505 | ov SLM8 **0.0397** | −21% |
| 40×40 s1 64→64 | blk SLM4 0.1128 | ov 8×2 SLM8 **0.0972** | −14% |
| 40×40 s1 51→51 | ov SLM1 0.1021 | ov 8×2 SLM8 **0.0859** | −16% |

**整网 busy**（`kernel_run --plan … --report --iters 20`，同会话 A/B）：

| 模型 | baseline | R39 | Δ |
|---|---:|---:|---:|
| yolov8n-pose | 11.966 | **10.627** | **−11.2%** |
| yolo11n-pose | 12.773 | **11.744** | **−8.1%** |
| mobilenetv3-small | 1.692 | **1.638** | **−3.2%** |

> 量化：这不是「换 kernel」，而是把 R25/R37 移植时漏掉的 `SLM_DIV_FACTOR`
> **真正接上**——R37 只接上了 blk 的一半，ov 的一半因数值 bug 被禁用。

---

## 5. 分析框架反查（`KernelFamily.ceiling` + `expectedOps`）

用「逐条缓存条目重算 expected/ratio」的方式（`refresh-expected`，零 GPU）反查，
发现并修复 3 个**事实性/一致性 bug**，另记录 1 个未决模型问题。判据是硬性的：
**任何测得 ops > 框架 ceiling（ratio>1）都说明模型是错的**（ceiling 必须是上界）。

### 5.1 `conv3x3_ov` ceiling 的通道粒度错（16 vs 32）——**已修**

`ov` 是 lane=输出通道、`OSV=32`（每 lane 2 通道），一个子组覆盖 **32** 个输出通道。
`expectedOps` 里的 WG 数用的是 `ceil(Cout/32)`，但注册表 ceiling 写的是
`wgCount(…, chPerWg=16)` = `ceil(Cout/16)`，**把 WG 数多算一倍**，使小网格上的
`gridFactor` 系统性偏高、ceiling 失真。改为 `chPerWg=32`。

### 5.2 两个 ceiling 把「work-group」当占用单位，而实际是「sub-group」——**已修**

`gridFactor(n_wg, eu)` 是为「1 WG = 1 子组」的 kernel 标定的。但：
- `conv_ov` + `SLM_DIV`：1 WG = `SLM` 个子组；
- `conv3x3_f16`（native）：local size `(40,8)` = 320 work-item = **20 个子组**。

用 WG 数会让小网格的 ceiling **低于实测**（native `80×80 32→16` ratio 曾 1.59；
ov `20×20 256→64` 修 5.1 后比值 1.51）。改为按**子组**计：ov 乘
`SLM_max = 2^floor(log2(min(8,Cin)))`，native 乘 `subPerWg = 40·8/16 = 20`。
修后 conv3×3 全族 **max ratio ≤ 0.81**（native 0.48 / ov 0.74 / blk 0.81），不再是自相矛盾。

### 5.3 `refresh-expected` 只覆盖 gemm/conv1x1——**已修**

`PlanModel::refreshExpected()` 原来只反解 `gemm`/`conv1x1`/`conv1x1_cat4` 的 key，
**完全不处理 conv3x3**。于是改了 conv3x3 的 ceiling 之后，缓存里的 `expected/ratio`
无法离线刷新，只能重跑 GPU 调优——正是「改了模型却看不出效果」的坑。补上 conv3x3，
并且统一改成「**优先用胜出族自己的 ceiling**」（与 `autotuneOp` 一致，而不是旧的
`expectedOps` 兜底）。

### 5.4 未决：非计算族的 roofline ceiling 仍偏低——**已记录，未改**

修完 conv3×3 后，缓存里 ratio>1.05 的条目集中在**非计算**族：

| kernel | ratio max | 说明 |
|---|---:|---|
| `ew_binary` / `ew_binary_v` | 1.68 / 1.30 | 内存 roofline `expectedOps` 偏低 |
| `slice_axis` | 1.53 | 同上 |
| `copy_c2` | 1.48 | 同上 |
| `softmax_axis` | 1.59 | 同上 |
| `conv3x3_f16` | 0.48 | ✅ 已修 |
| `conv3x3_ov` | 0.74 | ✅ 已修 |

这些是 `Bottleneck::Memory` 的族，`analyze_budget.py` 对 ratio≥1 已按 0 headroom 处理，
不会产生负值；但它们的 `expected` 作为「上界」仍偏低。**建议下一轮**：对内存族用
「工作集档位」标定（`scripts/analyze_cache.py` 已有工具），或直接把 ratio>1 记为
「模型刻度问题」而不参与排名。本轮按纪律（`docs/profiling-budget.md` §3.0）不改。

---

## 6. conv3×3 还能不能再攻？

按修好的框架重排（`ratio = measured / familyCeiling`），conv3×3 的两块：

- **小网格（20×20 / 40×40 的窄-中通道）**：SLM 已把 ratio 从 0.2–0.4 抬到 0.45–0.74；
  距离 ov 的**指令发射上限 20.3** 还有 ~1.4–2×，但主要缺口是**网格/延迟**，
  不是指令数——R24/R34 已多次证明「加 ILP / 改几何」在此无收益。
- **大网格（80×80 C64 等主导层）**：ratio 0.66–0.80，且 SLM 已接近最优点
  （8×2 SLM4 = 13.46、10×2 SLM1 = 13.59），**是当前最接近 quota 的一档**。
- **剩余可做**：`conv_cin3`（stem）ratio 仍 0.14–0.19，是 conv3×3 里离 quota 最远的；
  B 方向（拆 `Cout` / 波量化）在 batch=1 下已被 R24 否决。**建议**：stem 的
  Cin=3 全展开或换更激进的 lane 映射，是唯一还有结构空间的方向（量级小）。

> 反查也澄清了 R33 §3 的一个**结论偏差**：R33 说「native 窄通道是唯一值得做的方向」，
> 但那是在 **ov 的 SLM 候选被禁用**的前提下得出的。打开 SLM 后，ov 在小网格上
> 反而反超了 native/blk——**候选集的完整性本身就是分析结论的前提**。

---

## 7. 复现

```bash
# 构建（容器）
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake --build build -j4'

# 数值（SLM 1/2/4 全 PASS）
scripts/gpu_guard.sh run docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc '
    for s in 1 2 4; do OV_SLM=$s ./build/kernel_bench --op conv3x3ov \
      --conv-shape 128,64,40,40 --conv 8,2,1,32,16,1,1,0,3,1,16 --iters 5 --verify; done'

# 性能（SLM sweep）
#   OV_SLM=<n> ./build/kernel_bench --op conv3x3ov --conv-shape 256,64,20,20 \
#     --conv 8,1,1,32,16,1,1,0,3,1,16 --iters 10

# 整网 A/B（缓存）
#   INFVINO_TUNING_CACHE=config/tuning.json ./build/kernel_run \
#     --plan models/yolov8n-pose/model.plan --report --iters 20

# 重扫 conv3×3（候选集已含 SLM>1）
python3 scripts/autotune.py --model yolov8n-pose --ops conv3x3 --batch 3 --iters 8

# 离线重算 expected/ratio（零 GPU；现在覆盖 conv3x3）
./build/kernel_autotune --plan models/yolov8n-pose/model.plan \
  --cache config/tuning.json --refresh-expected

# 整网数值
python3 scripts/model_check.py  --model yolov8n-pose --repo $PWD --image infvino-dev:latest
python3 scripts/reuse_check.py --model yolov8n-pose --repo $PWD --image infvino-dev:latest
```
