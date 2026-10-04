# busy / net / e2e 预算与优化记分卡（分析框架）

> 目标：把「三种推理状态」统一成一个可归因的**预算方程**，用它回答两个问题：
> 1. **时间花在哪**（budget 瀑布）；
> 2. **每个优化部件买到了多少**（消融归因）。
>
> **标尺是硬件极限 / 中间标准（`expected_ops`），不是 OpenVINO。** OV 只是研发阶段
> 判断「能不能上线」的中间参照；本框架面向「优化到硬件极限」这一设计哲学。

---

## 1. 三种状态的定义（统一口径）

| 状态 | 含义 | 计时边界 | 来源 |
|---|---|---|---|
| **busy** | GPU kernel 执行时间之和 | 不含 launch / H2D-D2H / pre / post | `kernel_run --report`、`--profile-json` |
| **net** | 一次完整前向 | 输入 H2D + busy + 输出 D2H + f16 转换 | `infvino_bench` 的 `infer` |
| **e2e** | 从一帧图像到结果 | preprocess + net + postprocess | `infvino_bench` 的 `total` |

预算方程：

```
e2e  = pre + net + post
net  = busy + launch + copy/f16 + bubble
busy = Σ_node kernel_ms
```

> **口径纪律**：`net/e2e` 必须用**非 profiling** 的 `infvino_bench`；`busy`/host 分段
> 用 `kernel_run`（profiling 会逐节点 `clWaitForEvents`，改变 host 行为）。两者结合时
> `net-busy` 是「非 GPU 部分」的总量，其中 `enqueue/sync` 是 profiling 下的实测分量。

---

## 2. 工具链（三层）

```
[L0 采样]  GPU（安全容器，遵守 benchmark_protocol.md）
   kernel_run --plan … --report --profile-json out.json --profile-label <model>
      → 逐节点 GPU 时间 + 结构化计数 + 内存 + host 分段（机器可读）
   infvino_bench --config config/models.yaml --key <model> --iters N
      → e2e/pre/net/post 文本
        │
        ▼
[L1 分析]  scripts/analyze_budget.py（纯离线，无 GPU）
   读 profile-json + bench + config/tuning.json + model.plan
        │
        ▼
[L2 报告]  预算瀑布 + busy 归因 + 优化记分卡（ratio / headroom_ms）
   scripts/profile_ablation.py  → 优化消融矩阵（每个开关的 Δbusy/Δnet/Δe2e/Δfps）
```

### 2.1 Phase 1/2：预算与记分卡

```bash
# 采样（容器内；--profile-json 是 Phase 2 新增的机器可读输出）
docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 --device=/dev/dri/renderD128 \
  -v "$PWD":/workspace/infvino -w /workspace/infvino infvino-dev:latest bash -lc '
    export LD_LIBRARY_PATH=$PWD/build
    ./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 20 \
        --profile-json /tmp/prof.json --profile-label yolov8n-pose
    ./build/infvino_bench --config config/models.yaml --key yolov8n-pose --iters 80 > /tmp/bench.txt'

# 分析（需要 host 的 cv2/numpy；宿主 python3 即可）
python3 scripts/analyze_budget.py --model yolov8n-pose --plan models/yolov8n-pose/model.plan \
  --profile-json /tmp/prof.json --bench /tmp/bench.txt --tuning config/tuning.json
```

> 没有 `--profile-json` 时可用 `--report` 文本回退（Phase 1），但记分卡退化为
> 「签名级近似」；有 JSON 时是**逐节点精确 join**（signature → expected → ratio → headroom）。

### 2.2 Phase 3：优化消融矩阵

```bash
python3 scripts/profile_ablation.py --repo $PWD --image infvino-dev:latest \
  --models yolov8n-pose yolo11n-pose mobilenetv3-small \
  --iters 40 --iters-report 20 --out /tmp/ablation.json
```

每个开关一个独立容器（竞态隔离），逐模型跑 bench + profile，输出
`Δbusy / Δnet / Δe2e / Δfps`。支持的开关：

| 开关 | 消融对象 | 生效点 |
|---|---|---|
| `INFVINO_TUNING=off` | 自动调优 | 运行时 |
| `INFVINO_NO_FUSE_RES=1` | 残差融合（R33） | 运行时 |
| `INFVINO_NO_BLOCK_LAYOUT=1` | R36 持久 blocked 布局 | 运行时 |
| `INFVINO_NO_REORDER_DEDUP=1` | R36 同帧重排去重 | 运行时 |
| `INFVINO_NO_LAUNCH_CACHE=1` | P2 每节点 dispatch 缓存 | 运行时 |
| `INFVINO_NO_POOL=1` | P0 激活内存池 | 运行时 |
| `INFVINO_NO_POOL_OFFSET=1` | 激活池 byte-offset 子分配（本轮落地，默认开） | 运行时 |
| `INFVINO_PROGRAM_CACHE=<dir>` / `none` | 磁盘 kernel 二进制缓存（本轮落地；观测用，非消融） | 运行时 |
| `INFVINO_DW_PAD=1` | padded depthwise 候选 | 需 retune |
| `INFVINO_NO_FUSE_GENERAL=1` | 通用激活融合 | **plan 期**：须重生成 plan |

---

## 3. 指标

### 3.0 先做 roofline 判断，再读 ops/EU/cyc（重要前提）

`ops/EU/cyc` 只在**算术强度足够高、且数据通路以 packed FMA 为主**时才是有效标尺。
本机（Iris Xe / 单通道 LPDDR / 3.75 MB LLC）大量算子并不满足：

- `depthwise` / 小空间或小通道 `conv`：受**地址/边界指令数**约束（指令配额远低于 32）；
- `ew_*` / `copy_c` / `concat` / `slice` / `maxpool` / `softmax` / `resize` / `gap`：
  算术强度仅 ~0.25–0.5 FLOP/byte，正确上限是**内存 roofline**，再叠加**每 dispatch 的
  launch 地板**；
- `M=1` 的 GEMV / `bmm`：受**归约延迟 / 网格占用**约束。

因此，某个 kernel 的 `ops/EU/cyc` 很低，**不等于**「有同比例的可用时间可挖」，它可能是：

1. 该算子的**物理上限本来就低**（内存/指令/延迟受限）；
2. 它只是**更大模式的一部分**——可被生产者/消费者**融合**（激活/残差 epilogue、
   concat→conv、`gap`+分类头、逐元素折进 conv），此时单看它没有意义。

**纪律**：

- **不要**用单 kernel 的低 `ratio` 直接推断 headroom，更**不要**把某几个数硬编码成
  「硬件极限」（会导致系统性误判）；
- **要**把**端到端 + 具体模型**放在一起看：该层是否值得优化、能否融合；
- `ops/EU/cyc` 与 `ratio` 只用于**同一 family 内排序**；`headroom_ms` 是**相对量**，
  会被中间标准（`expected_ops`，本身是模型而非实测）的误差放大；
- 计算类（`conv`/`gemm`）的标尺是硬件极限；**非计算类**请改用 roofline / 指令配额 /
  launch 地板，并优先考虑融合与 dispatch 数。

- `measured_ops = FLOPs / (ms·1e-3) / (EU·clk)`：实测 `ops/EU/cyc`。
- `expected`：`config/tuning.json` 的中间标准（`TuningEntry.expected`）。
- `ratio = measured / expected`（1.0 = 达到这台机器的期望）。
- `headroom_ms = Σ_nodes ms·(1 − ratio)`（ratio<1 的节点）：**若都达到中间标准可省的 ms**。
  这是「离硬件极限还有多远」的直接量化，也是优化优先级排序的依据。

---

## 4. 参考实测（2026-10，Iris Xe 80EU / 1.3GHz，1080p 输入）

### 4.1 预算瀑布（`analyze_budget.py`，exact-node）

| 模型 | e2e | pre | net | busy | net−busy | post | headroom |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8n-pose | 19.31 | 2.09 | 17.13 | 13.44 | 3.68 | 0.015 | **5.60 ms** |
| yolo11n-pose | 20.72 | 2.19 | 18.46 | 14.62 | 3.84 | 0.015 | **5.72 ms** |
| mobilenetv3-small | 3.86 | 0.35 | 3.47 | 2.69 | 0.78 | 0.017 | **1.53 ms** |

yolov8 busy 归因：`conv3x3 8.92 ms(66%)` / `conv1x1 3.37 ms(25%)` / 小算子 1.16 ms(9%)；
headroom 也集中在 conv3x3（4.70 ms）。mobilenet headroom 主要在 conv1x1（1.06 ms，多为
N=1 的 GEMV 头层）。

### 4.2 优化消融矩阵（`profile_ablation.py`，Δ = 关闭 − baseline）

| 开关 | y8 Δe2e | y11 Δe2e | mb Δe2e | 判读 |
|---|---:|---:|---:|---|
| `TUNING=off` | +3.62 (+18.6%) | +4.81 (+23.4%) | +0.30 (+7.9%) | **自动调优是最大单项收益** |
| `NO_POOL=1` | +0.98 (+5.0%) | +1.53 (+7.4%) | +0.18 (+4.8%) | 内存池收益显著（且墙钟 > busy） |
| `NO_BLOCK_LAYOUT=1` | +0.11 (+0.6%) | +0.28 (+1.4%) | +0.00 | R36 布局：yolo 小收益，mb 不适用 |
| `NO_REORDER_DEDUP=1` | −0.07 | +0.20 | −0.01 | 同帧去重：在噪声内 |
| `NO_LAUNCH_CACHE=1` | −0.04 | +0.20 | +0.06 | P2 dispatch 缓存：小、mb 略明显 |
| `NO_FUSE_RES=1` | −0.11 | −0.11 | −0.01 | 残差融合：本代模型上在噪声内 |
| `DW_PAD=1` | −0.07 | +0.20 | +0.02 | 需 retune 才有意义，当前≈0 |

> **噪声**：本机单次 A/B 的 e2e 抖动约 ±0.1–0.2 ms，小于此的差值不应下结论。
> 结论与 `docs/openvino-gap-analysis.md` / README 里的 R 记录一致：**最大杠杆是自动调优
> 与内存池；布局/launch 缓存是小项；残差融合在本代三个模型上收益不显著。**

---

## 5. 已知口径问题（务必写进任何对外结论）

1. **profiling 改变 host 行为**：`busy` 干净，但 `sync/enqueue` 只在 profiling 下有意义；
   `net/e2e` 一律用非 profiling 的 `infvino_bench`。
2. **`headroom` 依赖中间标准的正确性**：`expected_ops` 是模型（非实测），会随 ISA 认知更新；
   ratio 只用于**排序与相对比较**，不代表绝对值。
3. **内容相关项**：`pre/net` 与图像内容无关；`post` 随检出数变化。固定输入或声明。
4. **plan 期融合**（`NO_FUSE_GENERAL`）无法用运行时开关消融，须重生成 plan 再比。
5. **节点对齐**：profile-json 的 `index` 是 **fusion 之后**的节点序，不要直接和 `model.plan`
   的行号对齐；用 `signature` 与 tuning 对齐。

---

## 6. 文件清单

| 文件 | 作用 |
|---|---|
| `kernel_run --profile-json` | L0：逐节点 GPU 时间 + 结构计数 + 内存 + host 分段（机器可读） |
| `scripts/analyze_budget.py` | L1/L2：预算瀑布 + busy 归因 + 中间标准记分卡（profile-json 或文本回退） |
| `scripts/profile_ablation.py` | L2：优化消融矩阵（busy/net/e2e 三态的 Δ 归因） |
| `scripts/analyze_cache.py` | L1/L2：非 compute-bound 小算子的**缓存命中/带宽**分析（工作集档位 + launch/内存拆分 + eff） |
| `config/tuning.json` | 中间标准 `expected_ops` 与实测 `ratio` 的数据源 |
| `docs/benchmark.md` | 数值/性能基准总览（与本文互补） |
| `docs/openvino-gap-analysis.md` | （历史）与 OV 的差距分析，仅作研发参照，非本框架标尺 |
