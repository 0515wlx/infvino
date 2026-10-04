# Round 43：硬上限 vs 告警 · MD API 接入 · 联合布局不动点推广

> 承接 R42 的评审意见，按「噪声 → 依赖/镜像 → 标尺语义 → MD API → 联合布局」顺序落地。
> 本文只记 R42 之后的增量；标尺的**语义修订**已同步进 `docs/round42-...md` §1。

---

## 1. 噪声（步骤 1/3）

- **根因确认**：LLC→DRAM 带宽断崖。copy 带宽在 footprint 4–8 MB 处从 ~90→~50→~20 GB/s，
  且**离散度在断崖处最大（~25–30%）**，深 DRAM（≥8 MB）反而 <5%。环境已排除（无其它
  容器 / load 0.18 / 温度 28–33°C / turbo on）。详见 `docs/round42-...md` §3.1。
- **落地**：
  - `scripts/gpu_clocks.sh lock|unlock|status`：锁 `gt_min/max_freq=RP0`（无免密 sudo 也能用，
    经 sysfs rw bind-mount 进容器写，已验证）。
  - `Autotuner::benchCandidate`：**median → min**（干扰只加时间），warmup 3→5；
    新增 `spread=p90/min−1` 输出；对胜出候选做**二次确认**（iters×3），win 者内部
    不一致即 `WARN`。
  - `kernel_bench` 结果行新增 `[min p90 spread]`，便于「同条件不一致立刻排查」。
  - `scripts/noise_check.sh`：连跑一个 4.2 MB footprint 的 stride-2 conv N 次，比较各次 min，
    spread≥10% 即 FAIL。实测 PASS（1.5%）。

## 2. 依赖 / dev 镜像（步骤 2）

`docker/Dockerfile` 的 Intel 源新增：`intel-metrics-discovery intel-metrics-discovery-dev
intel-gpu-tools libdrm-dev`。镜像已重建并**验证**：

| 工具 | 状态 |
|---|---|
| `intel-metrics-discovery` 1.12 | ✅ 头文件 `metrics_discovery_api.h` + `libigdmd.so.1`；`OpenMetricsDevice rc=0` |
| `intel_gpu_top`（IGT 1.26） | ⚠️ sandbox 里 `Failed to initialize PMU`（需 perf 权限，协议禁 `--privileged`） |
| `intel_gpu_frequency` | ⚠️ sandbox 里找不到 GPU chipset；改用 `gpu_clocks.sh`（sysfs） |
| `unitrace` | ❌ 无 apt 包，且构建宿主**无法访问 GitHub**；记录为「有网时可加」 |

> 结论：**MD API 是 sandbox 可用的那一个**；`intel_gpu_top`/`intel_gpu_frequency` 受限。

## 3. 标尺语义修订（步骤 3）

评审意见成立：**经验 derate 不能当硬上限**。已重构为三层（`docs/round42-...md` §1）：

| 层 | 定义 | 代码 |
|---|---|---|
| `hard_ceiling` | 只放 ISA 发射配额 `32·mad_frac` / roofline 下界（不可越） | `KernelFamily.hardCeiling` |
| `expected`（软） | 含 amort/gridFactor/占用 derate，仅排序 | `KernelFamily.ceiling` |
| `alarms` | 延迟/L3 复用/footprint/lane/网格 等**证据** | `kernel_diag`（R42 §2 设计） |

- `TuningEntry` 增 `hard_ceiling` / `hard_ratio`（**向后兼容**：旧缓存回退软值）；
  `kernel_autotune --expected` 同时打印 soft/hard；`refresh-expected` 一并重算。
- `hard_ratio = measured / hard_ceiling` 是「离物理极限」的唯一标尺；软 `ratio` 只做同族排序。
- 已实测写入：ov `hard_ceiling=20.3`（与 `kConvOvIssueCeiling` 一致），soft 随 shape 8–20。

## 4. MD API 工具（步骤 4）

新增 `src/tools/gpu_metrics.cpp`（CMake `find_library(igdmd)`，缺失则跳过）：

- `--list`：枚举 4 个 concurrent group / 27 个 OA metric set，暴露
  `EuActive` / `EuStall` / `EuThreadOccupancy` / `SlmBytesRead/Written` /
  `ShaderMemoryAccesses` / `L3ShaderThroughput` / L3 bank 等**正是判墙需要的计数器**。
- `--sample <Set> <ms>`：打开 IO stream 读原始 OA report（**已验证 OpenIoStream 可用**，
  但 `Activate` 在 sandbox 返回失败——OA 是受限资源；故当前只 dump 原始 report，
  解码（metric→equation ReadParams.ByteOffset）列为下一步）。
- 用途：与 `kernel_bench -DPROBE` 剖面**交叉验证**「哪堵墙」，把 R41/R42 的
  「推测」升级为硬件证据。

> 说明：`Activate` 失败与 `intel_gpu_top` 的 PMU 失败同源——sandbox 对硬件计数器设了限。
> 需要在放宽权限的环境（或宿主机上以 root 调 `perf_event_paranoid`）才能采到 OA。

### 4.1 深挖：OA 在本机**不可用**（kernel 配置）

- 用 `--cap-add=PERFMON --cap-add=SYS_ADMIN` + 把 `perf_event_paranoid` 降到 -1 后，
  `Activate` **成功**，但 `OpenIoStream` 返回 **rc=4 = `CC_CONCURRENT_GROUP_LOCKED`**。
- 根因：本机内核 **`CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS is not set`**（`/boot/config-$(uname -r)`
  实测），i915 不提供 OA/perf 通道 → OA concurrent group 不可用。
- 结论：**`gpu_metrics` 只能在能枚举、不能流式采样的状态**（除非重编内核）。
  因此 in-house `-DPROBE` 剖面仍是判墙主路径；MD API 作为「有合适内核时的证据补充」。
- 已把解码补全：`IMetricSet::CalculateMetrics(rawData,…)` + `TTypedValue_1_0`（按
  `ValueType` 打印 uint32/uint64/float/bool），连同 raw hex 兜底。在有 OA 的机器上即可直接用。

## 4b. kernel_diag（步骤 2）

新增 `scripts/kernel_diag.sh <Cin,Cout,H,W> [STRIDE OBW OBH SLM ACT]`：对 `conv3x3_ov`
一次跑 `PROBE=0/1/2/3/4`，输出墙剖面 + `hard_ratio` + 判决。实测两例：

```
$ scripts/kernel_diag.sh 64,64,160,160 2 6 2 4        # stride2 热点
  full 11.1  noInput 15.8  noWeight 12.0  noBoth 19.9  noStore 11.2
  hard_ratio = 0.55   verdict: INPUT-FEED latency bound (+79%), PF ineffective -> raise occupancy

$ scripts/kernel_diag.sh 16,8,160,160 1 8 2 1         # Cout=8
  full 3.4  noInput 4.4  noWeight 4.2  noBoth 5.4  noStore 3.7
  hard_ratio = 0.17   verdict: STRUCTURAL (lane waste / fixed overhead); feeds NOT the wall
```

这把 R41 的人工归因固化成一命令，并与 `hard_ratio`（物理距离）联动。

## 5. 三模型 retune（步骤 3）——系统实测

**先记联合布局推广（实现，承接 R42 §4）**：R38 的 (族,布局) 不动点原来只覆盖 conv3x3。
已推广到 conv1x1/depthwise：

1. autotune 对 `conv1x1_blk`/`depthwise_blk` 也写 `#blk`/`#non`/`#reorder` 分离条目，
   基础条目只存 kernel-only ms；
2. `resolveLayoutChoices` 去掉 `op=="conv3x3"` 过滤；
3. `nodeFamily`/`run()` 对 conv1x1/depthwise 也走 `choiceEntry`（此前直接 `tuning_.lookup`，绕过不动点）；
4. 修 conv1x1 输入下标（weights-first：`ins[1]` 才是激活）。

seeded 246 个 conv3x3/conv1x1/depthwise 签名，按批 retune（batch2/iters6），耗时 ~3 min，
无 HANG。结果：

- **`#blk/#non/#reorder` 已为 conv1x1(42)/conv3x3(31)/depthwise(15) 生成**（共 88 组），
  联合 (族,布局) 不动点首次在 conv3x1/depthwise 上被真正启用。tuning.json 342→513 条。
- 三模型 `model_check` **PASS**（y8 4.1e-4 / y11 7.9e-4 / mb 1.1e-2）、`reuse_check` **PASS**。

**busy（kernel_run --report --iters 20，锁频，3 rep 中位）**：

| 模型 | R39/R42 baseline | R43 retune | Δ |
|---|---:|---:|---:|
| yolov8n-pose | 10.643 | **10.546** | −0.9% |
| yolo11n-pose | 11.716 | **11.569** | −1.3% |
| mobilenetv3-small | 1.644 | 1.671 | **+1.6%** |

### 5.1 关键发现：隔离测量 ≠ 流水线表现（mb 回退的根因）

逐节点 profile 显示 mb 唯一回退在 `conv1x1`（+0.025 ms），且**9 个改动的 conv1x1 配置在
隔离 bench 上全部更快**（min 估计器，例如 `Cout576_N49_Cin96` 0.0352→0.0284、
`Cout64_N400_Cin64` 0.0264→0.0180），但 **in-pipeline 更慢**（`Cout576_N49_Cin96` 0.0916→0.0994）。
其中典型：`XB4→XB2`（更多 WG/更少复用）、`SLM1→SLM4`（更多在飞 → L3 压力更大）。

即：**单 kernel 隔离 bench（冷/空 cache）与它在真实流水线中的表现存在系统性偏差**——
autotune 目前按隔离 min 选，会把「隔离快、流水慢」的配置选进来。这是本轮最有价值的发现，
也解释了 R37 里「reorder 税 / 布局计费」之外的另一类选择偏差。

> 这不是 joint-layout 的 bug（layout report 显示 fsv16/ reorder / dispatch 与 baseline 一致）；
> 是**候选选择的目标函数**问题：应在尽量接近真实驻留状态（相邻 kernel 之后）下测，或对
> top-K 做「整网 busy」回验。

### 5.2 处置

- y8/y11 的 conv3x3/stem 改动是净正（stem `f16→cin3` 0.2249→0.2053 等），保留。
- mb 的 conv1x1 回退 **< 噪声阈值区间边缘**（+1.6% busy）。由于 sig-cache 跨模型共享，
  无法只回退 mb 的这几个签名而不动 y8/y11；**本轮以「验证系统」为目的保留 retune 结果**，
  并在下一步用「流水线感知 / 整网回验」重选（见 §7）。若需保守，回退一条命令即可
  （`git checkout config/tuning.json`）。

---

## 6. 验收 / 复现

```bash
scripts/gpu_clocks.sh lock
scripts/noise_check.sh 6                 # 期望 PASS（spread<10%）

# MD API
docker run --rm --memory=2g --device=/dev/dri/renderD128 \
  -v "$PWD":/workspace/infvino -w /workspace/infvino infvino-dev:latest \
  bash -lc './build/gpu_metrics --list | grep -E "EuActive|EuStall|L3|Slm|Dram"'

# 硬/软标尺
./build/kernel_autotune --plan models/yolov8n-pose/model.plan \
  --cache config/tuning.json --refresh-expected | tail
#   -> 表头 soft expected/ratio + hardC/hardR

# 回归
python3 scripts/model_check.py  --model yolov8n-pose --repo $PWD --image infvino-dev:latest
python3 scripts/reuse_check.py --model yolov8n-pose --repo $PWD --image infvino-dev:latest

scripts/gpu_clocks.sh unlock
```

## 7. 下一步

1. **流水线感知的候选选择**（本轮最重要的 open item）：现在 autotune 按「隔离 min」选，
   会选中「隔离快、整网慢」的配置（§5.1）。方向：把每个 shape 的 top-2/3 候选放到
   **真实相邻 kernel 之后**复测，或用 `kernel_run` 做整网 busy 回验；至少对 mb 的
   conv1x1 热签名这样跑一遍。
2. `gpu_metrics` 完整解码在有 OA 的内核上验证（本机需 `CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS=y`）。
3. 把 `kernel_diag` 接进 `autotune --report`（自动打印 hard_ratio + 墙判决）。
