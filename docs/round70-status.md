# R70：统一收尾 —— 开放项登记 + `--global` 硬信号 + 逐节点 spill 归因 + 残差落盘 + 三模型回归

> 用户决策：本轮不再新开研究线，而是把 R1–R69 累积的**遗留项统一收尾**（范围 A+B+C：
> 离线代码收尾 + 文档一致化 + GPU 回归）。结论先给：
>
> * **落地 3 项离线收尾**：逐节点 spill 归因暴露、`--global` 空操作硬信号、整网回验残差
>   数据集落盘；新增 [`docs/open-items.md`](open-items.md) 作为**开放项的单一权威清单**。
> * **三模型 GPU 回归全绿**：`model_check`/`reuse_check` 全 PASS，`mean_rel` 与 R68 逐位一致。
> * **C4「全量 retune 以激活布局 fixpoint」做了决定性实验**：mb（唯一布局耦合型模型）retune
>   后确实生成了 `#blk/#non/#reorder/#blkfsv16`（278 条），但整网 A/B **无可靠收益**
>   （base 中位 ≈1.378 ms vs retuned ≈1.390 ms）→ **生产 `config/tuning.json` 保持不动**，
>   与 R46/R47 的结论一致。
> * **有意未做**：`KernelFamily::launch` 声明化（数百行、跨三模型数值风险的重构）——已列入
>   `open-items.md` A2 并给出范围/前置，不在收尾轮硬做。

---

## 1. 本轮改动

### 1.1 逐节点 spill 归因暴露（open-items A1，R69 §6.3 遗留）

`L3Model` 早就能算**精确逐出归因** `L3Result::node_evict_ms`（「某节点把张量逐出、之后
真 miss」的字节折算 ms），但只在离线测试可见。本轮把它接到生产报告路径：

```bash
INFVINO_LAYOUT_REPORT=1 INFVINO_LAYOUT_SPILL=1 ./build-ct/kernel_run \
  --plan models/mobilenetv3-small/model.plan --report --iters 1
```

输出 `[layout] spill by node: total ...` + top-N（节点名 / op / ms / 占 spill 百分比）。
与 `layoutModelBreakdown()` **同源**（同一 `buildL3Access` + 同一默认 L3 配置），
**不参与选择**，打开关闭不影响任何数值/选择。

mb 实测：`spill=0.2318 ms`，但**仅 1 个节点有跨算子逐出归因**（`logits`，0.0001 ms）——
即 mb 的 spill **几乎全是冷启动 compulsory miss**，不是算子间干扰。这直接解释了
R69 §6「为什么当前三模型里新增内存受限候选缺乏证据」：**没有可归因的 spill 大户**。

### 1.2 `--global` 空操作硬信号（open-items C1，R44 #12 / R45 §67 遗留）

此前 `globalRetune` 未执行（无 profiling / 无短名单 / 短名单与本图不匹配）时返回 `0`，
与「执行了但没有净收益」无法区分，CI 只能 grep stderr。本轮：

* `PlanModel::globalRetune` 未执行时返回 **`-1`**（文档同步到 `PlanModel.hpp`）；
* `kernel_autotune --global` 打印 `global-retune: NO-OP` 并**退出码 3**
  （`INFVINO_GLOBAL_ALLOW_NOOP=1` 可容忍）；
* `scripts/autotune.py` 显式检测（`exit 4`），且**在「0 条目 → op 扫完」自然收尾之前**
  不误判（那批 `done` 为空、不算空操作）。

验证（已有 tuned 缓存、不 `--retune`）：

```text
rc=3
[global-retune] NO-OP: no candidate shortlist (...)
[kernel_autotune] refusing to report success for a NO-OP --global
```

且缓存 sha256 不变（空操作不写盘）。

### 1.3 整网回验残差数据集落盘（open-items C2，R47-tvm §7.1 遗留）

每次候选评估记录 `(sig, kernel, iso_pred, net_measured, accepted)` 到 JSONL
（`--residual <f>` 或 `INFVINO_GLOBAL_RESIDUAL=<f>`），作为学习型残差模型（iso→net）的
训练集。验证输出：

```json
{"sig":"conv1x1|Cout8_N1_Cin16_act2_f16","kernel":"conv1x1_gemv_f16","iso_pred":1.77796,"net_measured":1.5421,"accepted":false}
```

（`iso_pred` = 可加目标预测 ms，`net_measured` = 整网 median ms；残差 = 两者差。）

### 1.4 文档一致化（open-items H1/H2）

* `README.md` 文档表补齐 **round49–68**（此前只有 …48、69）并加入 `open-items.md`；
* 状态清单去掉过时项（「内存复用 byte-offset 子分配」早已落地却仍列未勾）并按
  `open-items.md` 归类剩余开放项；
* 新增 [`docs/open-items.md`](open-items.md)：把 R1–R69 散落的「下一步/待办/缺口」收进
  分类登记表（🔴未做 / 🟡暂缓 / ⚪负结果 / ✅已完成），**含每项的解除条件**。

---

## 2. 三模型 GPU 回归（范围 C 的数值/复用部分）

锁频（`gpu_clocks.sh lock`）与安全容器（`--memory`/`--pids-limit`/`--device=renderD128`/
逐命令 `timeout`），全部 `gpu_guard after` 无 HANG。

| 门 | y8 | y11 | mb |
|---|---|---|---|
| `model_check` | PASS `mean_rel=4.114e-04` | PASS `7.908e-04` | PASS `1.215e-02` |
| `reuse_check` | PASS | PASS | PASS |

`mean_rel` 与 R68/R69 基线**逐位相同** → 本轮改动**只增诊断/信号/日志，不改数值与选择**。

---

## 3. C4 实验：全量 retune 能不能「激活布局 fixpoint」？

### 3.1 现状确认

生产 `config/tuning.json`：**513 条全为 base，零 `#blk/#non/#reorder`**。因此
`resolveLayoutChoices` 的联合不动点在生产路径上**从不激活**（它要求 `#blk` 与 `#non`
同时存在——R47 §4.3 记录的残留）。

### 3.2 为什么不能用普通 `--retune` 分批

R46 §3.1 已知：`--retune` 每批从第一个签名重来（`sig_seen` 进程内），批次不推进。正确
路径是文档规定的 `--global`（`INFVINO_GLOBAL_PROGRESS=1` 用 per-plan 工件承载进度）。

### 3.3 实测（mobilenet，布局耦合型；ops = conv1x1,depthwise；锁频）

retune 确实生成 fixpoint 键：temp 缓存含 **278 条 `#` 键**（`#blk`/`#non`/`#reorder`/
`#blkfsv16`）。交错 A/B（`INFVINO_PLAN_TUNING=none` 隔离共享缓存效应，3 rep，`--iters 15`）：

| rep | base 缓存 busy (ms) | retuned 缓存 busy (ms) |
|---|---:|---:|
| 1 | 1.393 | 1.390 |
| 2 | 1.365 | 1.384 |
| 3 | 1.378 | 1.428 |
| 中位 | **1.378** | **1.390**（+0.9%） |

结论：**全量 retune + fixpoint 激活对 mb 无可靠收益（中位略差）**，与 R46/R47「整网回验
暂无可靠正收益」一致。yolo 更无耦合（R47：fsv16≈0、reorder 0.6–1.8%），无需重复。
→ **生产缓存保持不动**；C4 记为 ⚪（负结果，等待新证据才重开）。

---

## 4. 有意未做（诚实边界）

| 项 | 原因 |
|---|---|
| `KernelFamily::launch` 声明化（A2） | `PlanModel::run`/`makeEnqueue` 的**参数绑定按族高度特化**（缓冲槽、gws/lws 几何、options 回放），声明化是数百行重构，且触碰三模型 dispatch 数值路径。R48 §7.3-D 自己也把它列为设计缺口。**不在收尾轮硬做**；已在 `open-items.md` A2 记录范围与前置。 |
| 完整 blocked chain / conv3x3 布局链（B1/B3） | 高风险 kernel/布局重构，前置未满足（B1）；同 `open-items.md`。 |
| 逐节点 L3 定价 / 容量项（B4） | 三次负结果，保持关闭。 |

---

## 5. 复现

```bash
# 离线门（本机 host 无 cmake，经容器）
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake --build build-ct -j8 --target check'   # 6/6 PASS

# 逐节点 spill 归因（短跑、单配置）
docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc '
    export INFVINO_LAYOUT_REPORT=1 INFVINO_LAYOUT_SPILL=1
    timeout 60 ./build-ct/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 1'

# --global NO-OP 硬信号（已有 tuned 缓存、不 --retune → 应 rc=3）
docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc '
    timeout 90 ./build-ct/kernel_autotune --plan models/mobilenetv3-small/model.plan \
      --cache config/tuning.json --global --iters 1 --global-iters 1; echo rc=$?'

# 残差落盘（单签名）
docker run --rm --memory=3g --memory-swap=3g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino -v /tmp/opencode:/tmp/opencode \
  -w /workspace/infvino infvino-dev:latest bash -lc '
    ./build-ct/kernel_autotune --plan models/mobilenetv3-small/model.plan \
      --cache /tmp/opencode/t.json --op conv1x1 --limit 1 --iters 3 --retune \
      --global --global-limit 1 --residual /tmp/opencode/resid.jsonl && cat /tmp/opencode/resid.jsonl'

# 数值 / 复用回归（三模型）
for m in yolov8n-pose yolo11n-pose mobilenetv3-small; do
  python3 scripts/model_check.py --model $m --repo "$PWD" --image infvino-dev:latest
  python3 scripts/reuse_check.py --model $m --repo "$PWD" --image infvino-dev:latest
done
```

---

## 6. 改动文件

* `include/infvino/PlanModel.hpp`：`globalRetune` 返回值语义（`-1` = NO-OP）。
* `src/PlanModel.cpp`：逐节点 spill 归因报告（`INFVINO_LAYOUT_SPILL`）；`globalRetune`
  三处 NO-OP 返回 `-1`；残差数据集 `residualTap`。
* `src/tools/kernel_autotune.cpp`：`--global` NO-OP 退出码 3；`--residual`；区分
  「op 扫完的自然收尾」与「真空操作」。
* `scripts/autotune.py`：NO-OP 检测顺序（自然收尾优先）。
* `README.md`、`docs/open-items.md`（新增）、`docs/round70-status.md`（本文件）。
