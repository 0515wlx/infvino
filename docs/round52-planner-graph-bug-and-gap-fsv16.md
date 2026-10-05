# R52：Planner 图级 bug 定位与修复 —— `gap_fsv16` 契约 + 池/布局一致性探针

> 起点：R51 §5.1 的「图级正确性 bug」——D4+ 让 `gap` 直读 fsv16（联动 D5）后，mincut /
> 旧缓存路径的 `model_check` FAIL（mean_rel 0.507），三个 kernel 单元测试各自 PASS、
> 输入也正确，**问题只在组合进图后出现**，R51 未定位、按安全优先回退。
>
> 本文用**池/布局一致性探针**（`INFVINO_POOL_ALIAS_PROBE`）定位并修复根因，重新落地
> `gap_fsv16` 契约（补契约覆盖面），并给出数值 + 外部稳态 A/B 验证。

---

## 0. 结论先给

| 项 | 结果 |
|---|---|
| **根因** | 分配补齐集合（`mayBeFsv16` → `opCanReadFsv16`）**按 `family.supports` 过滤**；`gap_fsv16` 曾把 `supports` 设为**恒 false**（意图「永不产生候选」），导致含 gap 消费的 value 张量**不按补齐通道分配**；但 `planBlockedLayout`/mincut 的标记走 `nodeFamily`（**不看 supports**）仍把它标成 fsv16 → 生产者直写/消费者直读 `b_fs_yx_fsv16` **越界**，静默污染激活。 |
| **修复** | ① `gap_fsv16.supports` 改为真实 op 判据 `s.op=="gap"`（无候选由 `candidates==nullptr` 保证，与 supports 无关）；② 加**硬守卫** `mayMarkFsv16`：缓冲放不下补齐布局就拒绝标记；③ 加**探针** `poolAliasProbe`（容量 + 生产者直写能力双重校验）。 |
| **契约覆盖面** | 重新落地 `gap_fsv16` 布局契约族（`inIndex=0`、无候选），`gap`/`gap_r` 在输入持久 fsv16 时直读（`-DGAP_IN_FSV16`），mincut/不动点均能看到它。 |
| **数值** | 三模型 `model_check`：mb 默认 1.069e-2 / mincut 1.215e-2、**mincut+D5 1.203e-2（此前 0.507 FAIL）**、mb 默认+D5 1.069e-2、y8 4.114e-4、y11 7.908e-4，**全 PASS**；跨推理 `reuse_check` diff=0。 |
| **A/B** | mb `INFVINO_FUSE_SCALE`（D5）：默认 **−2.9%（7/8）**、mincut **−6.9%（8/8）**（交错、busy/locked）。 |
| **回归** | `tuning_test` PASS（新增 R52 契约族自检）；y8/y11 逐位不变、噪声内。 |

---

## 1. 根因：两套判据「标记 vs 分配」漂移

`gap_fsv16` 在 R51 的被回退实现里：

```cpp
f.supports = [](const OpSignature &) { return false; };   // 意图：永不产生候选
```

但 `supports` 在系统里不只决定候选枚举，还被**分配补齐集合**当作 op 归属判据：

- `PlanModel::allocateActivations`：
  ```
  mayBeFsv16(t) = 4-D && 非输出 && producer.canOutFsv16 && 每个 consumer 在激活槽能读 fsv16
  consumer 能读 fsv16 ⇔ ∃ family: layout.in==FSV16 && inIndex==slot
                              && (supports==nullptr || supports(sig))
  ```
  `gap_fsv16.supports` 恒 false ⇒ 含 gap 的 value 被判「不能持久 fsv16」⇒ **不按
  `ceil(C/16)*16` 补齐分配**（例如 C=120 只分 120×H×W）。

- `PlanModel::planBlockedLayout`（标记）：
  ```
  consumer 可读 fsv16 ⇔ nodeFamily(consumer)!=nullptr && nodeFamily(...)->layout.in==FSV16
                         && slot==inIndex
  nodeFamily(gap) = gap_fsv16   // 不看 supports
  ```
  ⇒ value 被**标记** fsv16。

于是出现「**标记 fsv16 但缓冲未补齐**」：C=120 时生产者按 `[C/16][H][W][16]` 写
`128×H×W` 个 half，而缓冲只有 `120×H×W` —— 越界写/读，激活被污染。R51 逐层快照里
「深度输出 `block.1.2` 与 numpy 不符、而三个 kernel 单元测试各自 PASS」正与此一致
（单元测试用的是**已补齐**的独立缓冲，掩盖了整网上分配与标记的漂移）。

> 教训与 R48/R49 一致：**契约必须单一真相源**。「无候选」应该由 `candidates==nullptr`
> 表达，**不能**借用 `supports=false` —— 后者是「本 op 是否属于本族」的判据，会被布局/
> 分配路径复用。

---

## 2. 修复

### 2.1 契约族判据修正（`src/KernelFamilies.cpp`）

```cpp
f.name = "gap_fsv16";
f.layout = {Layout::FSV16, Layout::NCHW, /*canOutFsv16=*/false, false};
f.layout.inIndex = 0;
f.supports = [](const OpSignature & s) { return s.op == "gap"; };  // 真实 op 判据
// 不设置 f.candidates ⇒ 永不产生候选（与 supports 解耦）
```

### 2.2 硬守卫 `mayMarkFsv16`（`src/PlanModel.cpp`）

`Tensor` 新增 `cap_elems`（底层缓冲实际元素数，parse/pool-arena/sub-buffer 都记录）。
标记前校验：

```cpp
if (!mayMarkFsv16(t)) continue;   // cap_elems < ceil(C/16)*16*H*W ⇒ 拒绝标记
```

`planBlockedLayout` 两处标记、`resolveLayoutMinCut` 落地前都过守卫（mincut 不满足则整体
回退 baseline）。这样即使**将来**再出现判据漂移，也只会「少一个优化」而不会**静默越界**。

### 2.3 池/布局一致性探针 `poolAliasProbe`

`INFVINO_POOL_ALIAS_PROBE=1` 打印详情；**任何违例都无条件打印**。校验：

1. 被标 fsv16 的张量 `cap_elems ≥ ceil(C/16)*16*H*W`；
2. 生产者**当前选中的 kernel**确实声明 `canOutFsv16`（防止「标记 fsv16 但生产者写 NCHW」）。

在 `resolveLayoutChoices` 末尾调用（任何布局落地后立即自检）。这正是定位本 bug 用的探针：
把「能力集合 vs 标记判据」的漂移变成一条显式 FAIL，而不是等到整网数值错乱。

---

## 3. 验证

### 3.1 探针

修复前（`supports=false` + 无守卫时的等价场景）：value 被标记但缓冲不足 →
`[pool-probe] BUG: ... marked fsv16 but buffer too small`。修复后：

```
[pool-probe] ok: /features/features.7/block/block.1/block.1.2/HardSwish_output_0 cap=25088 need=25088
[pool-probe] resolveLayoutChoices PASS: 0 violation(s)
```

（C=120 → `ceil(120/16)*16=128`，`128×14×14=25088`。）

### 3.2 数值（`model_check` 口径；fp16 输入 / onnxruntime CPU fp32 参考）

| 模型 | 路径 | mean_rel | 结果 |
|---|---|---|---|
| mobilenetv3-small | 默认 | 1.069e-2 | PASS |
| mobilenetv3-small | mincut | 1.215e-2 | PASS |
| mobilenetv3-small | 默认 + D5(`FUSE_SCALE`) | 1.069e-2 | PASS |
| mobilenetv3-small | **mincut + D5** | **1.203e-2** | **PASS（R51 曾 0.507 FAIL）** |
| yolov8n-pose | 默认 / mincut / +D5 | 4.114e-4 | PASS |
| yolo11n-pose | 默认 / mincut / +D5 | 7.908e-4 | PASS |

跨推理一致性（喂 A 再喂 B 的输出 vs 全新进程只喂 B）：mb（默认/+D5）、y11 **diff=0**。

### 3.3 外部稳态 A/B（同 binary，`INFVINO_FUSE_SCALE` 开/关，交错 ×8，`--iters 5` busy）

| 路径 | off（median） | on（median） | Δ | 胜场 |
|---|---:|---:|---:|---:|
| mobilenetv3-small 默认 | 1.630 | 1.583 | **−2.9%** | 7/8 |
| mobilenetv3-small mincut | 1.438 | 1.339 | **−6.9%** | 8/8 |

与 R51 §5.1 的预测（默认 −2.0%、mincut −6.9%）一致；此前「中性/FAIL」纯粹是
`gap_fsv16` 未接通 + 图级 bug。

### 3.4 回归门

| 检查 | 结果 |
|---|---|
| `tuning_test` | PASS（新增 R52 契约族自检：`gap_fsv16` 声明 / 无候选 / `supports` 可被分配补齐集合发现） |
| y8/y11 默认与 mincut | 与 R50/R51 逐位相同（4.114e-4 / 7.908e-4） |
| `config/tuning.json` | **未改动**（D5 仍 opt-in；无缓存重扫） |

---

## 4. 复现

```bash
# 构建
docker run --rm --memory=4g -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc 'cmake --build build-blk -j4 \
  --target kernel_run tuning_test kernel_numtest'

# 探针（任何布局落地后自检；verbose 打印每条）
docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 --device=/dev/dri/renderD128 \
  -v "$PWD":/workspace/infvino -w /workspace/infvino \
  -e INFVINO_FUSE_SCALE=1 -e INFVINO_POOL_ALIAS_PROBE=1 infvino-dev:latest bash -lc \
  'export LD_LIBRARY_PATH=$PWD/build-blk; \
   ./build-blk/kernel_run --plan models/mobilenetv3-small/model.plan --iters 1 2>&1 | grep pool-probe'

# 数值（含此前 FAIL 的组合）
INFVINO_FUSE_SCALE=1 INFVINO_LAYOUT_MINCUT=1 python3 scripts/model_check.py \
  --model mobilenetv3-small --repo "$PWD"
```

---

## 5. 下一步（回到 R48 §4bis 排序）

1. **D5 转正**：现在数值/收益都已确认（mb −2.9%/−6.9%）；转正需把
   `--op depthwise --retune` 的 `#blkfsv16` 并入 `config/tuning.json`（否则布局对
   「输出 fsv16」定价偏保守），并做一次正式的锁频外部 A/B。
2. **P1（高风险）conv3x3 的布局链**：`conv3x3_blk` 已声明 `canOutFsv16` 且
   `conv_blk.cl` 支持 `-DOUT_FSV16`，但缺 `#blkfsv16` 成本测量，且 R49 判其结构性弱于
   `conv3x3_ov`、暂未纳入 mincut。属「扩充算子族」的高风险项。
3. **P3/P4**：direct conv1x1（窄通道/小 N）、conv1x1 N=1 split-K、gemm staging、
   depthwise Memory/Launch 轴。
4. **契约机制加固**（本次已起步）：`poolAliasProbe` + `mayMarkFsv16` 已把「分配补齐集合
   ⊇ 标记集合」变成可自检的不变量；后续新增布局契约族应遵循「`supports` = 真实 op 判据、
   无候选用 `candidates=nullptr`」。
