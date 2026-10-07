# R71 实施：1×1 / depthwise / conv3×3 三项 + **blocked 链布局缺陷**（对 OV 差距的核心）

> 承接 `round71-ov-comparison.md`：逐层对照显示 OV 在 **1×1（1.5–2.4×）** 与 **depthwise（2–3×）**
> 上系统性更快，而 3×3 只快 7–11%。本文件记录按序实施这三项的结果，**重点是第 3 项——
> 一个靠逐层对照 OV 才暴露出来的 blocked 链布局缺陷**，它同时是 1×1 与 depthwise 落后的
> 共同根因。这是后续把系统做得比 OV 更好的关键参考。
>
> 口径：锁频 1300 MHz、安全容器、跑后 `gpu_guard` 无 HANG；`busy` 取 `kernel_run --report`
> 20 iter；数值门 `model_check`（vs onnxruntime）。

---

## 1. 1×1 候选谱（**已落地**：`e4f2a79`，mb busy −4.7%）

**根因**：`KernelFamilies.cpp` 的 `conv1x1_blk` 候选枚举有一个静默门槛

```cpp
if (slm > 1 && s.N < 64) continue;   // 小空间层的 split-K 候选被整类丢掉
```

而隔离实测它们最快：

| shape | 旧（被选中） | 新（split-K） | 加速 |
|---|---|---:|---:|
| `96→576 @7×7`（mb） | `XB2/slm1` 0.0433 ms | `XB4/slm4` **0.0211** | **2.05×** |
| `96→288 @7×7`（mb） | `XB2/slm1` 0.0235 | `XB4/slm4` **0.0136** | **1.73×** |
| `576→96 @7×7`（mb） | `XB2/slm1` 0.0284 | `XB4/slm4` 0.0235 | 1.21× |

**修复**：门槛放宽到 `s.N < 16`（只排除真正过小的层），并把 mb 的 `conv1x1`/`depthwise`
retune 后写回 `config/tuning.json`（只命中 mb 签名，yolo 不动）。

**结果**：mb busy **1.368 → 1.304 ms（−4.7%，3 rep 一致）**；`model_check`/`reuse_check`
PASS、`ctest` 6/6。y8/y11 中性（其 conv1x1 以 cat4 为主，见 §2）。
> 注：split-K 改变 fp16 K 累加顺序，mb `mean_rel` 1.215e-2 → 1.320e-2（仍 PASS，余量充足）。

---

## 2. cat4 → blocked 1×1（**部分落地**：管道就位，但净收益为 wash）

### 2.1 诊断
`conv1x1_cat4`（concat→1×1 融合）被路由到 **`gemm_f16`（tiled GEMM + CAT4 重定向）**：

| shape | 我们 cat4(gemm) | 我们 conv1x1_blk | OV `bfyx_f16_1x1` |
|---|---:|---:|---:|
| `384→256 @20×20`（y8） | 0.165 ms | **0.096** | 0.079 |
| 整网 | y8 gemm 2.96 ms / y11 gemm 3.86 ms（绝大多数是 cat4） | y11 blk 0.69 ms | OV 1×1 0.90/1.79 ms |

即：**我们已有更快的 blocked 1×1，但 cat4 融合把它路由到了 GEMM**。

### 2.2 已落地（默认不改变行为）
* `kernels/ops.cl`：`concat4` 新增 `OUT_FSV16`（`outer==1` 时直写 `b_fs_yx_fsv16`）。
* `KernelFamilies.cpp`：`small_concat4` 声明 `canOutFsv16=true`。
* `PlanModel`：`nodeFamily` 认识 `concat4`（生产者）、`smallKernelFor` 在输出被标记 fsv16 时
  加 `-DOUT_FSV16=1`；`INFVINO_NO_FUSE_CAT4=1` 可关闭 cat4 融合走 blocked 链。
* **链感知定价**（见 §3，本项的关键前提）。

### 2.3 结果：unfused 是 **wash**（不是胜）
| 模型 | fused（生产） | unfused（retune 后） |
|---|---:|---:|
| y8 | 10.50 ms | 10.53–10.60 |
| y11 | 11.34 | 11.40–11.47 |
| mb | 1.30 | 1.27–1.29 |

unfused y11 的 kernel 分解（链感知修复后）：`conv1x1_blk 0.69→1.26`、`gemm 3.86→2.61`，
**但 `concat4` 从 0.29 → 0.86 ms**（OV 的 `concatenation` 只 0.42）。**concat 物化的代价
吃掉了 GEMM→blk 的收益**。

### 2.4 真正的 1×1 赢法（下一步）
**融合的 blocked-cat4**：保留 cat4 融合（不物化 concat），让 `conv1x1_blk` 直接读 4 路
**fsv16** 源（通道块重定向，offset 需 16 对齐；不对齐则该源单独 reorder 或回退 gemm）。
预期 y8/y11 conv1x1 各 −1~2 ms（因为省掉 0.86 ms 的 concat）。需要：kernel 加 4 源 +
`opCanReadFsv16` 支持多输入槽 + dispatch 绑定。**这是 §3 布局缺陷的应用面。**

---

## 3. **blocked 链布局缺陷**（最重要；靠逐层对照 OV 才发现）

### 3.1 现象
OV 的 GPU kernel **全程跑在 `b_fs_yx_fsv16` 上**（y8 只有 3 个 `reorder_data_fast_b1`(f32)，
约 0.375 ms，且主要用于 f32→f16 边界）；`concatenation`、`bfyx_f16_1x1`、`bfyx_f16_depthwise`、
`bfyx_f16` 都在同一阻塞布局里直读直写，**零布局搬运**。

我们的图里，`resolveLayoutChoices` 的 (族,布局) 不动点会**卡在 NCHW 局部最优**：当某节点的
激活输入当前不是 fsv16 时，它按 `blk.ms + reorder.ms` 定价，于是选 `non`（NCHW），
**从不考虑「选 blk 会让生产者直写 fsv16，从而这趟 reorder 根本不必存在」**。结果：
* concat→1×1、跨族链（如 `conv1x1_blk → depthwise_blk`）在生产者当前不是 blk 时**断链**；
* 表现为每帧多趟 `reorder_bfyx_to_fsv16`，且 blocked kernel 用不上。

**证据（同一 unfused 图，只改这一处定价）**：

| | 链感知修复前 | 链感知修复后 |
|---|---:|---:|
| y11 fsv16 张量 | **3** | **21** |
| y11 `conv1x1_blk` | 0.69 ms | 1.26 ms |
| y11 `gemm_f16` | 3.86 ms | 2.61 ms |
| y11 `reorder` 调用 | 12（0.268 ms） | 10（0.175 ms） |

> 这正是「**没有 OV 就发现不了**」的地方：单看我们的 `ops/EU/cyc`，这些节点只是「离配额远」；
> 只有把 OV 的实际 kernel（`primitiveType` = `convolution_gpu_bfyx_f16_1x1` 等）与逐层时间
> join 起来，才看出差距的主因是**布局链断掉**而不是 kernel 算术本身。

### 3.2 已落地修复：链感知定价
`PlanModel::resolveLayoutChoices` 的坐标下降里，对 `blk` 的成本改为：

```
若「激活输入只有本节点一个消费者」且「其生产者能直写 fsv16」（conv1x1_blk/depthwise_blk/
concat4(outer==1)/ew_binary_ch）→ 认定选 blk 会令生产者被标记 fsv16 ⇒ reorder 计为 0
```

（`nodeCanOutFsv16` + `inUseCount==1`，见 `PlanModel.cpp` 的 R71 段）。对**当前三模型**的
**默认（fused）路径中性**（本就已链上的地方不受影响），但它让 unfused/cat4 链**能接上**
（fsv16 3→21）。这是后续所有布局链工作的地基。

### 3.3 仍未解决（后续研发参考）
1. **多消费者链**：本修复只覆盖单消费者。一个张量若被 blocked 与 NCHW 两类消费者共用，
   仍会被钉死 NCHW（需要 per-port `pref/firm/fusable` 与 reorder 的**分摊定价**，对应
   `open-items` A3）。
2. **生产者分组/整分量赋值**：应支持「一次给整条链赋 fsv16」的 move（对应 B1 完整 blocked chain）。
3. **concat 不物化**：§2.4 的融合 blocked-cat4。
4. **reorder 本身**：我们 concat 物化 0.86 vs OV 0.42；即必须让 concat/1×1 在同一 fsv16 布局
   内完成，而不是"算完再搬"。

> 一句话：**OV 的优势不是单个 kernel 更快，而是它的布局决策是"全局、跨算子、以 fsv16 为默认"，
> 我们的是"逐节点、保守、以 NCHW 为默认"**。把这个决策口径反过来，是比换 kernel 更根本的
> 系统级方向。

---

## 4. conv3×3（待做：`conv_blk` 补 `OBH`）

* OV 的 `convolution_gpu_bfyx_f16` 计算 **`OBW×OBH` 输出块**（selector 里 `OBH` 可 1/2/4…）。
* 我们移植的 **`kernels/conv_blk.cl` 完全没有 `OBH`**（每个 WG/迭代只出一行），候选谱里
  也只有 `OBW`。这是 3×3 剩余 7–11% 差距里最大的一块。
* 下一步：把 `OBH` 端口化（输出块高）+ 候选 `{OBW,OBH}`，对齐 OV 的 selector；预计窄通道/
  grid 饥饿层（`16→16@160×160`、`3→16@320`）受益最大。

---

## 5. 状态与提交

* **已提交**：`e4f2a79`（§1 候选谱 + mb retune）。
* **本轮代码（默认中性，管道/地基）**：`concat4 OUT_FSV16`、`small_concat4 canOutFsv16`、
  `nodeFamily`/`smallKernelFor` 的 concat4 支持、`INFVINO_NO_FUSE_CAT4`、**链感知定价**。
  默认路径 `model_check` 逐位不变（y8 4.114e-04 / y11 7.908e-04 / mb 1.320e-02）、`ctest` 6/6；
  实验路径（`INFVINO_NO_FUSE_CAT4=1` + unfused 缓存）三模型 `model_check` PASS。
* **未做（已定位，需专门一轮）**：融合 blocked-cat4（§2.4）、多消费者/整分量链（§3.3）、
  `conv_blk` OBH（§4）。

### 复现
```bash
# §1：mb retune + A/B
cp config/tuning.json /tmp/t.json
./build/kernel_autotune --plan models/mobilenetv3-small/model.plan --cache /tmp/t.json \
  --op conv1x1 --op conv1x1_cat4 --op depthwise --retune --iters 30
# §2/§3：unfused（链感知修复后）
INFVINO_NO_FUSE_CAT4=1 INFVINO_TUNING_CACHE=/tmp/t.json INFVINO_LAYOUT_REPORT=1 \
  ./build/kernel_run --plan models/yolo11n-pose/model.plan --report --iters 20
```
