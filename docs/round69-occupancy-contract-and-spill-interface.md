# R69：占用/访存契约声明化进注册表 —— 候选/内核 ↔ spill 的接口收口

> 承接 R68：L3 spill 成本模型已收敛为「可加主问题（实测 ms + reorder）+ 非可加全局模拟」，
> 并明确「当前三模型的增量在**候选/内核**，不在把已有候选选得更准」。本轮做的正是
> **候选/内核 ↔ spill 的接口**：把此前散在 `Tuning.cpp::occupancyStats` 的**占用几何**，
> 声明化进 `KernelFamily` 注册表（R48 §10.6-B 记录的「加族只加声明未落地」缺口），
> 并让 `buildL3Access` 的 spill 视图由该契约驱动、不再硬编码小算子名单。
>
> 结论先给：接口落地、默认开；**三模型选择逐位不变**（mincut fsv16/E、dispatch、
> reorder 全同），`model_check`/`reuse_check` PASS 且 `mean_rel` 与 R68 逐位相同。
> 唯一变化是**诊断用的 spill 数值**（y8 +0.328 ms、y11 +0.328 ms、mb +0.0001 ms），
> 来自此前未被计/被误算的族（`conv3x3_cin3`/`conv3x3_f16`/`gemm_sk`/`conv1x1_gemv`/
> 非 blk `depthwise`）。**是否新增内存受限候选：本轮结论是暂缓**（§6）。

---

## 1. 问题：候选 → spill 的接口是硬编码的

`L3Model` 的全局溢出模拟对每个节点要一个「在飞占用压力」R = `occupancyPressure(e,s)`。
R 由**选中候选的执行几何**决定（并发 WI 数 × 每 WG 触达字节）。但 R69 之前：

* `occupancyStats`（`Tuning.cpp`）是一个按 `s.op` 的 `if/else`，只覆盖 conv3x3（一律当
  `conv_ov` 解析 `-DOBW/–DOBH/–DSLM_DIV`）、`conv1x1_blk`、`depthwise_blk`、以及把
  gemm/conv1x1/cat4 混在一起的 GEMM 分支；
* 于是 **conv3x3 的另外三个 kernel**（`conv3x3_blk`、`conv3x3_f16`、`conv3x3_cin3`）
  全落进 ov 判据（blk 用了幻影 `OBH`、f16/cin3 的 `-DTX/-DCIN` 根本不被读）；
  **`gemm_sk`/`conv1x1_gemv`** 落进 GEMM 分支（`-DSK_*`/`-DGEMV_TM` 被忽略）；
  **非 blk 的 `depthwise_f16/v/vp`** 直接 `return false`（占用恒 0）；
* 新增族/候选**不会自动进入 spill 模型**——正是 R48 §10.6-B 批评的「四处硬编码」之一。

这与「注册表 = 候选 / 布局 / 上限的单一真相源」的定位不符：布局契约、上限都声明化了，
唯独**访存几何**没有。

## 2. 接口：`MemContract` + `KernelFamily::mem`

```cpp
// include/infvino/Tuning.hpp
struct MemContract {
  bool   valid        = false;
  double concurrent   = 0.0;   // 原始总 work-item 数（消费方按 8192 饱和夹取）
  double perWgBytes   = 0.0;   // 每个 work-group 触达的**输入激活**字节（权重常驻不计、输出不计）
};

// include/infvino/KernelFamily.hpp
std::function<MemContract(const OpSignature &, const std::string & options)> mem;
```

约定与 R47/R55 标定口径一致：`R = min(concurrent, 8192) × perWgBytes`。语义保持「在飞压力」
而非「单遍总字节」——所以 `perWgBytes` 只算输入 tile（权重/常量常驻，输出在写回后即离开
工作集），这与旧模型被标定的形态一致；把权重/输出也并进去只会破坏标定（§6）。

消费方（`occupancyPressure`/`occupancyThreads`）改为**注册表优先**：
按候选 `kernel` 名反查族 → 该族声明了 `mem` 就用它；否则回退到原有 legacy op 判据
（小算子/未知 kernel → 不计占用）。`buildL3Access` 里硬编码的 `isSmall` 名单被删除，
改为「无占用模型（`press<=0`）→ 用流式足迹（读+写）」，与契约自动对齐。

## 3. 各族访存几何（单一真相源，`KernelFamilies.cpp`）

| 族 | `concurrent`（原始 WI） | `perWgBytes`（输入激活） |
|---|---|---|
| `conv3x3_ov` | `ceil(W/OBW)·ceil(H/OBH)·ceil(Cout/32)·16·SLM` | `2·Cin·((OBW−1)S+3)·((OBH−1)S+3)` |
| `conv3x3_blk` | `ceil(W/OBW)·H·ceil(Cout/16)·16·SLM` | `2·Cin·((OBW−1)S+3)`（一行输入带） |
| `conv3x3_f16` | `ceil(W/TX)·ceil(H/TY)·ceil(Cout/CB)·(TX/TM)·TY` | `2·Cin·IN_ROWS·IN_COLS` |
| `conv3x3_cin3` | `ceil(W/128)·H·128` | `2·Cin·3·(128+2)`（3 行输入带） |
| `gemm_f16`/`gemm_cat4_f16` | `ceil(M/BM)·ceil(N/BN)·(BM/TM)·(BN/TN)` | `2·BK·(BM+BN)`（K-tile staging 窗口） |
| `gemm_sk_f16` | `ceil(N/TN)·ceil(M/TM)·SK_SG` | `2·K·(TM+TN)` |
| `conv1x1_gemv_f16` | `ceil(Cout/GEMV_TM)·16` | `2·Cin·(GEMV_TM+1)` |
| `conv1x1_blk` | `ceil(N/(XB·YB))·ceil(Cout/16)·16·SLM` | `2·Cin·XB·YB` |
| `depthwise_f16/v/vp` | `ceil(W/TW)·H·Cin`（TW=1 为 f16） | `2·((TW−1)S+K)·K` |
| `depthwise_blk` | `ceil(W/XB)·ceil(H/YB)·ceil(Cout/16)·16` | `2·16·((XB−1)S+K)·((YB−1)S+K)` |

修正的关键点（相对旧判据）：ov 的通道块用 `ceil(Cout/32)`（R39 已确认 osv32 语义）、
blk 去掉幻影 `OBH`、f16 用 `TX/TY/TM/CB`、gemm 的 WG 补上 `BM/TM` 因子、
split-K/GEMV 读自己的 knob、非 blk depthwise 首次计入。

## 4. 验证

| 门 | 结果 |
|---|---|
| `ctest`（6 套） | **6/6 PASS**（新增 R69 契约完整性 + 几何用例） |
| `model_check` y8 / y11 / mb | **PASS**，`mean_rel` = `4.114e-04 / 7.908e-04 / 1.215e-02`，与 R68 **逐位相同** |
| `reuse_check` y8 | **PASS**（跨推理一致性，diff 不变） |
| 选择（mincut fsv16 / E / dispatch / reorder） | 三模型**逐位不变**（新 vs `INFVINO_LEGACY_OCCUPANCY=1` 对照） |

模型分项（`INFVINO_LAYOUT_REPORT=1`，新 vs legacy）：

```text
y8  kernel=10.1851 reorder=0.0583 spill=1.2521 (legacy 0.9239) total=11.4955 (11.1672)
y11 kernel=10.7935 reorder=0.2082 spill=1.3650 (legacy 1.0367) total=12.3667 (12.0384)
mb  kernel=1.3323  reorder=0.0561 spill=0.2318 (legacy 0.2317) total=1.6202  (1.6201)
```

**spill 增量的逐族归因**（y8，用 `INFVINO_NO_MEM=<family>` 单个族回退 legacy）：

| 回退的族 | spill (ms) | Δ vs full |
|---|---:|---:|
| full | 1.2521 | — |
| `conv3x3_cin3` | 1.1109 | **−0.141** |
| `gemm_f16` | 1.1710 | **−0.081** |
| `conv3x3_f16` | 1.2168 | **−0.035** |
| `conv3x3_ov` | 1.2521 | 0.000 |
| `conv3x3_blk` | 1.2521 | 0.000 |

读数：增量集中在**旧判据误算的族**（cin3 的 3 行输入带、gemm 的 `BM/TM` WG 因子、
f16 的 TX/TY 几何），而 ov/blk 因 `concurrent` 早已饱和（8192）→ 修正不改变其压力。
mb 几乎不变，因为它的默认链几乎全是 `_blk`（旧判据本就准确）。

## 5. 开关 / 回退

| 开关 | 作用 |
|---|---|
| （默认） | 注册表声明优先 |
| `INFVINO_LEGACY_OCCUPANCY=1` | 强制回退到旧 `occupancyStats`（整份 A/B/回滚） |
| `INFVINO_NO_MEM=<family>` | 只回退某一族的契约（逐族归因/消融） |

默认开的选择依据：三模型的**选择逐位不变**、数值逐位不变，即「接口权威化」不引入任何
行为回归；spill 数值更完整地反映真实访存几何（诊断更真）。若未来某模型出现选择漂移，
`INFVINO_LEGACY_OCCUPANCY=1` 可立即回退。

## 6. 分析：是否新增内存受限候选？

本轮同时回答了用户提出的第三项（先分析）。用现在完整的契约观察：

1. **spill 不主导选择**。修正后 spill 占模型总分 y8 10.9%、y11 11.0%、mb 14.3%，且
   **选择完全未变**——说明当前三模型里「跨算子溢出」不是候选选择的分界；候选选择的
   分界仍是**单算子实测 ms**（autotune）+ 布局链（mincut）。
2. **新增内存受限候选的既有证据仍是负的**：R41 证 conv3x3 大层是「无 L1 + 128-GRF ILP」
   结构墙（写 kernel 无效）；D1 输出 tiling 已在 `conv1x1_blk` 试过（R48 §6，整网噪声内）；
   R67 证明把容量项折进目标函数会**一致回归**。映射到 spill 接口上，即：**没有证据表明
   存在某签名，其当前最优候选的内存行为是「可被一个新 kernel 显著改善」的**。
3. **要做的前置**（若要继续）：需要一个**逐节点 spill 归因**视图（现在 `L3Model` 已能算
   `node_evict_ms`，但 PlanModel 没暴露），先定位「谁是 spill 大户」，再判断其是否
   内存受限且有非受限替代；否则就是 R48 §7.3-E 的「候选方向与真实瓶颈正交」。

**建议**：本轮不新增内存受限候选。接口已就位——未来一旦出现（a）新的内存受限候选，或
（b）OA 计数器可用（R43 受内核 tracepoint 限制），把该候选的几何声明进 `mem` 即可自动
进入 spill 模型，无需再改 `Tuning.cpp`。下一步（若要动）应是**暴露逐节点 spill 归因**，
而不是盲加 kernel。

## 7. 改动文件

* `include/infvino/Tuning.hpp`：`MemContract`、`tuningOptionInt`、`occupancyPressure` 文档。
* `include/infvino/KernelFamily.hpp`：`KernelFamily::mem`。
* `src/Tuning.cpp`：`occupancyStats` 注册表优先 + legacy 回退 + 两个消融开关；`tuningOptionInt`。
* `src/KernelFamilies.cpp`：十个族的 `mem` 声明（§3）。
* `src/PlanModel.cpp`：`buildL3Access` 去掉硬编码 `isSmall`，由契约驱动。
* `tests/test_kernel_families.cpp`、`tests/test_rulers.cpp`：契约完整性 / 几何 / A/B 开关用例。

## 8. 复现

```bash
# 离线：契约 + 标尺 + L3 模型
docker run --rm -v "$PWD":/workspace/infvino -w /workspace/infvino infvino-dev:latest \
  bash -lc 'cmake --build build-blk -j8 --target check'

# 选择/分项：新 vs legacy（锁频；只读模型分项，不跑 kernel 计时）
for m in yolov8n-pose yolo11n-pose mobilenetv3-small; do
  ./build-blk/kernel_run --plan models/$m/model.plan --report --iters 1
  INFVINO_LEGACY_OCCUPANCY=1 ./build-blk/kernel_run --plan models/$m/model.plan --report --iters 1
done

# 数值回归
python3 scripts/model_check.py --model yolov8n-pose --repo "$PWD" --image infvino-dev:latest
```
