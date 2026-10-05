# R47：把 L3/DRAM 显式建模进标尺（不再当黑箱）

> 动机（用户）：LLC 溢出对性能有决定性影响，选择 kernel 时必须准确建模，而不是当黑箱。
> 项目哲学是追求极限——哪怕 ISA 不能直接控制、又难以测量，也要找出**相对准确的估计方法**。
>
> 本轮第一步：把内存层级（L3 容量 + DRAM 带宽断崖）显式做进**中间标准**（`expectedOps`）
> 与设备带宽曲线，使「离物理极限多远」这个判断不再把内存当不存在。

---

## 1. 设备带宽-足迹曲线（锁频实测，本轮重测）

`kernel_bench --op bandwidth --mb N`（copy，读+写），`scripts/gpu_clocks.sh lock` 下：

| footprint | 1 MB | 2 MB | 3 MB | 4 MB | 5 MB | 6 MB | 8 MB | 12 MB | 16 MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 145.4 | 138.8 | 111.5 | **56.1** | 35.5 | 28.2 | 22.2 | 20.7 | 20.3 |

**膝点在 3→4 MB**（有效 L3 ≈ 3.75 MB）：111 → 56 → 28 → 22，最终贴 DRAM 墙 ~20 GB/s。
（旧 `kBwCurve` 是 R30 未锁频值，1–4 MB 段偏低 ~1.5–2×；本轮替换为锁频值并扩展到 16 MB。）

---

## 2. 内存 roofline（通用）

新增 `memRooflineOps(flops, bytes, eu, clk, launch)`：

```
t_mem = launch + bytes / BW(bytes)          # BW 来自 §1，按足迹插值
memOps = flops / (EU · clk · t_mem)         # 与 ops/EU/cyc 同量纲
```

`BW(bytes)` 随足迹跨 L3 断崖而骤降 → 内存受限层的 `memOps` 自然被压到内存墙之下。

---

## 3. 接入标尺（conv / gemm / conv1x1）

`conv3x3` / `gemm`（及委托它的 `conv1x1`、`conv1x1_cat4`）的**软 `expected`** 改为：

```
expected = min( 族计算上限 , memRooflineOps(flops, bytes) )
```

`bytes` 取**单遍流**上界：输入（按 stride 放大空间）+ 输出 + 权重。落点：`Tuning.cpp::expectedOps`
与 `Autotuner.cpp::applyStandard` / `PlanModel.cpp::refreshExpected` 的取用处（`min`）。

---

## 4. 效果（mb，`--refresh-expected`，零 GPU）

| 签名 | measured | expected(old) | expected(new) | ratio(old→new) |
|---|---:|---:|---:|---:|
| `conv3x3\|W20H20s1p1_Cin64_Cout64` | 7.43 | 20.09 | **7.53** | 0.37 → **0.99** |
| `conv3x3\|W80H80s1p1_Cin64_Cout64` | 15.02 | 20.09 | 20.09 | 0.75（计算受限，不变）|
| `conv1x1\|Cout16_N3136_Cin16` | 0.47 | 2.10 | 2.10 | 0.22（prologue 受限，不变）|

即：**内存受限的层从"看似有 63% 余量"纠正为"已贴内存墙（ratio 0.99）"**；计算受限的层不受影响。
这直接消除了「把内存当成不存在」导致的**假余量**，让 `ratio`/`headroom` 排名可信。

---

## 5. 边界与下一步（诚实）

1. **单算子 `bytes/BW` 是朴素单遍流上界**；kernel 内若有 L1/SLM 复用，真实 DRAM 流量更少，
   实测**可以超过**该 roofline。因此它只进**软标尺**，**不作硬上限**（否则会被违反，重演 R39
   「内存族 ceiling 偏低、ratio>1」的问题）。
2. **跨算子 L3 争用**是本轮仍未建模的部分（也是非组合性的根源，见
   [`round47-full-flow-findings.md`](round47-full-flow-findings.md) §4.6）。本机 OA 计数器不可用
   （`CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS` 未开），无法直接测 L3/DRAM 字节。
   **可实现的估计方案（下一步）**：对每个激活张量做**复用距离**模型——统计其生产者与最后消费者
   之间新写入的字节量；若「复用距离 + 自身字节」> L3，则该张量在消费前已被逐出 → 记一次 DRAM 往返。
   把它作为 `predictNet` 的显式项，使**布局/候选选择**能看见「是否把工作集推过 L3」。
   （注意：张量字节在 fsv16（Cout%16==0）下不变，故该模型主要对**重排/额外物化**敏感。）
3. **layout 决策**应引入「工作集是否跨 L3」的约束（blocked chain / tiling 的目标函数）；
   这需要 2 的模型先落地并与实测校准。

---

## 6. 复现

```bash
scripts/gpu_clocks.sh lock
for mb in 1 2 3 4 5 6 8 12 16; do
  ./build-ct/kernel_bench --op bandwidth --mb $mb
done
scripts/gpu_clocks.sh unlock

# 标尺重算（零 GPU）
./build-ct/kernel_autotune --plan models/mobilenetv3-small/model.plan \
  --cache config/tuning.json --refresh-expected
```

---

## 7. 第二步：把 L3 溢出估计接进选择目标（`predictNet`）

**实现**（`PlanModel.cpp::globalRetune`，零 GPU、确定性）：

```
spill_bytes = Σ_t [ pressure(t) + bytes(t) > L3 ? bytes(t) : 0 ]      # 复用距离逐出
            + Σ_{blk 且输入未持久 fsv16} bytes(input)                   # reorder 缓冲占用 L3
spill_ms    = spill_bytes × (1/BW_DRAM − 1/BW_L3)     # 20 vs 130 GB/s
predictNet  = Σ 逐节点 kernel ms + Σ #reorder.ms + spill_ms
```

其中 `pressure(t)` = 张量 t 的生产者与最后消费者之间新写入的字节（用节点级前缀和）。
这让 `predictNet`（目标排序 / S2 契约门 / S3 chain move）**显式看见**「是否把工作集推过 L3」。
报告里打印 `base(L3 spill est)`，可观测。

**实测（mb 单进程 33 签名）**：`base(L3 spill est) ≈ 0.035 ms`（~2% busy）；S2 契约门现在把多个
族的 `gemm*` 候选判为预测更差而拒绝。

**诚实的局限**：
1. 张量字节在 kernel 选择下**不变**（fsv16 仅在 `Cout%16==0` 时启用、无通道补齐）→ spill 的
   **张量逐出项是赋值不变量**（在比较中相消）；只有 **reorder 缓冲项**随赋值变化，且与既有
   `#reorder.ms` 部分重叠。所以本步对选择的**边际影响有限**。
2. **仍缺**：① 每候选的 **tiling 感知 DRAM 流量**（OBW/OBH/BM/BN 决定 halo/重读次数）——
   但对**内存受限候选**，隔离实测 ms ≈ 流量/BW，已隐式包含；② **占用率/访问模式导致的
   跨算子 L3 争用**——这是端到端噪声的根源，本机 OA 不可用，需要校准实验或计数器才能精确。
3. `pressure` 用「节点级产出字节」近似「L3 驻留压力」，忽略张量实际重用（同张量多次读）与
   写合并缓冲，属**一阶估计**。

**结论**：这是「把 L3 当黑箱」→「一阶确定性估计」的第一步落地；它让标尺（§3）与代理目标（本节）
都显式含内存层级。要把它变成**准确实用**的选择依据，缺的是可用计数器或一套校准实验来标定
「每候选跨算子外溢」。在这些就位前，**保留隔离 `min` 作为选择主口径**，`predictNet` 仅用于
拒绝明显更差的候选（S2 门）。

---

## 8. 第三步：占用建模 + 强耦合的全局处理（L3 LRU 模拟）

**为什么不能逐算子相加**：瞬时占用会引入**强耦合**——候选 A 的在飞工作集会把候选 B 要读的输入
逐出 L3，B 的代价依赖 A 的选择。逐算子独立项会（a）无法表达这种相互依赖，（b）二次计费
（既算 `#reorder.ms` 又算张量逐出），（c）破坏可加性。**因此改为按节点顺序跑一遍全局缓存模拟。**

**实现**（`PlanModel.cpp::globalRetune`，零 GPU）：

```
状态：容量 = L3 的常驻张量集合（MRU 在尾）
每节点 i：
  press = min(WG_count, 80) × perWG_bytes        # 瞬时占用（并发 WG × 每 WG tile）
  while resident + press > L3: 逐出 LRU 头
  for each 输入激活: touch(read)                  # 未命中 → miss += bytes
  touch(输出, write)
spill_ms = miss_bytes × (1/BW_DRAM − 1/BW_L3)
```

- `perWG`：conv3x3 = `2·Cin·(obw+2)(obh+2)`；gemm/conv1x1 = `2·(bm·bk + bn·bk)`（按候选 options
  的 OBW/OBH/BM/BN/BK）；并发 WG ≈ min(grid, EU 数)。
- **耦合由模拟整体处理**：每次 `predictNet(asg)` 都在**整张图上重跑**一遍模拟，因此 A 的占用
  对 B 的影响会被一致地计入——这正是「全局、不可加」的部分。
- 报告打印 `base(L3 spill est)`；mb 实测 ≈ **0.227 ms**（约 busy 的 **13%**，比 step2 的 0.035 ms
  更贴近含占用后的真实压力）。

**诚实的边界**：
1. `perWG` 是**上界代理**（实际并发受寄存器/调度限制），是常量因子；`predictNet` 会把它当
   单一未标定系数的 proxy。**可标定**：用 `kernel_bench --op bandwidth` 的 footprint 扫描 +
   隔离实测来拟合该系数。
2. 逐出按**字节**近似（张量粒度，非 cache-line 粒度），忽略了组相联/流式写占位。
3. 仍然：**端到端噪声**（§见 findings §4.6）不因更准的模型而消失——模型提高的是**选择依据**的
   质量，不是测量精度。因此 `predictNet` 的正确定位仍是「排序 / 拒绝明显更差者」，最终裁决在
   测量可信度解决之前不宜依赖端到端。

**意义**：至此，选择目标里**显式包含**了「计算 roofline + 内存 roofline（L3 断崖）+ 跨算子 L3
占用耦合」，且耦合以全局模拟方式一致处理，而不是逐算子相加。这是让「选择」真正 LLC 感知的关键
结构；剩下的是**标定**（把第 1 条的常量因子拟合准）。

