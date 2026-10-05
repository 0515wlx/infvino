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
