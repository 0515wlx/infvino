# 多队列 / Out-of-Order / 软件事件图：本机可行性实测（负结果）

> 背景：`docs/command-buffer.md`（R35）证明 `cl_khr_command_buffer` 在本机 runtime
> 不可用，无法用「整帧录制重放」把 host 提交开销压掉。于是候选方向变成
> **软件事件依赖图 + 多 in-order 队列 / OOO 队列 + 软件记分板**：把帧内 DAG
> 表达成 event 依赖，提交到多个队列，让互相独立的 kernel 重叠执行。
>
> 本文用最小 OpenCL 探针回答一个前置问题：**本机（Iris Xe 80EU / NEO）到底
> 能不能让两个 kernel 并发执行？** 如果硬件/驱动层就串行，那么再精巧的软件
> 事件图/记分板都不会带来重叠收益。

## 0. 结论

**不能。** 在同一个 context、同一 device 上：

- 两个**互相独立**（不同 buffer、无数据依赖）的 kernel，无论放在
  **同一 in-order 队列**、**两个 in-order 队列**、还是**一个
  `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` 队列**，都**完全串行**：
  第二个 kernel 的 `START` 紧跟第一个的 `END`（相差 <0.01 ms），墙钟 ≈ 2× 单核。
- 驱动**声称**支持 OOO（`CL_DEVICE_QUEUE_PROPERTIES &
  CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` 为真），但实际不提供跨 kernel 并发。
- 用两个**不同 kernel**（排除 same-kernel 串行化）结果相同。

**推论**：`P0 软件事件依赖图` / `P2 软件 OoO 记分板` 在**本代硬件上无法产出
kernel 重叠收益**。它们只能影响 host 提交侧，而 `kernel_bench --op chain`
实测 host enqueue（~14 µs/dispatch）与 GPU（~12 µs）已大部分重叠
（wall−busy ≈ 6.9 µs/dispatch）。因此这两项应从路线图中**移除或降为「换硬件
后再评估」**，不应投入引擎重构。

## 1. 探针

`/tmp/opencode/mq_probe.c`（一次性实验，逻辑如下）：

```c
// 选择有 GPU 的平台（platform 1 "Intel(R) OpenCL Graphics"）
// 编译一个 latency-bound kernel（64 WI × 800 万次整数迭代，单核 ~400 ms），
// 两个不同 kernel（spin / spin2）、两个不同 buffer，互相无依赖。
// 三种提交方式，各跑一次，读 event profiling 的绝对 START/END：
//   (a) 同一 in-order 队列
//   (b) 两个 in-order 队列
//   (c) 一个 OUT_OF_ORDER 队列（无 wait list）
```

关键判据：`k1.START − k0.END`。若 `<0` 说明重叠。

## 2. 实测数据（2026-10，Iris Xe 80EU / NEO）

```
device: Intel(R) Iris(R) Xe Graphics   OUT_OF_ORDER support: YES
SAME queue : wall=801.970 ms   k0_gpu=400.927  k1_gpu=400.925   k1.START - k0.END = -0.006 ms
TWO queues : wall=801.991 ms   k0_gpu=400.926  k1_gpu=400.926   k1.START - k0.END = -0.007 ms
OOO queue  : wait=795.814 ms   k0_gpu=400.925  k1_gpu=394.756   k1.START - k0.END = -0.001 ms
=> same/two wall ratio = 1.00x
```

三种方式墙钟几乎相同（~802 ms = 2×400.9 ms），`k1` 只在 `k0` 结束后 ~0.006 ms
才启动。**没有任何并发**。

> 补充：`kernel_bench --op chain`（256 个 ~12 µs 的小 kernel，同一队列）显示
> per-dispatch `wall=18.69 µs, host=13.99 µs, gpu_busy=11.78 µs, wall−busy=6.90 µs`
> ——即 in-order 队列本身已经让 host 提交与 GPU 执行重叠，非重叠气泡只有 ~7 µs/次。

## 3. 对路线图的影响

| 原计划项 | 判决 | 原因 |
|---|---|---|
| P0 软件事件依赖图（多队列提交） | **移除** | 硬件/驱动串行，无重叠收益；host 侧已被 in-order 队列重叠 |
| P2 软件 OoO 记分板 | **移除**（换硬件再说） | 同上；`OUT_OF_ORDER` 标志位被驱动忽略 |
| P0 减 dispatch（融合/view alias） | **保留** | 唯一能把 per-dispatch 气泡×次数降下来的系统手段，但量级有限（见预算分析） |
| P1 内存 alias / 布局 | **保留但小** | 与并发无关 |

**真正的杠杆**：单 kernel 效率（conv3×3 occupancy/tiling、conv1×1/GEMM），
即算子族工作；以及 host 侧预/后处理与 GPU 的**流水线重叠**（多 Session，
非 kernel 并发，而是 CPU 预处理与 GPU 计算跨帧重叠）。

## 4. 复现

```bash
# 一次性探针（host，需 OpenCL headers + Intel GPU 驱动）
gcc -O2 /tmp/opencode/mq_probe.c -o /tmp/opencode/mq_probe -lOpenCL
scripts/gpu_guard.sh run /tmp/opencode/mq_probe
```
