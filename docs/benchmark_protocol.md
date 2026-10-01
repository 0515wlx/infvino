# GPU 基准安全协议（开发板）

> 适用对象：`infvino` 在这台开发板上做 **OpenCL/GPU 实验与基准** 时。
> 目的：避免重演历史上来之不易的 **i915 GPU HANG / 整机硬死机**，并且让排查有据可循。

## 1. 背景与已知事实

| 项 | 值 |
|---|---|
| SoC | Intel i5-1135G7（Iris Xe Graphics，80 EU，1.3 GHz）|
| 内核 | **PREEMPT_RT** `5.15.179-rt84` |
| 内存 | 7.5 GB，**与 iGPU 共享**（iGPU 无独立显存）|
| 症状 | 历史上反复重启（06:20/06:31/06:50/09:40/10:01/10:16…）|
| 日志证据 | `kern.log`/`syslog` 中见过 `[drm] GPU HANG ... in kernel_run` 与引擎复位 |
| 第三方 | `deploy-docker`（ROS，`restart=always` 且**无内存上限**）在段错误-重启循环，与本项目无关 |

**重要结论（排查过）**：两次「整机硬死机」在重启前 **没有任何 OOM/i915/panic 日志**，
属**硬锁死**而非干净 OOM；`CL_KERNEL_PRIVATE_MEM_SIZE = 0`（排除寄存器溢出/scratch 打爆内存）。
部分重启是 **宿主上无关脚本的 OOM**（人工重启），与本项目无关。

风险来源（按可能性）：

1. **PREEMPT_RT + i915** 在 GPU 长任务/异常时可能整机卡死（实时内核会放大 GPU 停顿的影响）。
2. **IGC JIT 编译**大量 kernel 配置时内存/时间开销集中（一次命令里循环编译多个配置时风险最高）。
3. **朴素/未验证的临时驱动**（自写 microbench）出 bug 时更易触发驱动异常。
4. iGPU 与 CPU 共享 7.5 GB，任何一侧的分配峰值都会互相挤压。

## 2. 强制规则（每次 GPU 实验前自查）

### R1. 容器必须限资源

```bash
docker run --rm \
  --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 \
  -v "$PWD":/workspace/infvino -w /workspace/infvino \
  infvino-dev:latest bash -lc '<cmd>'
```

- **只用 render 节点**（`renderD128`），不需要 `--privileged`，避免碰显示/compositor。
- 绝不允许无内存上限的容器跑 GPU 任务。

### R2. GPU 实验短促化

- 单个 kernel 目标 **< 100 ms**；**一条命令只跑一个配置**，配置之间留间隔。
- **不要**在一次命令里循环编译/运行大量 tile 配置——IGC 反复 JIT 是风险最高的时刻。
- 需要扫参时，优先**分批**（每次 1–3 个配置）并各自加 `timeout`。

### R3. 优先离线做 ISA 逆向

`ocloc` 编译/反汇编**不需要 GPU**：

```bash
ocloc compile -file kernels/gemm.cl -device tgl -options "-D... " -output k
ocloc disasm  -file k_tgllp.bin -device tgl -dump ./isa
# 反汇编在 ./isa/.text.<kernel>.asm
```

### R4. 用仓库自带、已验证的工具

- 基准：`kernel_bench`（`--op gemm/conv3x3/... --iters N`）。
- 端到端：`kernel_run --plan ... --report --iters N`。
- 少用临时 standalone 驱动；若必须，务必先做参数/边界断言并加 `timeout`。

### R5. 运行前后看内存

```bash
free -h            # 运行前确认 available
timeout 40 <cmd>   # 每个 GPU 进程都套 timeout
free -h            # 运行后再看一次
```

### R6. 发生 hang 时

- 先等：i915 通常会自动复位引擎（日志会出现 `GPU HANG` / `engine reset`）。
- 若整机卡死：手动重启；**记录时间点**，之后 `grep -a "GPU HANG\|oom\|panic" /var/log/{kern.log,syslog}` 复盘。
- 不要 `docker stop` 第三方的 `deploy-docker`（别人的容器）。

## 3. 安全的最小复现示例

```bash
# 数值 + 短基准（一个配置，<5s）
docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$PWD":/workspace/infvino \
  -w /workspace/infvino infvino-dev:latest bash -lc '
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j4 >/dev/null &&
    timeout 40 ./build/kernel_bench --op gemm --shape 4096,512,512 --iters 30 --verify'
```

## 4. 历史（保留以便对照）

- `softmax` 负 axis 未归一化 → 2.56 亿工作项假死；已修，并在 `kernel_run` 加 gws 安全阀（>3e8 报错）。
- 带宽测试 OOM（`--mb 1024`）；已把 `--mb` 上限钳到 256。
- 脚本/容器统一 `--memory=3g --memory-swap=3g`。
