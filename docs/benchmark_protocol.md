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
- **仓库自带的四个检查脚本（`kernel_check.py` / `model_check.py` /
  `numerical_check.py` / `engine_check.py`）已在 R23 修正**：此前它们用
  `--device=/dev/dri:/dev/dri --privileged` 且**没有** `timeout` 包裹，
  违反了本协议 R1/R2 —— 这是 2026-10-02 两次 GPU HANG 的间接诱因。
  现已统一为 `--device=/dev/dri/renderD128` + `--pids-limit=256` +
  `timeout` 包裹每一条 GPU 命令 + 跑后 `GPU HANG` 自检。

### R1b. 运行前后跑 `scripts/gpu_guard.sh`

```bash
scripts/gpu_guard.sh check          # 跑 GPU 前：内存 / dmesg 残留 HANG / 残留渲染进程
scripts/gpu_guard.sh run <cmd...>   # timeout 包裹 + 事后 HANG 检查
scripts/gpu_guard.sh after          # 手动跑后检查（出现 HANG 返回 3）
```

`dmesg` 在本机是 root-only（`kernel.dmesg_restrict=1`），守卫改读
`/var/log/kern.log` 与 `/var/log/syslog`。

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

- **2026-10-02 反复死机/重启（本轮事故）——已定位为两类不同的原因**：
  1. **GPU HANG（6 次，全部 `in kernel_run`）**：05:25:57、05:32:20、06:06:53
     等，`ecode 12:1:85dcfffb`。i915 在 `heartbeat` 里尝试复位引擎
     （`intel_engine_stop_cs → __intel_engine_reset_bh`）反复失败；PREEMPT_RT 下
     这种复位风暴会拖死整机。**关键证据：有个别重启完全没有 GPU HANG 记录**，
     所以 hang 不是每次重启的直接原因。
  2. **电源键强制关机**：06:20:10 的 syslog 明确写着
     `systemd-logind: Power key pressed → Powering Off → System is powering down`，
     也就是**有人按了电源键/长按强制重启**（这是卡死后的手动恢复手段，不是故障本身）。
     这也解释了为什么日志里会出现“干净 shutdown”记录。

  **排查结论**：
  - **不是 opencode/CLI 的内存泄漏**：每次都是 `kernel_run`（一个短命进程，跑完即退出，
    每次分配几十 MB）触发 GPU hang，与 CLI 常驻内存无关；`free` 一直有 6.8 GB 可用。
  - **不是某个新 kernel 的代码 bug**：把 R23 的 `bmm`/`concat-fusion` 单独回退后仍会 hang，
    而且历史 6 次 hang 覆盖了旧/新多套 kernel；`ecode` 每次都相同。
  - **真正的可控变量是「暴露量」而非「某个 kernel」**：本机 i915（NEO compute runtime
    + PREEMPT_RT 5.15）在**持续/反复提交 GPU 命令流**时会进入 hang；单次短核
    （一条命令、单个 kernel、1–3 iters）在 10+ 次实验中均未触发。
  - **第三方干扰**：`deploy-docker`（`violet0405/deploy-docker-image`，`privileged` +
    主机整个 `/dev` 映射 + 无内存上限 + `restart=always`）长期处于
    `component_conta` segfault→重启死循环，持续向内核灌故障事件。**已 stop 并
    `--restart=no`**，该干扰停止。

  **已落地的加固**：四个检查脚本去 `--privileged`、只挂 `renderD128`、逐命令
  `timeout`、跑后 `GPU HANG` 自检；新增 `scripts/gpu_guard.sh`（跑前查内存/残留
  hang/残留进程/`deploy-docker` 重启循环，跑后查 hang）。
  **操作建议**：GPU 实验务必「一条命令、单个配置、1–3 iters、连着 timeout」；
  一旦 `gpu_guard.sh after` 报 hang，就停下等 i915 自复位，不要连续叠加负载。
- **物理层外因**：本开发板为此类工控/开发板，存在**看门狗/电源键复位**机制；
  卡死后的正常恢复就是按电源键（日志里 06:20 即为此）。若频繁出现，应从
  硬件/散热/电源侧一并排查。
- `softmax` 负 axis 未归一化 → 2.56 亿工作项假死；已修，并在 `kernel_run` 加 gws 安全阀（>3e8 报错）。
- 带宽测试 OOM（`--mb 1024`）；已把 `--mb` 上限钳到 256。
- 脚本/容器统一 `--memory=3g --memory-swap=3g`。
- i915 GPU hang 记录（历史）：本开发板（i5-1135G7 + **PREEMPT_RT** `5.15.179-rt84`，
  7.5 GB 与 iGPU 共享）历史上有频繁重启与 `[drm] GPU HANG ... in kernel_run`。
