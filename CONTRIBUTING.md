# 贡献与仓库制度

infvino 是一个**相对独立的自研推理后端**。本文只约定协作、提交、文档与验证制度，方便长期维护；不涉及发布/许可流程。

---

## 1. 分支与同步

- `main` 始终保持**可构建**、可跑离线自检（`tuning_test`）。
- 有风险的改动（kernel 重写、内存池、布局、依赖升级）走短生命周期分支，自测通过后再合。
- 提交前先 `git pull --rebase`；**定期 push**，不要让本地成为单点。
- 重要节点打 tag（阶段收口 / 关键负结果），例如 `checkpoint-r36`。
- 本仓库**不设 CI**，原因见 §4；因此 `main` 的绿灯完全依赖下面的验证约定。

---

## 2. 提交信息（Conventional Commits，严格执行）

```
<type>(<scope>): <摘要>

<body>
```

- **type 单一**，取值：`feat` `fix` `perf` `refactor` `docs` `test` `build` `chore` `bench` `safety`。
- **scope 单一**，取值（按需扩充）：`conv` `gemm` `ops` `autotune` `mem` `layout` `fuse` `p2` `host` `engine` `isa` `env` `docker` `api`。
- **禁止** `perf/fuse`、`perf+docs`、`perf(p2+gemm)` 这类拼接：一个提交做多件事，就**拆成多个提交**。
- 摘要中英文皆可，简洁为主；性能/轮次类可带 `Rxx`（如 `perf(conv): R33 — ...`）。

`body` 推荐沿用本项目的实验报告结构：

```
问题/动机：
改动：
实测（设备 + 数值一致性 + 性能 A/B）：
回归/风险：
文档：
```

可选的 trailer（便于检索，将来可脚本化生成轮次索引）：

```
Round: R36
Result: positive | negative | mixed
Check: kernel_numtest BITEXACT; model_check PASS; reuse_check PASS
Doc: docs/block-layout.md
```

---

## 3. 「一轮优化 = 一个提交」的完成定义（DoD）

一轮优化只有满足以下全部条件才算完成：

1. 代码（或纯配置）改动；
2. **数值检验通过**：整网必须用**双输入**（每帧不同）并取**第二帧**；按需再跑算子级 / 库后端；
3. **同一提交更新文档**：结论写进对应 `docs/*.md`，并在 README 的文档表登记；
4. **负结果必须留档**（注明已回退 / 已关闭），不得静默丢弃；
5. 不引入未说明的回归。

机械性改动（格式化、依赖升级、生成物刷新、无关 bugfix）**拆成独立提交**，不要混进功能提交。

---

## 4. 验证制度（无 CI，人工执行）

> **为什么不设 CI**：核心验证是 GPU 性能，目前只有一块开发板；后续支持新模型时测试面会大改。当前阶段由开发者在本机/容器人工验证。CI 的缺席由下面的**固定命令 + 门禁脚本**补偿。

**构建（容器内）**

```bash
docker build -f docker/Dockerfile -t infvino-dev:latest .
docker run --rm -v "$PWD":/workspace/infvino -w /workspace/infvino infvino-dev:latest \
  bash -lc 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j'
```

**离线自检（无 GPU，改动后必跑）**

```bash
docker run --rm -v "$PWD":/workspace/infvino -w /workspace/infvino infvino-dev:latest \
  bash -lc 'ctest --test-dir build -R tuning_test --output-on-failure'
```

**GPU 验证（必须先读 `docs/benchmark_protocol.md`，并用 `gpu_guard` 包一层）**

```bash
scripts/gpu_guard.sh check
scripts/gpu_guard.sh run python3 scripts/kernel_check.py   --repo "$PWD" --image infvino-dev:latest
scripts/gpu_guard.sh run python3 scripts/model_check.py     --model yolov8n-pose --repo "$PWD" --image infvino-dev:latest
scripts/gpu_guard.sh run python3 scripts/numerical_check.py --repo "$PWD" --image infvino-dev:latest
scripts/gpu_guard.sh run python3 scripts/engine_check.py    --repo "$PWD" --image infvino-dev:latest
scripts/gpu_guard.sh run python3 scripts/reuse_check.py     --model yolov8n-pose --repo "$PWD" --image infvino-dev:latest
scripts/gpu_guard.sh after
```

- 改 `.cl` 后必须重跑对应算子级检验；数值要求**逐位一致**，或在噪声范围内并明确说明。
- 性能结论必须给**同会话 A/B**（或可复现的口径），不能只给单点数字。

---

## 5. 文档制度

- **单一事实源**：每轮结论只写在对应的 `docs/*.md`（专题 / Round 文件）；**README 只保留「现状 + 链接」**，不复制整段结论，避免多处漂移。
- 新增文档要登记进 **README 的文档表**。
- 代码与相关文档**同提交**；文档里的命令必须能直接复现。
- 设备/驱动/依赖或环境变化，更新 `docs/dependencies.md` 与 `docker/Dockerfile`。

---

## 6. 代码风格

- C++17；2 空格缩进；成员变量尾下划线（`device_`、`plan_path_`）；与现有代码保持一致。
- **命名空间统一为 `infvino`**。历史上引擎层的 `gk` 命名空间已废弃（见 `refactor(core)` 提交）。引擎原始设备描述为 `infvino::ClDeviceInfo`，公开门面的 `infvino::DeviceInfo` 在 `Device.hpp`。
- 公开头文件在 `include/infvino/`；**实现细节不要泄漏进公开签名**。
- 按 `-Wall -Wextra -Wpedantic` 处理编译告警；新增告警应消除。
- OpenCL kernel：文件名 `*.cl`，命名沿用现有前缀风格；从外部移植（如 OpenVINO）的代码需在 `THIRD_PARTY_NOTICES.md` 登记归属。
- 新增文件补版权头，格式与现有 `src/` / `include/` 一致。

---

## 7. 生成物政策

| 类别 | 政策 |
|---|---|
| 构建产物 `build*/` | 不入库（`.gitignore` 已覆盖） |
| 模型与 `models/*` | 不入库（`.gitkeep` 除外） |
| 调优缓存 `config/tuning.json` | 当前入库；重扫后更新它请**单独成提交或明确标注**，不夹带进功能提交 |

---

## 8. 安全（开发板）

开发板（i5-1135G7 + PREEMPT_RT）历史上多次因 `i915 GPU HANG` 整机硬死机。跑任何 GPU 负载前：

```bash
scripts/gpu_guard.sh check          # 跑前：驱动健康 / 内存 / 残留渲染
scripts/gpu_guard.sh run <cmd...>   # 用 timeout + 事后 HANG 检查包一层
scripts/gpu_guard.sh after          # 跑后：确认无 HANG（退出码 3 = 出现 HANG）
```

完整协议见 `docs/benchmark_protocol.md`。
