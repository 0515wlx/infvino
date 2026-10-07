#!/usr/bin/env python3
"""自动调优驱动：在容器内安全地分批跑 `kernel_autotune`（docs/autotuning.md）。

背景：本开发板（i5-1135G7 + PREEMPT_RT）在**持续/反复提交 GPU 命令流 + IGC 反复 JIT**
时容易触发 i915 GPU HANG（见 docs/benchmark_protocol.md）。因此这里**把一次大规模 tile
扫描拆成很多小批**：每批只调 `--limit N` 个签名、`--iters M`，跑完立即把结果写进
tuning.json（merge），一批失败/超时最多丢一批，不会前功尽弃，也避免一条命令里
循环编译上百个配置。

流程：
  1. 容器内构建 once；
  2. `--list` 拿到每个 op 的唯一签名数；
  3. 对每个 op，按 `--batch` 分批调用 kernel_autotune（`--limit batch`），
     每批之间打印进度；缓存文件贯穿所有批次（merge）。

用法:
  python3 scripts/autotune.py --model yolov8n-pose --repo $PWD
  python3 scripts/autotune.py --model mobilenetv3-small --ops conv1x1,depthwise --batch 4
  python3 scripts/autotune.py --model yolov8n-pose --list-only
"""
import argparse
import os
import subprocess
import sys

DOCKER_BASE = [
    "docker", "run", "--rm",
    "--memory=3g", "--memory-swap=3g", "--pids-limit=256",
    "--device=/dev/dri/renderD128",
]


def run(cmd):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--image", default="infvino-dev:latest")
    ap.add_argument("--model", default="yolov8n-pose")
    ap.add_argument("--ops", default="conv3x3,conv1x1,depthwise,gemm",
                    help="逗号分隔；空=全部")
    ap.add_argument("--batch", type=int, default=3,
                    help="每批调优的唯一签名数（越小越安全，默认 3）")
    ap.add_argument("--iters", type=int, default=15,
                    help="每个候选的计时迭代数（默认 15；越小越快但噪声越大）")
    ap.add_argument("--cache", default="config/tuning.json")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--bake", action="store_true",
                    help="额外把 plan 复制一份到 <model>/model.plan.baked（审计用）")
    ap.add_argument("--global", dest="global_", action="store_true",
                    help="R44: 隔离扫描后做整网 busy 坐标下降回验（目标函数改为端到端）")
    ap.add_argument("--global-topk", type=int, default=3)
    ap.add_argument("--global-iters", type=int, default=2)
    ap.add_argument("--global-rounds", type=int, default=2)
    ap.add_argument("--global-limit", type=int, default=0,
                    help="每批最多回验的签名数（0 = 用 --batch）")
    ap.add_argument("--global-margin", type=float, default=0.0,
                    help="隔离 margin 剪枝（默认 0=关；>0 有剪掉流水线更快候选的风险）")
    ap.add_argument("--global-budget", type=int, default=0,
                    help="整网执行次数总预算（0 = 不限；R47 口径改为整网执行次数，更贴近 GPU 风险）")
    ap.add_argument("--lock", action="store_true",
                    help="跑基准前 scripts/gpu_clocks.sh lock，结束后 unlock（推荐用于 --global）")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    plan = f"/workspace/infvino/models/{args.model}/model.plan"
    cache_abs = os.path.join(repo, args.cache)
    os.makedirs(os.path.dirname(cache_abs), exist_ok=True)

    # R47 fix: 绝对 --cache（如 /tmp/t.json）此前会被拼成容器内 `/workspace/infvino//tmp/t.json`
    # → 缓存写到错误位置甚至静默丢失（R46 文档正是这么用的）。改为 bind-mount 其所在目录，
    # 并把绝对路径**原样**传给容器；相对路径仍映射到仓库内。
    extra_mounts = []
    if os.path.isabs(args.cache):
        cache_in_container = args.cache
        extra_mounts = ["-v", f"{os.path.dirname(cache_abs)}:{os.path.dirname(cache_abs)}"]
    else:
        cache_in_container = f"/workspace/infvino/{args.cache}"
    dcmd = DOCKER_BASE + extra_mounts

    ops = [o for o in args.ops.split(",") if o]

    def in_container(inner: str, timeout: int = 240) -> int:
        cmd = dcmd + [
            "-v", f"{repo}:/workspace/infvino", "-w", "/workspace/infvino",
            args.image, "bash", "-lc",
            "set -e\n"
            "cmake -S /workspace/infvino -B /workspace/infvino/build-ct "
            "-DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1\n"
            "cmake --build /workspace/infvino/build-ct -j4 >/tmp/build.log 2>&1\n"
            + inner,
        ]
        return run(cmd)

    # 1) 签名清单
    for op in ops:
        rc = in_container(
            f"timeout 90 /workspace/infvino/build-ct/kernel_autotune "
            f"--plan {plan} --list --op {op} 2>/dev/null | grep 'unique tuning' || true")
        if rc != 0:
            print(f"[autotune] list failed for {op}", file=sys.stderr)
    if args.list_only:
        return 0

    # 2) 分批调优：一次容器构建，内部用**多个独立进程**分批（各自 timeout），
    #    每批 --limit <batch>；已调过的签名（cache 里 source=="tuned"）会被跳过，
    #    因此重复调用自然推进。这样既避免 IGC 单进程反复 JIT 上百配置，又不用每批重建。
    #
    #    每批之后做一次轻量 GPU HANG 自检；一旦发现 hang 立即停止。
    loop = [
        "set -e",
        "cmake -S /workspace/infvino -B /workspace/infvino/build-ct "
        "-DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1",
        "cmake --build /workspace/infvino/build-ct -j4 >/tmp/build.log 2>&1",
    ]
    # R44: 整网回验参数（每个批进程内先隔离扫描、再整网坐标下降；--global-limit 与 batch 对齐）。
    gargs = ""
    retune = ""
    if args.global_:
        glimit = args.global_limit if args.global_limit > 0 else args.batch
        gargs = (f"--global --global-topk {args.global_topk} --global-iters {args.global_iters} "
                 f"--global-rounds {args.global_rounds} --global-limit {glimit} "
                 f"--global-margin {args.global_margin} --global-budget {args.global_budget}")
        # 整网回验须重扫隔离候选；分批进度由 per-plan 工件承载（不能靠共享缓存的 tuned 跳过）。
        retune = "--retune"
        loop.append("export INFVINO_GLOBAL_PROGRESS=1")
        # R47: 打开整网回验报告（此前 autotune.py grep 了 'global-retune' 却没 export，
        # 导致回验日志从不出现——可观测性缺陷）。
        loop.append("export INFVINO_GLOBAL_RETUNE_REPORT=1")
        # 新一轮 campaign：清掉旧 per-plan 工件（否则所有节点都被判为「已回验」而全跳过）。
        loop.append(f"rm -f /workspace/infvino/models/{args.model}/model.plan.tuning.json")
    for op in ops:
        loop.append(f"echo '=== autotune {args.model} op={op} (batch={args.batch}) ==='")
        # 最多 64 批的安全上限（远超任何模型签名数）。
        loop.append(
            f"for b in $(seq 1 64); do\n"
            f"  out=$(timeout 200 /workspace/infvino/build-ct/kernel_autotune "
            f"--plan {plan} --cache {cache_in_container} "
            f"--op {op} --limit {args.batch} --iters {args.iters} {retune} {gargs} --expected 2>&1) || {{\n"
            f"    echo \"$out\"; echo '[autotune] batch failed; stop'; exit 1; }}\n"
            f"  echo \"$out\" | grep -E 'expected vs|ratio|wrote|global-retune|WARN' || true\n"
            f"  echo \"$out\" | grep -q '(0 entries this run' && {{ echo '[autotune] op done'; break; }}\n"
            f"  echo \"$out\" | grep -q 'global-retune: NO-OP' && {{ echo \"$out\"; echo '[autotune] --global NO-OP; stop'; exit 4; }}\n"
            f"  if tail -n 300 /var/log/kern.log /var/log/syslog 2>/dev/null | grep -qE 'GPU HANG|engine reset'; then echo '[autotune] GPU HANG'; exit 3; fi\n"
            f"done")
    loop.append("echo '[autotune] all batches complete'")
    # R45 P1#10: --global 的整网回验命令流远多于隔离扫描；跑前锁频、跑后解锁（推荐加 --lock）。
    locked = False
    if args.lock:
        subprocess.run([os.path.join(repo, "scripts", "gpu_clocks.sh"), "lock"], check=False)
        locked = True
    rc = run(dcmd + [
        "-v", f"{repo}:/workspace/infvino", "-w", "/workspace/infvino",
        args.image, "bash", "-lc", "\n".join(loop),
    ])
    if locked:
        subprocess.run([os.path.join(repo, "scripts", "gpu_clocks.sh"), "unlock"], check=False)
    if rc != 0:
        print(f"[autotune] run failed (rc={rc})", file=sys.stderr)
        subprocess.run([os.path.join(repo, "scripts", "gpu_guard.sh"), "after"])
        return rc

    if args.bake:
        baked = f"/workspace/infvino/models/{args.model}/model.plan.baked"
        in_container(
            f"/workspace/infvino/build-ct/kernel_autotune --plan {plan} "
            f"--cache {cache_in_container} --op conv3x3 --limit 0 --iters 1 "
            f"--bake {baked} >/dev/null 2>&1 || true")

    print(f"\n[autotune] done. cache = {args.cache} ({os.path.getsize(cache_abs) if os.path.exists(cache_abs) else 0} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
