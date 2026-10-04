#!/usr/bin/env python3
"""Phase 3: 优化消融矩阵 —— 用 busy/net/e2e 三态量化「每个优化部件买到了多少」。

对一组已有的运行时开关（环境变量）逐项做 A/B：同一容器、同一输入，分别跑
`infvino_bench`（net/e2e）与 `kernel_run --report --profile-json`（busy），
输出 Δbusy / Δnet / Δe2e / Δfps。用于回答「某个优化（自动调优 / 融合 / 布局 /
内存池 / launch 缓存）到底贡献了多少」。

**不测 OpenVINO**：标尺是自身 baseline 与硬件极限（见 docs/profiling-budget.md）。

安全：遵守 docs/benchmark_protocol.md —— 每个开关一个独立容器、renderD128、
内存/pid 限制、每条 GPU 命令 timeout、跑后检查 GPU HANG。

用法:
  python3 scripts/profile_ablation.py --repo $PWD --image infvino-dev:latest \
      --models yolov8n-pose yolo11n-pose mobilenetv3-small \
      --iters 40 --out /tmp/opencode/ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_budget import parse_bench, parse_profile_json  # noqa: E402

# 运行时开关消融表：(显示名, 环境变量, 说明)
TOGGLES = [
    ("baseline", {}, "无额外开关"),
    ("tuning-off", {"INFVINO_TUNING": "off"}, "关闭自动调优（走内置启发式）"),
    ("no-fuse-res", {"INFVINO_NO_FUSE_RES": "1"}, "关闭残差融合（R33）"),
    ("no-block-layout", {"INFVINO_NO_BLOCK_LAYOUT": "1"}, "关闭 R36 持久 blocked 布局"),
    ("no-reorder-dedup", {"INFVINO_NO_REORDER_DEDUP": "1"}, "关闭 R36 同帧重排去重"),
    ("no-launch-cache", {"INFVINO_NO_LAUNCH_CACHE": "1"}, "关闭 P2 每节点 dispatch 缓存"),
    ("no-pool", {"INFVINO_NO_POOL": "1"}, "关闭 P0 激活内存池（每张量一块）"),
    ("dw-pad", {"INFVINO_DW_PAD": "1"}, "启用 padded depthwise 候选（需 retune 才有意义）"),
]


def gpu_guard(repo: Path, cmd: list[str], timeout: int) -> int:
    """check → timeout 包裹执行 → after（出现 HANG 返回 3）。"""
    guard = str(repo / "scripts" / "gpu_guard.sh")
    if os.path.exists(guard):
        subprocess.run([guard, "check"], check=False)
    try:
        rc = subprocess.run(["timeout", str(timeout), *cmd]).returncode
    except FileNotFoundError:
        rc = subprocess.run(cmd).returncode
    if os.path.exists(guard):
        subprocess.run([guard, "after"], check=False)
    return rc


def run_toggle(repo: Path, image: str, workdir: Path, name: str, env: dict,
               models: list[str], iters: int, iters_report: int, timeout: int) -> bool:
    out = workdir / name
    out.mkdir(parents=True, exist_ok=True)
    inner = ["export LD_LIBRARY_PATH=$PWD/build"]
    for k in models:
        inner.append(f"timeout 90 ./build/infvino_bench --config config/models.yaml --key {k} "
                     f"--iters {iters} > /out/{name}/bench_{k}.txt 2>&1")
        inner.append(f"timeout 60 ./build/kernel_run --plan models/{k}/model.plan --report "
                     f"--iters {iters_report} --profile-json /out/{name}/prof_{k}.json "
                     f"--profile-label {k} > /out/{name}/kr_{k}.txt 2>&1")
    docker = ["docker", "run", "--rm", "--memory=2g", "--memory-swap=2g", "--pids-limit=256",
              "--device=/dev/dri/renderD128",
              "-v", f"{repo}:/workspace/infvino", "-v", f"{workdir}:/out",
              "-w", "/workspace/infvino"]
    for key, val in env.items():
        docker += ["-e", f"{key}={val}"]
    docker += [image, "bash", "-lc", "\n".join(inner)]
    print(f"[ablation] {name}: {env or '(none)'}", flush=True)
    rc = gpu_guard(repo, docker, timeout)
    if rc != 0:
        print(f"[ablation] WARN: {name} exit={rc}", file=sys.stderr)
    return rc == 0


def collect(workdir: Path, name: str, models: list[str]) -> dict:
    res = {}
    for k in models:
        row = {}
        bf = workdir / name / f"bench_{k}.txt"
        if bf.exists():
            st = parse_bench(bf.read_text())
            row["e2e_ms"] = st.get("e2e_ms")
            row["net_ms"] = st.get("net_ms")
        pf = workdir / name / f"prof_{k}.json"
        if pf.exists() and pf.stat().st_size > 0:
            try:
                rep = parse_profile_json(str(pf))
                row["busy_ms"] = rep.get("busy_ms")
                row["structural"] = rep.get("structural", {})
            except Exception as e:  # noqa: BLE001
                print(f"[ablation] parse {pf} failed: {e}", file=sys.stderr)
        res[k] = row
    return res


def delta(base: dict, cur: dict) -> dict:
    out = {}
    for key in ("busy_ms", "net_ms", "e2e_ms"):
        b, c = base.get(key), cur.get(key)
        if b and c:
            out[key] = c - b
            out[key + "_pct"] = 100.0 * (c - b) / b
    if cur.get("e2e_ms"):
        out["fps"] = 1000.0 / cur["e2e_ms"]
    if base.get("e2e_ms") and cur.get("e2e_ms"):
        out["fps_pct"] = (base["e2e_ms"] / cur["e2e_ms"] - 1.0) * 100.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--image", default="infvino-dev:latest")
    ap.add_argument("--models", nargs="*", default=["yolov8n-pose", "yolo11n-pose", "mobilenetv3-small"])
    ap.add_argument("--iters", type=int, default=40, help="infvino_bench 迭代数")
    ap.add_argument("--iters-report", type=int, default=20, help="kernel_run 迭代数")
    ap.add_argument("--workdir", default="/tmp/opencode/ablation")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--toggles", nargs="*", default=None, help="只跑指定开关名（默认全部）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo, workdir = Path(args.repo).resolve(), Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    toggles = [t for t in TOGGLES if not args.toggles or t[0] in args.toggles]

    raw = {}
    for name, env, _desc in toggles:
        run_toggle(repo, args.image, workdir, name, env, args.models, args.iters,
                   args.iters_report, args.timeout)
        raw[name] = collect(workdir, name, args.models)

    base = raw.get("baseline", {})
    table = {"baseline": base, "delta": {}}
    for name, _env, desc in toggles:
        if name == "baseline":
            continue
        table["delta"][name] = {"desc": desc,
                                "models": {k: delta(base.get(k, {}), raw[name].get(k, {}))
                                           for k in args.models}}

    # ---- 打印归因表 ----
    print("\n=== 优化消融矩阵（Δ = 该开关关闭 − baseline；正=关闭后更慢=该优化有收益）===")
    for name, _env, desc in toggles:
        if name == "baseline":
            continue
        print(f"\n## {name}  ({desc})")
        print(f"  {'model':20s} {'Δbusy':>9s} {'Δnet':>9s} {'Δe2e':>9s} {'Δfps':>8s}")
        for k in args.models:
            d = table["delta"][name]["models"][k]
            def f(x, pct=None):
                if x is None:
                    return "  n/a"
                s = f"{x:+.3f}"
                return s + (f"({pct:+.1f}%)" if pct is not None else "")
            print(f"  {k:20s} {f(d.get('busy_ms'), d.get('busy_ms_pct')):>9s}"
                  f" {f(d.get('net_ms'), d.get('net_ms_pct')):>9s}"
                  f" {f(d.get('e2e_ms'), d.get('e2e_ms_pct')):>9s}"
                  f" {f(d.get('fps_pct')):>8s}")

    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "image": args.image,
               "iters": args.iters, "results": raw, "attribution": table["delta"]}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"\n[ablation] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
