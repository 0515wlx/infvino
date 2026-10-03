#!/usr/bin/env python3
"""OpenVINO 对照基线：自动路由到**最新稳定版**，不硬编码版本号。

用法::

    # 只解析当前最新稳定版（查 PyPI，排除预发布）
    python3 scripts/openvino_baseline.py resolve

    # 在隔离 venv 中安装解析出的版本，并对 ONNX 模型跑 GPU 基线
    python3 scripts/openvino_baseline.py run --device GPU

    # 指定版本 / 私有镜像 / 复用已有 venv
    python3 scripts/openvino_baseline.py run --version 2026.4.1
    python3 scripts/openvino_baseline.py run --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    python3 scripts/openvino_baseline.py run --venv .venv-ov --no-install

设计要点：
  * 版本解析始终交给 `resolve_latest_stable()`（PyPI JSON + PEP 440 预发布过滤），
    代码与文档都不得写死版本；`--version` 仅用于复现历史基线。
  * 安装与运行都在独立 venv（默认仓库根 `.venv-ov/`），避免污染开发环境。
  * 运行 GPU 基线时遵守 `docs/benchmark_protocol.md` 的安全约定（默认迭代数很小）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_PYPI_JSON = "https://pypi.org/pypi/openvino/json"
DEFAULT_DEVICES = ["GPU"]
REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# 版本解析（不硬编码）
# --------------------------------------------------------------------------- #
def _is_stable(version: str) -> bool:
    """True 表示 `version` 是稳定版（非 pre-release / dev）。"""
    try:
        from packaging.version import Version  # type: ignore

        return not Version(version).is_prerelease
    except Exception:
        # 无 packaging 时的保守回退：只接受纯数字点分版本。
        import re

        return re.fullmatch(r"\d+(?:\.\d+)*", version) is not None


def _version_key(version: str):
    try:
        from packaging.version import Version  # type: ignore

        return Version(version)
    except Exception:
        import re

        return tuple(int(p) for p in re.findall(r"\d+", version))


def resolve_latest_stable(pypi_json: str = DEFAULT_PYPI_JSON) -> str:
    """返回 OpenVINO 的最新稳定版（排除预发布），从 PyPI 动态解析。"""
    with urllib.request.urlopen(pypi_json, timeout=30) as resp:  # noqa: S310
        data = json.load(resp)
    versions = [v for v in data.get("releases", {}) if _is_stable(v)]
    if not versions:
        raise RuntimeError(f"无法从 {pypi_json} 解析出任何稳定版 OpenVINO")
    return max(versions, key=_version_key)


# --------------------------------------------------------------------------- #
# venv / 安装
# --------------------------------------------------------------------------- #
def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def ensure_venv(venv: Path) -> Path:
    py = _venv_python(venv)
    if not py.exists():
        print(f"[ov-baseline] 创建 venv: {venv}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    return py


def pip_install(py: Path, spec: str, index_url: str | None) -> None:
    cmd = [str(py), "-m", "pip", "install", "--upgrade", spec]
    if index_url:
        cmd += ["--index-url", index_url]
    print(f"[ov-baseline] {py} -m pip install {spec}")
    subprocess.check_call(cmd)


# --------------------------------------------------------------------------- #
# GPU 基线（在装了 openvino 的解释器里执行）
# --------------------------------------------------------------------------- #
def _measure(ov, model_path: Path, device: str, iters: int, warmup: int) -> dict:
    import numpy as np

    core = ov.Core()
    model = core.read_model(str(model_path))

    inp = model.input(0)
    pshape = inp.partial_shape
    if not pshape.is_static:
        raise RuntimeError(
            f"{model_path.name}: 输入 {inp.any_name} 是动态 shape {pshape}，"
            "请在脚本中固定输入尺寸后再跑基线"
        )
    shape = [int(d.get_length()) for d in pshape]
    data = np.random.default_rng(0).random(shape).astype(np.float32)

    compiled = core.compile_model(model, device)
    infer = compiled.create_infer_request()

    for _ in range(warmup):
        infer.infer({inp: data})

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        infer.infer({inp: data})
        times_ms.append((time.perf_counter() - t0) * 1e3)

    times_ms.sort()
    n = len(times_ms)
    return {
        "input_shape": shape,
        "iters": n,
        "mean_ms": sum(times_ms) / n,
        "median_ms": times_ms[n // 2],
        "min_ms": times_ms[0],
    }


def run_bench(args: argparse.Namespace) -> int:
    try:
        import openvino as ov  # type: ignore
    except ImportError:
        print(
            "[ov-baseline] 当前解释器没有 openvino。请用 `run` 子命令（会建 venv 安装），"
            "或先激活装了 openvino 的环境。",
            file=sys.stderr,
        )
        return 2

    device = args.device
    core = ov.Core()
    try:
        full_name = core.get_property(device, "FULL_DEVICE_NAME")
    except Exception:
        full_name = ""

    model_paths = [Path(m) for m in args.models]
    results = {}
    for mp in model_paths:
        if not mp.exists():
            print(f"[ov-baseline] 跳过不存在的模型: {mp}", file=sys.stderr)
            continue
        stats = _measure(ov, mp, device, args.iters, args.warmup)
        results[mp.stem] = stats
        print(
            f"[ov-baseline] {mp.stem:24s} mean={stats['mean_ms']:8.3f} ms "
            f"median={stats['median_ms']:8.3f} ms min={stats['min_ms']:8.3f} ms"
        )

    payload = {
        "openvino_version": ov.__version__,
        "device": device,
        "device_full_name": full_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[ov-baseline] 写入 {out}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_models() -> list[str]:
    found = sorted(str(p) for p in (REPO_ROOT / "models").glob("*.onnx"))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_res = sub.add_parser("resolve", help="解析最新稳定版并打印")
    p_res.add_argument("--pypi-json", default=DEFAULT_PYPI_JSON)
    p_res.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_run = sub.add_parser("run", help="解析/安装 OpenVINO 并跑 GPU 基线")
    p_run.add_argument("--version", default=None, help="覆盖解析结果（复现历史基线用）")
    p_run.add_argument("--index-url", default=None, help="pip/ PyPI 镜像地址")
    p_run.add_argument("--pypi-json", default=DEFAULT_PYPI_JSON)
    p_run.add_argument("--venv", default=str(REPO_ROOT / ".venv-ov"))
    p_run.add_argument("--no-install", action="store_true", help="只用当前环境，不建 venv")
    p_run.add_argument("--device", default="GPU")
    p_run.add_argument("--models", nargs="*", default=None)
    p_run.add_argument("--iters", type=int, default=20)
    p_run.add_argument("--warmup", type=int, default=5)
    p_run.add_argument("--output", default=str(REPO_ROOT / "build-ct" / "openvino-baseline.json"))

    # 内部：在目标解释器里真正跑基线
    p_inner = sub.add_parser("_bench", help=argparse.SUPPRESS)
    p_inner.add_argument("--device", default="GPU")
    p_inner.add_argument("--models", nargs="*", required=True)
    p_inner.add_argument("--iters", type=int, default=20)
    p_inner.add_argument("--warmup", type=int, default=5)
    p_inner.add_argument("--output", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "resolve":
        ver = resolve_latest_stable(args.pypi_json)
        if args.json:
            print(json.dumps({"latest_stable": ver}))
        else:
            print(ver)
        return 0

    if args.cmd == "_bench":
        return run_bench(args)

    # run
    version = args.version or resolve_latest_stable(args.pypi_json)
    print(f"[ov-baseline] OpenVINO 目标版本（最新稳定）: {version}")
    models = args.models or _default_models()
    if not models:
        print(
            "[ov-baseline] 未找到 ONNX 模型。先运行 scripts/export_models.py 或 --models 指定。",
            file=sys.stderr,
        )
        return 2

    if args.no_install:
        py = Path(sys.executable)
    else:
        venv = Path(args.venv)
        py = ensure_venv(venv)
        pip_install(py, f"openvino=={version}", args.index_url)

    cmd = [
        str(py),
        str(Path(__file__).resolve()),
        "_bench",
        "--device",
        args.device,
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--output",
        args.output,
        "--models",
        *models,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
