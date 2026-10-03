#!/usr/bin/env python3
# Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
#
# 跨推理一致性检查（P0/P1 回归护栏）。
#
# 目的：抓「把每帧都会变的激活错误地缓存成只算一次」这类 bug——单输入重复跑的测试
# 永远发现不了（历史上的 blkInput 缓存陈旧 bug 就是这样漏掉的）。
#
# 方法：同一进程先喂 A 再喂 B，dump 第二帧(B)；另起一个全新进程只喂 B，dump；
#       两者应一致（差异仅来自 fp16 输出舍入）。
#
# 用法:
#   python3 scripts/reuse_check.py --model yolov8n-pose --repo $PWD --image infvino-dev:latest
import argparse
import os
import subprocess
import sys

import numpy as np


def gen_input(path, shape, seed):
    rng = np.random.default_rng(seed)
    rng.random(shape).astype(np.float16).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-pose")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--image", default="infvino-dev:latest")
    ap.add_argument("--workdir", default="/tmp/reuse_check")
    args = ap.parse_args()

    wd = args.workdir
    os.makedirs(wd, exist_ok=True)
    onnx_path = os.path.join(args.models_dir, args.model + ".onnx")
    # 推断输入尺寸
    import onnx
    from onnx import shape_inference
    m = shape_inference.infer_shapes(onnx.load(onnx_path))
    dims = m.graph.input[0].type.tensor_type.shape.dim
    shape = [x.dim_value if x.dim_value > 0 else 1 for x in dims]
    gen_input(os.path.join(wd, "A.bin"), shape, 1)
    gen_input(os.path.join(wd, "B.bin"), shape, 2)

    inner = [
        "set -e",
        "cmake -S /workspace/infvino -B /tmp/b -DCMAKE_BUILD_TYPE=Release >/tmp/c.log 2>&1",
        "cmake --build /tmp/b -j2 --target reuse_check kernel_run >/tmp/b.log 2>&1",
        "export LD_LIBRARY_PATH=/tmp/b",
        f"timeout 60 /tmp/b/reuse_check --config /workspace/infvino/config/models.yaml "
        f"--key {args.model} --a /work/A.bin --b /work/B.bin --dump-b /work/reuse_B.bin",
        f"timeout 60 /tmp/b/kernel_run --plan /work/{args.model}/model.plan "
        f"--kernel-dir /workspace/infvino/kernels --input /work/B.bin --output /work/single_B.bin --iters 1",
        "if dmesg 2>/dev/null | grep -q 'GPU HANG'; then exit 3; fi",
    ]
    # 需要 plan
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "onnx2plan.py"),
                    "--onnx", onnx_path, "--out-dir", os.path.join(wd, args.model)], check=True)

    subprocess.run(["docker", "run", "--rm", "--memory=3g", "--memory-swap=3g", "--pids-limit=256",
                    "--device=/dev/dri/renderD128",
                    "-v", f"{os.path.abspath(args.repo)}:/workspace/infvino", "-w", "/workspace/infvino",
                    "-v", f"{os.path.abspath(wd)}:/work", args.image, "bash", "-lc",
                    "\n".join(inner)], check=True)

    reuse = np.fromfile(os.path.join(wd, "reuse_B.bin"), dtype=np.float32)
    single = np.fromfile(os.path.join(wd, "single_B.bin"), dtype=np.float16).astype(np.float32)
    if reuse.shape != single.shape:
        print(f"shape mismatch {reuse.shape} vs {single.shape} -> FAIL")
        return 1
    d = np.abs(reuse - single)
    scale = np.abs(single).max() + 1e-9
    rel = d.mean() / scale
    print(f"\n=== {args.model} 跨推理一致性 ===")
    print(f"  2nd-frame(A->B) vs fresh-proc(B): mean={d.mean():.4e} max={d.max():.4e} "
          f"scale_rel={rel:.3e} -> {'PASS' if rel < 5e-3 else 'FAIL'}")
    return 0 if rel < 5e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
