#!/usr/bin/env python3
"""模型端到端数值检验 + ops/EU/cyc（自研 OpenCL kernel）。

流程：
  1. 生成确定性输入（fp16 舍入后的同一份数据喂 onnxruntime 与自研 kernel）；
  2. onnxruntime(CPU,FP32) 参考；
  3. onnx2plan 生成 plan，kernel_run 在容器内跑自研 kernel；
  4. **相对误差** mean/amax 判据；并从 kernel_run --report 读总时间算 ops/EU/cyc。

用法:
  python3 model_check.py --model yolov8n-pose --repo $PWD --image infvino-dev:latest
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import onnx
from onnx import shape_inference

EU, CLOCK_GHZ = 80, 1.3
MEAN_REL_TOL, MAX_REL_TOL = 2e-2, 5e-2

DEFAULT_MODELS = ["yolov8n-pose", "yolo11n-pose", "mobilenetv3-small"]


def conv_flops(onnx_path):
    m = shape_inference.infer_shapes(onnx.load(onnx_path))
    shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
              for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
    inits = {i.name: i for i in m.graph.initializer}
    total = 0
    for n in m.graph.node:
        if n.op_type == "Conv":
            w = inits.get(n.input[1])
            if w is None:
                continue
            cout, cin_g, kh, kw = list(w.dims)
            o = shapes.get(n.output[0])
            if o and len(o) == 4:
                total += 2 * cout * o[2] * o[3] * cin_g * kh * kw
        elif n.op_type == "Gemm":
            a, b = shapes.get(n.input[0]), shapes.get(n.input[1])
            if a and b:
                total += 2 * a[-1] * b[0]
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-pose")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--image", default="infvino-dev:latest")
    ap.add_argument("--workdir", default="/tmp/mcheck")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    onnx_path = os.path.join(args.models_dir, args.model + ".onnx")
    if not os.path.isabs(onnx_path):
        onnx_path = os.path.abspath(onnx_path)
    wd = args.workdir
    plan_dir = os.path.join(wd, args.model)
    os.makedirs(plan_dir, exist_ok=True)

    # 1) deterministic input (fp16) + onnxruntime reference
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in in_meta.shape]
    rng = np.random.default_rng(0)
    x16 = rng.random(shape).astype(np.float16)
    xin = os.path.join(wd, "input.bin")
    x16.tofile(xin)
    ref = sess.run(None, {in_meta.name: x16.astype(np.float32)})[0]
    np.save(os.path.join(wd, "ref.npy"), ref)
    print(f"[ref] {args.model} input{shape} -> output{ref.shape}")

    # 2) plan
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "onnx2plan.py"),
                    "--onnx", onnx_path, "--out-dir", plan_dir], check=True)

    # 3) run in container (cmake build + kernel_run)
    # Safety (docs/benchmark_protocol.md): only the render node, no --privileged,
    # memory + pid limits, and every GPU command wrapped in `timeout` so a wedged
    # kernel cannot pin the host. `gpu_guard` checks dmesg for a GPU HANG and aborts.
    inner = [
        "set -e",
        "cmake -S /workspace/infvino -B /tmp/build -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1",
        "cmake --build /tmp/build -j2 >/tmp/build.log 2>&1",
        f"timeout 60 /tmp/build/kernel_run --plan /work/{args.model}/model.plan "
        f"--input /work/input.bin --output /work/out.bin --iters {args.iters} --report "
        "| tee /work/krun.log",
        "if dmesg 2>/dev/null | grep -q 'GPU HANG'; then echo '[model_check] GPU HANG detected'; exit 3; fi",
    ]
    subprocess.run(["docker", "run", "--rm",
                    "--memory=3g", "--memory-swap=3g", "--pids-limit=256",
                    "--device=/dev/dri/renderD128",
                    "-v", f"{args.repo}:/workspace/infvino", "-w", "/workspace/infvino",
                    "-v", f"{os.path.abspath(wd)}:/work", args.image, "bash", "-lc",
                    "\n".join(inner)], check=True, capture_output=False)

    # 解析 kernel_run 报告的 "total kernel time"
    kernel_total_ms = 0.0
    if os.path.exists(os.path.join(wd, "krun.log")):
        for line in open(os.path.join(wd, "krun.log")):
            m = re.search(r"total kernel time:\s*([\d.]+)\s*ms", line)
            if m:
                kernel_total_ms = float(m.group(1))

    # 4) compare
    got = np.fromfile(os.path.join(wd, "out.bin"), dtype=np.float16).reshape(ref.shape)
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    diff = np.abs(g - r)
    mean_rel = float(diff.mean() / (np.abs(r).mean() + 1e-12))
    max_rel = float(diff.max() / (np.abs(r).max() + 1e-12))
    ok = (mean_rel < MEAN_REL_TOL) and (max_rel < MAX_REL_TOL)
    print(f"\n=== {args.model} end-to-end (相对误差) ===")
    print(f"  mean_rel={mean_rel:.3e} max_rel(amax)={max_rel:.3e} "
          f"max_abs={float(diff.max()):.3e} -> {'PASS' if ok else 'FAIL'}")

    # 5) ops/EU/cyc（单流，kernel 自身时间之和）
    gf = conv_flops(onnx_path) / 1e9
    total_ms = kernel_total_ms
    if total_ms > 0:
        fps = 1000.0 / total_ms
        oeuc = gf * 1e9 * fps / (EU * CLOCK_GHZ * 1e9)
        print(f"  conv+gemm GFLOPs={gf:.3f}  kernel total={total_ms:.3f} ms  "
              f"-> {1000.0/total_ms:.1f} fps  ops/EU/cyc={oeuc:.2f} ({oeuc/16*100:.1f}% of 16)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
