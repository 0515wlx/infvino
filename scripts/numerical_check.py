#!/usr/bin/env python3
"""数值检验：infvino 库后端 (ClBackend / 自研 OpenCL kernel) 输出 vs onnxruntime 参考。

在 **宿主机**运行（需 numpy + onnxruntime，且能用 docker 调起镜像）。
流程：
  1. 为每个模型生成确定性输入 (seeded uniform, f32)，写 input.bin；
  2. 用 onnxruntime(CPU, FP32) 计算参考输出 -> ref .npy；
  3. 需要时先用 onnx2plan 生成 plan；
  4. 在容器内构建并运行 infvino_numtest，dump 库后端输出 -> .bin；
  5. 比对：**统一用相对误差**（mean 与 amax 两个口径），不使用余弦相似度。
     后端为 FP16 kernel，阈值按 FP16 放宽。

用法：
  python3 numerical_check.py --repo /path/to/infvino \
      --image infvino-dev:latest --workdir /tmp/numtest
"""
import argparse
import os
import subprocess
import sys

import numpy as np

MODELS = [
    # name, onnx, shape
    ("yolov8n-pose",      "yolov8n-pose.onnx",      (1, 3, 640, 640)),
    ("yolo11n-pose",      "yolo11n-pose.onnx",      (1, 3, 640, 640)),
    ("mobilenetv3-small", "mobilenetv3-small.onnx", (1, 3, 224, 224)),
]

MEAN_REL_TOL, MAX_REL_TOL = 2e-2, 5e-2


def rel_error_stats(got, ref):
    """相对误差（不使用余弦相似度）：
       mean_rel = mean|got-ref| / mean|ref|
       max_rel  = max|got-ref|  / max|ref|   (amax 口径)
    """
    g = np.asarray(got, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    diff = np.abs(g - r)
    mean_rel = float(diff.mean() / (np.abs(r).mean() + 1e-12))
    max_rel = float(diff.max() / (np.abs(r).max() + 1e-12))
    return mean_rel, max_rel, float(diff.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--image", default="infvino-dev:latest")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--workdir", default="/tmp/numtest")
    ap.add_argument("--mean-rel", type=float, default=MEAN_REL_TOL)
    ap.add_argument("--max-rel", type=float, default=MAX_REL_TOL)
    args = ap.parse_args()

    import onnxruntime as ort

    os.makedirs(args.workdir, exist_ok=True)
    models_dir = os.path.abspath(args.models_dir)
    repo = os.path.abspath(args.repo)

    jobs = []
    for name, onnx, shape in MODELS:
        path = os.path.join(models_dir, onnx)
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        rng = np.random.default_rng(0)
        inp = rng.random(shape).astype(np.float32)
        inp.tofile(os.path.join(args.workdir, f"in_{name}.bin"))

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        refs = sess.run(None, {sess.get_inputs()[0].name: inp})
        for i, r in enumerate(refs):
            np.save(os.path.join(args.workdir, f"ref_{name}_out{i}.npy"), r)

        # 保证 plan 存在
        plan = os.path.join(models_dir, name, "model.plan")
        if not os.path.exists(plan):
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "onnx2plan.py"),
                            "--onnx", path, "--out-dir", os.path.join(models_dir, name)], check=True)

        jobs.append((name, len(refs)))
        print(f"[ref] {name}: onnxruntime outputs " + ", ".join(str(r.shape) for r in refs))

    # 容器内一次构建 + 跑所有模型的库后端
    inner = [
        "set -e",
        "cmake -S /workspace/infvino -B /tmp/build -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1",
        "cmake --build /tmp/build -j2 >/tmp/build.log 2>&1",
    ]
    for name, _ in jobs:
        inner.append(
            f"timeout 60 /tmp/build/infvino_numtest --config /workspace/infvino/config/models.yaml --key {name} "
            f"--input /work/in_{name}.bin --dump /work/lib_{name} "
            f"> /work/log_{name}.txt 2>&1 || echo 'RUNFAIL {name}'")
    inner.append("if dmesg 2>/dev/null | grep -q 'GPU HANG'; then "
                 "echo '[numerical_check] GPU HANG detected'; exit 3; fi")
    subprocess.run(["docker", "run", "--rm", "--memory=3g", "--memory-swap=3g",
                    "--pids-limit=256", "--device=/dev/dri/renderD128",
                    "-v", f"{repo}:/workspace/infvino", "-w", "/workspace/infvino",
                    "-v", f"{os.path.abspath(args.workdir)}:/work",
                    args.image, "bash", "-lc", "\n".join(inner)], check=True)

    print("\n=== numerical comparison (vs onnxruntime FP32) ===")
    print("  相对误差定义: mean_rel = mean|got-ref| / mean|ref| ; "
          "max_rel(amax) = max|got-ref| / max|ref|")
    print("  FP16 kernel: mean_rel<%.0e 且 max_rel<%.0e" % (args.mean_rel, args.max_rel))
    all_pass = True
    for name, nout in jobs:
        for i in range(nout):
            ref = np.load(os.path.join(args.workdir, f"ref_{name}_out{i}.npy"))
            f = os.path.join(args.workdir, f"lib_{name}_out{i}.bin")
            if not os.path.exists(f):
                print(f"  {name:20s} out{i}: MISSING ({f})")
                all_pass = False
                continue
            got = np.fromfile(f, dtype=np.float32).reshape(ref.shape)
            mean_rel, max_rel, max_abs = rel_error_stats(got, ref)
            ok = (mean_rel < args.mean_rel) and (max_rel < args.max_rel)
            all_pass &= ok
            print(f"  {name:20s} out{i}: mean_rel={mean_rel:.3e} "
                  f"max_rel(amax)={max_rel:.3e} max_abs={max_abs:.3e} "
                  f"[mean<{args.mean_rel:.0e},max<{args.max_rel:.0e}] -> {'PASS' if ok else 'FAIL'}")
    print("\nRESULT:", "ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
