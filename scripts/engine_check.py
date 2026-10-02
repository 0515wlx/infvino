#!/usr/bin/env python3
"""引擎级检验：验证 **预处理 + ClBackend** 与 onnxruntime 参考一致。

与 numerical_check.py 的区别：这里喂的是**图片**，走与引擎完全一致的预处理
（letterbox/mean/std），因此同时覆盖 Preprocessor。

流程：
  1. 生成确定性 BGR 图片，保存 PNG；
  2. numpy/cv2 复刻 Preprocessor -> blob，onnxruntime 参考；
  3. 容器内 infvino_numtest --image 跑引擎，dump 原始输出；
  4. 相对误差比对（FP16 阈值）。

用法:
  python3 engine_check.py --repo $PWD --image infvino-dev:latest
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import yaml


def rel_error_stats(got, ref):
    g = np.asarray(got, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    diff = np.abs(g - r)
    return (float(diff.mean() / (np.abs(r).mean() + 1e-12)),
            float(diff.max() / (np.abs(r).max() + 1e-12)),
            float(diff.max()))


def preprocess(bgr, cfg):
    """复刻 infvino Preprocessor（src/Preprocess.cpp）。"""
    iw, ih = cfg["imgsz"]
    letterbox = cfg.get("letterbox", True)
    if letterbox:
        scale = min(iw / bgr.shape[1], ih / bgr.shape[0])
        rw, rh = int(round(bgr.shape[1] * scale)), int(round(bgr.shape[0] * scale))
        pw, ph = iw - rw, ih - rh
        pl, pt = pw // 2, ph // 2
        resized = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.copyMakeBorder(resized, pt, ph - pt, pl, pw - pl,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
    else:
        canvas = cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)

    x = canvas.astype(np.float32)
    if cfg.get("normalize", True):
        x = x / 255.0
    if cfg.get("to_rgb", True):
        x = x[..., ::-1]
    blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # NCHW

    mean = cfg.get("mean")
    std = cfg.get("std")
    if mean or std:
        for c in range(blob.shape[1]):
            m = mean[c] if mean and c < len(mean) else 0.0
            s = std[c] if std and c < len(std) else 1.0
            if s == 0:
                s = 1.0
            blob[0, c] = (blob[0, c] - m) / s
    return blob


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--keys", nargs="*", default=["yolov8n-pose", "yolo11n-pose", "mobilenetv3-small"])
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--image", default="infvino-dev:latest")
    ap.add_argument("--workdir", default="/tmp/enginecheck")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--mean-rel", type=float, default=2e-2)
    ap.add_argument("--max-rel", type=float, default=5e-2)
    args = ap.parse_args()

    import onnxruntime as ort

    os.makedirs(args.workdir, exist_ok=True)
    repo = os.path.abspath(args.repo)
    cfg_all = yaml.safe_load(open(args.config))

    # 确定性图片（uint8 BGR）
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)
    img_path = os.path.join(args.workdir, "engine_input.png")
    cv2.imwrite(img_path, img)

    jobs = []
    for key in args.keys:
        mcfg = dict(cfg_all["models"][key])
        onnx_path = os.path.join(repo, mcfg["path"])
        if not os.path.exists(onnx_path):
            print(f"[skip] {key}: {onnx_path} not found")
            continue
        plan_path = os.path.join(repo, mcfg["plan"])
        if not os.path.exists(plan_path):
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "onnx2plan.py"),
                            "--onnx", onnx_path, "--out-dir", os.path.dirname(plan_path)], check=True)

        blob = preprocess(img, mcfg)
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ref = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
        np.save(os.path.join(args.workdir, f"eng_ref_{key}.npy"), ref)
        jobs.append((key, ref.shape))
        print(f"[ref] {key}: blob{blob.shape} -> out{ref.shape}")

    inner = [
        "set -e",
        "cmake -S /workspace/infvino -B /tmp/build -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1",
        "cmake --build /tmp/build -j2 >/tmp/build.log 2>&1",
    ]
    for key, _ in jobs:
        inner.append(
            f"timeout 60 /tmp/build/infvino_numtest --config /workspace/infvino/config/models.yaml --key {key} "
            f"--image /work/engine_input.png --dump /work/eng_{key} "
            f"> /work/eng_log_{key}.txt 2>&1 || echo 'RUNFAIL {key}'")
    inner.append("if dmesg 2>/dev/null | grep -q 'GPU HANG'; then "
                 "echo '[engine_check] GPU HANG detected'; exit 3; fi")
    subprocess.run(["docker", "run", "--rm", "--memory=3g", "--memory-swap=3g",
                    "--pids-limit=256", "--device=/dev/dri/renderD128",
                    "-v", f"{repo}:/workspace/infvino", "-w", "/workspace/infvino",
                    "-v", f"{os.path.abspath(args.workdir)}:/work",
                    args.image, "bash", "-lc", "\n".join(inner)], check=True)

    print("\n=== engine check (Preprocess + ClBackend vs onnxruntime) ===")
    all_pass = True
    for key, shape in jobs:
        ref = np.load(os.path.join(args.workdir, f"eng_ref_{key}.npy"))
        f = os.path.join(args.workdir, f"eng_{key}_out0.bin")
        if not os.path.exists(f):
            print(f"  {key:20s} MISSING ({f})")
            all_pass = False
            continue
        got = np.fromfile(f, dtype=np.float32).reshape(ref.shape)
        mean_rel, max_rel, max_abs = rel_error_stats(got, ref)
        ok = (mean_rel < args.mean_rel) and (max_rel < args.max_rel)
        all_pass &= ok
        print(f"  {key:20s} mean_rel={mean_rel:.3e} max_rel(amax)={max_rel:.3e} "
              f"max_abs={max_abs:.3e} -> {'PASS' if ok else 'FAIL'}")
    print("\nRESULT:", "ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
