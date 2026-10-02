#!/usr/bin/env python3
"""数值检验：自研 kernel（OpenCL）输出 vs numpy FP32 参考。

流程（与 scripts/numerical_check.py 同构）：
  1. 宿主机生成确定性 FP16 输入 A/B（seeded uniform），写 *.bin；
  2. kernel_numtest 在容器内跑自研 kernel，dump FP16 输出；
  3. numpy 用 FP32 计算参考，比对：**相对误差 mean_rel 与 max_rel(amax)**（不使用余弦）。

用法:
  python3 kernel_check.py --repo /path/to/infvino \\
      --image infvino-dev:latest --workdir /tmp/kcheck
"""
import argparse
import os
import subprocess
import sys

import numpy as np

# 相对误差阈值（FP16 kernel vs FP32 参考）：mean 与 amax 两个口径。
MEAN_REL_TOL = 1e-2
MAX_REL_TOL = 5e-2

# 代表性 conv GEMM shape（M=H*W, N=Cout, K=Cin*kh*kw）。
DEFAULT_SHAPES = [
    (6400, 64, 64, "1x1 s8 yolo11 head"),
    (1600, 128, 128, "1x1 s16"),
    (400, 256, 256, "1x1 s32"),
    (1024, 1024, 1024, "square"),
    (4096, 512, 512, "square-large"),
]


def gen_inputs(shape, workdir):
    M, N, K = shape
    rng = np.random.default_rng(0)
    a = rng.random((M, K)).astype(np.float16)
    b = rng.random((K, N)).astype(np.float16)
    pa = os.path.join(workdir, f"a_{M}x{N}x{K}.bin")
    pb = os.path.join(workdir, f"b_{M}x{N}x{K}.bin")
    a.tofile(pa)
    b.tofile(pb)
    ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    return pa, pb, ref


# R29: attention bmm + softmax numeric coverage.
DEFAULT_BMM = [
    (1, 2, 400, 32, 400, "attn QK^T"),
    (1, 2, 64, 400, 400, "attn PV"),
]
BMM_VARIANTS = [
    ("bmm", ""),
    ("bmm2", ""),
    ("bmm_t", "-DBMM_TM=4 -DBMM_TN=8 -DBMM_UK=4"),
    ("bmm_t", "-DBMM_TM=8 -DBMM_TN=4 -DBMM_UK=4"),
]
DEFAULT_SOFTMAX = [
    (800, 400, 1, "attn"),
    (1, 16, 33600, "dfl"),
]
SOFTMAX_VARIANTS = [
    ("softmax_axis", ""),
    ("softmax_axis_r", "-DSM_WGS=64"),
]


def gen_bmm(shape, workdir):
    B0, B1, M, K, N, _ = shape
    rng = np.random.default_rng(0)
    a = rng.random((B0 * B1, M, K)).astype(np.float16)
    b = rng.random((B0 * B1, K, N)).astype(np.float16)
    pa = os.path.join(workdir, f"bmmA_{B0}x{B1}x{M}x{K}x{N}.bin")
    pb = os.path.join(workdir, f"bmmB_{B0}x{B1}x{M}x{K}x{N}.bin")
    a.tofile(pa)
    b.tofile(pb)
    ref = np.einsum("bmk,bkn->bmn", a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    return pa, pb, ref


def gen_softmax(shape, workdir):
    outer, axdim, inner, _ = shape
    rng = np.random.default_rng(0)
    x = rng.standard_normal((outer, axdim, inner)).astype(np.float16)
    px = os.path.join(workdir, f"sm_{outer}x{axdim}x{inner}.bin")
    x.tofile(px)
    xs = x.astype(np.float32)
    mx = xs.max(axis=1, keepdims=True)
    e = np.exp(xs - mx)
    ref = (e / e.sum(axis=1, keepdims=True)).astype(np.float16)
    return px, ref


DEFAULT_DEPTHWISE = [
    (64, 80, 80, 3, 1, 1, 1, "dw3x3 s1"),
    (128, 40, 40, 3, 1, 1, 1, "dw3x3 s1"),
    (16, 112, 112, 3, 2, 1, 0, "dw3x3 s2"),
    (96, 14, 14, 5, 2, 2, 0, "dw5x5 s2"),
]
DW_VARIANTS = [
    ("depthwise_f16", ""),
    ("depthwise_v", "-DDW_TW=4"),
    ("depthwise_v", "-DDW_TW=8"),
]


def _act(a, code):
    if code == 1:
        return a / (1.0 + np.exp(-a))
    if code == 2:
        return a * np.clip(a + 3.0, 0.0, 6.0) / 6.0
    if code == 3:
        return np.maximum(a, 0.0)
    if code == 4:
        return np.clip(a + 3.0, 0.0, 6.0) / 6.0
    return a


def gen_depthwise(shape, workdir):
    C, H, W, K, S, P, act, _ = shape
    rng = np.random.default_rng(0)
    x = rng.standard_normal((C, H, W)).astype(np.float16)
    w = (rng.standard_normal((C, K, K)).astype(np.float16) * 0.2).astype(np.float16)
    b = rng.standard_normal(C).astype(np.float16)
    tag = f"{C}x{H}x{W}x{K}x{S}x{P}x{act}"
    px, pw, pb = (os.path.join(workdir, f"dw{tag}.{ext}") for ext in ("x", "w", "b"))
    x.tofile(px); w.tofile(pw); b.tofile(pb)
    Ho, Wo = (H + 2 * P - K) // S + 1, (W + 2 * P - K) // S + 1
    xp = np.pad(x.astype(np.float32), ((0, 0), (P, P), (P, P)))
    acc = np.zeros((C, Ho, Wo), np.float32)
    for kh in range(K):
        for kw in range(K):
            acc += xp[:, kh:kh + Ho * S:S, kw:kw + Wo * S:S] * w[:, kh, kw].astype(np.float32)[:, None, None]
    ref = _act(acc + b.astype(np.float32)[:, None, None], act).astype(np.float16)
    return px, pw, pb, ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--image", default="infvino-dev:latest")
    ap.add_argument("--workdir", default="/tmp/kcheck")
    ap.add_argument("--tiles", default="", help="GEMM: BM,BN,BK,TM,TN[,VEC2,PAD]")
    ap.add_argument("--conv", default="", help="conv3x3 cfg 覆盖（默认用 C++ 默认值）")
    ap.add_argument("--skip-gemm", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    gemm_jobs, conv_jobs, c1x1_jobs = [], [], []

    if not args.skip_gemm:
        for M, N, K, label in DEFAULT_SHAPES:
            pa, pb, ref = gen_inputs((M, N, K), args.workdir)
            gemm_jobs.append((M, N, K, pa, pb, ref, label))
            print(f"[ref] gemm {M}x{N}x{K:5d} {label}")

    for shape in DEFAULT_CONVS:
        Cin, Cout, H, W, s, p, label = shape[:7]
        cfg = shape[7] if len(shape) > 7 else ""
        px, pw, pb, ref = gen_conv(shape, args.workdir)
        conv_jobs.append((shape, px, pw, pb, ref, cfg))
        print(f"[ref] conv {Cin}x{Cout}x{H}x{W}s{s} {label}")

    for shape in DEFAULT_CONV1X1:
        Cin, Cout, H, W, label = shape
        px, pw, ref = gen_conv1x1(shape, args.workdir)
        c1x1_jobs.append((shape, px, pw, ref))
        print(f"[ref] conv1x1 {Cin}x{Cout}x{H}x{W} {label}")

    gemv_jobs = []
    for Cin, Cout, label in DEFAULT_GEMV:
        shape = (Cin, Cout, 1, 1, label)
        px, pw, ref = gen_conv1x1(shape, args.workdir)
        gemv_jobs.append((shape, px, pw, ref))
        print(f"[ref] conv1x1g {Cin}x{Cout} {label}")

    ov_jobs = []
    for shape in DEFAULT_CONVS_OV:
        Cin, Cout, H, W, s, p, label = shape[:7]
        px, pw, pb, ref = gen_conv(shape, args.workdir)
        ov_jobs.append((shape, px, pw, pb, ref))
        print(f"[ref] conv3x3ov {Cin}x{Cout}x{H}x{W}s{s} {label}")

    bmm_jobs = []
    for shape in DEFAULT_BMM:
        pa, pb, ref = gen_bmm(shape, args.workdir)
        for kern, opts in BMM_VARIANTS:
            bmm_jobs.append((shape, kern, opts, pa, pb, ref))
        print(f"[ref] bmm {shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]} {shape[5]}")

    sm_jobs = []
    for shape in DEFAULT_SOFTMAX:
        px, ref = gen_softmax(shape, args.workdir)
        for kern, opts in SOFTMAX_VARIANTS:
            sm_jobs.append((shape, kern, opts, px, ref))
        print(f"[ref] softmax {shape[0]}x{shape[1]}x{shape[2]} {shape[3]}")

    dw_jobs = []
    for shape in DEFAULT_DEPTHWISE:
        px, pw, pb, ref = gen_depthwise(shape, args.workdir)
        for kern, opts in DW_VARIANTS:
            dw_jobs.append((shape, kern, opts, px, pw, pb, ref))
        print(f"[ref] depthwise {shape[0]}x{shape[1]}x{shape[2]} K{shape[3]} {shape[7]}")

    tiles_arg = f"--tiles {args.tiles}" if args.tiles else ""
    inner = [
        "set -e",
        "cmake -S /workspace/infvino -B /tmp/build -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1",
        "cmake --build /tmp/build -j2 >/tmp/build.log 2>&1",
        "gpu_hang=0",
    ]
    for M, N, K, pa, pb, _, _ in gemm_jobs:
        inner.append(
            f"timeout 30 /tmp/build/kernel_numtest --op gemm --m {M} --n {N} --k {K} {tiles_arg} "
            f"--input-a /work/{os.path.basename(pa)} --input-b /work/{os.path.basename(pb)} "
            f"--dump /work/out_{M}x{N}x{K}.bin || echo RUNFAIL gemm {M}x{N}x{K}"
        )
    for shape, px, pw, pb, _, cfg in conv_jobs:
        Cin, Cout, H, W, s, p, _ = shape[:7]
        tag = f"{Cin}x{Cout}x{H}x{W}s{s}"
        cfg_arg = f"--conv {cfg}" if cfg else (f"--conv {args.conv}" if args.conv else "")
        inner.append(
            f"timeout 30 /tmp/build/kernel_numtest --op conv3x3 --cin {Cin} --cout {Cout} --h {H} --w {W} "
            f"--stride {s} --pad {p} {cfg_arg} "
            f"--input-x /work/{os.path.basename(px)} --input-w /work/{os.path.basename(pw)} "
            f"--input-bias /work/{os.path.basename(pb)} --dump /work/out_conv_{tag}.bin "
            f"|| echo RUNFAIL conv {tag}"
        )
    for shape, px, pw, _ in c1x1_jobs:
        Cin, Cout, H, W, _ = shape
        tag = f"{Cin}x{Cout}x{H}x{W}"
        inner.append(
            f"timeout 30 /tmp/build/kernel_numtest --op conv1x1 --cin {Cin} --cout {Cout} --h {H} --w {W} "
            f"--input-x /work/x1x1_{tag}.bin --input-w /work/w1x1_{tag}.bin "
            f"--dump /work/out_c1x1_{tag}.bin || echo RUNFAIL conv1x1 {tag}"
        )
    for shape, px, pw, _ in gemv_jobs:
        Cin, Cout, H, W, _ = shape
        tag = f"{Cin}x{Cout}x{H}x{W}"
        inner.append(
            f"timeout 30 /tmp/build/kernel_numtest --op conv1x1g --cin {Cin} --cout {Cout} "
            f"--input-x /work/x1x1_{tag}.bin --input-w /work/w1x1_{tag}.bin "
            f"--dump /work/out_gemv_{tag}.bin || echo RUNFAIL conv1x1g {tag}"
        )
    for shape, px, pw, pb, _ in ov_jobs:
        Cin, Cout, H, W, s, p, label = shape[:7]
        tag = f"{Cin}x{Cout}x{H}x{W}s{s}"
        inner.append(
            f"timeout 30 /tmp/build/kernel_numtest --op conv3x3 --ov --cin {Cin} --cout {Cout} --h {H} --w {W} "
            f"--stride {s} --pad {p} "
            f"--input-x /work/{os.path.basename(px)} --input-w /work/{os.path.basename(pw)} "
            f"--input-bias /work/{os.path.basename(pb)} --dump /work/out_convov_{tag}.bin "
            f"|| echo RUNFAIL conv3x3ov {tag}"
        )
    inner.append("if dmesg 2>/dev/null | grep -q 'GPU HANG'; then "
                 "echo '[kernel_check] GPU HANG detected'; exit 3; fi")
    for shape, kern, opts, pa, pb, _ in bmm_jobs:
        B0, B1, M, K, N, _ = shape
        tag = f"{kern}_{B0}x{B1}x{M}x{K}x{N}"
        optarg = f'--opts "{opts}"' if opts else ""
        inner.append(
            f"timeout 30 /tmp/build/kernel_numtest --op bmm --kernel {kern} {optarg} "
            f"--b0 {B0} --b1 {B1} --m {M} --n {N} --k {K} "
            f"--input-a /work/{os.path.basename(pa)} --input-b /work/{os.path.basename(pb)} "
            f"--dump /work/out_{tag}.bin || echo RUNFAIL bmm {tag}"
        )
    for shape, kern, opts, px, _ in sm_jobs:
        outer, axdim, inn, _ = shape
        tag = f"{kern}_{outer}x{axdim}x{inn}"
        optarg = f'--opts "{opts}"' if opts else ""
        inner.append(
            f"timeout 30 /tmp/build/kernel_numtest --op softmax --kernel {kern} {optarg} "
            f"--outer {outer} --axdim {axdim} --inner {inn} "
            f"--input-x /work/{os.path.basename(px)} --dump /work/out_{tag}.bin "
            f"|| echo RUNFAIL softmax {tag}"
        )
    for shape, kern, opts, px, pw, pb, _ in dw_jobs:
        C, H, W, K, S, P, act, _ = shape
        tag = f"{kern}_{C}x{H}x{W}x{K}x{S}x{P}x{act}"
        optarg = f'--opts "{opts}"' if opts else ""
        inner.append(
            f"timeout 30 /tmp/build/kernel_numtest --op depthwise --kernel {kern} {optarg} "
            f"--cin {C} --h {H} --w {W} --dw-k {K} --stride {S} --pad {P} --act {act} "
            f"--input-x /work/{os.path.basename(px)} --input-w /work/{os.path.basename(pw)} "
            f"--input-bias /work/{os.path.basename(pb)} --dump /work/out_{tag}.bin "
            f"|| echo RUNFAIL depthwise {tag}"
        )
    run(["docker", "run", "--rm",
         "--memory=3g", "--memory-swap=3g", "--pids-limit=256",
         "--device=/dev/dri/renderD128",
         "-v", f"{args.repo}:/workspace/infvino", "-w", "/workspace/infvino",
         "-v", f"{os.path.abspath(args.workdir)}:/work",
         args.image, "bash", "-lc", "\n".join(inner)])

    print("\n=== kernel numerical comparison (vs numpy FP32) ===")
    ok_all = True
    for M, N, K, _, _, ref, label in gemm_jobs:
        got = np.fromfile(os.path.join(args.workdir, f"out_{M}x{N}x{K}.bin"),
                          dtype=np.float16).reshape(M, N)
        ok, msg = _cmp(got, ref)
        ok_all &= ok
        print(f"  gemm {M}x{N}x{K:<5d} {label:20s} {msg} -> {'PASS' if ok else 'FAIL'}")
    for shape, _, _, _, ref, _ in conv_jobs:
        Cin, Cout, H, W, s, p, label = shape[:7]
        tag = f"{Cin}x{Cout}x{H}x{W}s{s}"
        path = os.path.join(args.workdir, f"out_conv_{tag}.bin")
        if not os.path.exists(path):
            ok_all = False
            print(f"  conv {Cin}x{Cout}x{H}x{W}s{s} {label:20s} MISSING -> FAIL")
            continue
        got = np.fromfile(path, dtype=np.float16).reshape(ref.shape)
        ok, msg = _cmp(got, ref)
        ok_all &= ok
        print(f"  conv {Cin}x{Cout}x{H}x{W}s{s} {label:20s} {msg} -> {'PASS' if ok else 'FAIL'}")
    for shape, _, _, ref in c1x1_jobs:
        Cin, Cout, H, W, label = shape
        path = os.path.join(args.workdir, f"out_c1x1_{Cin}x{Cout}x{H}x{W}.bin")
        if not os.path.exists(path):
            ok_all = False
            print(f"  conv1x1 {Cin}x{Cout}x{H}x{W} {label:20s} MISSING -> FAIL")
            continue
        got = np.fromfile(path, dtype=np.float16).reshape(ref.shape)
        ok, msg = _cmp(got, ref)
        ok_all &= ok
        print(f"  conv1x1 {Cin}x{Cout}x{H}x{W} {label:20s} {msg} -> {'PASS' if ok else 'FAIL'}")
    for shape, _, _, ref in gemv_jobs:
        Cin, Cout, H, W, label = shape
        path = os.path.join(args.workdir, f"out_gemv_{Cin}x{Cout}x{H}x{W}.bin")
        if not os.path.exists(path):
            ok_all = False
            print(f"  conv1x1g {Cin}x{Cout} {label:20s} MISSING -> FAIL")
            continue
        got = np.fromfile(path, dtype=np.float16).reshape(ref.shape)
        ok, msg = _cmp(got, ref)
        ok_all &= ok
        print(f"  conv1x1g {Cin}x{Cout} {label:20s} {msg} -> {'PASS' if ok else 'FAIL'}")
    for shape, _, _, _, ref in ov_jobs:
        Cin, Cout, H, W, s, p, label = shape[:7]
        tag = f"{Cin}x{Cout}x{H}x{W}s{s}"
        path = os.path.join(args.workdir, f"out_convov_{tag}.bin")
        if not os.path.exists(path):
            ok_all = False
            print(f"  conv3x3ov {Cin}x{Cout}x{H}x{W}s{s} {label:16s} MISSING -> FAIL")
            continue
        got = np.fromfile(path, dtype=np.float16).reshape(ref.shape)
        ok, msg = _cmp(got, ref)
        ok_all &= ok
        print(f"  conv3x3ov {Cin}x{Cout}x{H}x{W}s{s} {label:16s} {msg} -> {'PASS' if ok else 'FAIL'}")
    for shape, kern, opts, _, _, ref in bmm_jobs:
        B0, B1, M, K, N, label = shape
        tag = f"{kern}_{B0}x{B1}x{M}x{K}x{N}"
        path = os.path.join(args.workdir, f"out_{tag}.bin")
        if not os.path.exists(path):
            ok_all = False
            print(f"  bmm {kern:10s} {M}x{N}x{K} B{B0}.{B1} MISSING -> FAIL")
            continue
        got = np.fromfile(path, dtype=np.float16).reshape(ref.shape)
        ok, msg = _cmp(got, ref)
        ok_all &= ok
        print(f"  bmm {kern:10s} {M}x{N}x{K} B{B0}.{B1} {label:12s} {msg} -> {'PASS' if ok else 'FAIL'}")
    for shape, kern, opts, _, ref in sm_jobs:
        outer, axdim, inn, label = shape
        tag = f"{kern}_{outer}x{axdim}x{inn}"
        path = os.path.join(args.workdir, f"out_{tag}.bin")
        if not os.path.exists(path):
            ok_all = False
            print(f"  softmax {kern:16s} {outer}x{axdim}x{inn} MISSING -> FAIL")
            continue
        got = np.fromfile(path, dtype=np.float16).reshape(ref.shape)
        ok, msg = _cmp(got, ref)
        ok_all &= ok
        print(f"  softmax {kern:16s} {outer}x{axdim}x{inn} {label:6s} {msg} -> {'PASS' if ok else 'FAIL'}")
    for shape, kern, opts, _, _, _, ref in dw_jobs:
        C, H, W, K, S, P, act, label = shape
        tag = f"{kern}_{C}x{H}x{W}x{K}x{S}x{P}x{act}"
        path = os.path.join(args.workdir, f"out_{tag}.bin")
        if not os.path.exists(path):
            ok_all = False
            print(f"  depthwise {kern:14s} {C}x{H}x{W} K{K} MISSING -> FAIL")
            continue
        got = np.fromfile(path, dtype=np.float16).reshape(ref.shape)
        ok, msg = _cmp(got, ref)
        ok_all &= ok
        print(f"  depthwise {kern:14s} {C}x{H}x{W} K{K}s{S} {label:10s} {msg} -> {'PASS' if ok else 'FAIL'}")
    print("\nRESULT:", "ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


def _cmp(got, ref):
    """相对误差判据（不使用余弦相似度）：mean_rel 与 max_rel(amax) 双阈值。"""
    g = np.asarray(got, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    diff = np.abs(g - r)
    mean_rel = float(diff.mean() / (np.abs(r).mean() + 1e-12))
    max_rel = float(diff.max() / (np.abs(r).max() + 1e-12))
    ok = (mean_rel < MEAN_REL_TOL) and (max_rel < MAX_REL_TOL)
    return ok, (f"mean_rel={mean_rel:.3e} max_rel(amax)={max_rel:.3e} "
                f"max_abs={float(diff.max()):.3e}")


# 代表性 conv3x3 shape (Cin,Cout,H,W,stride,pad,label[,cfg 覆盖])。
DEFAULT_CONVS = [
    (64, 64, 80, 80, 1, 1, "p3 head 3x3 s1"),
    (32, 64, 80, 80, 1, 1, "c2f 3x3 s1"),
    (64, 64, 40, 40, 1, 1, "p4 3x3 s1"),
    (16, 32, 160, 160, 2, 1, "stem 3x3 s2", "40,8,1,32,8,2,1,0,3,1,16"),
]

# 代表性 conv1x1 shape (Cin,Cout,H,W)。
DEFAULT_CONV1X1 = [
    (64, 64, 80, 80, "c2f 1x1"),
    (384, 128, 40, 40, "neck 1x1"),
    (512, 256, 20, 20, "sppf 1x1"),
]

# 代表性 HW==1 split-K GEMV shape (Cin,Cout) —— mobilenet SE / classifier。
DEFAULT_GEMV = [
    (576, 1024, "classifier.0"),
    (1024, 1000, "classifier.3"),
    (240, 64, "se fc1"),
]

# OpenVINO os_iyx_osv32 port (kernels/conv_ov.cl) 代表性 shape (Cin,Cout,H,W,stride,pad,label)。
DEFAULT_CONVS_OV = [
    (64, 64, 80, 80, 1, 1, "p3 head 3x3 s1 ov"),
    (16, 32, 160, 160, 2, 1, "stem 3x3 s2 ov"),
]


def ref_conv3x3(x, w, b, stride, pad):
    """numpy FP32 参考：x[Cin,H,W] w[Cout,Cin,3,3] b[Cout] -> y[Cout,Hout,Wout]."""
    from numpy.lib.stride_tricks import sliding_window_view
    Cin, H, W = x.shape
    Cout = w.shape[0]
    Hout = (H + 2 * pad - 3) // stride + 1
    Wout = (W + 2 * pad - 3) // stride + 1
    xp = np.pad(x.astype(np.float32), ((0, 0), (pad, pad), (pad, pad)))
    xv = sliding_window_view(xp, (3, 3), axis=(1, 2))[:, ::stride, ::stride]  # [Cin,Hout,Wout,3,3]
    assert xv.shape[1] == Hout and xv.shape[2] == Wout, (xv.shape, Hout, Wout)
    y = np.einsum('oikm,iyxkm->oyx', w.astype(np.float32), xv, optimize=True)
    return (y + b.astype(np.float32)[:, None, None]).astype(np.float16)


def gen_conv1x1(shape, workdir):
    """1x1 conv == GEMM: y[Cout,HW] = W[Cout,Cin] * x[Cin,HW]."""
    Cin, Cout, H, W, _ = shape
    rng = np.random.default_rng(13)
    x = (rng.random((Cin, H, W)) * 2 - 1).astype(np.float16)
    w = ((rng.random((Cout, Cin)) - 0.5) * 0.2).astype(np.float16)
    tag = f"{Cin}x{Cout}x{H}x{W}"
    px = os.path.join(workdir, f"x1x1_{tag}.bin")
    pw = os.path.join(workdir, f"w1x1_{tag}.bin")
    x.tofile(px)
    w.tofile(pw)
    ref = (w.astype(np.float32) @ x.reshape(Cin, -1).astype(np.float32))
    return px, pw, ref.reshape(Cout, H, W).astype(np.float16)


def gen_conv(shape, workdir):
    Cin, Cout, H, W, s, p = shape[:6]
    rng = np.random.default_rng(7)
    x = (rng.random((Cin, H, W)) * 2 - 1).astype(np.float16)
    w = ((rng.random((Cout, Cin, 3, 3)) - 0.5) * 0.2).astype(np.float16)
    b = ((rng.random(Cout) - 0.5) * 0.1).astype(np.float16)
    tag = f"{Cin}x{Cout}x{H}x{W}s{s}"
    px, pw, pb = (os.path.join(workdir, f"{pre}_{tag}.bin") for pre in ("x", "w", "b"))
    x.tofile(px)
    w.tofile(pw)
    b.tofile(pb)
    return px, pw, pb, ref_conv3x3(x, w, b, s, p)


def run(cmd):
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
