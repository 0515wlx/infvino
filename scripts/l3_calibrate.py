#!/usr/bin/env python3
# Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
#
# l3_calibrate.py —— L3 替换策略等价模型的**自标定工具**（基础设施）。
#
# 目的：把本机 GPU 上**自行实测**的「保留率 vs 流式污染足迹 vs 重用次数」曲线，
# 拟合成 L3Model 所需的等价工程模型参数，并写出 `config/l3_calibration.json`。
#
# 重要声明（provenance）：
#   * 本工具只消费**本仓库自研微基准**（`kernel_bench --op l3retain`）的实测输出；
#   * 不读取、不使用、不引用任何芯片厂商的内部文档 / 未公开寄存器规格；
#   * 产出的模型是**行为等价**的工程近似（用于编译器成本函数），**不是**对具体微架构
#     实现（tag RAM、way 选择逻辑、替换位）的复刻或断言。见 THIRD_PARTY_NOTICES.md。
#
# 用法：
#   # 1) 采集（可用 docker，锁频见 scripts/gpu_clocks.sh）
#   ./build/kernel_bench --op l3retain --hot-lines 16384 --hot-gws 8192 \
#       --hot-sweep 1,4,16 --wi 1024,2048,4096,8192 --fp 4,8,16,24,32,48 > /tmp/l3retain.csv
#   # 2) 拟合
#   python3 scripts/l3_calibrate.py --raw /tmp/l3retain.csv --out config/l3_calibration.json
#   # 或直接由本脚本调用采集命令（把完整命令放在 --run 之后）：
#   python3 scripts/l3_calibrate.py --run \
#       ./build/kernel_bench --op l3retain --hot-lines 16384 --hot-gws 8192 \
#       --hot-sweep 1,4,16 --wi 1024,2048,4096,8192 --fp 4,8,16,24,32,48

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HOT_SET_BYTES_DEFAULT = 16384 * 64          # 1 MB hot set used by the probe
PHYS_L3_SYSFS = "/sys/devices/system/cpu/cpu0/cache/index3/size"


def read_physical_l3_bytes():
    """Return the CPU's L3 size (which equals the SoC GPU L3 on this machine)."""
    try:
        with open(PHYS_L3_SYSFS) as f:
            m = re.match(r"\s*(\d+)\s*([KMG])?", f.read().strip(), re.I)
        if not m:
            return None
        v = int(m.group(1))
        unit = (m.group(2) or "").upper()
        return v * {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[unit]
    except OSError:
        return None


def parse_raw(text):
    """Parse l3retain CSV -> {iters: [(R_A_bytes, retained), ...]}."""
    groups = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("iters,"):
            continue
        parts = line.split(",")
        if len(parts) != 5:
            continue
        try:
            iters = int(parts[0])
            r_mb = float(parts[1])
            retained = float(parts[4])
        except ValueError:
            continue
        groups.setdefault(iters, []).append((r_mb * 1e6, retained))
    # collapse duplicate R_A by mean retained
    out = {}
    for iters, rows in groups.items():
        agg = {}
        for r, v in rows:
            agg.setdefault(round(r, 1), []).append(v)
        out[iters] = sorted((r, sum(vs) / len(vs)) for r, vs in agg.items())
    return out


def r_at_retained(rows, level=0.5):
    """Interpolate the aggressor footprint where retained crosses `level` (censored at max)."""
    prev = None
    for r, v in rows:
        if prev is not None:
            (pr, pv) = prev
            if (pv - level) * (v - level) <= 0 and pv != v:
                t = (level - pv) / (v - pv)
                return pr + t * (r - pr)
        prev = (r, v)
    # never crosses: return the last R (censored lower bound)
    return rows[-1][0]


def fit_power_law(xs, ys):
    """Least-squares fit y = c * x^a (log-log). Returns (c, a) or None."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return math.exp(b), a


def main():
    ap = argparse.ArgumentParser(description="Self-calibrate the L3 replacement (equivalent) model.")
    ap.add_argument("--raw", help="file with `kernel_bench --op l3retain` CSV output")
    ap.add_argument("--run", nargs=argparse.REMAINDER,
                    help="command to run the probe (everything after --run)")
    ap.add_argument("--out", default="config/l3_calibration.json")
    ap.add_argument("--level", type=float, default=0.5, help="retention level for the knee (default 0.5)")
    ap.add_argument("--hot-set-bytes", type=int, default=HOT_SET_BYTES_DEFAULT)
    args = ap.parse_args()

    if args.run:
        cmd = args.run
        print(f"[l3_calibrate] run: {' '.join(cmd)}", file=sys.stderr)
        text = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    elif args.raw:
        text = open(args.raw).read()
    else:
        text = sys.stdin.read()

    groups = parse_raw(text)
    if not groups:
        sys.exit("[l3_calibrate] no rows parsed (need l3retain CSV)")

    knees = {it: r_at_retained(rows, args.level) for it, rows in groups.items()}
    iters_sorted = sorted(knees)
    fit = fit_power_law([float(it) for it in iters_sorted], [knees[it] for it in iters_sorted])
    c0_bytes, alpha = (fit if fit else (knees[iters_sorted[0]], 0.0))
    phys = read_physical_l3_bytes()

    report = {
        "provenance": {
            "tool": "scripts/l3_calibrate.py",
            "source": "infvino's own microbenchmark `kernel_bench --op l3retain`",
            "vendor_documents_used": False,
            "statement": ("Equivalent engineering model of L3 replacement behaviour, "
                          "self-calibrated from on-device measurements. Not a reverse "
                          "engineering of, nor a claim about, any specific vendor "
                          "microarchitecture (tag RAM / way-select / replacement bits)."),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "machine": {"physical_l3_bytes": phys, "hot_set_bytes": args.hot_set_bytes},
        "retention_knee": {
            "level": args.level,
            "r50_bytes": {str(it): knees[it] for it in iters_sorted},
            "reuse_protect_ratio": {
                str(it): (knees[it] / knees[iters_sorted[0]]) for it in iters_sorted
            },
        },
        "fit": {
            "model": "knee(reuse) = c0_bytes * reuse^alpha",
            "c0_bytes": c0_bytes,
            "alpha": alpha,
        },
        # Recommendation consumed by docs / humans: the measured behaviour (heavily-reused
        # data resists streaming pollution far beyond the nominal capacity) matches the
        # evict-first NRU policy in L3Model, not strict LRU.
        "recommended_policy": "nru",
        "tuning_anchors": {
            "l3_physical_bytes": phys if phys else 8 * 1024 * 1024,
            "pollution_knee_bytes_reuse1": knees[iters_sorted[0]],
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[l3_calibrate] wrote {args.out}")
    print(f"  physical L3            : {phys} B")
    for it in iters_sorted:
        print(f"  R_50(reuse={it:<3})       : {knees[it]/1e6:6.2f} MB"
              f"   (x{knees[it]/knees[iters_sorted[0]]:.2f} vs reuse=1)")
    print(f"  fit knee = {c0_bytes/1e6:.2f} MB * reuse^{alpha:.3f}")
    print(f"  recommended policy     : nru  (see docs/round60-*.md)")


if __name__ == "__main__":
    main()
