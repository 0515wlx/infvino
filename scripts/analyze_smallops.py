#!/usr/bin/env python3
"""分析三个模型里「剩下的 kernel」（非 conv/gemm）的流量与内存 roofline 上限。

Round 30：这些算子算术强度 ~0.25–0.5 FLOP/byte，碰不到 FMA，正确的标尺是
    T_min = launch + bytes / BW_copy(footprint)
本脚本解析 model.plan，按算子累计读写字节与单节点上限时间；若再给一份
`kernel_run --report` 的输出，则打印实测 vs 上限。

用法:
  python3 analyze_smallops.py --models-dir models
  python3 analyze_smallops.py --plan models/yolo11n-pose/model.plan --report /tmp/r.txt
"""
import argparse
import math
import os
import re
from collections import defaultdict

# 与 src/Tuning.cpp 的 R30 中间标准保持一致（Iris Xe / 单通道 LPDDR 实测）。
BW_PTS = [(4e3, 3.0), (16e3, 8.1), (64e3, 22.7), (256e3, 46.4), (512e3, 63.3),
          (1e6, 89.7), (2e6, 62.4), (4e6, 44.6), (8e6, 21.0)]
LAUNCH_US = 3.5
SMALL_OPS = {"ew_binary", "ew_binary_bcast", "ew_unary", "copy_c", "slice_axis",
             "concat4", "maxpool", "resize_nn", "permute_0213", "softmax_axis",
             "gap", "bmm"}


def copy_bw_gbps(fp):
    if fp <= BW_PTS[0][0]:
        return BW_PTS[0][1]
    if fp >= BW_PTS[-1][0]:
        return BW_PTS[-1][1]
    for i in range(len(BW_PTS) - 1):
        x0, y0 = BW_PTS[i]
        x1, y1 = BW_PTS[i + 1]
        if x0 <= fp <= x1:
            t = math.log(fp / x0) / math.log(x1 / x0)
            return y0 * (y1 / y0) ** t
    return 20.0


def parse_plan(path):
    tensors, nodes = {}, []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] in ("tensor", "input"):
            tensors[t[1]] = [int(x) for x in t[2:]]
        elif t[0] == "node":
            attrs = {}
            for a in t[4:]:
                if "=" in a:
                    k, v = a.split("=", 1)
                    attrs[k] = v
            nodes.append((t[1], t[2].split(","), t[3].split(","), attrs))
    return tensors, nodes


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def numel(t, name):
    return prod(t[name]) if name in t else 0


def a(a):
    return {k: int(v) for k, v in a.items() if re.fullmatch(r"-?\d+", v)}


def cost(kind, ai, t, raw=None):
    """返回 (out_elems, rw_bytes)。只覆盖剩余 kernel。"""
    if kind == "ew_binary":
        n = ai.get("n", 0)
        bs = ai.get("b_scalar", 0)
        if raw and "bdims" in raw:            # 通用广播：b 的 distinct 值未知，保守按 n
            return n, 2 * (2 * n + n)
        return n, 2 * (2 * n + (0 if bs else n))
    if kind == "ew_unary":
        n = ai.get("n", 0)
        return n, 4 * n
    if kind == "concat4":
        out = (ai.get("ca", 0) + ai.get("cb", 0) + ai.get("cc", 0) + ai.get("cd", 0)) \
            * ai.get("outer", 1) * ai.get("inner", 1)
        return out, 4 * out
    if kind == "copy_c":
        out = ai.get("HW", 0) * ai.get("cnt", 0)
        return out, 4 * out
    if kind == "slice_axis":
        out = ai.get("outer", 1) * ai.get("len", 0) * ai.get("inner", 1)
        return out, 4 * out
    if kind == "permute_0213":
        out = ai.get("D1", 1) * ai.get("D2", 1) * ai.get("I", 1)
        return out, 4 * out
    if kind == "resize_nn":
        c, h, w, s = ai.get("C", 0), ai.get("H", 0), ai.get("W", 0), ai.get("S", 1)
        out = c * h * s * w * s
        return out, 2 * (c * h * w + out)
    if kind == "maxpool":
        out = ai.get("C", 0) * ai.get("Hout", 0) * ai.get("Wout", 0)
        k = ai.get("K", 1)
        return out, 2 * (out * k * k + out)   # 每输出 K×K 带边界读
    if kind == "softmax_axis":
        out = ai.get("outer", 1) * ai.get("axdim", 1) * ai.get("inner", 1)
        return out, 8 * out                   # 三趟读 + 一趟写
    if kind == "gap":
        c, hw = ai.get("C", 0), ai.get("HW", 0)
        return c, 2 * (c * hw + c)
    if kind == "bmm":
        b0, b1, m, k, n = (ai.get(x, 1) for x in ("B0", "B1", "M", "K", "N"))
        out = b0 * b1 * m * n
        return out, None                      # 计算受限，单独处理
    return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--plan")
    ap.add_argument("--report")
    args = ap.parse_args()

    # 可选：解析 kernel_run --report 的实测 ms（按 tag 聚合）。
    measured = defaultdict(float)
    if args.report and os.path.exists(args.report):
        for line in open(args.report):
            m = re.match(r"\s+(\S+)\s+([\d.]+)\s+ms", line)
            if m:
                measured[m.group(1).split("@")[0]] += float(m.group(2))

    plans = [args.plan] if args.plan else [
        os.path.join(args.models_dir, m, "model.plan")
        for m in ("yolov8n-pose", "yolo11n-pose", "mobilenetv3-small")]
    for plan in plans:
        if not os.path.exists(plan):
            print(f"[skip] {plan}")
            continue
        tensors, nodes = parse_plan(plan)
        agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # nodes, out, bytes, ceil_us(sum)
        for kind, ins, outs, araw in nodes:
            if kind not in SMALL_OPS:
                continue
            ai = a(araw)
            out, byts = cost(kind, ai, tensors, araw)
            s = agg[kind]
            s[0] += 1
            s[1] += out
            s[2] += byts or 0
            if byts:
                s[3] += LAUNCH_US + byts / (copy_bw_gbps(byts) * 1e9) * 1e6
        print(f"\n================ {os.path.dirname(plan)} ================")
        print(f"{'op':14} {'nodes':>5} {'out_elems':>11} {'MB(rw)':>9} "
              f"{'ceil_us':>9} {'meas_ms':>9} {'eff':>5}")
        for kind, (n, out, byts, t_us) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
            if byts <= 0:
                t_us = float("nan")   # bmm：计算受限，不用内存模型
            tag = "permute" if kind == "permute_0213" else kind
            meas = measured.get(tag, 0.0)
            eff = (t_us / (meas * 1000)) if (meas > 0 and byts > 0) else float("nan")
            print(f"{kind:14} {n:5d} {out:11.0f} {byts/1e6:9.2f} "
                  f"{t_us:9.1f} {meas:9.3f} {eff:5.2f}")
        print("  ceil_us = 单节点上限（launch + bytes/BW）；eff = 上限/实测（>1 还有余量）")


if __name__ == "__main__":
    raise SystemExit(main())
