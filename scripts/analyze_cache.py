#!/usr/bin/env python3
"""非 compute-bound 小算子的**缓存命中/带宽**分析（离线，不碰 GPU）。

背景：`docs/budget-analysis-3models.md` §9 指出，剩余空间的大头是物理/结构受限。
对算术强度 ~0.25–0.5 FLOP/byte 的小算子，正确标尺是**内存 roofline**，而 roofline
本身强烈依赖**工作集是否命中缓存**：Iris Xe 只有 ~3.75 MB LLC（无独立大 L2），
一旦工作集超过 LLC 就退化成 DRAM 流式（R30 实测：1 MB 时 89.7 GB/s，8 MB 时 21 GB/s）。

本脚本逐节点（profile-json 精确 join）计算：
  * bytes      ：该节点读+写的总字节（与 R30 / `expectedOps` 同模型）
  * achieved   ：实测带宽 = bytes / ms
  * roofline   ：**按该节点工作集插值**的 copy 带宽上限（R30 曲线）
  * eff        ：achieved / roofline（1.0 = 贴住该工作集的理论带宽）
  * 缓存档位   ：工作集 ≤ L1(默认 192 KB) / ≤ LLC(默认 3.75 MB) / > LLC

用法（推荐：用 kernel_run --profile-json 的逐节点数据）：
  python3 scripts/analyze_cache.py --model yolov8n-pose \
      --plan models/yolov8n-pose/model.plan --profile-json /tmp/prof.json
  python3 scripts/analyze_cache.py --models-dir models --profile-dir /tmp/final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_smallops import BW_PTS, copy_bw_gbps, parse_plan, a, cost  # noqa: E402

SMALL_OPS = {"ew_binary", "ew_binary_bcast", "ew_unary", "copy_c", "slice_axis",
             "concat4", "maxpool", "resize_nn", "permute_0213", "softmax_axis",
             "gap", "bmm"}
LAUNCH_US = 3.5   # 每 dispatch 的 launch 地板（R30 实测，与 expectedOps 同源）
# 设备缓存档位（Iris Xe / Xe-LP；可 --l1/--llc 覆盖）。仅用于**分档提示**，
# 不参与任何「硬件极限」结论——缓存大小是设备事实，不是性能上限。
L1_DEFAULT = 192 * 1024
LLC_DEFAULT = 3750 * 1024


def node_bytes(kind, raw):
    ai = a(raw)
    out, byts = cost(kind, ai, {}, raw)
    return byts


def tier(fp, l1, llc):
    if fp <= l1:
        return "L1"
    if fp <= llc:
        return "LLC"
    return "DRAM"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--profile-json", default=None)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--profile-dir", default=None, help="批量：<dir>/prof_<model>.json")
    ap.add_argument("--l1", type=int, default=L1_DEFAULT)
    ap.add_argument("--llc", type=int, default=LLC_DEFAULT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    jobs = []
    if args.plan and args.profile_json:
        jobs = [(args.model or args.plan, args.plan, args.profile_json)]
    else:
        models = ["yolov8n-pose", "yolo11n-pose", "mobilenetv3-small"]
        for m in models:
            plan = os.path.join(args.models_dir, m, "model.plan")
            prof = os.path.join(args.profile_dir or "", f"prof_{m}.json")
            if os.path.exists(plan) and os.path.exists(prof):
                jobs.append((m, plan, prof))

    all_out = {}
    for model, plan, prof in jobs:
        _tensors, nodes = parse_plan(plan)
        profd = json.load(open(prof))
        # 节点按 signature 对齐 plan 的顺序不可靠（fusion 后 index 不同）；
        # 用 profile 节点自带的 op/tag + ms，按 op 聚合，再对 plan 里同 op 的
        # 每节点 bytes 取「最大值」作为工作集代表（同一 op 形状可能不同）。
        by_op = defaultdict(lambda: {"ms": 0.0, "n": 0, "bytes": 0.0, "fp_max": 0.0, "sigs": set()})
        for nd in profd.get("nodes", []):
            op = nd.get("op", "")
            if op not in SMALL_OPS:
                continue
            ms = float(nd.get("ms_per_frame", 0.0))
            if ms <= 0:
                continue
            by_op[op]["ms"] += ms
            by_op[op]["n"] += 1
            sig = nd.get("signature", "")
            if sig:
                by_op[op]["sigs"].add(sig)

        # plan 侧：每个 op 的 bytes（同 op 多形状时取最大工作集，并累计总量）
        plan_bytes = defaultdict(lambda: {"sum": 0.0, "max": 0.0, "n": 0})
        for kind, ins, outs, raw in nodes:
            if kind not in SMALL_OPS:
                continue
            b = node_bytes(kind, raw)
            if not b:
                continue
            plan_bytes[kind]["sum"] += b
            plan_bytes[kind]["max"] = max(plan_bytes[kind]["max"], b)
            plan_bytes[kind]["n"] += 1

        rows = []
        tot_ms = tot_bytes = 0.0
        for op, d in by_op.items():
            pb = plan_bytes.get(op, {"sum": 0.0, "max": 0.0})
            # 实测带宽按「该 op 实际搬运的总字节」/ 总 ms；工作集按单节点最大。
            bytes_total = pb["sum"] if pb["sum"] else 0.0
            ms = d["ms"]
            achieved = bytes_total / (ms * 1e-3) / 1e9 if ms > 0 and bytes_total > 0 else 0.0
            fp = pb["max"]
            roof = copy_bw_gbps(fp) if fp > 0 else 0.0
            # 把 launch 地板与内存时间分开，区分「缓存/带宽受限」vs「launch/网格受限」。
            t_launch_us = LAUNCH_US * d["n"]
            t_mem_us = bytes_total / (roof * 1e9) * 1e6 if roof > 0 else 0.0
            t_min_us = t_launch_us + t_mem_us
            eff = (t_min_us / (ms * 1000.0)) if ms > 0 and t_min_us > 0 else 0.0
            bottleneck = "launch/grid" if t_launch_us >= t_mem_us else "mem/cache"
            rows.append({"op": op, "n": d["n"], "ms": round(ms, 3),
                         "MB": round(bytes_total / 1e6, 2), "fp_KB": round(fp / 1e3, 1),
                         "achieved_GBs": round(achieved, 1), "roofline_GBs": round(roof, 1),
                         "eff": round(eff, 3), "tier": tier(fp, args.l1, args.llc) if fp else "-",
                         "t_launch_us": round(t_launch_us, 1), "t_mem_us": round(t_mem_us, 1),
                         "bottleneck": bottleneck})
            tot_ms += ms
            tot_bytes += bytes_total

        rows.sort(key=lambda r: -r["ms"])
        res = {"model": model, "rows": rows, "total_small_ms": round(tot_ms, 3),
               "total_small_MB": round(tot_bytes / 1e6, 2),
               "achieved_GBs": round(tot_bytes / (tot_ms * 1e-3) / 1e9, 1) if tot_ms > 0 else 0.0,
               "l1": args.l1, "llc": args.llc}
        all_out[model] = res

        if not args.json:
            print(f"\n================ {model} ================")
            print(f"  小算子合计 {res['total_small_ms']:.3f} ms, {res['total_small_MB']:.2f} MB(rw), "
                  f"整体 achieved {res['achieved_GBs']:.1f} GB/s  "
                  f"(L1≤{args.l1//1024}KB, LLC≤{args.llc//1024}KB)")
            print(f"  {'op':16} {'n':>3} {'ms':>7} {'MB':>7} {'fp_KB':>8} {'ach GB/s':>9} "
                  f"{'roof':>6} {'t_lnch':>7} {'t_mem':>7} {'eff':>5} {'tier':>5} {'bottleneck':>11}")
            for r in rows:
                print(f"  {r['op']:16} {r['n']:3d} {r['ms']:7.3f} {r['MB']:7.2f} "
                      f"{r['fp_KB']:8.1f} {r['achieved_GBs']:9.1f} {r['roofline_GBs']:6.1f} "
                      f"{r['t_launch_us']:7.1f} {r['t_mem_us']:7.1f} {r['eff']:5.2f} "
                      f"{r['tier']:>5} {r['bottleneck']:>11}")
            print("  说明：ach=总字节/总ms；roof=按单节点最大工作集插值的 copy 带宽；")
            print("        t_lnch=launch地板×节点数，t_mem=bytes/roof，eff=(t_lnch+t_mem)/ms；")
            print("        eff≈1 表示已贴住「该工作集 + launch 地板」的现实上限；bottleneck 指明是")
            print("        launch/网格 还是 内存/缓存 受限（后者才与「缓存命中」直接相关）。")
    if args.json:
        print(json.dumps(all_out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
