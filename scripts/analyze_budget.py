#!/usr/bin/env python3
"""infvino busy/net/e2e 预算与优化记分卡（Phase 1/2，纯离线，不碰 GPU）。

把三层推理状态拆成一个可归因的「预算瀑布」，并把 GPU busy 按**算子 / 数据通路**
归因，再结合 `config/tuning.json` 的**中间标准**（expected_ops）给出每个节点的
`ops/EU/cyc`、`ratio = 实测/期望` 与可挖余量 `headroom_ms`。

设计原则（团队结论）：**不以 OpenVINO 为标尺**。物理硬件极限 / 中间标准才是标尺；
本工具只回答「时间花在哪、离这台机器的极限还有多少」。

数据来源（二选一，推荐 profile-json）：
  --profile-json  `kernel_run --report --profile-json out.json` 的 JSON
                  → 逐节点精确 join（signature→expected→ratio→headroom）
  --report        `kernel_run --report` 文本（Phase 1 回退；只有签名级近似）
  --bench         `infvino_bench` 文本（e2e/pre/net/post）
  --plan          model.plan（FLOPs / 结构）
  --tuning        config/tuning.json（中间标准）

用法:
  python3 scripts/analyze_budget.py --model yolov8n-pose --plan models/yolov8n-pose/model.plan \
      --profile-json /tmp/prof.json --bench /tmp/bench.txt
  python3 scripts/analyze_budget.py --model yolov8n-pose --report /tmp/kr.txt   # Phase 1 回退
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

EU_DEFAULT = 80
CLK_GHZ_DEFAULT = 1.3
PEAK = 32.0  # FP16 packed FLOP/EU/cyc 理论峰值

_CONV3X3 = {"conv3x3", "conv3x3ov", "conv3x3blk"}
_CONV1X1 = {"conv1x1", "conv1x1g", "conv1x1cat4", "gemm"}
_DEPTHWISE = {"depthwise", "depthwise_v", "depthwise_vp"}


def rollup_of(key: str) -> str:
    if key == "conv3x3":
        return "conv3x3"
    if key == "conv1x1":
        return "conv1x1"
    if key == "depthwise":
        return "depthwise"
    if key == "conv_general":
        return "conv_general"
    return "smallops"


def subpath_of(tag: str) -> str:
    return {"conv3x3ov": "ov", "conv3x3blk": "blk", "conv3x3": "native"}.get(tag.split("@")[0], "")


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


# --------------------------------------------------------------------------- #
def parse_plan(path: str):
    """返回 (rows, family_flops, family_nodes, plan_signatures)。"""
    tensors, inits, nodes = {}, {}, []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if p[0] in ("tensor", "input"):
            tensors[p[1]] = tuple(int(x) for x in p[2:])
        elif p[0] == "init":
            inits[p[1]] = tuple(int(x) for x in p[3:])
        elif p[0] == "node":
            attrs = {}
            for kv in p[4:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    attrs[k] = v
            nodes.append({"op": p[1], "ins": p[2].split(","), "outs": p[3].split(","), "attrs": attrs})

    rows = []
    for n in nodes:
        op, ins, outs, a = n["op"], n["ins"], n["outs"], n["attrs"]
        oshape = tensors.get(outs[0])
        key, fl, sig = op, 0.0, None
        if op == "conv3x3" and len(ins) > 1 and ins[1] in inits and oshape:
            w = inits[ins[1]]
            fl = 2.0 * w[0] * w[1] * (w[2] if len(w) > 2 else 3) * (w[3] if len(w) > 3 else 3) * oshape[2] * oshape[3]
            sig = (f"conv3x3|W{a.get('Wout')}H{a.get('Hout')}s{a.get('stride',1)}p{a.get('pad',1)}"
                   f"_Cin{w[1]}_Cout{w[0]}_act{a.get('act',0)}_f16")
            key = "conv3x3"
        elif op == "conv1x1" and ins and ins[0] in inits:
            w = inits[ins[0]]
            if oshape and w[0]:
                nn = _prod(oshape) // w[0]
                fl = 2.0 * w[0] * w[1] * nn
                sig = f"conv1x1|Cout{w[0]}_N{nn}_Cin{w[1]}_act{a.get('act',0)}_f16"
            key = "conv1x1"
        elif op == "conv_general" and len(ins) > 1 and ins[1] in inits and oshape:
            w = inits[ins[1]]
            k = w[2] if len(w) > 2 else 3
            g = int(a.get("G", 1))
            cin = w[1] * g
            fl = 2.0 * w[0] * w[1] * k * k * oshape[2] * oshape[3]
            if g == cin:
                key = "depthwise"
                sig = (f"depthwise|W{a.get('Wout')}H{a.get('Hout')}s{a.get('S',1)}p{a.get('P',0)}"
                       f"_Cin{cin}_Cout{cin}_K{k}_G{g}_act{a.get('act',0)}_f16")
            else:
                key = "conv_general"
        elif op == "gemm" and ins and ins[0] in inits:
            w = inits[ins[0]]
            bshape = tensors.get(ins[1]) or (1,)
            nn = _prod(bshape[1:]) if len(bshape) > 1 else 1
            fl = 2.0 * w[0] * w[1] * nn
            key = "conv1x1"
        elif op == "gap":
            sig = f"gap|C{a.get('C')}_HW{a.get('HW')}_act0_f16"
        rows.append({"key": key, "flops": fl, "sig": sig})

    fam_flops, fam_nodes, sigs = defaultdict(float), defaultdict(int), set()
    for r in rows:
        fam_flops[r["key"]] += r["flops"]
        fam_nodes[r["key"]] += 1
        if r["sig"]:
            sigs.add(r["sig"])
    return rows, dict(fam_flops), dict(fam_nodes), sigs


def _host_from(wall, sync, enq):
    return {"wall_ms": wall, "sync_ms": sync, "enqueue_ms": enq,
            "host_total_ms": max(0.0, wall - sync),
            "setarg_ms": max(0.0, wall - sync - enq)}


def parse_report(text: str) -> dict:
    out = {"tags": [], "device": "", "eu": EU_DEFAULT, "clk_ghz": CLK_GHZ_DEFAULT}
    for line in text.splitlines():
        m = re.search(r"device:\s*(.*?)\s*EU=(\d+)\s*clk=(\d+)MHz", line)
        if m:
            out["device"], out["eu"], out["clk_ghz"] = m.group(1).strip(), int(m.group(2)), int(m.group(3)) / 1000.0
        m = re.match(r"activation pool:\s*requested\s*([\d.]+)\s*MB\s*->\s*allocated\s*([\d.]+)\s*MB\s*\((\d+)\s*buffers,\s*reuse\s*([\d.]+)%\)", line)
        if m:
            out["pool"] = {"requested_mb": float(m.group(1)), "allocated_mb": float(m.group(2)),
                           "buffers": int(m.group(3)), "reuse_pct": float(m.group(4))}
        m = re.match(r"total kernel time:\s*([\d.]+)\s*ms\s*\(iters=(\d+)\)", line)
        if m:
            out["busy_ms"], out["iters"] = float(m.group(1)), int(m.group(2))
        m = re.match(r"host segmentation:\s*wall=([\d.]+)\s*busy=([\d.]+)\s*enqueue=([\d.]+)\s*sync=([\d.]+)", line)
        if m:
            out["host"] = _host_from(float(m.group(1)), float(m.group(4)), float(m.group(3)))
        m = re.match(r"structural:\s*nodes=(\d+)\s*launched=(\d+)\s*alias/skip=(\d+)\s*dispatches=([\d.]+)\s*reorder=([\d.]+)\(([\d.]+)ms\)\s*fusions\(res=(\d+),concat=(\d+)\)\s*fsv16=(\d+)", line)
        if m:
            out["structural"] = {"plan_nodes": int(m.group(1)), "dispatched_nodes": int(m.group(2)),
                                 "skipped_nodes": int(m.group(3)), "dispatches_per_frame": float(m.group(4)),
                                 "reorder_calls_per_frame": float(m.group(5)), "reorder_ms_per_frame": float(m.group(6)),
                                 "fusions_res": int(m.group(7)), "fusions_concat": int(m.group(8)),
                                 "fsv16_tensors": int(m.group(9))}
        m = re.match(r"\s+(\S+)\s+([\d.]+)\s+ms\s+x(\d+)\s*$", line)
        if m:
            out["tags"].append({"tag": m.group(1), "ms": float(m.group(2)), "calls": int(m.group(3))})
    return out


def parse_profile_json(path: str) -> dict:
    d = json.load(open(path))
    t = d.get("timing_ms", {})
    m = re.search(r"EU=(\d+)\s*clk=(\d+)MHz", d.get("device", ""))
    mem = d.get("memory", {})
    req, alc = mem.get("requested_bytes", 0), mem.get("allocated_bytes", 0)
    rep = {
        "device": d.get("device", ""),
        "eu": int(m.group(1)) if m else EU_DEFAULT,
        "clk_ghz": int(m.group(2)) / 1000.0 if m else CLK_GHZ_DEFAULT,
        "iters": d.get("iters"),
        "busy_ms": t.get("busy"),
        "host": _host_from(t.get("wall", 0), t.get("sync", 0), t.get("enqueue", 0)),
        "tags": [{"tag": o["tag"], "ms": o["ms_per_frame"], "calls": o["calls_per_frame"]}
                 for o in d.get("ops", [])],
        "structural": d.get("structural"),
        "profile": d,
    }
    if req:
        rep["pool"] = {"requested_mb": req / 1e6, "allocated_mb": alc / 1e6,
                       "buffers": mem.get("buffers", 0),
                       "reuse_pct": 100.0 * (1 - alc / req) if req else 0.0}
    return rep


def parse_bench(text: str) -> dict:
    out = {}
    for pat, keys in (
        (r"latency\(ms\) total:\s*mean=([\d.]+).*?->\s*([\d.]+)\s*fps", ("e2e_ms", "e2e_fps")),
        (r"latency\(ms\) infer:\s*mean=([\d.]+).*?->\s*([\d.]+)\s*fps", ("net_ms", "net_fps")),
    ):
        m = re.search(pat, text)
        if m:
            out[keys[0]], out[keys[1]] = float(m.group(1)), float(m.group(2))
    m = re.search(r"breakdown\s*:\s*pre=([\d.]+)\s*infer=([\d.]+)\s*post=([\d.]+)", text)
    if m:
        out["pre_ms"], out["net_breakdown_ms"], out["post_ms"] = map(float, m.groups())
    return out


def parse_tuning(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        return json.load(open(path)).get("entries", {})
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
def build(model, plan_path, report_text, bench_text, tuning_path, profile_path) -> dict:
    if profile_path:
        rep = parse_profile_json(profile_path)
    else:
        rep = parse_report(report_text)
    eu, clk = rep["eu"], rep["clk_ghz"]
    euc = eu * clk * 1e9
    busy = rep.get("busy_ms") or sum(t["ms"] for t in rep["tags"])

    op_ms, op_calls, sub_ms = defaultdict(float), defaultdict(int), defaultdict(float)
    for t in rep["tags"]:
        k = t["tag"].split("@")[0]
        kk = "conv3x3" if k in _CONV3X3 else ("conv1x1" if k in _CONV1X1 else ("depthwise" if k in _DEPTHWISE else k))
        op_ms[kk] += t["ms"]
        op_calls[kk] += t["calls"]
        if subpath_of(t["tag"]):
            sub_ms[subpath_of(t["tag"])] += t["ms"]

    fam_flops, fam_nodes, plan_sigs = {}, {}, set()
    if plan_path and os.path.exists(plan_path):
        _, fam_flops, fam_nodes, plan_sigs = parse_plan(plan_path)

    ops_rows = []
    for k, ms in sorted(op_ms.items(), key=lambda kv: -kv[1]):
        fl = fam_flops.get(k, 0.0)
        row = {"op": k, "rollup": rollup_of(k), "ms": round(ms, 3), "calls": op_calls[k],
               "share_pct": round(100.0 * ms / busy, 1) if busy else 0.0}
        if fl > 0 and ms > 0:
            row["gflops"] = round(fl / 1e9, 4)
            row["measured_ops"] = round(fl / (ms * 1e-3) / euc, 3)
            row["nodes"] = fam_nodes.get(k, 0)
        if k == "conv3x3":
            row["subpath_ms"] = {a: round(b, 3) for a, b in sub_ms.items()}
        ops_rows.append(row)

    compute_flops = sum(fam_flops.get(k, 0.0) for k in ("conv3x3", "conv1x1", "depthwise", "conv_general"))
    compute_ms = sum(op_ms.get(k, 0.0) for k in ("conv3x3", "conv1x1", "depthwise", "conv_general"))
    overall_ops = compute_flops / (compute_ms * 1e-3) / euc if compute_ms > 0 else 0.0
    small_ms = sum(ms for k, ms in op_ms.items() if rollup_of(k) == "smallops")

    # ---- 中间标准记分卡 ----
    entries = parse_tuning(tuning_path)
    sc = {"mode": "exact-node" if (rep.get("profile") and rep["profile"].get("nodes")) else "isolated-signature",
          "families": {}, "headroom_ms": 0.0, "worst": [],
          "matched": 0, "unmatched": 0, "tuning_total": len(entries)}
    fam_agg = defaultdict(lambda: {"ms": 0.0, "headroom": 0.0, "n": 0, "ratios": []})
    if sc["mode"] == "exact-node":
        for n in rep["profile"]["nodes"]:
            sig, ms = n.get("signature"), float(n.get("ms_per_frame", 0))
            if not sig or ms <= 0:
                continue
            e = entries.get(sig)
            if not e:
                sc["unmatched"] += 1
                continue
            sc["matched"] += 1
            r = float(e.get("ratio", 0.0))
            fam = "conv1x1" if sig.split("|")[0] == "gemm" else sig.split("|")[0]
            head = ms * (1.0 - r) if r < 1.0 else 0.0
            agg = fam_agg[fam]
            agg["ms"] += ms; agg["headroom"] += head; agg["n"] += 1; agg["ratios"].append(r)
            sc["headroom_ms"] += head
            sc["worst"].append({"signature": sig, "op": n.get("op", ""), "tag": n.get("tag", ""),
                                "kernel": e.get("kernel", ""), "ratio": round(r, 3), "ms": round(ms, 4),
                                "ops": round(float(e.get("ops_per_eu_cyc", 0)), 3),
                                "expected_ops": round(float(e.get("expected_ops_per_eu_cyc", 0)), 3)})
    else:
        for key, e in entries.items():
            if key not in plan_sigs:
                continue
            r = e.get("ratio")
            if r is None:
                continue
            r = float(r); sc["matched"] += 1
            fam = "conv1x1" if key.split("|")[0] == "gemm" else key.split("|")[0]
            ms = float(e.get("ms", 0.0))
            head = ms * (1.0 - r) if r < 1.0 else 0.0
            agg = fam_agg[fam]
            agg["ms"] += ms; agg["headroom"] += head; agg["n"] += 1; agg["ratios"].append(r)
            sc["headroom_ms"] += head
            sc["worst"].append({"signature": key, "op": key.split("|")[0], "tag": "", "kernel": e.get("kernel", ""),
                                "ratio": round(r, 3), "ms": round(ms, 4),
                                "ops": round(float(e.get("ops_per_eu_cyc", 0)), 3),
                                "expected_ops": round(float(e.get("expected_ops_per_eu_cyc", 0)), 3)})
    for fam, v in fam_agg.items():
        rs = sorted(v["ratios"])
        sc["families"][fam] = {"n": v["n"], "ratio_min": round(rs[0], 3),
                               "ratio_median": round(rs[len(rs) // 2], 3), "ratio_max": round(rs[-1], 3),
                               "ms": round(v["ms"], 3), "headroom_ms": round(v["headroom"], 3)}
    sc["headroom_ms"] = round(sc["headroom_ms"], 3)
    sc["worst"].sort(key=lambda x: x["ratio"])
    sc["worst"] = sc["worst"][:12]

    return {
        "model": model, "device": rep["device"], "iters": rep.get("iters"),
        "states": parse_bench(bench_text) if bench_text else {},
        "kernel_busy_ms": round(busy, 3),
        "host": rep.get("host", {}), "pool": rep.get("pool", {}),
        "structural": rep.get("structural", {}),
        "dispatches_per_frame": sum(t["calls"] for t in rep["tags"]),
        "ops": ops_rows,
        "compute_gflops": round(compute_flops / 1e9, 3),
        "compute_ops_eu_cyc": round(overall_ops, 3),
        "smallops_ms": round(small_ms, 3),
        "smallops_share_pct": round(100.0 * small_ms / busy, 1) if busy else 0.0,
        "scorecard": sc,
    }


def render(r: dict) -> str:
    L = []
    st, h, p, sc, so = r["states"], r["host"], r["pool"], r["scorecard"], r["structural"]
    L.append(f"# 预算报告 · {r['model']}")
    L.append(f"device: {r['device']}   iters={r['iters']}   dispatches/帧={r['dispatches_per_frame']:.0f}")
    L.append("# 读 ops/EU/cyc 前先做 roofline 判断：内存/指令/launch 受限的层算子低 ratio≠可挖；")
    L.append("#   很多 kernel 只是更大模式的一部分（可融合），只看单 kernel 不合理，须结合端到端。")
    L.append("#   详见 docs/profiling-budget.md §3.0。")
    L.append("")
    L.append("## 1. 三层预算瀑布")
    if st:
        e2e = st.get("e2e_ms", 1) or 1
        net = st.get("net_ms", 0)
        b = r["kernel_busy_ms"]
        L.append(f"  e2e   {st.get('e2e_ms',0):8.3f} ms  ({st.get('e2e_fps',0):7.2f} fps)")
        L.append(f"  ├─ pre  {st.get('pre_ms',0):8.3f} ms  ({100*st.get('pre_ms',0)/e2e:5.1f}%)")
        L.append(f"  ├─ net  {net:8.3f} ms  ({100*net/e2e:5.1f}%)")
        L.append(f"  │   ├─ busy      {b:8.3f} ms  (net 的 {100*b/net if net else 0:5.1f}%)   [GPU kernel]")
        L.append(f"  │   ├─ net-busy  {max(0.0,net-b):8.3f} ms  (launch+copy+f16+bubble)")
        if h:
            L.append(f"  │   │   其中 profiling 下 enqueue={h.get('enqueue_ms',0):.3f} sync={h.get('sync_ms',0):.3f} ms")
        L.append(f"  └─ post {st.get('post_ms',0):8.3f} ms")
    else:
        L.append(f"  (未提供 --bench)  GPU busy={r['kernel_busy_ms']:.3f} ms")
    L.append("")
    L.append("## 2. GPU busy 归因（逐算子 / 数据通路）")
    L.append(f"  {'op':12s} {'ms':>8s} {'share':>7s} {'calls':>6s} {'GFLOPs':>9s} {'ops/EU/cyc':>11s}")
    for o in r["ops"]:
        L.append(f"  {o['op']:12s} {o['ms']:8.3f} {o['share_pct']:6.1f}% {o['calls']:6.0f}"
                 f" {o.get('gflops',0):9.4f} {o.get('measured_ops',0):11.3f}")
        if "subpath_ms" in o:
            L.append("      └ 通路: " + "  ".join(f"{k}={v}" for k, v in o["subpath_ms"].items()))
    L.append(f"  计算类合计 {r['compute_gflops']:.3f} GFLOPs / {compute_ms(r):.3f} ms"
             f" -> {r['compute_ops_eu_cyc']:.3f} ops/EU/cyc ({r['compute_ops_eu_cyc']/PEAK*100:.1f}% of {PEAK:.0f})")
    L.append(f"  小算子(内存/launch 受限) 合计 {r['smallops_ms']:.3f} ms ({r['smallops_share_pct']:.1f}% busy)")
    if so:
        L.append(f"  结构: 节点 {so.get('plan_nodes')} → launch {so.get('dispatched_nodes')}"
                 f"（alias/skip {so.get('skipped_nodes')}）; 融合 res={so.get('fusions_res')} concat={so.get('fusions_concat')};"
                 f" reorder {so.get('reorder_calls_per_frame'):.0f}/帧({so.get('reorder_ms_per_frame'):.3f}ms); fsv16 张量 {so.get('fsv16_tensors')}")
    if p:
        L.append(f"  内存池: requested {p.get('requested_mb',0):.1f} -> allocated {p.get('allocated_mb',0):.1f} MB "
                 f"({p.get('buffers',0)} buffers, reuse {p.get('reuse_pct',0):.0f}%)")
    L.append("")
    L.append(f"## 3. 优化记分卡（中间标准 expected_ops · {sc['mode']}）")
    L.append(f"  tuning 命中 {sc['matched']} / 未命中 {sc['unmatched']}（全库 {sc['tuning_total']}）;"
             f" ratio<1 可挖余量 ≈ {sc['headroom_ms']:.3f} ms")
    L.append(f"  {'family':12s} {'n':>4s} {'ms':>8s} {'headroom':>9s} {'ratio min':>10s} {'median':>8s} {'max':>8s}")
    for fam, v in sorted(sc["families"].items(), key=lambda kv: -kv[1]["headroom_ms"]):
        L.append(f"  {fam:12s} {v['n']:4d} {v['ms']:8.3f} {v['headroom_ms']:9.3f}"
                 f" {v['ratio_min']:10.3f} {v['ratio_median']:8.3f} {v['ratio_max']:8.3f}")
    L.append("  最差节点（ratio 升序）：")
    for x in sc["worst"]:
        L.append(f"    {x['ratio']:.3f}  {x['signature']}  [{x['kernel']}]  {x['ms']} ms  {x['ops']}/{x['expected_ops']}")
    return "\n".join(L)


def compute_ms(r):
    return sum(o["ms"] for o in r["ops"] if o["rollup"] in ("conv3x3", "conv1x1", "depthwise", "conv_general"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--report", default=None, help="kernel_run --report 文本（Phase1 回退）")
    ap.add_argument("--profile-json", default=None, help="kernel_run --profile-json（推荐）")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--tuning", default="config/tuning.json")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.report and not args.profile_json:
        ap.error("需要 --profile-json 或 --report 之一")
    r = build(args.model, args.plan, open(args.report).read() if args.report else "",
              open(args.bench).read() if args.bench else None, args.tuning, args.profile_json)
    if args.out:
        json.dump(r, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(r, indent=2, ensure_ascii=False) if args.json else render(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
