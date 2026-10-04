#!/usr/bin/env python3
"""Block layout 离线规划/分析器（P1-layout 的自动化大脑）。

问题：`conv3x3_blk`（OV 阻塞式 conv 移植，kernels/conv_blk.cl）读的是
`b_fs_yx_fsv16` 布局。当前 `PlanModel::blkInput()` 对**每个** blocked conv 的输入
都跑一次 `reorder_bfyx_to_fsv16`（bfyx -> fsv16），即使它的生产者本身也是一个
blocked conv、本就该直接产出 fsv16。本脚本在**不改运行时**的前提下，读 plan +
自动调优缓存 tuning.json，算出：

  1. 每个 conv3x3 实际会选哪个 kernel（autotune 命中 → blk/ov/native）；
  2. 哪些**激活张量**可以持久化为 fsv16 —— 生产者是 blocked conv，且它的**所有**
     消费者也是 blocked conv（输入槽 0），且不是网络输出、Cout 是 16 的倍数；
  3. 持久化后能省掉多少次 reorder（含同帧同张量去重）。

这不是纸上谈兵：输出同时是运行时 `PlanModel::planBlockedLayout()` 的规格说明，
也是「布局自动化」的决策模型——**布局选择完全由 autotune 的 kernel 选择驱动**，
不再需要人手工按模型/按层标注。

用法：
  python3 scripts/analyze_layout.py                       # 扫 models/*/model.plan
  python3 scripts/analyze_layout.py models/yolov8n-pose/model.plan
  python3 scripts/analyze_layout.py --json --out /tmp/layout.json
"""
import argparse
import glob
import json
import os
import sys

FP16_BYTES = 2


def parse_plan(path):
    """解析 plan 文本（格式见 include/infvino/PlanModel.hpp）。"""
    tensors, nodes, inputs, outputs = {}, [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            kind = tok[0]
            if kind in ("input", "tensor", "init"):
                # name (file) d0 d1 ...
                name = tok[1]
                rest = tok[2:]
                if kind == "init":
                    rest = tok[3:]  # skip file
                dims = [int(x) for x in rest]
                tensors[name] = dims
                if kind == "input":
                    inputs.append(name)
            elif kind == "node":
                op, ins, outs = tok[1], tok[2], tok[3]
                attrs = {}
                for kv in tok[4:]:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        attrs[k] = v
                nodes.append({
                    "op": op,
                    "ins": [] if ins == "-" else ins.split(","),
                    "outs": [] if outs == "-" else outs.split(","),
                    "attr": attrs,
                })
            elif kind == "output":
                outputs.append(tok[1])
    return tensors, nodes, inputs, outputs


def load_tuning(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return d.get("entries", {})


def sig_conv3x3(Wout, Hout, stride, pad, Cin, Cout, act):
    return "conv3x3|W{}H{}s{}p{}_Cin{}_Cout{}_act{}_f16".format(
        Wout, Hout, stride, pad, Cin, Cout, act)


def attrint(node, key, default):
    try:
        return int(node["attr"].get(key, default))
    except (TypeError, ValueError):
        return default


def analyze(plan_path, tuning, verbose=True):
    tensors, nodes, inputs, outputs = parse_plan(plan_path)
    output_set = set(outputs)

    # 生产者 / 消费者
    producer = {}
    for ni, n in enumerate(nodes):
        for o in n["outs"]:
            producer[o] = ni
    consumers = {}  # tensor -> list of (node_index, input_slot)
    for ni, n in enumerate(nodes):
        for slot, t in enumerate(n["ins"]):
            if t != "-":
                consumers.setdefault(t, []).append((ni, slot))

    # 1) 每个 conv3x3 会选哪个 kernel
    conv_info = {}
    for ni, n in enumerate(nodes):
        if n["op"] != "conv3x3":
            continue
        out = n["outs"][0]
        x = n["ins"][0]
        if x not in tensors or out not in tensors:
            continue
        xd, od = tensors[x], tensors[out]
        Cin = xd[-3] if len(xd) >= 3 else 0
        Cout = od[-3] if len(od) >= 3 else 0
        Wout, Hout = attrint(n, "Wout", od[-1]), attrint(n, "Hout", od[-2])
        stride, pad, act = attrint(n, "stride", 1), attrint(n, "pad", 1), attrint(n, "act", 0)
        has_res = len(n["ins"]) > 3 and n["ins"][3] != "-"
        sig = sig_conv3x3(Wout, Hout, stride, pad, Cin, Cout, act)
        e = tuning.get(sig)
        kernel, source = None, "heuristic"
        if e is not None:
            kernel, source = e.get("kernel"), "tuned"
            # 与 PlanModel dispatch 一致：带残差的 conv3x3 只有 conv3x3_ov 支持 RES。
            if has_res and kernel != "conv3x3_ov":
                kernel, source, e = None, "heuristic", None
        if kernel is None:
            # 计划里没有 blk= 属性（见 onnx2plan.py）→ 回退启发式：默认走 ov。
            kernel = "conv3x3_ov"
        will_blk = (kernel == "conv3x3_blk")
        conv_info[ni] = {
            "sig": sig, "kernel": kernel, "source": source,
            "Cin": Cin, "Cout": Cout, "Hout": Hout, "Wout": Wout,
            "stride": stride, "act": act, "in": x, "out": out,
            "ms": (e or {}).get("ms"), "ops": (e or {}).get("ops_per_eu_cyc"),
            "will_blk": will_blk,
        }

    # 2) 持久 fsv16 张量：生产者为 blocked conv，所有消费者也是 blocked conv(槽0)，
    #    非输出，Cout%16==0（fsv16 的通道按 16 补齐，未补齐会越界）。
    fsv16 = set()
    skipped_reason = {}
    for ni, ci in conv_info.items():
        if not ci["will_blk"]:
            continue
        t = ci["out"]
        if t in output_set:
            skipped_reason[t] = "network output"
            continue
        if ci["Cout"] % 16 != 0:
            skipped_reason[t] = "Cout%16!=0"
            continue
        cons = consumers.get(t, [])
        if not cons:
            skipped_reason[t] = "no consumers"
            continue
        ok = True
        for cn, slot in cons:
            c = nodes[cn]
            if c["op"] != "conv3x3" or slot != 0 or not conv_info.get(cn, {}).get("will_blk"):
                ok = False
                break
        if ok:
            fsv16.add(t)
        else:
            skipped_reason[t] = "has non-blocked consumer"

    # 3) reorder 次数（三档）
    blk_convs = [ni for ni, ci in conv_info.items() if ci["will_blk"]]
    # (a) 现状：每个 blocked conv 一次 reorder（同张量多次消费会重复——blkInput 每调用一次就重排）
    cur_reorders = len(blk_convs)
    blk_inputs = {conv_info[ni]["in"] for ni in blk_convs}
    # (b) 仅同帧去重：每个被 blocked conv 消费的张量只重排一次
    dedup_reorders = len(blk_inputs)
    # (c) 去重 + 持久 fsv16：输入本身就已是 fsv16 的直接读，零 reorder
    opt_reorders = len({t for t in blk_inputs if t not in fsv16})

    return {
        "plan": plan_path,
        "nodes": len(nodes),
        "conv3x3": len(conv_info),
        "blk_convs": blk_convs,
        "conv_info": conv_info,
        "fsv16": sorted(fsv16),
        "skipped": skipped_reason,
        "cur_reorders": cur_reorders,
        "dedup_reorders": dedup_reorders,
        "opt_reorders": opt_reorders,
        "saved_dedup": cur_reorders - dedup_reorders,
        "saved_persist": dedup_reorders - opt_reorders,
        "consumers": consumers,
    }


def print_report(r, tuning_path):
    print("=" * 78)
    print("plan:", r["plan"], " (%d nodes, %d conv3x3)" % (r["nodes"], r["conv3x3"]))
    print("tuning:", tuning_path)
    print("-" * 78)
    by_kernel = {}
    for ci in r["conv_info"].values():
        by_kernel[ci["kernel"]] = by_kernel.get(ci["kernel"], 0) + 1
    print("conv3x3 kernel 分布:", ", ".join("%s=%d" % kv for kv in sorted(by_kernel.items())))
    print("blocked conv:", len(r["blk_convs"]))

    if r["blk_convs"]:
        print("-" * 78)
        print("blocked conv（输入张量 / 形状 / 来源）:")
        for ni in r["blk_convs"]:
            ci = r["conv_info"][ni]
            mark = "->FSV16" if ci["out"] in r["fsv16"] else ""
            print("  %-42s s%d %dx%d Cin%-4d Cout%-4d %-8s %s" % (
                ci["in"][-38:], ci["stride"], ci["Wout"], ci["Hout"],
                ci["Cin"], ci["Cout"], ci["source"], mark))

    print("-" * 78)
    print("可持久化为 fsv16 的激活张量: %d" % len(r["fsv16"]))
    for t in r["fsv16"]:
        print("  ", t)

    if r["skipped"]:
        print("-" * 78)
        print("blocked 输出但未持久化（原因）:")
        for t, why in sorted(r["skipped"].items()):
            print("  %-46s %s" % (t[-44:], why))

    print("-" * 78)
    print("reorder 次数：现状 %d" % r["cur_reorders"])
    print("  - 同帧去重        -> %d（省 %d）" % (r["dedup_reorders"], r["saved_dedup"]))
    print("  - + 持久 fsv16    -> %d（再省 %d）" % (r["opt_reorders"], r["saved_persist"]))
    tot = r["saved_dedup"] + r["saved_persist"]
    print("  合计省 %d / %d（%.0f%%）" % (
        tot, r["cur_reorders"],
        100.0 * tot / r["cur_reorders"] if r["cur_reorders"] else 0.0))
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="block layout 离线规划/分析")
    ap.add_argument("plans", nargs="*", help="plan 文件；缺省扫 models/*/model.plan")
    ap.add_argument("--tuning", default=None,
                    help="tuning.json（默认 config/tuning.json 或 $INFVINO_TUNING_CACHE）")
    ap.add_argument("--json", action="store_true", help="额外输出 JSON 摘要")
    ap.add_argument("--out", default=None, help="JSON 写入路径")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tuning_path = args.tuning or os.environ.get("INFVINO_TUNING_CACHE") or \
        os.path.join(repo, "config", "tuning.json")
    tuning = load_tuning(tuning_path)

    plans = args.plans or sorted(glob.glob(os.path.join(repo, "models", "*", "model.plan")))
    if not plans:
        print("no plan found", file=sys.stderr)
        return 1

    summary = {}
    for p in plans:
        r = analyze(p, tuning)
        print_report(r, tuning_path)
        summary[os.path.relpath(p, repo)] = {
            "conv3x3": r["conv3x3"],
            "blk_convs": len(r["blk_convs"]),
            "fsv16_tensors": r["fsv16"],
            "cur_reorders": r["cur_reorders"],
            "dedup_reorders": r["dedup_reorders"],
            "opt_reorders": r["opt_reorders"],
            "saved_dedup": r["saved_dedup"],
            "saved_persist": r["saved_persist"],
        }
    if args.json or args.out:
        text = json.dumps(summary, indent=2, ensure_ascii=False)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
            print("wrote", args.out)
        else:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
