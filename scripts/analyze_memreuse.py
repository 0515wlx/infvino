#!/usr/bin/env python3
# Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
#
# P0 配套：对 plan 做生存期分析，估算激活内存的可复用比例。
# 不需要 GPU，也不依赖 numpy。
import argparse
import collections
import re
import sys


def parse_plan(path):
    tensors = {}      # name -> dims (tuple[int])
    is_init = set()
    inputs = set()
    outputs = set()
    nodes = []        # (op, [ins], [outs])
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        kind = p[0]
        if kind == "input":
            tensors[p[1]] = tuple(int(x) for x in p[2:])
            inputs.add(p[1])
        elif kind == "init":
            dims = tuple(int(x) for x in p[3:])
            tensors[p[1]] = dims
            is_init.add(p[1])
        elif kind == "tensor":
            tensors[p[1]] = tuple(int(x) for x in p[2:])
        elif kind == "node":
            ins = p[2].split(",")
            outs = p[3].split(",")
            nodes.append((p[1], ins, outs))
        elif kind == "output":
            outputs.add(p[1])
    return tensors, is_init, inputs, outputs, nodes


def numel(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def analyze(path):
    tensors, is_init, inputs, outputs, nodes = parse_plan(path)
    N = len(nodes)

    # 每个张量的生产者与最后消费者
    prod = {}
    first_use = {}
    last_use = {}
    for i, (op, ins, outs) in enumerate(nodes):
        for t in ins:
            if t == "-":
                continue
            if t not in first_use:
                first_use[t] = i
            last_use[t] = i
        for o in outs:
            if o != "-":
                prod[o] = i

    intervals = []  # (birth, death, bytes, name)
    for name, dims in tensors.items():
        if name in is_init:
            continue
        b = numel(dims) * 2  # fp16
        if name in inputs:
            birth, death = 0, N - 1
        elif name in outputs:
            birth, death = first_use.get(name, 0), N - 1
        elif name in first_use:
            birth, death = prod.get(name, 0), last_use[name]
        else:
            continue  # 未使用的中间张量（应被裁剪）
        intervals.append((birth, death, b, name))

    # 按 survival 累加峰值
    events = collections.defaultdict(list)
    for birth, death, b, _ in intervals:
        events[birth].append(b)
        events[death + 1].append(-b)
    cur = peak = 0
    for i in range(N + 1):
        for d in events.get(i, []):
            cur += d
        peak = max(peak, cur)

    total = sum(b for _, _, b, _ in intervals)
    return dict(
        path=path,
        intervals=intervals,
        num_act=len(intervals),
        total=total,
        peak=peak,
        reuse=(1.0 - peak / total) if total else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="+", help="plan 文件路径")
    ap.add_argument("--top", type=int, default=0, help="额外列出占用最大的前 N 个张量")
    args = ap.parse_args()

    print(f"{'model':<22} {'act tensors':>11} {'sum alloc':>10} {'peak live':>10} {'reuse':>7}")
    print("-" * 66)
    for p in args.plans:
        try:
            r = analyze(p)
        except Exception as e:  # noqa: BLE001
            print(f"{p}: ERROR {e}", file=sys.stderr)
            continue
        name = p.split("/")[-2] if "/" in p else p
        print(f"{name:<22} {r['num_act']:>11} {r['total']/1e6:>9.1f}M "
              f"{r['peak']/1e6:>9.1f}M {r['reuse']*100:>6.0f}%")
        if args.top:
            for _, _, b, nm in sorted(r["intervals"], key=lambda x: -x[2])[: args.top]:
                print(f"      {b/1e6:8.2f}MB  {nm}")


if __name__ == "__main__":
    main()
