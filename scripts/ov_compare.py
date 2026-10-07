#!/usr/bin/env python3
"""ov_compare.py —— OpenVINO 逐层 profiling 与 infvino 逐节点实测按 **算子形状精确 join**。

回答「逐层谁快、快在哪类形状」，并把两侧放进同一套 ops/EU/cyc 量纲。

用法：
  .venv-ov/bin/python scripts/ov_layer_profile.py --model yolov8n-pose --device GPU \
      --out /tmp/ov_y8.json
  python3 scripts/ov_compare.py --ov /tmp/ov_y8.json --our /tmp/y8.p20.json

约定：
  * 键 = (Cin, Cout, Hout, Wout, stride, K, kind)；`depthwise` 的 K 归一到 0（按 shape 匹配）。
  * `conv1x1_cat4`（infvino 把 concat 融进 1x1）归一到 `conv1x1`，与 OV 的独立 Concat+1x1 对齐。
  * ops/EU/cyc 按**每层**算（键会聚合重复层，故用节点数还原每层耗时）。
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict

EU, CLK = 80, 1.3e9


def flops_for(Cin, Cout, H, W, K, kind, N=None):
    if kind == "depthwise":
        return 2.0 * Cout * H * W * K * K
    if kind == "conv1x1":
        return 2.0 * Cout * (N if N else H * W) * Cin
    return 2.0 * Cout * H * W * Cin * K * K


def our_key(sig):
    """返回 (key7, flops) 或 None。"""
    if sig.startswith("conv3x3|"):
        m = re.search(r"W(\d+)H(\d+)s(\d+)p(\d+)_Cin(\d+)_Cout(\d+)_", sig)
        if m:
            W, H, st, p, Cin, Cout = map(int, m.groups())
            return (Cin, Cout, H, W, st, 3, "conv3x3"), flops_for(Cin, Cout, H, W, 3, "conv3x3")
    if sig.startswith("depthwise|"):
        m = re.search(r"W(\d+)H(\d+)s(\d+)p(\d+)_Cin(\d+)_Cout(\d+)_K(\d+)", sig)
        if m:
            W, H, st, p, Cin, Cout, K = map(int, m.groups())
            return (Cin, Cout, H, W, st, 0, "depthwise"), flops_for(Cin, Cout, H, W, K, "depthwise")
    if sig.startswith("conv1x1|") or sig.startswith("conv1x1_cat4|"):
        m = re.search(r"Cout(\d+)_N(\d+)_Cin(\d+)_", sig)
        if m:
            Cout, N, Cin = map(int, m.groups())
            s = int(round(N ** 0.5))
            return (Cin, Cout, s, s, 1, 1, "conv1x1"), flops_for(Cin, Cout, s, s, 1, "conv1x1", N)
    return None


def ov_key(n):
    """返回 (key7, flops) 或 None。只处理卷积类节点。"""
    typ = n.get("type", "")
    if "Convolution" not in typ:
        return None
    ins = n.get("in_shapes") or []
    x = ins[0] if ins else None
    o = n.get("out_shape")
    if not (x and o and len(x) == 4 and len(o) == 4):
        return None
    Cin = x[1]
    Cout_o, Ho, Wo = o[1], o[2], o[3]
    st = (n.get("strides") or [1, 1])[0]
    typ = n.get("type", "")
    K, groups = 0, 1
    w = ins[1] if len(ins) > 1 else None
    if w:
        if len(w) == 4:
            _O, I, Kh, Kw = w
            K, groups = Kh, max(1, Cin // max(1, I))
        elif len(w) == 5:
            G, _O, _I, Kh, Kw = w
            K, groups = Kh, G
    if K == 0:
        K = 3 if "GroupConvolution" in typ else 1
    Keff = K
    if "GroupConvolution" in typ or (groups == Cin and Cin == Cout_o and K > 1):
        kind, K = "depthwise", 0
    elif K == 1:
        kind = "conv1x1"
    elif K == 3:
        kind = "conv3x3"
    else:
        kind = f"conv{K}x{K}"
    return (Cin, Cout_o, Ho, Wo, st, K, kind), flops_for(
        Cin, Cout_o, Ho, Wo, Keff, kind, N=Ho * Wo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ov", required=True)
    ap.add_argument("--our", required=True)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    ov = json.load(open(args.ov))
    ours = json.load(open(args.our))

    our_ms, our_n, our_fl = defaultdict(float), defaultdict(int), defaultdict(float)
    for n in ours["nodes"]:
        r = our_key(n["signature"])
        if r:
            k, f = r
            our_ms[k] += n.get("ms_per_frame", 0.0)
            our_n[k] += 1
            our_fl[k] += f

    ov_us, ov_n, ov_fl = defaultdict(float), defaultdict(int), defaultdict(float)
    for n in ov["nodes"]:
        r = ov_key(n)
        if r:
            k, f = r
            ov_us[k] += n["us"]
            ov_n[k] += 1
            ov_fl[k] += f

    keys = sorted(set(our_ms) | set(ov_us),
                  key=lambda k: -(our_ms.get(k, 0.0) + ov_us.get(k, 0.0) / 1000.0))
    total_our = sum(our_ms.values())
    total_ov = sum(ov_us.values()) / 1000.0
    print(f"===== {ov['model']} : OpenVINO vs infvino（相同 shape 精确 join）=====")
    print(f"  OV 逐层合计 {total_ov:8.3f} ms | infvino 对应节点合计 {total_our:8.3f} ms "
          f"| OV/infvino = {total_ov/total_our if total_our else 0:.3f}")
    print(f"  {'shape (Cin->Cout @HxW s)':40s} {'kind':10s} {'our_ms':>8} {'ov_ms':>8} "
          f"{'ov/our':>7} {'our_ops':>8} {'ov_ops':>8}")
    for k in keys[:args.top]:
        Cin, Cout, H, W, st, K, kind = k
        on = our_n.get(k, 0)
        vn = ov_n.get(k, 0)
        om = our_ms.get(k, 0.0)
        vm = ov_us.get(k, 0.0) / 1000.0
        our_ops = (our_fl.get(k, 0.0) / on) / (EU * CLK * (om / on) * 1e-3) if on and om else 0.0
        ov_ops = (ov_fl.get(k, 0.0) / vn) / (EU * CLK * (vm / vn) * 1e-3) if vn and vm else 0.0
        r = (vm / om) if om > 0 else float("nan")
        print(f"  {f'{Cin}->{Cout} @{H}x{W} s{st}':40s} {kind:10s} {om:8.4f} {vm:8.4f} "
              f"{r:7.2f} {our_ops:8.2f} {ov_ops:8.2f}")

    print("\n-- 分类聚合 --")
    cats = defaultdict(lambda: [0.0, 0.0, 0])
    for k in set(our_ms) | set(ov_us):
        kind = k[6]
        cats[kind][0] += our_ms.get(k, 0.0)
        cats[kind][1] += ov_us.get(k, 0.0) / 1000.0
        cats[kind][2] += 1
    for kind, (om, vm, n) in sorted(cats.items(), key=lambda kv: -kv[1][0]):
        r = vm / om if om else float("nan")
        print(f"  {kind:12s} our={om:8.3f} ms  ov={vm:8.3f} ms  ov/our={r:5.2f}  ({n} shapes)")


if __name__ == "__main__":
    main()
