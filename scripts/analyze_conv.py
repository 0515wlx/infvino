#!/usr/bin/env python3
"""分析三个模型里的卷积层：FLOPs、算术强度(AI)、roofline 定位。

用于 Round 2 决策：im2col→GEMM vs 直接卷积。
输出每个模型按 FLOPs 排序的 conv 表，以及汇总。

用法:
  python3 analyze_conv.py --models-dir models
"""
import argparse
import os

import numpy as np
import onnx
from onnx import shape_inference

MODELS = ["yolov8n-pose.onnx", "yolo11n-pose.onnx", "mobilenetv3-small.onnx"]


def get_shapes(model):
    """返回 name -> shape（int 或 None）。"""
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    shapes = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        dims = [d.dim_value if d.HasField("dim_value") else None
                for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def conv_layers(path):
    model = onnx.load(path)
    shapes = get_shapes(model)
    inits = {i.name: i for i in model.graph.initializer}
    rows = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        w = inits.get(node.input[1])
        if w is None:
            continue
        cout, cin_g, kh, kw = list(w.dims)
        a = {x.name: onnx.helper.get_attribute_value(x) for x in node.attribute}
        strides = list(a.get("strides", [1, 1]))
        groups = int(a.get("group", 1))
        pads = list(a.get("pads", [0, 0, 0, 0]))
        xin = shapes.get(node.input[0])
        xout = shapes.get(node.output[0])
        if not xin or not xout or len(xin) != 4 or len(xout) != 4:
            continue
        cin = xin[1]
        hin, win = xin[2], xin[3]
        hout, wout = xout[2], xout[3]
        if None in (cin, hin, win, hout, wout):
            continue
        flops = 2 * cout * hout * wout * cin_g * kh * kw
        wbytes = cout * cin_g * kh * kw * 2
        ibytes = cin * hin * win * 2
        obytes = cout * hout * wout * 2
        ai_min = flops / (wbytes + ibytes + obytes)
        ai_wstream = flops / (wbytes + obytes)  # weights stream from DRAM (large layers)
        rows.append(dict(
            name=node.name or node.output[0], cin=cin, cout=cout, kh=kh, kw=kw,
            stride=strides[0], group=groups, hout=hout, wout=wout,
            flops=flops, ai_min=ai_min, ai_wstream=ai_wstream,
            wkb=wbytes / 1024, depthwise=(groups == cin and cin > 1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    for m in MODELS:
        p = os.path.join(args.models_dir, m)
        if not os.path.exists(p):
            print(f"[skip] {p}")
            continue
        rows = conv_layers(p)
        total = sum(r["flops"] for r in rows)
        print(f"\n================ {m} ================")
        print(f"conv layers={len(rows)}  total conv FLOPs={total/1e9:.3f} G")
        rows.sort(key=lambda r: -r["flops"])
        print(f"{'layer':26s} {'Cin':>4} {'Cout':>5} {'k':>2} {'s':>1} {'g':>4} "
              f"{'HW':>7} {'GFLOPs':>8} {'cum%':>5} {'AI_min':>8} {'AI_wstr':>8}")
        cum = 0.0
        for r in rows:
            cum += r["flops"]
            print(f"{r['name'][:26]:26s} {r['cin']:>4} {r['cout']:>5} {r['kh']:>2} {r['stride']:>1} "
                  f"{r['group']:>4} {r['hout']:>3}x{r['wout']:<3} {r['flops']/1e9:>8.3f} "
                  f"{cum/total*100:>5.1f} {r['ai_min']:>8.0f} {r['ai_wstream']:>8.0f}")
        # top-N to cover 90% flops
        cum = 0.0
        n90 = 0
        for r in rows:
            cum += r["flops"]
            n90 += 1
            if cum / total >= 0.9:
                break
        print(f"-> top {n90} layers cover 90% of conv FLOPs")
        dw = [r for r in rows if r["depthwise"]]
        print(f"-> depthwise layers={len(dw)} "
              f"({sum(r['flops'] for r in dw)/total*100:.1f}% of FLOPs)")


if __name__ == "__main__":
    raise SystemExit(main())
