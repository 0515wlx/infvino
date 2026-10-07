#!/usr/bin/env python3
"""OV 逐层 profiling：把 OpenVINO 的 per-node 真实耗时导出为 JSON。

用法：
  .venv-ov/bin/python scripts/ov_layer_profile.py --model yolov8n-pose --device GPU \
      --iters 20 --warmup 5 --out /tmp/ov_y8.json

输出每个节点的 node_name / node_type / exec_type / real_time(us)，以及按 node_type 聚合。
供 scripts/analyze_walls.py 的 OV 对照路径使用（同一套绑定墙模型）。
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型名（models/<name>.onnx）或 onnx 路径")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import openvino as ov
    import numpy as np

    mp = Path(args.model)
    if mp.suffix != ".onnx":
        mp = REPO / "models" / f"{args.model}.onnx"
    core = ov.Core()
    model = core.read_model(str(mp))
    inp = model.input(0)
    iname = inp.get_any_name()
    shape = [int(d.get_length()) for d in inp.partial_shape]
    data = np.random.default_rng(0).random(shape).astype(np.float32)

    compiled = core.compile_model(model, args.device, {"PERF_COUNT": True})
    req = compiled.create_infer_request()

    # 逐 op 元信息（类型 + 静态形状 + 卷积属性），按 friendly_name 建索引，供逐层对照。
    opmeta = {}
    for op in model.get_ordered_ops():
        try:
            meta = {"type": op.get_type_name()}
            try:
                meta["out_shape"] = [int(d) for d in op.get_output_shape(0)]
            except Exception:
                meta["out_shape"] = None
            ins = []
            for i in range(op.get_input_size()):
                try:
                    ins.append([int(d) for d in op.get_input_shape(i)])
                except Exception:
                    ins.append(None)
            meta["in_shapes"] = ins
            for attr in ("get_strides", "get_dilations", "get_pads_begin", "get_pads_end"):
                fn = getattr(op, attr, None)
                if fn is not None:
                    try:
                        meta[attr] = [int(v) for v in fn()]
                    except Exception:
                        pass
            opmeta[(op.get_friendly_name(), meta["type"])] = meta
        except Exception:
            continue
    for _ in range(args.warmup):
        req.infer({iname: data})

    # 端到端计时（与 openvino_baseline.py 同口径：wall）
    wall = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        req.infer({iname: data})
        wall.append((time.perf_counter() - t0) * 1e3)
    wall.sort()

    # 逐层 profiling（real_time 是纯设备/内核时间，单位 us）
    info = req.get_profiling_info()
    nodes = []
    for p in info:
        m = opmeta.get((p.node_name, p.node_type), {})
        if not m:
            # 融合/重命名后 profiling 名可能与模型 op 名不完全相等：做前缀回退匹配。
            for (kname, ktype), meta in opmeta.items():
                if ktype == p.node_type and (kname.startswith(p.node_name) or
                                             p.node_name.startswith(kname)):
                    m = meta
                    break
        nodes.append({
            "name": p.node_name,
            "type": p.node_type,
            "exec": p.exec_type,
            "us": float(p.real_time.total_seconds() * 1e6),
            "us_cpu": float(p.cpu_time.total_seconds() * 1e6),
            "index": int(getattr(p, "execution_index", 0)),
            "out_shape": m.get("out_shape"),
            "in_shapes": m.get("in_shapes"),
            "strides": m.get("get_strides"),
            "dilations": m.get("get_dilations"),
            "pads_begin": m.get("get_pads_begin"),
            "pads_end": m.get("get_pads_end"),
        })
    by_type = defaultdict(float)
    cnt_type = defaultdict(int)
    for n in nodes:
        by_type[n["type"]] += n["us"]
        cnt_type[n["type"]] += 1

    out = {
        "model": mp.stem,
        "device": args.device,
        "openvino_version": ov.__version__,
        "input_shape": shape,
        "wall_ms": {"mean": sum(wall) / len(wall), "median": wall[len(wall) // 2], "min": wall[0]},
        "profiled_total_us": sum(n["us"] for n in nodes),
        "n_nodes": len(nodes),
        "by_type_us": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "n_by_type": dict(cnt_type),
        "nodes": nodes,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[ov-prof] {mp.stem} {args.device}: wall median={out['wall_ms']['median']:.3f} ms, "
          f"profiled_total={out['profiled_total_us']/1000:.3f} ms, {len(nodes)} nodes")
    print("  by type (ms):")
    for t, us in list(out["by_type_us"].items())[:15]:
        print(f"    {t:28s} {us/1000:8.3f}  ({cnt_type[t]})")


if __name__ == "__main__":
    sys.exit(main())
