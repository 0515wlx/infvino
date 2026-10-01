#!/usr/bin/env python3
"""导出 ultralytics YOLO 模型为 ONNX（infvino 用）。

依赖: ultralytics, onnx  (在宿主机 venv 中运行即可，不需要 OpenVINO)
用法:
    python3 export_models.py --out models
    python3 export_models.py --out models --models yolov8n-pose yolo11n-pose

导出后：
  1) 把打印的 yaml 片段粘贴到 config/models.yaml；
  2) 用 onnx2plan.py 生成自研 kernel 的执行计划（后端必需）：
       python3 scripts/onnx2plan.py --onnx models/<name>.onnx --out-dir models/<name>
"""
import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models", help="ONNX 输出目录")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--models",
        nargs="*",
        default=["yolov8n-pose", "yolo11n-pose"],
        help="权重名（不含 .pt）",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from ultralytics import YOLO

    try:
        import onnx
    except ImportError:
        onnx = None

    for name in args.models:
        try:
            model = YOLO(f"{name}.pt")
            path = model.export(
                format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True, opset=args.opset
            )
            dst = os.path.join(args.out, name + ".onnx")
            os.replace(path, dst)

            meta = {}
            if onnx is not None:
                try:
                    meta = {kv.key: kv.value for kv in onnx.load(dst).metadata_props}
                except Exception:
                    meta = {}

            task = meta.get("task", "detect")
            imgsz = meta.get("imgsz", f"[{args.imgsz}, {args.imgsz}]")
            print(f"\n[{name}] -> {dst}")
            print(f"  task={task}  imgsz={imgsz}  kpt_shape={meta.get('kpt_shape', '-')}")
            print("  # 粘贴到 config/models.yaml:")
            print(f"  {name}:")
            print(f"    path: models/{name}.onnx")
            print(f"    plan: models/{name}/model.plan")
            print(f"    task: {task}")
            print(f"    imgsz: {imgsz}")
            if meta.get("kpt_shape"):
                print(f"    kpt_shape: {meta['kpt_shape']}")
            print("    num_classes: <填类别数>")
            print("    device: AUTO")
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
