#!/usr/bin/env python3
"""ONNX -> 自研 kernel 执行计划（plan）转换器。

输出一个文本 plan（格式见 include/infvino/PlanModel.hpp）以及所有 initializer 的 fp16 .bin，
全部放在 --out-dir 下。init 文件按 basename 引用（相对 plan 所在目录），因此整个目录可整体搬迁。

用法:
  python3 onnx2plan.py --onnx models/yolov8n-pose.onnx --out-dir models/yolov8n-pose
  # 生成 models/yolov8n-pose/model.plan + init_*.bin
"""
import argparse
import os

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

FP16 = np.float16


def t2np(t):
    return numpy_helper.to_array(t)


class Converter:
    def __init__(self, onnx_path, bindir):
        self.model = shape_inference.infer_shapes(onnx.load(onnx_path))
        self.bindir = bindir
        self.inits = {i.name: t2np(i) for i in self.model.graph.initializer}
        # Constant 节点 -> initializer
        for n in self.model.graph.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        self.inits[n.output[0]] = numpy_helper.to_array(a.t)
        self.shapes = {}
        for vi in list(self.model.graph.value_info) + list(self.model.graph.input) + list(self.model.graph.output):
            d = [x.dim_value for x in vi.type.tensor_type.shape.dim]
            self.shapes[vi.name] = d
        self.lines = []
        self.declared = set()
        self.binfiles = {}  # name -> filename

    # ---- helpers ----
    def shape(self, name):
        s = self.shapes.get(name)
        if s is None:
            if name in self.inits:
                s = list(self.inits[name].shape)
                self.shapes[name] = s
            else:
                raise KeyError(f"unknown shape for {name}")
        return s

    def declare(self, name, dims):
        if name in self.declared:
            return
        self.declared.add(name)
        self.lines.append("tensor {} {}".format(name, " ".join(str(d) for d in dims)))

    def declare_init(self, name):
        if name in self.declared:
            return
        arr = self.inits[name].astype(FP16)
        base = f"init_{name.replace('/', '_')}.bin"
        arr.tofile(os.path.join(self.bindir, base))
        self.binfiles[name] = base
        self.declared.add(name)
        # init 路径按 basename 记录，运行期相对 plan 所在目录解析。
        self.lines.append("init {} {} {}".format(
            name, base, " ".join(str(d) for d in arr.shape)))

    def attr(self, node, name, default=None):
        for a in node.attribute:
            if a.name == name:
                return onnx.helper.get_attribute_value(a)
        return default

    def node_line(self, kind, ins, outs, **attrs):
        a = " ".join(f"{k}={v}" for k, v in attrs.items() if v is not None)
        self.lines.append("node {} {} {} {}".format(kind, ",".join(ins) or "-", ",".join(outs), a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out-dir", required=True, help="plan 与 init .bin 的输出目录")
    ap.add_argument("--plan-name", default="model.plan")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    c = Converter(args.onnx, args.out_dir)
    g = c.model.graph

    graph_inputs = [i.name for i in g.input if i.name not in c.inits]
    assert len(graph_inputs) == 1, graph_inputs
    xin = graph_inputs[0]
    c.declared.add(xin)
    c.lines.append("input {} {}".format(xin, " ".join(str(d) for d in c.shape(xin))))

    # --- 融合 Conv+SiLU：Conv -> Sigmoid -> Mul(Conv, Sigmoid) ---
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    fused_sigmoid = set()
    fused_mul = set()
    conv_act = {}  # conv原始输出名 -> 融合后应写出的张量名(Mul输出)
    for n in g.node:
        if n.op_type != "Conv":
            continue
        conv_out = n.output[0]
        cs = consumers.get(conv_out, [])
        sig = next((x for x in cs if x.op_type == "Sigmoid"), None)
        if sig is None:
            continue
        for m in consumers.get(sig.output[0], []):
            if m.op_type == "Mul" and set(m.input) == {conv_out, sig.output[0]}:
                conv_act[conv_out] = m.output[0]
                fused_sigmoid.add(sig.name)
                fused_mul.add(m.name)
                break

    def emit_conv(n):
        x, w = n.input[0], n.input[1]
        b = n.input[2] if len(n.input) > 2 else None
        wshape = c.shape(w)
        cout, cin, kh, kw = wshape[0], wshape[1], wshape[2], wshape[3]
        strides = c.attr(n, "strides", [1, 1])
        pads = c.attr(n, "pads", [0, 0, 0, 0])
        sh, sw = strides
        ph, pw = pads[0], pads[1]
        assert kh == kw, (kh, kw)
        raw_out = n.output[0]
        fused = raw_out in conv_act
        out = conv_act.get(raw_out, raw_out)
        oshape = c.shape(raw_out)
        act = 1 if fused else 0
        c.declare_init(w)
        if b:
            c.declare_init(b)
        c.declare(out, oshape)
        groups = c.attr(n, "group", 1)
        # 3x3 groups=1 走调优过的专用 kernel（含 stride2；s2 用更小的 CINC/TX
        # 以免输入 halo 超出 SLM）。其余走通用 kernel。
        use_direct = (kh == 3 and groups == 1 and sh in (1, 2))
        if use_direct:
            # cfg = TX,TY,TM,CB,CINC（STRIDE/PAD/ACT 由 PlanModel 从节点属性注入）
            # s2 的输入 halo 是 s1 的两倍：用 TM=1 / CB=32 提高通道复用、降低输入重读
            # 放大（Cout/CB），CINC=8 保证 halo 不超 SLM。
            cfg = None if sh == 1 else "64,8,1,32,8,2"
            c.node_line("conv3x3", [x, w, b or "-"], [out], stride=sh, pad=ph,
                        Hout=oshape[2], Wout=oshape[3], act=(act or None), cfg=cfg)
            return
        # 1x1 且 groups=1 -> gemm
        if kh == 1 and groups == 1:
            cur = out
            if b or fused:
                cur = out + "__t1"
                c.declare(cur, oshape)
            c.node_line("gemm", [w, x], [cur])
            if b:
                dst = out if not fused else out + "__t2"
                if fused:
                    c.declare(dst, oshape)
                c.node_line("bias_add", [cur, b], [dst], HW=oshape[2] * oshape[3])
                cur = dst
            if fused:
                c.node_line("ew_unary", [cur], [out], op=1, n=int(np.prod(oshape)))
            return
        # 通用（含 depthwise / 5x5 / stride2）
        c.node_line("conv_general", [x, w, b or "-"], [out], K=kh, S=sh, P=ph,
                    G=groups, Hout=oshape[2], Wout=oshape[3], act=(act or None))

    for n in g.node:
        op = n.op_type
        if n.name in fused_sigmoid or n.name in fused_mul:
            continue
        if op == "Constant":
            continue
        for ii in n.input:
            if ii in c.inits:
                c.declare_init(ii)
        if op == "Conv":
            emit_conv(n)
        elif op in ("Sigmoid", "Relu", "HardSwish", "HardSigmoid"):
            code = {"Sigmoid": 0, "Relu": 2, "HardSwish": 3, "HardSigmoid": 4}[op]
            out = n.output[0]
            c.declare(out, c.shape(out))
            c.node_line("ew_unary", [n.input[0]], [out], op=code, n=int(np.prod(c.shape(out))))
        elif op in ("Add", "Sub", "Mul", "Div"):
            code = {"Add": 0, "Sub": 1, "Mul": 2, "Div": 3}[op]
            a, b = n.input[0], n.input[1]
            out = n.output[0]
            oshape = c.shape(out)
            c.declare(out, oshape)
            asha, bsha = c.shape(a), c.shape(b)
            b_scalar = 0
            bdims = None
            if list(asha) != list(oshape) or (b in c.inits and list(c.inits[b].shape) != list(oshape)):
                rank = len(oshape)

                def align(s):
                    return [1] * (rank - len(s)) + list(s)

                def strides(s, aligned):
                    st = [0] * rank
                    acc = 1
                    for i in range(rank - 1, -1, -1):
                        st[i] = 0 if (aligned[i] == 1 and oshape[i] != 1) else acc
                        acc *= aligned[i]
                    return st

                a_al, b_al = align(asha), align(bsha)
                a_st = strides(asha, a_al)
                b_st = strides(bsha, b_al)
                bdims = (oshape, a_st, b_st)
            if b in c.inits:
                c.declare_init(b)
                if c.inits[b].size == 1 and list(asha) == list(oshape):
                    b_scalar = 1
            c.node_line("ew_binary", [a, b], [out], op=code,
                        n=int(np.prod(oshape)), b_scalar=b_scalar,
                        bdims=(",".join(",".join(str(x) for x in g_) for g_ in bdims) if bdims else None))
        elif op == "Concat":
            axis = c.attr(n, "axis", 1)
            out = n.output[0]
            oshape = c.shape(out)
            rank = len(oshape)
            if axis < 0:
                axis += rank
            outer = int(np.prod(oshape[:axis]))
            inner = int(np.prod(oshape[axis + 1:]))
            dims = [c.shape(i)[axis] for i in n.input]
            ins = list(n.input)
            while len(ins) < 4:
                ins.append("-")
                dims.append(0)
            for i in n.input:
                if i in c.inits:
                    c.declare_init(i)
            c.declare(out, oshape)
            c.node_line("concat4", ins[:4], [out], ca=dims[0], cb=dims[1], cc=dims[2],
                        cd=dims[3], outer=outer, inner=inner)
        elif op == "Split":
            x = n.input[0]
            axis = c.attr(n, "axis", 0)
            xshape = c.shape(x)
            if axis < 0:
                axis += len(xshape)
            if axis == 1:
                c0 = 0
                for out in n.output:
                    oshape = c.shape(out)
                    cnt = oshape[axis]
                    c.declare(out, oshape)
                    c.node_line("copy_c", [x], [out], HW=int(np.prod(oshape[axis + 1:])),
                                c0=c0, cnt=cnt, dst_off=0)
                    c0 += cnt
            else:
                outer = int(np.prod(xshape[:axis])) if axis > 0 else 1
                inner = int(np.prod(xshape[axis + 1:])) if axis + 1 < len(xshape) else 1
                start = 0
                for out in n.output:
                    oshape = c.shape(out)
                    length = oshape[axis]
                    c.declare(out, oshape)
                    c.node_line("slice_axis", [x], [out], outer=outer, axdim=xshape[axis],
                                inner=inner, start=start, len=length)
                    start += length
        elif op == "Slice":
            x = n.input[0]
            starts = c.inits[n.input[1]].tolist()
            ends = c.inits[n.input[2]].tolist()
            axes = c.inits[n.input[3]].tolist() if len(n.input) > 3 else list(range(len(starts)))
            steps = c.inits[n.input[4]].tolist() if len(n.input) > 4 else [1] * len(starts)
            assert len(starts) == 1 and steps[0] == 1, (starts, steps)
            axis = axes[0]
            xshape = c.shape(x)
            if axis < 0:
                axis += len(xshape)
            start = int(starts[0])
            end = int(ends[0])
            if end > 1 << 30 or end == 9223372036854775807:
                end = xshape[axis]
            out = n.output[0]
            oshape = c.shape(out)
            outer = int(np.prod(xshape[:axis]))
            inner = int(np.prod(xshape[axis + 1:]))
            c.declare(out, oshape)
            c.node_line("slice_axis", [x], [out], outer=outer, axdim=xshape[axis],
                        inner=inner, start=start, len=end - start)
        elif op == "MaxPool":
            x = n.input[0]
            out = n.output[0]
            xshape = c.shape(x)
            oshape = c.shape(out)
            K = c.attr(n, "kernel_shape", [5, 5])[0]
            S = c.attr(n, "strides", [1, 1])[0]
            pads = c.attr(n, "pads", [0, 0, 0, 0])
            c.declare(out, oshape)
            c.node_line("maxpool", [x], [out], C=xshape[1], H=xshape[2], W=xshape[3],
                        Hout=oshape[2], Wout=oshape[3], K=K, S=S, P=pads[0])
        elif op == "Resize":
            x = n.input[0]
            out = n.output[0]
            xshape = c.shape(x)
            oshape = c.shape(out)
            assert oshape[2] % xshape[2] == 0
            S = oshape[2] // xshape[2]
            c.declare(out, oshape)
            c.node_line("resize_nn", [x], [out], C=xshape[1], H=xshape[2], W=xshape[3], S=S)
        elif op in ("Reshape", "Flatten"):
            out = n.output[0]
            oshape = c.shape(out)
            c.declare(out, oshape)
            c.node_line("reshape", [n.input[0]], [out])
        elif op == "Transpose":
            perm = c.attr(n, "perm", None)
            out = n.output[0]
            oshape = c.shape(out)
            xshape = c.shape(n.input[0])
            assert perm[0] == 0, perm
            mode = 0 if perm == [0, 2, 1, 3] else 1
            assert perm in ([0, 2, 1, 3], [0, 1, 3, 2]), perm
            c.declare(out, oshape)
            c.node_line("permute_0213", [n.input[0]], [out], D1=xshape[1], D2=xshape[2],
                        I=int(np.prod(xshape[3:])), mode=mode)
        elif op == "Softmax":
            axis = c.attr(n, "axis", 1)
            x = n.input[0]
            out = n.output[0]
            xshape = c.shape(x)
            if axis < 0:
                axis += len(xshape)
            axdim = xshape[axis]
            outer = int(np.prod(xshape[:axis])) if axis > 0 else 1
            inner = int(np.prod(xshape[axis + 1:])) if axis + 1 < len(xshape) else 1
            c.declare(out, c.shape(out))
            c.node_line("softmax_axis", [x], [out], outer=outer, axdim=axdim, inner=inner)
        elif op == "MatMul":
            a, b = n.input[0], n.input[1]
            out = n.output[0]
            asha, bsha = c.shape(a), c.shape(b)
            c.declare(out, c.shape(out))
            c.node_line("bmm", [a, b], [out], B0=asha[0], B1=asha[1], M=asha[2], K=asha[3], N=bsha[3])
        elif op == "GlobalAveragePool":
            x = n.input[0]
            out = n.output[0]
            xshape = c.shape(x)
            c.declare(out, c.shape(out))
            c.node_line("gap", [x], [out], C=xshape[1], HW=xshape[2] * xshape[3])
        elif op == "Gemm":
            a, b = n.input[0], n.input[1]
            out = n.output[0]
            wshape = c.shape(b) if c.attr(n, "transB", 0) else c.shape(a)
            xshape = c.shape(a)
            K = xshape[1]
            N = wshape[0]
            c.declare_init(b)
            c.declare(out, c.shape(out))
            if len(n.input) > 2 and n.input[2] in c.inits:
                bi = n.input[2]
                c.declare_init(bi)
                tmp = out + "__pre"
                c.declare(tmp, c.shape(out))
                c.node_line("gemm", [b, a], [tmp])
                c.node_line("bias_add", [tmp, bi], [out], HW=1)
            else:
                c.node_line("gemm", [b, a], [out])
        else:
            raise NotImplementedError(f"{op} ({n.name})")

    # 输出
    for o in g.output:
        c.declare(o.name, c.shape(o.name))
        c.lines.append(f"output {o.name}")

    plan = os.path.join(args.out_dir, args.plan_name)
    with open(plan, "w") as f:
        f.write("\n".join(c.lines) + "\n")
    print(f"[onnx2plan] wrote {plan}: {len(c.lines)} lines, {len(c.binfiles)} inits")


if __name__ == "__main__":
    main()
