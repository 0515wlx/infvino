#!/usr/bin/env python3
# analyze_walls.py —— R71 绑定墙分析（纯离线）。
#
# 把 `kernel_autotune --wall-report` 的逐签名墙判决与 `kernel_run --profile-json`
# 的逐节点实测 ms 按 signature **join**，得到**按调用次数加权的墙预算**：
#   计算墙（FMA 指令配额 / 寄存器 ILP / SLM 带宽 / SLM 容量）
#   带宽墙（GPU 私有 L3 / 共享 LLC / DRAM）
#   以及 dispatch/网格墙（launch）。
#
# 为什么需要 join：wall-report 是「每唯一签名一行」，而 profile-json 是「每节点」。
# 同一签名可能被多个层复用（如 4× 80×80 conv3x3），只有 join 才能还原真实时间占比。
#
# 用法：
#   kernel_autotune --plan models/<m>/model.plan --wall-report > /tmp/wall_<m>.txt
#   kernel_run --plan models/<m>/model.plan --report --profile-json /tmp/<m>.json
#   python3 scripts/analyze_walls.py --model <m> --wall /tmp/wall_<m>.txt --profile /tmp/<m>.json
import argparse
import json
import sys
from collections import defaultdict

WALLS = ["FMA-issue", "reg-ILP", "SLM-bw", "SLM-cap", "GPU-L3", "shared-LLC", "DRAM", "launch"]


def parse_wall_report(path):
    """signature -> dict(kernel, ms, meas, hard, ratio, foot_mb, binding, tier, llc_mb)."""
    rows = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("--") or line.startswith("=="):
            continue
        toks = line.split()
        if len(toks) != 10:
            continue
        if toks[0] == "signature":
            continue
        sig, kernel, ms, meas, hard, ratio, foot, binding, tier, llc = toks
        rows[sig] = dict(kernel=kernel, ms=float(ms), meas=float(meas), hard=float(hard),
                         ratio=float(ratio), foot_mb=float(foot), binding=binding,
                         tier=tier, llc_mb=float(llc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wall", required=True)
    ap.add_argument("--profile", required=True)
    args = ap.parse_args()

    walls = parse_wall_report(args.wall)
    prof = json.load(open(args.profile))
    nodes = prof["nodes"]

    # call-weighted join
    by_wall = defaultdict(lambda: [0.0, 0])
    by_tier = defaultdict(lambda: [0.0, 0])
    per_sig = defaultdict(float)
    unjoined = 0.0
    total = 0.0
    for n in nodes:
        ms = n.get("ms_per_frame", 0.0)
        total += ms
        sig = n["signature"]
        w = walls.get(sig)
        if w is None:
            unjoined += ms
            continue
        by_wall[w["binding"]][0] += ms
        by_wall[w["binding"]][1] += 1
        by_tier[w["tier"]][0] += ms
        by_tier[w["tier"]][1] += 1
        per_sig[sig] += ms

    print(f"===== {args.model} =====  busy(profile) = {total:.3f} ms, "
          f"unjoined = {unjoined:.3f} ms")
    print("\n-- wall budget (call-weighted) --")
    for wk in WALLS:
        if wk in by_wall:
            ms, n = by_wall[wk]
            print(f"  {wk:11s} {ms:8.3f} ms  {100*ms/total:5.1f}%  ({n} nodes)")
    print("\n-- memory tier budget --")
    for tk in ["GPU-L3", "shared-LLC", "DRAM"]:
        if tk in by_tier:
            ms, n = by_tier[tk]
            print(f"  {tk:11s} {ms:8.3f} ms  {100*ms/total:5.1f}%  ({n} nodes)")

    # 计算墙的「离墙差距」——FMA-issue/reg-ILP 节点按 ms 排序，看硬上限与实测。
    print("\n-- top compute-bound nodes (measured ms, wallR = measured/hard) --")
    comp = [(ms, sig) for sig, ms in per_sig.items() if walls[sig]["binding"] in
            ("FMA-issue", "reg-ILP", "SLM-bw")]
    comp.sort(reverse=True)
    print(f"  {'ms':>8} {'measured':>9} {'hard':>7} {'wallR':>6} {'footMB':>7}  {'binding':<9} signature")
    for ms, sig in comp[:15]:
        w = walls[sig]
        print(f"  {ms:8.3f} {w['meas']:9.2f} {w['hard']:7.2f} {w['ratio']:6.2f} "
              f"{w['foot_mb']:7.2f}  {w['binding']:<9} {sig}")

    # 可回收 headroom（若贴住硬上限）——注意：物理上多不可达（见 R41）。
    hrs = 0.0
    for n in nodes:
        w = walls.get(n["signature"])
        if w and w["hard"] > 0 and w["ratio"] < 1.0:
            hrs += n["ms_per_frame"] * (1.0 - w["ratio"])
    print(f"\n== 若所有节点贴住其绑定墙硬上限，可省 {hrs:.3f} ms "
          f"({100*hrs/total:.1f}% busy) [R41: conv 侧多为结构/延迟墙，非可回收] ==")


if __name__ == "__main__":
    sys.exit(main())
