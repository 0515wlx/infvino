#!/usr/bin/env bash
# noise_check.sh —— 同条件重复测量，检测 LLC/DRAM 噪声（R42）。
#
# 背景：噪声主因是工作集跨 3.75 MB L3 边界时的 DRAM 带宽骤降（见
# docs/round42-rulers-noise-and-tooling.md §3）。同一 binary、同一 shape 连跑会
# 出现偶发慢尾。判据：取每次的 **min**（内禀成本），min 之间的离散度应 < ~10%。
#
# 用法:
#   scripts/noise_check.sh [REPEAT] [MODEL_CFG]
#   # 默认跑一个 4.2 MB footprint 的 stride-2 conv（正好压在 L3 边界）
#
# 建议在跑基准前执行；若 FAIL：先 scripts/gpu_clocks.sh lock、确认无其它容器/负载，
# 再复测；仍 FAIL 则说明该 shape 本身受 LLC 边界影响，报告中应标注。
set -u
REPEAT=${1:-6}
IMG=${INFVINO_IMAGE:-infvino-dev:latest}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 时钟检查（host sysfs；只读，不需要 root）
CLK="/sys/class/drm/card0"
if [ -e "$CLK/gt_min_freq_mhz" ]; then
  mn=$(cat "$CLK/gt_min_freq_mhz"); mx=$(cat "$CLK/gt_max_freq_mhz")
  if [ "$mn" != "$mx" ]; then
    echo "[noise_check] NOTE: GPU clocks not pinned (min=$mn max=$mx) -> run scripts/gpu_clocks.sh lock" >&2
  else
    echo "[noise_check] GPU clocks pinned at $mx MHz"
  fi
fi

echo "[noise_check] $REPEAT repeats of a boundary-footprint conv (4.2 MB, out80 s2 64->64)"
vals=$(docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$ROOT":/workspace/infvino -w /workspace/infvino "$IMG" bash -lc '
    export LD_LIBRARY_PATH=$PWD/build
    for i in $(seq 1 '"$REPEAT"'); do
      OV_SLM=4 ./build/kernel_bench --op conv3x3ov --conv-shape 64,64,160,160 \
        --conv 6,2,1,32,16,2,1,1,3,1,16 --iters 20 2>/dev/null \
        | grep -oE "min [0-9.]+" | grep -oE "[0-9.]+"
    done')

if [ -z "$vals" ]; then echo "[noise_check] FAIL: no measurements" >&2; exit 2; fi
echo "[noise_check] per-repeat min (ms): $(echo $vals | tr '\n' ' ')"
stats=$(echo "$vals" | awk '
  {v[n++]=$1; if($1<mn||n==1)mn=$1; if($1>mx)mx=$1; s+=$1}
  END{printf "min=%.4f max=%.4f spread=%.1f%%", mn, mx, (mx/mn-1)*100}')
echo "[noise_check] $stats"
sp=$(echo "$vals" | awk '{v[n++]=$1; if($1<mn||n==1)mn=$1; if($1>mx)mx=$1} END{printf "%.1f",(mx/mn-1)*100}')
if awk -v s="$sp" 'BEGIN{exit !(s<10.0)}'; then
  echo "[noise_check] PASS (spread < 10%)"
else
  echo "[noise_check] FAIL (spread >= 10%): investigate LLC/DRAM or environment" >&2
  exit 1
fi
