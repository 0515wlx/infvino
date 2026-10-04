#!/usr/bin/env bash
# kernel_diag.sh —— 对一个 conv3x3 shape 做「墙归因」：跑 baseline + feed/store 隔离
# 探针，输出真实间距（hard_ratio）与告警，回答「差在哪堵墙、该造什么候选」。
#
# 依赖 R41/R43 的 `-DPROBE`（0=full, 1=noInput, 2=noWeight, 3=noBoth, 4=noStore）与
# `-DPF`。全部默认关闭、仅诊断用，数值不参与生产。
#
# 用法:
#   scripts/kernel_diag.sh <Cin,Cout,H,W> [STRIDE] [OBW] [OBH] [SLM] [ACT]
# 例:
#   scripts/kernel_diag.sh 64,64,160,160 2 6 2 4      # stride2 大层（R41 热点）
#   scripts/kernel_diag.sh 16,8,160,160 1 8 2 1       # Cout=8（lane 浪费）
set -u
SHAPE=${1:?usage: kernel_diag.sh Cin,Cout,H,W [STRIDE OBW OBH SLM ACT]}
STRIDE=${2:-1}; OBW=${3:-8}; OBH=${4:-2}; SLM=${5:-1}; ACT=${6:-1}
IMG=${INFVINO_IMAGE:-infvino-dev:latest}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ISA 指令配额（ocloc 实测 mad_frac × 32），用作 hard ceiling。
HARD_OV=20.3   # conv3x3_ov（R24/R41：297 mad / 432 instr ≈ 0.688，cap 20.3）

raw=$(docker run --rm --memory=2g --memory-swap=2g --pids-limit=256 \
  --device=/dev/dri/renderD128 -v "$ROOT":/workspace/infvino -w /workspace/infvino "$IMG" \
  bash -lc "export LD_LIBRARY_PATH=\$PWD/build
    for p in 0 1 2 3 4; do
      line=\$(OV_SLM=$SLM timeout 60 ./build/kernel_bench --op conv3x3ov \
        --conv-shape $SHAPE --conv $OBW,$OBH,1,32,16,$STRIDE,1,$ACT,3,1,16,0,0,0,0,0,0,\$p \
        --iters 10 2>/dev/null | tail -1)
      ops=\$(echo \"\$line\" | grep -oE 'ops/EU/cyc=[ 0-9.]*' | grep -oE '[0-9.]+')
      ms=\$(echo \"\$line\" | grep -oE '[0-9.]+ ms' | head -1 | grep -oE '[0-9.]+')
      sp=\$(echo \"\$line\" | grep -oE 'spread [+-][0-9]+' | grep -oE '[+-][0-9]+')
      echo \"\$p \$ops \$ms \$sp\"
    done")

echo "[kernel_diag] conv3x3_ov  shape=$SHAPE  s=$STRIDE  OBW=$OBW OBH=$OBH SLM=$SLM act=$ACT"
echo "  hard_ceiling(ov) = $HARD_OV ops/EU/cyc  (ISA 指令发射配额)"
printf "%s\n" "$raw" | awk -v hard="$HARD_OV" '
  $1=="0"{full=$2; ms=$3; sp=$4}
  $1=="1"{noin=$2} $1=="2"{now=$2} $1=="3"{noboth=$2} $1=="4"{nostore=$2}
  END{
    printf "  %-10s %6s\n","full",full; printf "  %-10s %6s\n","noInput",noin;
    printf "  %-10s %6s\n","noWeight",now; printf "  %-10s %6s\n","noBoth",noboth;
    printf "  %-10s %6s\n","noStore",nostore;
    hr = (hard>0)? full/hard : 0;
    printf "\n  hard_ratio = %.2f  (measured %.1f / hard %.1f)   min=%.3f ms spread=%s%%\n", hr, full, hard, ms, sp;
    printf "  --- verdict ---\n";
    if (nostore > full*1.25) printf "  * STORE/DRAM-BW bound: noStore=%.1f vs full=%.1f (+%.0f%%) -> fuse consumer / persist layout\n", nostore, full, (nostore/full-1)*100;
    if (noboth > hard*0.55) {
      src = (noin-full > now-full) ? "INPUT" : "WEIGHT";
      printf "  * %s-FEED latency bound: noBoth=%.1f (+%.0f%%), noInput=%.1f noWeight=%.1f, PF is ineffective -> raise occupancy (SLM_DIV)\n", src, noboth, (noboth/full-1)*100, noin, now;
    }
    if (noboth < hard*0.35) printf "  * STRUCTURAL (fixed overhead / lane waste / grid): noBoth=%.1f < 0.35*hard=%.1f -> check Cout%%32, small Cin/grid; feeds are NOT the wall\n", noboth, hard*0.35;
    if (noboth > hard*0.85 && full > hard*0.75) printf "  * issue-saturated: no same-family candidate will help\n";
  }'
