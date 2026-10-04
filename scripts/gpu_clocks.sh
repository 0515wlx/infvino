#!/usr/bin/env bash
# gpu_clocks.sh —— 锁定/恢复 iGPU 频率，消除 DVFS 抖动（性能测量用）。
#
# 背景：i915 默认 DVFS，短 kernel 会在 100–1300 MHz 间爬频；同一 binary 重复测量
# 会看到 ±10–20% 的漂移（见 docs/round41 §噪声）。把 gt_min_freq_mhz 钉到 RP0
# 可让测量只反映 kernel，而不是调频器。
#
# 用法:
#   scripts/gpu_clocks.sh status     # 打印当前 min/max/act/RP0/RPn
#   scripts/gpu_clocks.sh lock       # min=max=RP0（跑基准前）
#   scripts/gpu_clocks.sh unlock     # min=RPn, max=RP0（恢复默认）
#
# 权限：写 /sys 需要 root。本机 host 无免密 sudo，但 docker 里是 root，且 sysfs
# 文件可以 rw bind-mount 进去（已验证）。因此脚本默认走 docker；若本身是 root
# 则直接写。
set -u
D=/sys/class/drm/card0
IMG=${GPU_CLOCKS_IMAGE:-ubuntu:24.04}
[ -e "$D/gt_min_freq_mhz" ] || { echo "[gpu_clocks] $D not found (no i915 sysfs)" >&2; exit 2; }

inner() {
  # $1 = lock|unlock|status
  local act=$1
  local min max rp0 rpn
  rp0=$(cat /c/gt_RP0_freq_mhz 2>/dev/null || echo 1300)
  rpn=$(cat /c/gt_RPn_freq_mhz 2>/dev/null || echo 100)
  case "$act" in
    lock)   echo "$rp0" > /c/gt_max_freq_mhz; echo "$rp0" > /c/gt_min_freq_mhz;;
    unlock) echo "$rpn" > /c/gt_min_freq_mhz; echo "$rp0" > /c/gt_max_freq_mhz;;
  esac
  echo "[gpu_clocks] $act -> min=$(cat /c/gt_min_freq_mhz) max=$(cat /c/gt_max_freq_mhz) act=$(cat /c/gt_act_freq_mhz) RP0=$rp0 RPn=$rpn"
}

run() {
  if [ "$(id -u)" = "0" ]; then
    mkdir -p /tmp/gpu_clocks && for f in gt_min_freq_mhz gt_max_freq_mhz gt_act_freq_mhz gt_RP0_freq_mhz gt_RPn_freq_mhz; do ln -sf "$D/$f" "/tmp/gpu_clocks/$f" 2>/dev/null; done
    # root fallback writes directly
    local rp0 rpn; rp0=$(cat "$D/gt_RP0_freq_mhz"); rpn=$(cat "$D/gt_RPn_freq_mhz")
    case "$1" in
      lock) echo "$rp0" > "$D/gt_max_freq_mhz"; echo "$rp0" > "$D/gt_min_freq_mhz";;
      unlock) echo "$rpn" > "$D/gt_min_freq_mhz"; echo "$rp0" > "$D/gt_max_freq_mhz";;
    esac
    echo "[gpu_clocks] $1 -> min=$(cat $D/gt_min_freq_mhz) max=$(cat $D/gt_max_freq_mhz)"
    return
  fi
  command -v docker >/dev/null || { echo "[gpu_clocks] no docker and not root" >&2; exit 2; }
  docker run --rm --entrypoint bash \
    -v "$D/gt_min_freq_mhz:/c/gt_min_freq_mhz:rw" \
    -v "$D/gt_max_freq_mhz:/c/gt_max_freq_mhz:rw" \
    -v "$D/gt_act_freq_mhz:/c/gt_act_freq_mhz:ro" \
    -v "$D/gt_RP0_freq_mhz:/c/gt_RP0_freq_mhz:ro" \
    -v "$D/gt_RPn_freq_mhz:/c/gt_RPn_freq_mhz:ro" \
    "$IMG" -c "$(declare -f inner); inner $1"
}

case "${1:-status}" in
  lock|unlock|status) run "$1";;
  *) echo "usage: $0 lock|unlock|status" >&2; exit 2;;
esac
