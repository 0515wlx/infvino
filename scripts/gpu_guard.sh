#!/usr/bin/env bash
# gpu_guard —— 开发板 GPU 安全守卫（配合 docs/benchmark_protocol.md）。
#
# 用法:
#   scripts/gpu_guard.sh check        # 跑 GPU 前：检查 i915 健康 / 内存 / 残留渲染
#   scripts/gpu_guard.sh after        # 跑 GPU 后：检查是否出现 GPU HANG（退出码 3）
#   scripts/gpu_guard.sh run <cmd...> # 包一层 timeout + 事后 HANG 检查
#
# 背景：本开发板（i5-1135G7 + PREEMPT_RT）历史上多次因 `i915 GPU HANG ... in
# kernel_run` 而整机硬死机。GPU HANG 本身是驱动可恢复事件，但 heartbeat 复位
# 反复失败会拖死实时内核。守卫的目的是：
#   1. 跑前确认驱动没有处于 hang/reset 状态；
#   2. 跑后立即发现 HANG，避免继续叠加负载。
set -u

DMESG_HANG_PAT='GPU HANG|engine reset|Resetting .*rcs0|GuC.*reset'

# dmesg is root-only (kernel.dmesg_restrict=1); read the syslog files instead.
hang_tail() {
  { tail -n 400 /var/log/kern.log 2>/dev/null; tail -n 400 /var/log/syslog 2>/dev/null; }
}

check() {
  local rc=0
  local avail
  avail=$(free -m | awk '/^Mem:/{print $7}')
  echo "[gpu_guard] available RAM: ${avail} MB"
  if [ "${avail:-0}" -lt 500 ]; then
    echo "[gpu_guard] WARN: low memory (<500 MB available)" >&2
    rc=1
  fi
  if [ -d /sys/class/drm ] && ls /sys/class/drm/card*/device/gt0 2>/dev/null | head -1 >/dev/null; then
    :
  fi
  if hang_tail | grep -qE "$DMESG_HANG_PAT"; then
    echo "[gpu_guard] WARN: recent GPU HANG/reset in kernel log — do not run GPU work" >&2
    rc=1
  fi
  local stale
  stale=$(pgrep -x kernel_run 2>/dev/null; pgrep -x kernel_bench 2>/dev/null; \
          pgrep -x infvino_bench 2>/dev/null; pgrep -x infvino_numtest 2>/dev/null)
  if [ -n "$stale" ]; then
    echo "[gpu_guard] WARN: stale render process:" >&2
    echo "$stale" >&2
    rc=1
  fi
  # Known host-destabiliser: the third-party deploy-docker container runs
  # privileged + maps the whole /dev + has no memory limit and, when it is in a
  # crash/restart loop, floods the kernel with fault events (can hard-lock the
  # PREEMPT_RT host even with no GPU HANG). Warn if it is running/restarting.
  if command -v docker >/dev/null 2>&1; then
    local dd
    dd=$(docker ps --filter name=deploy-docker --format '{{.Status}}' 2>/dev/null)
    if echo "$dd" | grep -qi 'Restarting\|Up '; then
      echo "[gpu_guard] WARN: deploy-docker is running/restarting (known host destabiliser): $dd" >&2
      echo "[gpu_guard]       consider: docker stop deploy-docker" >&2
    fi
  fi
  [ "$rc" = 0 ] && echo "[gpu_guard] OK to run GPU work."
  return "$rc"
}

after() {
  if hang_tail | grep -qE "$DMESG_HANG_PAT"; then
    echo "[gpu_guard] FAIL: GPU HANG detected after run — STOP GPU work, capture kernel log" >&2
    hang_tail | grep -E "$DMESG_HANG_PAT" | tail -5 >&2
    return 3
  fi
  echo "[gpu_guard] no GPU HANG after run."
  return 0
}

case "${1:-check}" in
  check) shift || true; check ;;
  after) shift || true; after ;;
  run)
    shift
    if [ "$#" -eq 0 ]; then echo "usage: gpu_guard.sh run <cmd...>" >&2; exit 2; fi
    check || echo "[gpu_guard] proceeding despite warnings"
    timeout "${GPU_TIMEOUT:-120}" "$@"
    rc=$?
    after || true
    exit "$rc"
    ;;
  *) echo "usage: gpu_guard.sh {check|after|run <cmd>}" >&2; exit 2 ;;
esac
