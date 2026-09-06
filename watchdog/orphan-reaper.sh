#!/bin/bash
# Hermes Gateway 孤儿进程看门狗
# 背景: 2026-08-28 假死事故 —— 55个 gateway 孤儿进程(RSS合计12.5G)把 2G 内存机器挤爆。
# 原因: --replace 手动启动的进程不在 systemd cgroup 内, 且 PID 文件互相覆盖导致
#       Hermes 自带的 orphan reaper 看不见它们。本脚本直接扫进程表, 不依赖 PID 文件。
# 规则: 1. 保护 gateway.pid 里的当前进程和启动 <180s 的新进程(重启窗口)
#       2. 其余 "hermes_cli.main gateway run" 进程: SIGTERM -> 10s -> SIGKILL

LOG="/var/log/hermes-orphan-reaper.log"
PATTERN="hermes_cli.main gateway run"
MIN_AGE="${MIN_AGE:-180}"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# 当前受保护的 PID: gateway.pid + systemd 主进程
protected=""
if [ -f /root/.hermes/gateway.pid ]; then
    protected=$(python3 -c "import json;print(json.load(open('/root/.hermes/gateway.pid'))['pid'])" 2>/dev/null)
fi
svc_pid=$(systemctl show hermes-gateway.service -p MainPID --value 2>/dev/null)
[ "$svc_pid" != "0" ] && protected="$protected $svc_pid"

# 排除脚本自身及其调用链, 避免把运行看门狗的 shell 误判为孤儿
self_pid=$$
self_tree="$self_pid"
# 收集祖先链 (ppid 向上追到 init)
ppid=$(ps -o ppid= -p "$self_pid" 2>/dev/null | tr -d ' ')
while [ -n "$ppid" ] && [ "$ppid" != "1" ] && [ "$ppid" != "0" ]; do
    self_tree="$self_tree $ppid"
    ppid=$(ps -o ppid= -p "$ppid" 2>/dev/null | tr -d ' ')
done

killed=0
for pid in $(pgrep -f "$PATTERN"); do
    # 跳过受保护进程
    for p in $protected; do
        [ "$pid" = "$p" ] && continue 2
    done
    # 跳过硬看门狗自身及其调用链 (防止自杀)
    for s in $self_tree; do
        [ "$pid" = "$s" ] && continue 2
    done
    # 跳过启动 <180s 的进程(重启保护窗口)
    elapsed=$(stat -c %Y "/proc/$pid" 2>/dev/null)
    now=$(date +%s)
    [ -z "$elapsed" ] && continue
    age=$((now - elapsed))
    [ "$age" -lt "$MIN_AGE" ] && continue

    rss=$(awk '/VmRSS/{print int($2/1024)"MB"}' "/proc/$pid/status" 2>/dev/null)
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)
    log "ORPHAN pid=$pid age=${age}s rss=$rss cmd=$cmd -> SIGTERM"
    kill -TERM "$pid" 2>/dev/null && killed=$((killed+1))
done

# 给 SIGTERM 10 秒宽限, 仍存活则 SIGKILL
if [ "$killed" -gt 0 ]; then
    sleep 10
    for pid in $(pgrep -f "$PATTERN"); do
        for p in $protected; do
            [ "$pid" = "$p" ] && continue 2
        done
        for s in $self_tree; do
            [ "$pid" = "$s" ] && continue 2
        done
        elapsed=$(stat -c %Y "/proc/$pid" 2>/dev/null)
        [ -n "$elapsed" ] || continue
        age=$(( $(date +%s) - elapsed ))
        [ "$age" -lt "$MIN_AGE" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            log "ESCALATE pid=$pid -> SIGKILL"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
    log "SUMMARY: terminated=$killed"
else
    log "OK: no orphans"
fi
