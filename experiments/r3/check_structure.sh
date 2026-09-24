#!/bin/bash
echo "=== now ==="
date '+%H:%M:%S'
echo "=== R3 output tree ==="
find "$(ls -d /home/*/3dgs-renderer-benchmark/results/reference_v1/r3 2>/dev/null | head -1)" -maxdepth 2 2>/dev/null | sort
echo "=== all recent log files ==="
find /home /tmp -name "*.log" -mmin -360 -exec ls -la {} \; 2>/dev/null | head -20
echo "=== full_run.log ==="
LOG="$(find /home /tmp -name 'full_run.log' 2>/dev/null | head -1)"
if [ -n "$LOG" ]; then tail -30 "$LOG"; else echo "no full_run.log"; fi
echo "=== similar processes (r3, python, gpu) ==="
ps -eo pid,ppid,etime,time,pcpu,stat,args --sort=-pcpu 2>/dev/null | grep -iE "python|r3|gpu" | grep -v grep | head -8
