#!/bin/bash
# Full R3 status report
REPO="$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)"
R3DIR="$REPO/results/reference_v1/r3"
echo "DATE=$(date '+%Y-%m-%d %H:%M:%S')"
echo "REPO=$REPO"
echo "--- processes ---"
ps -eo pid,etime,time,pcpu,stat,args | grep -E "r3_certificate|launch_r3" | grep -v grep | head -5
echo "--- runner log (last 15 lines) ---"
tail -15 "$R3DIR/5000/runner.log" 2>/dev/null
echo "--- iter count ---"
grep -c "iter 50" "$R3DIR/5000/runner.log" 2>/dev/null || echo 0
echo "--- output files ---"
find "$R3DIR" -type f 2>/dev/null | head -30
