#!/bin/bash
pgrep -af "p14_o1.py" || echo "no p14_o1 processes"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== room stage A summary ==="
grep "cell.*h32\|D0 done" /root/epic05_p14/logs/o1a_room.log
echo "=== bicycle stage A summary ==="
grep "cell.*h32\|D0 done" /root/epic05_p14/logs/o1a_bicycle.log
