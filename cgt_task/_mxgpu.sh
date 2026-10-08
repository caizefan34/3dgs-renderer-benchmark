echo "=== GPU free mem ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== unknown 2594xxx procs ==="
ps -o pid,etime,cmd -p 2594768,2594769,2594770 2>/dev/null | grep -v grep
