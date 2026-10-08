nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | head
echo "--- leftover python procs ---"
ps aux | grep -E 'dpgr0|dpc0' | grep -v grep | head
