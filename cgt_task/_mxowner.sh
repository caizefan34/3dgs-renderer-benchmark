echo "=== compute-apps (gpu_uuid,pid,mem) ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
echo "=== all python/torch procs ==="
ps -eo pid,user,etime,cmd | grep -iE "python|vllm" | grep -v grep | head -40
