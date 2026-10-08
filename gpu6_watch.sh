for i in $(seq 1 25); do
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 6
  sleep 1
done | sort | uniq -c
