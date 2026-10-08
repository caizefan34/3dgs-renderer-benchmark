echo "=== GPU6 procs over 30s ==="
for i in $(seq 1 30); do
  n=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 6 | wc -l)
  m=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 6)
  echo "t$i procs=$n $m"
  sleep 1
done
echo "=== GPU7 procs over 10s ==="
for i in $(seq 1 10); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 7 | wc -l)
  echo "t$i procs=$n"
  sleep 1
done
