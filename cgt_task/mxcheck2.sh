cd /home/liaoyuanjun/cgt_realscene
echo "=== procs ==="
pgrep -af "run_g1.sh|g1_compressibility|launch_all" 2>&1 | head
echo "=== gpu 5,6,7 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '6,8p'
echo "=== launch_all.out ==="
cat logs/g1/launch_all.out 2>&1
echo "=== gpu5.out ==="
cat logs/g1/gpu5.out 2>&1 | head
echo "=== gpu7.out ==="
cat logs/g1/gpu7.out 2>&1 | head