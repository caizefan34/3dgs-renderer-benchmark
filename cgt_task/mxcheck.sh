echo "=== run_g1 procs ==="
pgrep -af "run_g1.sh|g1_compressibility" 2>&1 | head
echo "=== gpu 5,6,7 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '6,8p'
echo "=== gpu5.out ==="
cat /home/liaoyuanjun/cgt_realscene/logs/g1/gpu5.out 2>&1 | head
echo "=== gpu7.out ==="
cat /home/liaoyuanjun/cgt_realscene/logs/g1/gpu7.out 2>&1 | head