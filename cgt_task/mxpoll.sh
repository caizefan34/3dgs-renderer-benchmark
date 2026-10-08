cd /home/liaoyuanjun/cgt_realscene
echo "=== running combos (logs) ==="
for g in 5 6 7; do echo "GPU$g:"; tail -3 logs/g1/gpu$g.out 2>&1; done
echo "=== json produced ==="
ls -la logs/g1/*.json 2>&1
echo "=== gpu util ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '6,8p'