cd /home/liaoyuanjun/cgt_realscene
echo "=== prior state ==="
pgrep -af "run_g1.sh|g1_compressibility|launch_all" 2>&1 | head
ls -la logs/g1/ 2>&1
echo "=== relaunch with mkdir first ==="
mkdir -p logs/g1
setsid bash launch_all.sh > logs/g1/launch_all.out 2>&1 < /dev/null &
sleep 5
echo "=== after launch ==="
pgrep -af "run_g1.sh|g1_compressibility" 2>&1 | head
echo "--- gpu 5,6,7 ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '6,8p'
echo "--- launch_all.out ---"
cat logs/g1/launch_all.out 2>&1