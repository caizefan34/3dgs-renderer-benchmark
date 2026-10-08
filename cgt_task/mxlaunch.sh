cd /home/liaoyuanjun/cgt_realscene
mkdir -p logs/g1
nohup bash run_g1.sh 5 "room:5000:r5k,room:10000:r10k" > logs/g1/gpu5.out 2>&1 &
echo "GPU5 PID $!"
nohup bash run_g1.sh 6 "room:15000:r15k,room:30000:r30k" > logs/g1/gpu6.out 2>&1 &
echo "GPU6 PID $!"
nohup bash run_g1.sh 7 "garden:30000:garden,bicycle:30000:bicycle" > logs/g1/gpu7.out 2>&1 &
echo "GPU7 PID $!"
sleep 3
echo "=== launched processes ==="
pgrep -af run_g1.sh | head
echo "=== gpu status ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '6,8p'