echo "=== g1_compressibility fix check (MX) ==="
sed -n '125,140p' /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g1_compressibility.py
echo "=== GPU3 + all GPU pids ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
echo "=== map: which pid on which gpu ==="
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "=== g3 garden full log ==="
cat /home/liaoyuanjun/cgt_realscene/final/logs/g3r_garden_r30k.log
