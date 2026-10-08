set -e
cd /root/3dgs-renderer-benchmark
echo "===== GPU free check ====="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "===== SMOKE: room, 1200 iters, GPU 3 ====="
rm -rf /root/gsc0_smoke
/root/miniforge3/envs/gsplat/bin/python /root/gsc0_trace.py \
  --scene room --gpu 3 --out /root/gsc0_smoke --iterations 1200 2>&1 | tail -40
echo "===== smoke outputs ====="
ls -la /root/gsc0_smoke /root/gsc0_smoke/traces 2>&1