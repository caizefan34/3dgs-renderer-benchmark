echo "=== GPU compute apps 5,6,7 ==="
for g in 5 6 7; do echo -n "GPU$g: "; nvidia-smi -i $g --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | tr "\n" " "; echo; done
echo "=== my python ==="
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python -c "import torch,gsplat,numpy,scipy;print(torch.__version__, gsplat.__version__, numpy.__version__, scipy.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
echo "=== room ckpts ==="
ls /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_multiphase_v2/checkpoints/
echo "=== garden ==="
find /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_garden_30k -name '*.pt' 2>&1
echo "=== bicycle ==="
find /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_bicycle_30k -name '*.pt' 2>&1
echo "=== data dirs ==="
ls /tmp/3dgs-renderer-benchmark/data/datasets/mipnerf360/ 2>&1
echo "=== room30k assets ==="
ls /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/ 2>&1