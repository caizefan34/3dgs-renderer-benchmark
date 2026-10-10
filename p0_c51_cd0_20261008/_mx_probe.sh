#!/bin/bash
# C51J P0 resource baseline probe (read-only, audit-only)
set +e
echo "=== HOST ==="
hostname; date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "=== STORAGE ==="
df -B1 /mnt/storage_pool/liaoyuanjun 2>/dev/null || df -B1 /mnt/storage_pool 2>/dev/null || echo "storage_pool not mounted"
echo "=== INODES ==="
df -i /mnt/storage_pool/liaoyuanjun 2>/dev/null | tail -2
echo "=== OUTPUT DIR EXISTS? ==="
ls -ld /mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX 2>/dev/null || echo "output dir not yet created"
echo "=== GPU STATUS ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "=== GPU COMPUTE APPS (exclusivity) ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -20
echo "=== A100 COUNT ==="
nvidia-smi --query-gpu=name --format=csv,noheader | grep -c A100
echo "=== PYTHON/TORCH ==="
python3 -c "import torch; print('torch='+torch.__version__, 'cuda='+str(torch.version.cuda), 'gpus='+str(torch.cuda.device_count()))" 2>/dev/null || echo "torch import failed"
echo "=== GSPLAT ==="
python3 -c "import gsplat; print('gsplat='+gsplat.__version__)" 2>/dev/null || echo "gsplat import failed"
echo "=== NVCC ==="
nvcc --version 2>/dev/null | grep release || echo "no nvcc"
echo "=== 132-JOB RAW RESULTS PRESENT? ==="
find /mnt/storage_pool/liaoyuanjun -maxdepth 4 -iname "*confirmatory-accel15*" 2>/dev/null | head -10
find /mnt/storage_pool/liaoyuanjun -maxdepth 4 -iname "*132*" 2>/dev/null | head -10
echo "=== DONE ==="
