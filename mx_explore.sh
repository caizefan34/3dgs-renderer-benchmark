#!/bin/bash
echo "=== REPO ==="
ls /home/liaoyuanjun/3dgs-renderer-benchmark/
echo "=== SCRIPTS ==="
ls /home/liaoyuanjun/3dgs-renderer-benchmark/scripts/
echo "=== BASELINE ==="
ls /home/liaoyuanjun/3dgs-renderer-benchmark/baseline/
echo "=== WORKTREE ==="
ls /tmp/gsplat_systems_revalidation/baseline/reference_v1/
echo "=== WORKTREE ROOT ==="
ls /tmp/gsplat_systems_revalidation/
echo "=== TORCH ==="
python3 -c 'import torch; print("torch", torch.__version__); import gsplat; print("gsplat", gsplat.__version__)'
echo "=== CKPT ==="
ls /home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
