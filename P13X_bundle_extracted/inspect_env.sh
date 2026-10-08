#!/bin/bash
# P13-X remote environment inspection (read-only)
echo "=== fused-ssim .so sha256 (gsplat env) ==="
sha256sum /root/miniforge3/envs/gsplat/lib/python3.10/site-packages/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so

echo; echo "=== P12 room c0fused run dir ==="
ls -la /root/epic05_p12/runs/p12a/p12a_c0f_s42_room/ | head -25

echo; echo "=== p12a manifest (first 1200 chars) ==="
head -c 1200 /root/epic05_p12/manifests/p12a_manifest.json

echo; echo; echo "=== P12 preflight dir ==="
ls /root/epic05_p12/preflight/ 2>/dev/null | head

echo; echo "=== python env version ==="
/root/miniforge3/envs/gsplat/bin/python -c "import torch, gsplat; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('gsplat file', gsplat.__file__)" 2>&1 | tail -3

echo; echo "=== GPU now ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo; echo "=== C0 worktree top ==="
ls /root/epic05_frozen/higs_c0_final30k_worktree/ | head -15

echo; echo "=== room dataset cameras ==="
ls /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room/ | head
ls /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room/sparse/0/ 2>/dev/null
