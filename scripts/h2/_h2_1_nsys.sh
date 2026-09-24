#!/bin/bash
# H2-1 nsys kernel inventory for mx - uses full nsys path
set -e
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
NSYS=/home/liaoyuanjun/miniforge3/pkgs/nsight-compute-2026.2.1.5-h60dd89f_0/nsight-compute-2026.2.1/host/target-linux-x64/nsys
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx:/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_scene_cuda
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=2

mkdir -p /tmp/h2_1_nsys

echo "=== Running nsys profile for B2 kernel inventory ==="
$NSYS profile -t cuda,nvtx --stats=true -o /tmp/h2_1_nsys/h2_1_b2 \
  --force-overwrite=true \
  $PY /tmp/h2_1_profile.py --out-dir /tmp/h2_1_nsys --scene room --cam-idx 0 --nsys-mode --gpu 0 --max-long-side 2048 2>&1 | tee /tmp/h2_1_nsys/nsys_output.txt

echo "=== nsys done ==="
ls -la /tmp/h2_1_nsys/
echo "=== Kernel stats from nsys ==="
cat /tmp/h2_1_nsys/nsys_output.txt | grep -A 100 "CUDA Kernel Statistics" | head -120
