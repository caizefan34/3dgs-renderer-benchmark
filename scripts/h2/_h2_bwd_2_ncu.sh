#!/bin/bash
# H2-BWD-2 NCU occupancy probe
export HOME=/tmp/ncu_home
mkdir -p $HOME
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx:/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_scene_cuda
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export TORCH_EXTENSIONS_DIR=/tmp/higs_h2_bwd_cf/cache
export HIGS_PX_RUNTIME=2
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
OUT=/tmp/higs_h2_bwd_2
for V in baseline scalar_adjoint; do
  echo "===== NCU variant=$V ====="
  /usr/bin/ncu --section Occupancy --csv -k regex:higs_blend_bwd --launch-skip 3 --launch-count 1 --target-processes all "$PY" /tmp/higs_h2_bwd_2/ncu_probe.py --variant "$V" --source /tmp/higs_h2_bwd_cf/source --core-so /tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so > $OUT/ncu_$V.csv 2> $OUT/ncu_$V.err
  echo "rc=$? err:"; head -3 $OUT/ncu_$V.err
  echo "--- csv occupancy/eligible rows ---"
  grep -i "achieved\|eligible\|warps_active\|theoretical" $OUT/ncu_$V.csv 2>/dev/null | head -25
done
echo "===== NCU probe done ====="
