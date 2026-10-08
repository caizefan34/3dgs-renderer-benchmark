#!/bin/bash
set -e
export TMPDIR=/dev/shm
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/usr/bin:/bin:$PATH
export LD_LIBRARY_PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3.10
SC=/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN
CFG=/mnt/storage_pool/liaoyuanjun/fused_profile0/profile
CKPT=/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused
cd $SC
$PY scripts/bwd1_workload.py --config $CFG/mat_room.yaml --checkpoint $CKPT/fp1_room_2026-10-06-01-01-33/checkpoints/final.pt --scene room --view 0 --out raw/workload_room.json > logs/wl_room3.log 2>&1
$PY scripts/bwd1_workload.py --config $CFG/mat_bicycle.yaml --checkpoint $CKPT/fp1_bicycle_2026-10-06-01-01-33/checkpoints/final.pt --scene bicycle --view 0 --out raw/workload_bicycle.json > logs/wl_bicycle2.log 2>&1
$PY scripts/bwd1_workload.py --config $CFG/mat_garden.yaml --checkpoint $CKPT/fp1_garden_2026-10-06-01-32-48/checkpoints/final.pt --scene garden --view 0 --out raw/workload_garden.json > logs/wl_garden3.log 2>&1
echo ALL_DONE
