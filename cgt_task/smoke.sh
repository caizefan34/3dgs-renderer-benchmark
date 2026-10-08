#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
cd /home/liaoyuanjun/cgt_realscene/src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
timeout 300 $PY g1_compressibility.py --scene room --ckpt /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/checkpoints/iter_30000.pt --data /tmp/3dgs-renderer-benchmark/data --views 1 --iterations 60 --max-tiles-per-view 20 --out /home/liaoyuanjun/cgt_realscene/logs/smoke.json 2>&1 | tail -30

