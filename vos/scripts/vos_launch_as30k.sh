#!/bin/bash
# Decisive quality-attribution run: pure A_s (single-tensor Adam, NO VOS)
# 30K on room. If A_s's final PSNR lands ~0.1-0.35 dB below C0 like the VOS
# arms do, the quality cost is the torch-internal A_s-vs-A_f envelope
# (inherent to the semantics VOS reproduces bitwise), not a VOS defect.
set -u
export PY=/root/miniforge3/envs/gsplat/bin/python
export D=/root/vos_traces
cd "$D"
CUDA_VISIBLE_DEVICES=3 $PY vos_train_driver.py --scene room \
  --iterations 30000 --gpu 3 --as-mode --final-eval all \
  --outdir "$D"/vos_runs/as_room_30k > "$D"/logs/vos_as_room_30k.log 2>&1 &
echo "AS_30K pid=$! LAUNCHED $(date -u +%FT%TZ)"
