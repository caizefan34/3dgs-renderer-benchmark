#!/bin/bash
# Validate the upgraded trainer (phase timing + harness files) on GPU 2, FUNCTIONAL_ONLY.
set -u
export CUDA_VISIBLE_DEVICES=2
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/usr/bin:/bin:$PATH
ROOT=/home/liaoyuanjun/3dgs-renderer-benchmark
OUT=/mnt/storage_pool/liaoyuanjun/final30k_trainer_check
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$ROOT"
for arm in b1a c0; do
  d="$OUT/${arm}_room_2k"
  mkdir -p "$d"
  echo "=== CHECK $arm/room 2K $(date +%H:%M:%S) ==="
  $PY "$ROOT/final30k_trainer.py" --arm "$arm" --scene room --iterations 2000 \
      --outdir "$d" --gpu 0 --final-eval subset --timing-grade FUNCTIONAL_ONLY \
      > "$OUT/log_${arm}.txt" 2>&1
  echo "=== rc=$? $(date +%H:%M:%S) ==="
done
echo VALIDATION_DONE
