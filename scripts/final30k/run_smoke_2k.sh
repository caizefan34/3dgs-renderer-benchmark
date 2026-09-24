#!/bin/bash
# Gate E: matched 2K B1A vs C0_V3_FINAL30K smoke — 3 scenes x 2 arms, GPU 2 (FUNCTIONAL_ONLY)
set -u
export CUDA_VISIBLE_DEVICES=2
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/usr/bin:/bin:$PATH
ROOT=/home/liaoyuanjun/3dgs-renderer-benchmark
OUT=/mnt/storage_pool/liaoyuanjun/final30k_smoke
mkdir -p "$OUT"
cd "$ROOT"

for scene in room bicycle garden; do
  for arm in b1a c0; do
    d="$OUT/${arm}_${scene}_2k"
    if [ -f "$d/results.json" ]; then
      echo "SKIP $arm/$scene (results.json exists)"
      continue
    fi
    mkdir -p "$d"
    echo "=== RUN $arm/$scene 2K  $(date +%H:%M:%S) ==="
    $PY "$ROOT/final30k_trainer.py" --arm "$arm" --scene "$scene" \
        --iterations 2000 --outdir "$d" --gpu 0 \
        --final-eval subset --timing-grade FUNCTIONAL_ONLY \
        > "$OUT/log_${arm}_${scene}.txt" 2>&1
    rc=$?
    echo "=== DONE $arm/$scene rc=$rc  $(date +%H:%M:%S) ==="
    if [ $rc -ne 0 ]; then
      echo "FAILED: tail of log:"
      tail -25 "$OUT/log_${arm}_${scene}.txt"
    fi
  done
done
echo "ALL SMOKE RUNS COMPLETE $(date +%H:%M:%S)"
