#!/bin/bash
# FINAL-30K: 26-run phase launcher (13 scenes x {B1A, C0_V3_FINAL30K})
#
# Dynamic GPU scheduling per Addendum B:
#  - before EVERY launch, re-scan all 8 GPUs (never assume prior state)
#  - choose the GPU with the most free memory; prefer clean (no foreign jobs)
#  - run FINAL_30K on the chosen GPU (single consumer per GPU)
#  - contamination snapshot before each run (for provenance/quality label)
#  - skip runs whose results.json already exists (idempotent resume)
#  - timing_grade reflects the GPU state at launch (FUNCTIONAL_ONLY unless clean)
set -u
PY=/mnt/storage_pool/liaoyuanjin/higs-13scene-env/bin/python
export PATH=/mnt/storage_pool/liaoyuanjin/higs-13scene-env/bin:/usr/bin:/bin:$PATH
ROOT=/home/liaoyuanjin/3dgs-renderer-benchmark
OUT=/mnt/storage_pool/liaoyuanjin/final30k_runs
SNAP=$OUT/.snapshots
mkdir -p "$OUT" "$SNAP"
cd "$ROOT"

SCENES="bicycle bonsai counter drjohnson flowers garden kitchen playroom room stump train treehill truck"
ARMS="b1a c0"

snapshot() {
  local f="$SNAP/${1}.snap.txt"
  {
    echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
    echo "== nvidia-smi =="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "== compute apps =="
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null
    echo "== ps aux (gpu) =="
    ps aux | grep -Ei "python|cuda|train" | grep -v grep | head -40
  } > "$f"
}

pick_gpu() {
  # return index of GPU with most free memory, output on stdout
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | sort -t, -k3,3nr -k2,2n | awk -F, 'NR==1{print $1}'
}

run_one() {
  local arm="$1"; local scene="$2"
  local d="$OUT/${arm}_${scene}"
  if [ -f "$d/results.json" ]; then
    echo "SKIP $arm/$scene (results.json present)"
    return 0
  fi
  local gpu
  gpu=$(pick_gpu)
  snapshot "${arm}_${scene}_pre"
  echo "=== RUN $arm/$scene on GPU $gpu  $(date +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES="$gpu" $PY "$ROOT/final30k_trainer.py" \
      --arm "$arm" --scene "$scene" --iterations 30000 \
      --outdir "$d" --gpu 0 --final-eval all --timing-grade FUNCTIONAL_ONLY \
      > "$OUT/log_${arm}_${scene}.txt" 2>&1
  local rc=$?
  echo "=== DONE $arm/$scene rc=$rc $(date +%H:%M:%S) ==="
  if [ $rc -ne 0 ]; then
    echo "FAILED $arm/$scene rc=$rc (tail):"
    tail -25 "$OUT/log_${arm}_${scene}.txt"
  fi
}

for scene in $SCENES; do
  for arm in $ARMS; do
    run_one "$arm" "$scene"
  done
done
echo "ALL 26 RUNS COMPLETE $(date +%H:%M:%S)"
