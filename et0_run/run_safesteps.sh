#!/bin/bash
# ET0 Phase 3: safestep for all 6 cells with cohort tau_H. EPIC-05 A100 ONLY.
set -u
ROOT=/root/epic05_moonshot/et0
OUT=/root/epic05_moonshot/et0/full
PY=/root/miniforge3/envs/gsplat/bin/python
cd "$ROOT" || exit 1
TAU=$($PY -c "import json; print(json.load(open('$OUT/ET0_TAU_H.json'))['cohort_tau_H'])")
echo "[$(date +%H:%M:%S)] safestep phase, tau_H=$TAU"
CELLS=("room 10000" "room 15000" "room 20000" "bicycle 10000" "bicycle 15000" "bicycle 20000")
for c in "${CELLS[@]}"; do
  set -- $c; S=$1; K=$2; cell=${S}_${K}
  log="$OUT/safestep_${cell}.log"
  if [ -f "$OUT/ET0_SAFESTEP_${cell}.json" ]; then echo "SKIP $cell (exists)"; continue; fi
  echo "[$(date +%H:%M:%S)] safestep $cell ..."
  for attempt in 1 2 3; do
    CUDA_VISIBLE_DEVICES=1 $PY et0_cell.py --scene $S --ckpt $K --outdir "$OUT" \
        --mode safestep --tau-h "$TAU" > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$OUT/ET0_SAFESTEP_${cell}.json" ]; then
      echo "OK $cell (attempt $attempt)"; break
    fi
    echo "RETRY $cell attempt=$attempt rc=$rc"; sleep 10
  done
done
echo "[$(date +%H:%M:%S)] safestep phase complete"
