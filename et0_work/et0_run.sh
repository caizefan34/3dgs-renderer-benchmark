#!/bin/bash
# ET0 full driver (FROZEN prereg). Runs on EPIC-05 A100 ONLY.
# Phase 1: sweep all 6 cells (parallel, 1/GPU)
# Phase 2: cohort tau_H = median H(alpha=1) over all cells/probes
# Phase 3: safestep all 6 cells (parallel)
# Phase 4: aggregate -> ET0_SUMMARY.json
# Phase 5: verdict -> ET0_VERDICT.json
set -u
ROOT=/root/epic05_moonshot/et0
OUT=/root/epic05_moonshot/et0/out
PY=/root/miniforge3/envs/gsplat/bin/python
mkdir -p "$OUT"
cd "$ROOT" || exit 1

CELLS=("room 10000" "room 15000" "room 20000" "bicycle 10000" "bicycle 15000" "bicycle 20000")
GPUS=(0 1 2 3 4 5)

echo "=== [$(date +%H:%M:%S)] PHASE 1: sweep ==="
for i in "${!CELLS[@]}"; do
  set -- ${CELLS[$i]}; scene=$1; it=$2; gpu=${GPUS[$i]}
  cell=${scene}_${it}
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY et0_cell.py --scene $scene --ckpt $it \
     --outdir "$OUT" --mode sweep > "$OUT/sweep_${cell}.log" 2>&1 &
  echo "launched sweep cell=$cell gpu=$gpu pid=$!"
done
wait
echo "=== [$(date +%H:%M:%S)] PHASE 1 done ==="

echo "=== [$(date +%H:%M:%S)] PHASE 2: cohort tau_H ==="
$PY - <<'PYEOF'
import json, glob, numpy as np
hs=[]
for f in sorted(glob.glob("/root/epic05_moonshot/et0/out/ET0_H1_*.json")):
    hs += json.load(open(f))["H_at_alpha1"]
tau=float(np.median(hs)) if hs else None
open("/root/epic05_moonshot/et0/tau_H.txt","w").write(str(tau))
print("cohort_tau_H", tau, "n_H", len(hs))
PYEOF
TAU=$(cat /root/epic05_moonshot/et0/tau_H.txt)
echo "tau_H=$TAU"

echo "=== [$(date +%H:%M:%S)] PHASE 3: safestep ==="
for i in "${!CELLS[@]}"; do
  set -- ${CELLS[$i]}; scene=$1; it=$2; gpu=${GPUS[$i]}
  cell=${scene}_${it}
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY et0_cell.py --scene $scene --ckpt $it \
     --outdir "$OUT" --mode safestep --tau-h $TAU > "$OUT/safestep_${cell}.log" 2>&1 &
  echo "launched safestep cell=$cell gpu=$gpu pid=$!"
done
wait
echo "=== [$(date +%H:%M:%S)] PHASE 3 done ==="

echo "=== [$(date +%H:%M:%S)] PHASE 4: aggregate ==="
CELLS_CSV=$(printf "%s," "${CELLS[@]}" | sed 's/ $//' | tr ' ' '_')
$PY et0_cell.py --mode aggregate --outdir "$OUT" --cells "$CELLS_CSV" > "$OUT/aggregate.log" 2>&1
echo "aggregate rc=$?"

echo "=== [$(date +%H:%M:%S)] PHASE 5: verdict ==="
$PY et0_analyze.py "$OUT/ET0_SUMMARY.json" > "$OUT/verdict.log" 2>&1
echo "verdict rc=$?"
echo "=== [$(date +%H:%M:%S)] ET0 DRIVER DONE ==="
cat "$OUT/ET0_VERDICT.json" 2>/dev/null