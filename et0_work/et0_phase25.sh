#!/bin/bash
# ET0 Phase 2-5 for full/. Parallel safesteps (1/GPU) + bash retry.
# EPIC-05 A100 ONLY. Idempotent: skips cells whose SAFESTEP json exists.
set -u
ROOT=/root/epic05_moonshot/et0
OUT=/root/epic05_moonshot/et0/full
PY=/root/miniforge3/envs/gsplat/bin/python
cd "$ROOT" || exit 1
CELLS=("room 10000" "room 15000" "room 20000" "bicycle 10000" "bicycle 15000" "bicycle 20000")
GPUS=(0 1 2 3 4 5)

echo "[$(date +%H:%M:%S)] PHASE 2: cohort tau_H"
$PY - <<'PYEOF'
import json, glob, numpy as np
hs=[]
for f in sorted(glob.glob("/root/epic05_moonshot/et0/full/ET0_H1_*.json")):
    hs += json.load(open(f))["H_at_alpha1"]
tau=float(np.median(hs)) if hs else None
json.dump({"cohort_tau_H": tau, "n_H1": len(hs)},
          open("/root/epic05_moonshot/et0/full/ET0_TAU_H.json","w"), indent=2)
print("cohort_tau_H", tau, "n_H", len(hs))
PYEOF
TAU=$($PY -c "import json; print(json.load(open('/root/epic05_moonshot/et0/full/ET0_TAU_H.json'))['cohort_tau_H'])")
echo "tau_H=$TAU"

echo "[$(date +%H:%M:%S)] PHASE 3: safesteps (parallel, 1/GPU)"
run_ss () {
  local scene=$1 it=$2 gpu=$3 cell=${scene}_${it}
  [ -f "$OUT/ET0_SAFESTEP_${cell}.json" ] && { echo "SKIP $cell (exists)"; return 0; }
  for a in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu $PY et0_cell.py --scene $scene --ckpt $it \
       --outdir "$OUT" --mode safestep --tau-h "$TAU" > "$OUT/safestep_${cell}.log" 2>&1
    if [ -f "$OUT/ET0_SAFESTEP_${cell}.json" ]; then echo "OK $cell (attempt $a)"; return 0; fi
    echo "RETRY $cell attempt=$a rc=$?"; sleep 8
  done
  echo "FAIL $cell"; return 1
}
for i in "${!CELLS[@]}"; do
  set -- ${CELLS[$i]}; run_ss $1 $2 ${GPUS[$i]} &
done
wait
echo "[$(date +%H:%M:%S)] PHASE 3 done"

echo "[$(date +%H:%M:%S)] PHASE 4: aggregate"
CELLS_CSV=$(printf "%s," "${CELLS[@]}" | sed 's/ $//' | tr ' ' '_')
$PY et0_cell.py --mode aggregate --outdir "$OUT" --cells "$CELLS_CSV" > "$OUT/aggregate.log" 2>&1
echo "aggregate rc=$?"

echo "[$(date +%H:%M:%S)] PHASE 5: verdict"
$PY et0_analyze.py "$OUT/ET0_SUMMARY.json" > "$OUT/verdict.log" 2>&1
echo "verdict rc=$?"
echo "=== VERDICT ==="
cat "$OUT/ET0_VERDICT.json" 2>/dev/null
echo "=== DONE $(date +%H:%M:%S) ==="
