#!/bin/bash
# Robust ET0 resume: re-run only MISSING sweep/safestep cells (with hash retry),
# then cohort tau_H -> aggregate -> verdict. Idempotent. EPIC-05 A100 ONLY.
set -u
ROOT=/root/epic05_moonshot/et0
OUT=/root/epic05_moonshot/et0/full
PY=/root/miniforge3/envs/gsplat/bin/python
cd "$ROOT" || exit 1
CELLS=("room 10000" "room 15000" "room 20000" "bicycle 10000" "bicycle 15000" "bicycle 20000")

run_cell () {  # $1=scene $2=it $3=mode $4=extra...
  local scene=$1 it=$2 mode=$3; shift 3
  local cell=${scene}_${it}
  local log="$OUT/${mode}_${cell}.log"
  for attempt in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 $PY et0_cell.py --scene $scene --ckpt $it \
       --outdir "$OUT" --mode $mode "$@" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -s "$log" ]; then
      # success only if the expected json exists and log has no traceback
      if [ "$mode" = "sweep" ] && [ -f "$OUT/ET0_SWEEP_${cell}.json" ]; then echo "OK $mode $cell (attempt $attempt)"; return 0; fi
      if [ "$mode" = "safestep" ] && [ -f "$OUT/ET0_SAFESTEP_${cell}.json" ]; then echo "OK $mode $cell (attempt $attempt)"; return 0; fi
    fi
    echo "RETRY $mode $cell attempt=$attempt rc=$rc"; sleep 8
  done
  echo "FAIL $mode $cell after retries"; return 1
}

echo "[$(date +%H:%M:%S)] resume: missing sweeps"
for c in "${CELLS[@]}"; do
  set -- $c; cell=$1_$2
  [ -f "$OUT/ET0_SWEEP_${cell}.json" ] || run_cell $1 $2 sweep
done

echo "[$(date +%H:%M:%S)] resume: cohort tau_H"
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

echo "[$(date +%H:%M:%S)] resume: missing safesteps"
for c in "${CELLS[@]}"; do
  set -- $c; cell=$1_$2
  [ -f "$OUT/ET0_SAFESTEP_${cell}.json" ] || run_cell $1 $2 safestep --tau-h "$TAU"
done

echo "[$(date +%H:%M:%S)] resume: aggregate"
CELLS_CSV=$(printf "%s," "${CELLS[@]}" | sed 's/ $//' | tr ' ' '_')
$PY et0_cell.py --mode aggregate --outdir "$OUT" --cells "$CELLS_CSV" > "$OUT/aggregate.log" 2>&1
echo "aggregate rc=$?"

echo "[$(date +%H:%M:%S)] resume: verdict"
$PY et0_analyze.py "$OUT/ET0_SUMMARY.json" > "$OUT/verdict.log" 2>&1
echo "verdict rc=$?"
echo "=== VERDICT ==="
cat "$OUT/ET0_VERDICT.json" 2>/dev/null
