#!/bin/bash
# ET0 full pipeline: 6 cells. sweep -> cohort tau_H -> safestep -> aggregate.
set -e
cd /root/epic05_moonshot/et0
P=/root/miniforge3/envs/gsplat/bin/python
OUT=/root/epic05_moonshot/et0/full
rm -rf $OUT
mkdir -p $OUT
CELLS="room_10000 room_15000 room_20000 bicycle_10000 bicycle_15000 bicycle_20000"

echo "=== PHASE 1: SWEEP (all cells) ==="
for C in $CELLS; do
  S=$(echo $C | cut -d_ -f1); K=$(echo $C | cut -d_ -f2)
  echo "--- sweep $C ---"
  $P et0_cell.py --scene $S --ckpt $K --outdir $OUT > $OUT/sweep_$C.log 2>&1
  echo "  exit=$? done $C"
done

echo "=== PHASE 2: cohort tau_H = median H(alpha=1) over ALL cells ==="
$P - <<'PYEOF'
import json, glob, numpy as np
h1 = []
for f in sorted(glob.glob("/root/epic05_moonshot/et0/full/ET0_H1_*.json")):
    d = json.load(open(f))
    h1.extend([x for x in d["H_at_alpha1"] if x is not None])
tau = float(np.median(h1)) if h1 else None
json.dump({"cohort_tau_H": tau, "n_H1": len(h1), "H1_all": h1},
          open("/root/epic05_moonshot/et0/full/ET0_TAU_H.json", "w"), indent=2)
print("cohort_tau_H =", tau, "n =", len(h1))
PYEOF
TAU=$($P -c "import json; print(json.load(open('/root/epic05_moonshot/et0/full/ET0_TAU_H.json'))['cohort_tau_H'])")
echo "tau_H=$TAU"

echo "=== PHASE 3: SAFESTEP (all cells) ==="
for C in $CELLS; do
  S=$(echo $C | cut -d_ -f1); K=$(echo $C | cut -d_ -f2)
  echo "--- safestep $C ---"
  $P et0_cell.py --scene $S --ckpt $K --outdir $OUT --mode safestep --tau-h $TAU > $OUT/safestep_$C.log 2>&1
  echo "  exit=$? done $C"
done

echo "=== PHASE 4: AGGREGATE ==="
$P et0_cell.py --outdir $OUT --mode aggregate --cells "$CELLS" 2>&1 | tail -50
echo "=== ALL DONE ==="
