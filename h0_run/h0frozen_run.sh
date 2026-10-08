#!/usr/bin/env bash
# H0-FROZEN driver: 4 frozen cells, one per GPU, in parallel.
set -euo pipefail
PY=/root/miniforge3/envs/gsplat/bin/python
H0=/root/epic05_h0
ROOT=/root/epic05_h0frozen
mkdir -p "$ROOT/cells"
echo "[h0frozen] $(date -u +%FT%TZ) start"

declare -A GPU=( [room_15000]=0 [room_25000]=1 [bicycle_15000]=2 [bicycle_25000]=3 )
pids=()
for cell in room_15000 room_25000 bicycle_15000 bicycle_25000; do
  scene=${cell%_*}; iter=${cell#*_}
  outdir="$ROOT/cells/$cell"
  mkdir -p "$outdir"
  (
    export CUDA_VISIBLE_DEVICES=${GPU[$cell]}
    echo "[h0frozen] launch $cell on GPU ${GPU[$cell]}"
    cd "$H0"
    "$PY" h0_rollout.py --scene "$scene" --iter "$iter" \
      --arms NATIVE,E2D,OPA --Ks 4,8 --folds A,B --offsets 0,1,2,3 --time \
      --out "$outdir/H0_ROLLOUTS.json"
    echo "[h0frozen] done $cell"
  ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[h0frozen] $(date -u +%FT%TZ) rollouts done (fail=$fail)"
exit $fail
