#!/bin/bash
# P15-P0 Step 2 driver: source identity check (no GPU needed).
set -u
ROOT=/root/epic05_p15/p0
P=/root/miniforge3/envs/gsplat/bin/python
mkdir -p "$ROOT/artifacts" "$ROOT/logs" "$ROOT/exports" "$ROOT/summaries"
cd "$ROOT/scripts"
$P p15_p0_identity.py \
  --p14-scripts /root/epic05_p14/scripts \
  --p14-root /root/epic05_p14 \
  --p15-root "$ROOT" \
  > "$ROOT/logs/identity.log" 2>&1
rc=$?
tail -n 25 "$ROOT/logs/identity.log"
echo "identity rc=$rc"
exit $rc
