#!/bin/bash
# P12-B campaign launcher: 13 fresh pairs (A0_FUSED vs A1_FUSED).
# PRECONDITION: the P12-A publication 30K window is CLOSED (all 26 P12-A runs
# finished). Host-level rule: P12-A and P12-B full campaigns never overlap.
set -u
CAMP=/root/epic05_p12
if [ ! -f "$CAMP/runs/p12a/scheduler_summary.json" ]; then
  echo "REFUSING: P12-A window is not closed (no scheduler_summary.json)."
  exit 1
fi
mkdir -p "$CAMP/runs/p12b" "$CAMP/logs"
cd "$CAMP"
setsid nohup /root/miniforge3/envs/gsplat/bin/python p12_pair_scheduler.py \
  --manifest "$CAMP/manifests/p12b_manifest.json" \
  --trainer "$CAMP/publication_trainer_p12.py" \
  --python /root/epic05_py_wrapper.sh \
  --out "$CAMP/runs/p12b" > "$CAMP/logs/p12b_scheduler.log" 2>&1 < /dev/null &
echo "P12B_SCHEDULER_PID=$!"
