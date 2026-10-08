#!/bin/bash
# P12-B garden pair contamination rerun (unchanged; fresh back-to-back pair).
set -u
CAMP=/root/epic05_p12
cd "$CAMP"
setsid nohup /root/miniforge3/envs/gsplat/bin/python p12_pair_scheduler.py \
  --manifest "$CAMP/manifests/p12b_garden_rerun_manifest.json" \
  --trainer "$CAMP/publication_trainer_p12.py" \
  --python /root/epic05_py_wrapper.sh \
  --out "$CAMP/runs/p12b" > "$CAMP/logs/p12b_garden_rerun.log" 2>&1 < /dev/null &
echo "GARDEN_RERUN_PID=$!"
