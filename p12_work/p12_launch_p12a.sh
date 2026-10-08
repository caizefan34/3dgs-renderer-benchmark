#!/bin/bash
# P12-A campaign launcher: 13 fresh pairs (GMAIN_FUSED vs C0_FUSED),
# preregistered alternating arm order, back-to-back on pinned physical GPUs
# (GPU 4 excluded: foreign co-tenant). P12-B must NOT be launched until the
# P12-A publication window is CLOSED.
set -u
CAMP=/root/epic05_p12
mkdir -p "$CAMP/runs/p12a" "$CAMP/logs"
cd "$CAMP"
setsid nohup /root/miniforge3/envs/gsplat/bin/python p12_pair_scheduler.py \
  --manifest "$CAMP/manifests/p12a_manifest.json" \
  --trainer "$CAMP/publication_trainer_p12.py" \
  --python /root/epic05_py_wrapper.sh \
  --out "$CAMP/runs/p12a" > "$CAMP/logs/p12a_scheduler.log" 2>&1 < /dev/null &
echo "P12A_SCHEDULER_PID=$!"
