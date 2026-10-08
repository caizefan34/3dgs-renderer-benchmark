#!/bin/bash
# P18-P0-R1 post-fix pipeline: rebuild trace .so (slot-label fix), re-capture
# room+bicycle (frozen/off/on), re-verify records with the GPU exact-path checker
# (expect 100% claimed-row agreement), and re-run the within-process neutrality probe.
set -uo pipefail
cd /root/epic05_p18_r1
PY=/root/miniforge3/envs/gsplat/bin/python

echo "===== REBUILD (build_trace.sh) ====="
bash scripts/build_trace.sh || { echo BUILD_FAILED; exit 1; }

echo "===== RE-CAPTURE (frozen/off/on) ====="
bash scripts/p18_r1_capture_all.sh

echo "===== RE-VERIFY room (GPU exact path) ====="
CUDA_VISIBLE_DEVICES=3 $PY scripts/p18_r1_verify_records.py room 0 room || { echo VERIFY_ROOM_FAILED; exit 1; }

echo "===== RE-VERIFY bicycle (GPU exact path) ====="
CUDA_VISIBLE_DEVICES=3 $PY scripts/p18_r1_verify_records.py bicycle 0 bicycle || { echo VERIFY_BICYCLE_FAILED; exit 1; }

echo "===== NEUTRALITY (new build) ====="
CUDA_VISIBLE_DEVICES=3 $PY scripts/p18_r1_neutral.py || { echo NEUTRAL_FAILED; exit 1; }

echo "FIX_PIPELINE_DONE"
