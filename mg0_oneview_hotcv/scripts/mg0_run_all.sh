#!/bin/bash
# MG0 OneView-GS / HotCV — full pipeline driver (sequential, GPU 2 = bcca92b9).
# Every step is logged to /root/epic05_mg0/logs/<step>.log and aborts on failure.
set -u
D=/root/epic05_mg0
PY=/root/miniforge3/envs/gsplat/bin/python
export CUDA_VISIBLE_DEVICES=2
export MG0_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '3p' | tr -d ' ')"
mkdir -p "$D/artifacts" "$D/controls" "$D/logs" "$D/runs"
cd "$D/scripts" || exit 1

cont() {
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu \
      --format=csv > "$D/logs/cont_$1.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv > "$D/logs/cont_apps_$1.csv"
}

run() {
  name="$1"; shift
  echo "=== [$name] $(date -u +%H:%M:%S) ==="
  cont "${name}_start"
  "$@" > "$D/logs/${name}.log" 2>&1
  rc=$?
  cont "${name}_end"
  if [ "$rc" -ne 0 ]; then
    echo "=== [$name] FAILED rc=$rc ==="
    tail -40 "$D/logs/${name}.log"
    exit "$rc"
  fi
  echo "=== [$name] OK $(date -u +%H:%M:%S) ==="
}

echo "MG0 pipeline start $(date -u) on GPU $MG0_GPU_UUID"
cont pipeline_start

$PY -m py_compile mg0_core.py mg0_stage0_audit.py mg0_stage1_cameras.py \
    mg0_variance.py mg0_cost.py mg0_gate.py mg0_onestep.py mg0_finalize.py \
    || { echo "SYNTAX ERROR"; exit 2; }

# Stage 0 + 1
run s0_room    $PY mg0_stage0_audit.py room
run s0_bicycle $PY mg0_stage0_audit.py bicycle
run s1_cameras $PY mg0_stage1_cameras.py

# Stages 2-4
run v_room_15000    $PY mg0_variance.py room 15000
run v_bicycle_15000 $PY mg0_variance.py bicycle 15000
run v_room_20000    $PY mg0_variance.py room 20000
run v_bicycle_20000 $PY mg0_variance.py bicycle 20000
run v_room_25000    $PY mg0_variance.py room 25000
run v_bicycle_25000 $PY mg0_variance.py bicycle 25000

# Merge + cost + gate
run finalize_1 $PY mg0_finalize.py
run cost       $PY mg0_cost.py
run gate       $PY mg0_gate.py

# Stage 6 (gated inside the script itself)
run onestep $PY mg0_onestep.py

# Final verdict + manifest
run finalize_2 $PY mg0_finalize.py

cont pipeline_end
echo "MG0 pipeline DONE $(date -u)"
