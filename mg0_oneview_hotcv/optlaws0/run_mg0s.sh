#!/bin/bash
# OPTLAWS0 / MG0-S — sidecar driver (EPIC-05, GPU 2 = bcca92b9), sequential.
# Runs AFTER MG0 Stage 1 has frozen the 64-camera set. Does not touch MG0
# scripts or gates; mg0_core is imported read-only as the frozen definition.
set -u
D=/root/epic05_mg0
PY=/root/miniforge3/envs/gsplat/bin/python
export CUDA_VISIBLE_DEVICES=2
export MG0_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '3p' | tr -d ' ')"
export MG0_SCRIPTS_DIR="$D/scripts"
mkdir -p "$D/artifacts" "$D/logs"

cont() {
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu \
      --format=csv > "$D/logs/optlaws0_cont_$1.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv > "$D/logs/optlaws0_cont_apps_$1.csv"
}

run() {
  name="$1"; shift
  echo "=== [MG0S:$name] $(date -u +%H:%M:%S) ==="
  cont "${name}_start"
  "$@" > "$D/logs/mg0s_${name}.log" 2>&1
  rc=$?
  cont "${name}_end"
  if [ "$rc" -ne 0 ]; then
    echo "=== [MG0S:$name] FAILED rc=$rc ==="
    tail -40 "$D/logs/mg0s_${name}.log"
    exit "$rc"
  fi
  echo "=== [MG0S:$name] OK $(date -u +%H:%M:%S) ==="
}

cd "$D/optlaws0" || exit 1
$PY -m py_compile mg0s_variance.py mg0s_lowrank.py mg0s_gate.py || { echo "SYNTAX ERROR"; exit 2; }

if [ ! -f "$D/artifacts/MG0_CAMERA_SET.json" ]; then
  echo "MG0 Stage 1 (MG0_CAMERA_SET.json) missing — run MG0 Stage 1 first."
  exit 3
fi

echo "MG0-S start $(date -u) on GPU $MG0_GPU_UUID"
cont pipeline_start

for s in room bicycle; do
  for c in 20000 25000; do
    run "var_${s}_${c}"     $PY mg0s_variance.py "$s" "$c"
    run "lowrank_${s}_${c}" $PY mg0s_lowrank.py  "$s" "$c"
  done
done

run gate $PY mg0s_gate.py

cont pipeline_end
echo "MG0-S DONE $(date -u) — MG0-S row IDs are frozen; OC0 may now run."