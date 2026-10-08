#!/bin/bash
# OPTLAWS0 / OC0 — sidecar driver (EPIC-05, GPU 2 = bcca92b9), sequential.
#
# Preconditions enforced here:
#   * the MG0 pipeline is no longer running on this GPU;
#   * MG0-S has frozen the row ids (MG0S_VARIANCE_{scene}_{ckpt}.npz present),
#     because OC0 row/camera ids may only be frozen AFTER MG0-S;
#   * the instrumented OC0 copy of the frozen worktree builds cleanly.
# MG0 scripts, MG0 gates, the production renderer, the loss and the resolution
# are untouched; mg0_core is imported read-only.
set -u
D=/root/epic05_mg0
PY=/root/miniforge3/envs/gsplat/bin/python
export CUDA_VISIBLE_DEVICES=2
export MG0_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '3p' | tr -d ' ')"
export MG0_SCRIPTS_DIR="$D/scripts"
export MG0_ROOT="$D"
export OC0_WORKTREE="$D/c0_worktree_oc0"
mkdir -p "$D/artifacts" "$D/artifacts/oc0" "$D/logs"

cont() {
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu \
      --format=csv > "$D/logs/oc0_cont_$1.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv > "$D/logs/oc0_cont_apps_$1.csv"
}

run() {
  name="$1"; shift
  echo "=== [OC0:$name] $(date -u +%H:%M:%S) ==="
  cont "${name}_start"
  "$@" > "$D/logs/oc0_${name}.log" 2>&1
  rc=$?
  cont "${name}_end"
  if [ "$rc" -ne 0 ]; then
    echo "=== [OC0:$name] FAILED rc=$rc ==="
    tail -40 "$D/logs/oc0_${name}.log"
    exit "$rc"
  fi
  echo "=== [OC0:$name] OK $(date -u +%H:%M:%S) ==="
}

cd "$D/optlaws0" || exit 1
$PY -m py_compile oc0_common.py oc0_selection.py oc0_gradient_rows.py \
    oc0_patch_renderer.py oc0_replay.py oc0_renderer_features.py \
    oc0_parity.py oc0_contamination.py oc0_analysis.py oc0_gate.py \
    || { echo "SYNTAX ERROR"; exit 2; }

if pgrep -f "mg0_run_all.sh" > /dev/null 2>&1; then
  echo "MG0 pipeline still running — OC0 must wait (priority order in DSH_TASK)."
  exit 3
fi

for s in room bicycle; do
  for c in 20000 25000; do
    f="$D/artifacts/MG0S_VARIANCE_${s}_${c}.npz"
    if [ ! -f "$f" ]; then
      echo "MG0-S artifact missing: $f — run run_mg0s.sh first."
      exit 4
    fi
  done
done

echo "OC0 start $(date -u) on GPU $MG0_GPU_UUID"
cont pipeline_start

# ---- 1. instrumented copy of the frozen worktree (additive publication only)
run patch $PY oc0_patch_renderer.py

# ---- 2. freeze row + camera selection from the MG0-S variance ranking ------
run selection $PY oc0_selection.py

# ---- 3. gradient conflict labels (reuses the MG0 gradient machinery) -------
for s in room bicycle; do
  for c in 20000 25000; do
    run "grad_${s}_${c}" $PY oc0_gradient_rows.py "$s" "$c"
  done
done

# ---- 4. parity (MUST pass before any renderer feature is accepted) ---------
for s in room bicycle; do
  for c in 20000 25000; do
    run "par_ref_${s}_${c}" $PY oc0_parity.py reference "$s" "$c"
    run "par_rep_${s}_${c}" $PY oc0_parity.py replay "$s" "$c"
  done
done
run parity $PY oc0_parity.py finalize

# ---- 5. renderer-state features from the exact native replay ---------------
for s in room bicycle; do
  for c in 20000 25000; do
    run "feat_${s}_${c}" $PY oc0_renderer_features.py "$s" "$c"
  done
done

# ---- 6. contamination audit, analysis, mechanical gate ---------------------
run contamination $PY oc0_contamination.py
run analysis      $PY oc0_analysis.py
run gate          $PY oc0_gate.py

cont pipeline_end
echo "OC0 DONE $(date -u)"
echo "Required artifacts:"
ls -1 "$D/artifacts/OC0_SELECTION.json" \
      "$D/artifacts/OC0_INSTRUMENTATION_PARITY.json" \
      "$D/artifacts/OC0_CONTAMINATION.json" \
      "$D/artifacts/oc0/"OC0_RENDERER_FEATURES_*.parquet \
      "$D/artifacts/OC0_GRADIENT_ROWS_"*.npz 2>/dev/null
echo "STOP — return raw artifacts for independent analysis."