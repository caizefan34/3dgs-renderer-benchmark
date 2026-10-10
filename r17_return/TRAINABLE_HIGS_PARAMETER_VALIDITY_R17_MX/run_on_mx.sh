#!/usr/bin/env bash
# R17 master runner — MX (A100-40GB).
# Runs preflight -> P0..P6 -> collect. Never modifies renderer sources, binaries,
# or checkpoints. Every step's exit code is recorded; a failing step does NOT
# stop the remaining steps (independent measurements), and nothing is marked
# executed unless its deliverable file exists and parses.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_on_mx.sh
# Optional environment overrides: HIGS_PROJECT, R17_OUT, R17_CKPT, R15_CAPTURE,
# R16_OUT, R17_CAMERAS_JSON, R17_P3_CELLS, R17_P4_PROTOCOLS.
set -uo pipefail

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
SCRIPTS="$HERE/scripts"

export HIGS_PROJECT="${HIGS_PROJECT:-/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx}"
export R17_OUT="${R17_OUT:-/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX}"
export R17_CKPT="${R17_CKPT:-/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt}"
export R15_CAPTURE="${R15_CAPTURE:-/mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX/captured_projected_inputs.pt}"
export R16_OUT="${R16_OUT:-/mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$HIGS_PROJECT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$R17_OUT"/logs "$R17_OUT"/raw "$R17_OUT"/oracle "$R17_OUT"/captures
LOGS="$R17_OUT/logs"

# Same environment R15/R16 used on MX.
if [ -f /home/liaoyuanjun/miniforge3/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/liaoyuanjun/miniforge3/bin/activate /mnt/storage_pool/liaoyuanjun/higs-13scene-env || \
    echo "WARN: conda env activation returned non-zero"
fi
PY="${PYTHON:-python3}"

{
  echo "== R17 runner =="
  date -u
  echo "HIGS_PROJECT=$HIGS_PROJECT"
  echo "R17_OUT=$R17_OUT"
  echo "R17_CKPT=$R17_CKPT"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "python: $($PY -c 'import sys; print(sys.executable, sys.version)' 2>&1)"
  nvidia-smi 2>&1 | head -25
  df -h "$R17_OUT" 2>&1 | tail -2
} | tee "$LOGS/runner.log"

: > "$LOGS/step_exit_codes.txt"

run_step() {
  local name="$1"; shift
  echo ""
  echo "########## STEP $name ##########"
  ( "$@" ) 2>&1 | tee "$LOGS/$name.log"
  local code=${PIPESTATUS[0]}
  echo "########## STEP $name EXIT $code ##########"
  echo "${name} ${code}" >> "$LOGS/step_exit_codes.txt"
  return 0
}

run_step preflight            $PY "$SCRIPTS/r17_preflight.py"
run_step p0_arithmetic        $PY "$SCRIPTS/r17_p0_arithmetic_audit.py"
run_step p1_parameter_contract $PY "$SCRIPTS/r17_p1_parameter_contract.py"
run_step p2_camera_identity   $PY "$SCRIPTS/r17_p2_camera_identity.py"
run_step p3_controlled        $PY "$SCRIPTS/r17_p3_controlled.py" --cells "${R17_P3_CELLS:-A,B,B2,B3,C,C2,D,D2}"
run_step p4_benchmark         $PY "$SCRIPTS/r17_p4_benchmark.py" --protocols "${R17_P4_PROTOCOLS:-B1,B4}"
run_step p5_safety_gate       $PY "$SCRIPTS/r17_p5_safety_gate.py"
run_step p6_verdict           $PY "$SCRIPTS/r17_p6_verdict.py"
run_step collect              $PY "$SCRIPTS/r17_collect.py"
$PY "$SCRIPTS/build_step_codes.py" "$LOGS"

echo ""
echo "==== R17 RUN FINISHED ===="
echo "Deliverables: $R17_OUT"
cat "$LOGS/step_exit_codes.txt"
