#!/usr/bin/env bash
# R18 master runner — validated benchmark (correctness + performance).
# Runs preflight -> P0 -> P1 -> P2 -> P3 -> collect. Never modifies renderer
# sources, binaries or checkpoints. A failing step does NOT stop the rest.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_on_mx.sh
set -uo pipefail

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
SCRIPTS="$HERE"   # this runner lives inside the scripts/ dir

export HIGS_PROJECT="${HIGS_PROJECT:-/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx}"
export R18_OUT="${R18_OUT:-/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_VALIDATED_BENCHMARK_R18_MX}"
export R18_CKPT="${R18_CKPT:-/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-/mnt/storage_pool/liaoyuanjun/tmp}"
export PY="${PYTHON:-/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3}"
export PYTHON="${PYTHON:-/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3}"
# conda nvcc activate hook trips `set -u` if these are unset; pre-define empty.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
export PYTHONPATH="$HIGS_PROJECT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$R18_OUT"/logs "$R18_OUT"/raw "$R18_OUT"/captures
LOGS="$R18_OUT/logs"

{
  echo "== R18 runner =="
  date -u
  echo "HIGS_PROJECT=$HIGS_PROJECT"
  echo "R18_OUT=$R18_OUT"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "python: $($PY -c 'import sys; print(sys.executable, sys.version)' 2>&1)"
  nvidia-smi 2>&1 | head -25
  df -h "$R18_OUT" 2>&1 | tail -2
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

run_step preflight            $PY "$SCRIPTS/r18_preflight.py"
run_step p0_integrity         $PY "$SCRIPTS/r18_p0_integrity.py"
run_step p1_correctness       $PY "$SCRIPTS/r18_p1_correctness.py" --configs "${R18_P1_CONFIGS:-580K_B1,580K_B4,4M_B1,4M_B4}"
run_step p2_performance       $PY "$SCRIPTS/r18_p2_performance.py" --configs "${R18_P2_CONFIGS:-580K_540p_B1,580K_540p_B4,4M_540p_B1,4M_540p_B4,4M_1080p_B4}"
run_step p3_verdict           $PY "$SCRIPTS/r18_p3_verdict.py"
run_step collect              $PY "$SCRIPTS/r18_collect.py"
$PY "$SCRIPTS/build_step_codes.py" "$LOGS"

echo ""
echo "==== R18 RUN FINISHED ===="
echo "Deliverables: $R18_OUT"
cat "$LOGS/step_exit_codes.txt"
[stderr]