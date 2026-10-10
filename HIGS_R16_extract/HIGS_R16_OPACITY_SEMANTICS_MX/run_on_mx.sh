#!/usr/bin/env bash
set -euo pipefail
PROJECT=${HIGS_PROJECT:-/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx}
R15=${R15_CAPTURE:-/mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX/captured_projected_inputs.pt}
OUT=${R16_OUT:-/mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT"
test -s "$R15"
python3 "$(dirname "$(realpath "$0")")/r16_replay.py" --source-root "$PROJECT" --capture "$R15" --out "$OUT" 2>&1 | tee "$OUT/run.log"
