#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx
ENTRY=${1:-"$(dirname "$(realpath "$0")")/reference/R8_WORKING_ENTRYPOINT.py"}
OUT=${2:-/mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX}
SAMPLES=${HIGS_ORACLE_N:-1000}
mkdir -p "$OUT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python3 "$(dirname "$(realpath "$0")")/capture_higs.py" \
  --source-root "$PROJECT" --entrypoint "$ENTRY" \
  --out "$OUT/captured_projected_inputs.pt" --samples "$SAMPLES" --seed 42 \
  2>&1 | tee "$OUT/capture.log"
test -s "$OUT/captured_projected_inputs.pt"
python3 "$(dirname "$(realpath "$0")")/oracle_higs.py" \
  --capture "$OUT/captured_projected_inputs.pt" --source-root "$PROJECT" \
  --out-dir "$OUT/oracle" 2>&1 | tee "$OUT/oracle.log"
echo "Report: $OUT/oracle/oracle_summary.json"
