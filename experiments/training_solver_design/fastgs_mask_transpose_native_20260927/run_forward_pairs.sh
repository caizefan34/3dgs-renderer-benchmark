#!/usr/bin/env bash
set -euo pipefail
cd /root/codex_fastgs_audit_20260927
export CUDA_VISIBLE_DEVICES=3
PY=/root/codex_fastgs_audit_20260927/.venv/bin/python
OUT=/root/fastgs_mask_transpose_native_20260927
for pair in 1 2 3; do
  "$PY" "$OUT/forward_cost_probe.py" > "$OUT/forward_gpu3_${pair}_native.json"
  FASTGS_EXTENSION_ROOT=/root/codex_fastgs_mask_transpose_trial_20260927 "$PY" "$OUT/forward_cost_probe.py" > "$OUT/forward_gpu3_${pair}_candidate.json"
done