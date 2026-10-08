#!/bin/bash
# P15-P0 Step 4 driver: offline analysis + report + ledger + manifest.
# Offline only (no GPU, no training). Analysis stops with
# P0_EXPORT_INVALID if any export failed validation; the report still
# documents the outcome.
set -u
ROOT=/root/epic05_p15/p0
P=/root/miniforge3/envs/gsplat/bin/python
cd "$ROOT/scripts"
$P p15_p0_analyze.py \
  --exports "$ROOT/exports" \
  --summaries "$ROOT/summaries" \
  --out "$ROOT/artifacts/P15_P0_PARETO.json" \
  --verdict-out "$ROOT/artifacts/P15_P0_VERDICT.json" \
  2>&1 | tee "$ROOT/logs/analyze.log"
rc_a=${PIPESTATUS[0]}
echo "analyze rc=$rc_a"
$P p15_p0_report.py \
  --pareto "$ROOT/artifacts/P15_P0_PARETO.json" \
  --out "$ROOT/artifacts/P15_P0_REPORT.md" \
  --root "$ROOT" \
  2>&1 | tee "$ROOT/logs/report.log"
rc_r=${PIPESTATUS[0]}
echo "report rc=$rc_r"
[ $rc_a -eq 0 ] || exit $rc_a
exit $rc_r
