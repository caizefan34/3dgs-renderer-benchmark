#!/usr/bin/env bash
# Master post-processing pipeline: runs after all 12 seed4344 runs complete.
# Order: aggregate (9-pair stats + verdict) -> first-divergence -> derived ->
#        report + manifest. All idempotent. Prints the terminal verdict at the end.
set -u
Q=/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
RUNROOT=/dev/shm/c0_quality_stability/runs
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
cd "$Q" || exit 1
echo "[pipeline] $(date -u +%H:%M:%SZ) collect seed4344 checkpoints"
mkdir -p "$Q/checkpoints"
for d in "$RUNROOT"/*/; do
  for ck in "$d"ckpt_lite_30000.pt; do
    [ -f "$ck" ] || continue
    base=$(basename "$d"); cp -n "$ck" "$Q/checkpoints/${base}.ckpt_lite_30000.pt" 2>/dev/null || true
  done
done
echo "checkpoints in campaign: $(ls "$Q/checkpoints"/*.ckpt_lite_30000.pt 2>/dev/null | wc -l)"
echo "[pipeline] $(date -u +%H:%M:%SZ) aggregate"
$PY scripts/aggregate_quality.py --outdir "$Q" || { echo "AGGREGATE FAILED"; exit 1; }
echo "[pipeline] first_divergence"
$PY scripts/first_divergence.py --outdir "$Q" || { echo "FIRST_DIV FAILED"; exit 1; }
echo "[pipeline] derived"
$PY scripts/build_derived.py --outdir "$Q" || { echo "DERIVED FAILED"; exit 1; }
echo "[pipeline] report + manifest"
$PY scripts/build_report.py --outdir "$Q" || { echo "REPORT FAILED"; exit 1; }
echo "[pipeline] DONE $(date -u +%H:%M:%SZ)"
echo "=== VERDICT ==="
$PY -c "import json;v=json.load(open('$Q/QUALITY_VERDICT.json'));print('verdict=',v['verdict']);print('mean_dPSNR=',v['criteria']['mean_d_psnr'],'CI=',v['criteria']['dPSNR_ci95'])"
