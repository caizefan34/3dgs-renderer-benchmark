#!/bin/bash
# Wait for 30000_B to complete, then finish the full pipeline
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
PARALLEL_DIR="/mnt/storage_pool/liaoyuanjun/r3_1_parallel"
MERGED_DIR="/mnt/storage_pool/liaoyuanjun/r3_1_full"
LOG="/mnt/storage_pool/liaoyuanjun/finalize.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Waiting for 30000_B ==="
while [ ! -f "$PARALLEL_DIR/30000_B/certificate_correctness.json" ]; do
    sleep 120
    log "  Still waiting... $(tail -1 $PARALLEL_DIR/30000_B/runner_rerun.log 2>/dev/null)"
done
log "=== 30000_B COMPLETE ==="

cd "$REPO"

log "=== Re-merging with 30000_B ==="
"$PYTHON" experiments/r3/merge_r3_1_subbatches.py 2>&1 | tee -a "$LOG"

log "=== Re-analyzing ==="
"$PYTHON" experiments/r3/r3_analyze.py --input-dir "$MERGED_DIR" --output "$MERGED_DIR" 2>&1 | tee -a "$LOG"

log "=== Per-window decisions ==="
for w in 5000 15000 30000; do
    "$PYTHON" experiments/r3/r3_decision.py --input "$MERGED_DIR/$w" 2>&1 | tee -a "$LOG"
done

log "=== Cross-window decision ==="
"$PYTHON" experiments/r3/r3_decision.py --input "$MERGED_DIR" 2>&1 | tee -a "$LOG"

log "=== Float64 forensics (full 90 measurements) ==="
"$PYTHON" experiments/r3/r3_float64_forensics.py --input-dir "$MERGED_DIR" --output "$MERGED_DIR/float64_forensics.json" 2>&1 | tee -a "$LOG"

log "=== Copy to audit package ==="
AUDIT="$REPO/audit_packages/candidate_c_r3_1_final"
cp "$MERGED_DIR/aggregated_summary.json" "$AUDIT/analysis/"
cp "$MERGED_DIR/final_decision.json" "$AUDIT/analysis/"
cp "$MERGED_DIR/float64_forensics.json" "$AUDIT/analysis/"
for w in 5000 15000 30000; do
    cp "$MERGED_DIR/$w/final_decision.json" "$AUDIT/analysis/decision_$w.json"
done
cp "$REPO/reports/r3_1/vec_gate_report.json" "$AUDIT/analysis/" 2>/dev/null || true

log "=== Build audit package ==="
"$PYTHON" audit_packages/build_r3_1_audit.py 2>&1 | tee -a "$LOG"

log "=== PIPELINE COMPLETE ==="
date
