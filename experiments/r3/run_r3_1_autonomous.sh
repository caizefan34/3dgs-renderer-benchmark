#!/bin/bash
# R3.1 Full Autonomous Pipeline - launched via nohup, runs unattended
# Waits for all 9 sub-batches, then runs merge→analyze→decision→forensics→audit
set -uo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
PARALLEL_DIR="/mnt/storage_pool/liaoyuanjun/r3_1_parallel"
MERGED_DIR="/mnt/storage_pool/liaoyuanjun/r3_1_full"
LOG="/mnt/storage_pool/liaoyuanjun/r3_1_pipeline.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== R3.1 Pipeline started ==="

cd "$REPO"

# Phase 0: Wait for all 9 sub-batches to complete
log "Phase 0: Waiting for all 9 sub-batches..."
while true; do
    complete=0
    for d in 5000_A 5000_B 5000_C 15000_A 15000_B 15000_C 30000_A 30000_B 30000_C; do
        if [ -f "$PARALLEL_DIR/$d/certificate_correctness.json" ]; then
            complete=$((complete+1))
        fi
    done
    log "  Sub-batches complete: $complete/9"
    if [ "$complete" -eq 9 ]; then
        log "  All 9 sub-batches complete!"
        break
    fi
    sleep 120
done

# Phase 1: Merge sub-batch outputs
log "Phase 1: Merging sub-batch outputs..."
"$PYTHON" experiments/r3/merge_r3_1_subbatches.py 2>&1 | tee -a "$LOG"
log "  Merge complete."

# Phase 2: Aggregate analysis
log "Phase 2: Running r3_analyze.py..."
"$PYTHON" experiments/r3/r3_analyze.py --input-dir "$MERGED_DIR" --output "$MERGED_DIR" 2>&1 | tee -a "$LOG"
log "  Analysis complete."

# Phase 3: Per-window decision
log "Phase 3: Running per-window decisions..."
for w in 5000 15000 30000; do
    log "  Decision for window $w..."
    "$PYTHON" experiments/r3/r3_decision.py --input "$MERGED_DIR/$w" 2>&1 | tee -a "$LOG"
done

# Phase 4: Cross-window decision
log "Phase 4: Running cross-window decision..."
"$PYTHON" experiments/r3/r3_decision.py --input "$MERGED_DIR" 2>&1 | tee -a "$LOG"

# Phase 5: Float64 forensics
log "Phase 5: Running float64 forensics..."
"$PYTHON" experiments/r3/r3_float64_forensics.py --input-dir "$MERGED_DIR" --output "$MERGED_DIR/float64_forensics.json" 2>&1 | tee -a "$LOG"

# Phase 6: Copy analysis to audit package
log "Phase 6: Copying analysis to audit package..."
AUDIT="$REPO/audit_packages/candidate_c_r3_1_final"
cp "$MERGED_DIR/aggregated_summary.json" "$AUDIT/analysis/" 2>/dev/null || log "  WARN: aggregated_summary.json not found"
cp "$MERGED_DIR/final_decision.json" "$AUDIT/analysis/" 2>/dev/null || log "  WARN: final_decision.json not found"
cp "$MERGED_DIR/float64_forensics.json" "$AUDIT/analysis/" 2>/dev/null || log "  WARN: float64_forensics.json not found"
for w in 5000 15000 30000; do
    cp "$MERGED_DIR/$w/final_decision.json" "$AUDIT/analysis/decision_$w.json" 2>/dev/null || log "  WARN: decision_$w.json not found"
done
cp "$REPO/reports/r3_1/vec_gate_report.json" "$AUDIT/analysis/" 2>/dev/null || true

# Phase 7: Build audit package
log "Phase 7: Building audit package..."
"$PYTHON" audit_packages/build_r3_1_audit.py 2>&1 | tee -a "$LOG"

log "=== R3.1 Pipeline COMPLETE ==="
log "Results: $MERGED_DIR"
log "Audit:   $AUDIT"
log "=== DONE ==="
