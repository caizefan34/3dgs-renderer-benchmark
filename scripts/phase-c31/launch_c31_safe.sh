#!/usr/bin/env bash
# C31 Safe Launcher — FULLY SEQUENTIAL, no GPU sharing.
# P1 (DDP) runs first with exclusive use of all GPUs.
# P2 (camera isolation) and P3 (screening candidates) use GPU 0 only.
# Each phase guards against failure so later phases still run.

cd ~/3dgs-renderer-benchmark
mkdir -p results/phase-c31 logs/phase-c31

export PYTHONPATH=src:$PYTHONPATH
NOW=$(date +%Y%m%d_%H%M%S)
FAILED_PHASES=""

echo "=== C31 Safe Launcher (Sequential) at $NOW ==="
echo "GPU count: $(nvidia-smi -L | wc -l)"
echo "Free memory check:"
free -g | head -2
echo ""

# ── PHASE 1: DDP Scaling (exclusive: N=1 GPU0, N=2 GPU0-1, N=4 GPU0-3, N=8 GPU0-7) ──
echo "[P1] DDP scaling (N=1,2,4,8 sequential, all GPUs)..."
nohup python3 -u scripts/phase-c31/c31_a_ddp.py \
    --out results/phase-c31/c31_a_nNVAL.json \
    --warmup 50 --measured 150 \
    --port-offset 0 \
    > logs/phase-c31/c31_a_${NOW}.log 2>&1 &
P1_PID=$!
echo "  Launched (PID: $P1_PID)"
if wait $P1_PID; then
    echo "  P1 DDP complete."
else
    echo "  [WARN] P1 DDP FAILED (see logs/phase-c31/c31_a_${NOW}.log)"
    FAILED_PHASES="$FAILED_PHASES P1"
fi

# Ensure all GPUs are free before single-GPU phases
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# ── PHASE 2: Camera isolation (GPU0 only) ──
echo ""
echo "[P2] Camera isolation experiment (GPU0)..."
nohup python3 -u scripts/phase-c31/c31_b_camera_isolate.py \
    --out results/phase-c31/c31_b_camera_isolate.json \
    --gpu 0 --warmup 15 --n 50 --n-profile 5 \
    > logs/phase-c31/c31_b_${NOW}.log 2>&1 &
P2_PID=$!
echo "  Launched (PID: $P2_PID)"
if wait $P2_PID; then
    echo "  P2 Camera isolation complete."
else
    echo "  [WARN] P2 Camera isolation FAILED (see logs/phase-c31/c31_b_${NOW}.log)"
    FAILED_PHASES="$FAILED_PHASES P2"
fi

# ── PHASE 3: Remaining candidates (GPU0, sequential) ──
echo ""
echo "[P3] Remaining candidates on GPU0 (sequential)..."

CANDIDATES=(
    "d_param_cadence"
    "e_birth"
    "gh_camera"
    "ij_convergence"
)

for cand in "${CANDIDATES[@]}"; do
    script="scripts/phase-c31/c31_${cand}.py"
    if [ -f "$script" ]; then
        echo "  Running C31-${cand}..."
        if python3 -u "$script" \
            --out "results/phase-c31/c31_${cand}.json" \
            --gpu 0 \
            >> "logs/phase-c31/c31_${cand}_${NOW}.log" 2>&1; then
            echo "  C31-${cand} complete."
        else
            echo "  [WARN] C31-${cand} FAILED (see logs/phase-c31/c31_${cand}_${NOW}.log)"
            FAILED_PHASES="$FAILED_PHASES ${cand}"
        fi
    else
        echo "  Script $script not found, skipping."
    fi
    sleep 3
done

echo ""
echo "=== C31 launcher finished ==="
if [ -n "$FAILED_PHASES" ]; then
    echo "  FAILED phases:${FAILED_PHASES}"
else
    echo "  All phases complete."
fi
echo "  Results: results/phase-c31/"
echo "  Logs:    logs/phase-c31/"
echo "========================================"
