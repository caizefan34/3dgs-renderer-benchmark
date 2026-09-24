#!/usr/bin/env bash
# C31 Batch Launcher — runs all screening candidates on 8 GPUs
set -e
cd ~/3dgs-renderer-benchmark
mkdir -p results/phase-c31 logs/phase-c31

export PYTHONPATH=src:$PYTHONPATH
NOW=$(date +%Y%m%d_%H%M%S)

echo "=== C31 Screening Launch at $NOW ==="
echo "GPU count: $(nvidia-smi -L | wc -l)"
echo ""

# --- C31-A: Multi-GPU Scaling Baseline ---
echo "[C31-A] Starting multi-GPU scaling baseline..."
for N in 1 2 4 8; do
    if [ $N -gt $(nvidia-smi -L | wc -l) ]; then break; fi
    nohup python3 -u scripts/phase-c31/c31_a_multigpu.py \
        --out results/phase-c31/c31_a_n${N}.json \
        --warmup 50 --measured 100 \
        > logs/phase-c31/c31_a_n${N}_${NOW}.log 2>&1 &
    echo "  Launched N=$N (PID: $!)"
done
echo ""

# --- C31-D: Parameter Cadence ---
echo "[C31-D] Starting parameter cadence analysis..."
nohup python3 -u scripts/phase-c31/c31_d_param_cadence.py \
    --out results/phase-c31/c31_d_param_cadence.json --steps 500 \
    > logs/phase-c31/c31_d_${NOW}.log 2>&1 &
echo "  Launched (PID: $!)"

# --- C31-E: Gaussian Birth ---
echo "[C31-E] Starting birth state analysis..."
nohup python3 -u scripts/phase-c31/c31_e_birth.py \
    --out results/phase-c31/c31_e_birth.json --steps 500 \
    > logs/phase-c31/c31_e_${NOW}.log 2>&1 &
echo "  Launched (PID: $!)"

# --- C31-F: State Co-Design ---
echo "[C31-F] Starting state co-design analysis..."
nohup python3 -u scripts/phase-c31/c31_f_state.py \
    --out results/phase-c31/c31_f_state.json \
    --warmup 50 --measured 100 \
    > logs/phase-c31/c31_f_${NOW}.log 2>&1 &
echo "  Launched (PID: $!)"

# --- C31-GH: Camera Utility ---
echo "[C31-GH] Starting camera utility analysis..."
nohup python3 -u scripts/phase-c31/c31_gh_camera.py \
    --out results/phase-c31/c31_gh_camera.json --gpu 0 --max-cameras 50 \
    > logs/phase-c31/c31_gh_${NOW}.log 2>&1 &
echo "  Launched (PID: $!)"

# --- C31-IJ: Convergence ---
echo "[C31-IJ] Starting convergence analysis..."
nohup python3 -u scripts/phase-c31/c31_ij_convergence.py \
    --out results/phase-c31/c31_ij_convergence.json --steps 500 --gpu 0 \
    > logs/phase-c31/c31_ij_${NOW}.log 2>&1 &
echo "  Launched (PID: $!)"

echo ""
echo "=== All C31 candidates launched. Monitor with: ==="
echo "  tail -f logs/phase-c31/c31_*.log"
echo "  python3 scripts/phase-c31/_check_progress.py results/phase-c31/"
echo "================================"
