#!/usr/bin/env bash
set -euo pipefail

REPO=/root/codex_fastgs_audit_20260927
LOG_DIR=/root/fastgs_sparse_adjoint_pilot_eval_20260927
EXPECTED_COMMIT=44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f
EXPECTED_CHECKPOINT=d91ba09a5ab040373cc59a93b6864949cb3cc1b5d5b2dea17c79223fef7e927b

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum checkpoints/room_big_4099.pth | cut -d ' ' -f1)" = "$EXPECTED_CHECKPOINT"
test -d /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room/images
test ! -e output/room_sparse_adj_eval_p1_10k
test ! -e output/room_sparse_adj_eval_p75_10k
test ! -e "$LOG_DIR"

gpu_lines=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
printf 'GPU preflight (index, utilization percent, memory MiB):\n%s\n' "$gpu_lines"
for gpu in 3 4 5; do
    line=$(printf '%s\n' "$gpu_lines" | awk -F, -v id="$gpu" '$1 + 0 == id { print $2 "," $3 }')
    test -n "$line"
    read -r util memory <<< "$(printf '%s\n' "$line" | awk -F, '{ gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); print $1, $2 }')"
    if (( util != 0 || memory != 0 )); then
        echo "GPU $gpu became busy (util=$util%, memory=$memory MiB); aborting."
        exit 3
    fi
done
if pgrep -af '[p]ython.*train_sparse_adj.py'; then
    echo "A sparse-adjoint run already exists; aborting."
    exit 4
fi

mkdir -p "$LOG_DIR"
common=(
    --source_path /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room --eval
    --images images --resolution -1 --iterations 10000
    --start_checkpoint checkpoints/room_big_4099.pth
    --save_iterations 4612 10000 --checkpoint_iterations 10000
    --test_iterations 10000 --highfeature_lr 0.02 --grad_abs_thresh 0.0004
)

env CUDA_VISIBLE_DEVICES=3 FASTGS_SPARSE_ADJOINT_P=1.0 FASTGS_SPARSE_ADJOINT_SEED=20260927 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    /usr/bin/time -p -o "$LOG_DIR/p1.time" .venv/bin/python train_sparse_adj.py \
    "${common[@]}" --model_path output/room_sparse_adj_eval_p1_10k \
    >"$LOG_DIR/p1.log" 2>&1 </dev/null &
p1_pid=$!
printf '%s\n' "$p1_pid" > "$LOG_DIR/p1.pid"

env CUDA_VISIBLE_DEVICES=4 FASTGS_SPARSE_ADJOINT_P=0.75 FASTGS_SPARSE_ADJOINT_SEED=20260927 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    /usr/bin/time -p -o "$LOG_DIR/p75.time" .venv/bin/python train_sparse_adj.py \
    "${common[@]}" --model_path output/room_sparse_adj_eval_p75_10k \
    >"$LOG_DIR/p75.log" 2>&1 </dev/null &
p75_pid=$!
printf '%s\n' "$p75_pid" > "$LOG_DIR/p75.pid"

printf 'started p1 pid=%s GPU=3 and p75 pid=%s GPU=4; logs=%s\n' "$p1_pid" "$p75_pid" "$LOG_DIR"
