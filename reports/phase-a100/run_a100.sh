#!/bin/bash
# A100 Experiment Runner ? 3DGS phase-a100 evidence chain
# Usage: bash run_a100.sh [step]
#   step=0: prepare datasets + symlinks
#   step=1: smoke test (forward pass)
#   step=2: forward correctness
#   step=3: backward/gradient correctness
#   step=4: GT quality audit
#   step=5: 500-step training (room, bicycle, garden ? tile16/tile20/tile32)
#   step=6: 30K training (room baseline validation)
#   step=7: 30K training (bicycle/garden critical runs)
#   step=all: full pipeline (requires free GPUs)

set -e
export PATH="$HOME/miniforge3/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO="/mnt/storage_pool/3dgs-renderer-benchmark/repo"
cd "$REPO"

echo "=== A100 Experiment Runner ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CONDA: anysplat"
echo "REPO: $REPO"
echo "COMMIT: $(git rev-parse HEAD)"
echo ""

step="${1:-all}"

prepare_datasets() {
    echo "=== Step 0: Preparing datasets ==="
    mkdir -p "$REPO/data/official/mipnerf360"
    mkdir -p "$REPO/data/datasets/mipnerf360"

    # Symlink or copy dataset
    # Expected structure:
    # data/official/mipnerf360/{scene}/  ? point_cloud.ply + cameras.json
    # data/datasets/mipnerf360/{scene}/images/  ? GT images
    # 
    # Source: /mnt/storage_pool/liaoyuanjun/data/ or wherever datasets are
    # Currently data must be provided externally
    echo "  Dataset paths configured."
    echo "  TODO: Copy Mip-NeRF 360 datasets to:"
    echo "    $REPO/data/official/mipnerf360/{scene}/"
    echo "    $REPO/data/datasets/mipnerf360/{scene}/"
}

smoke_test() {
    echo "=== Step 1: Smoke test ==="
    conda run -n anysplat python3 -c "
import torch, gsplat
print('gsplat:', gsplat.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
# Quick rasterization test
means = torch.randn(100, 3, device='cuda')
quats = torch.randn(100, 4, device='cuda')
scales = torch.randn(100, 3, device='cuda').exp()
opacities = torch.sigmoid(torch.randn(100, 1, device='cuda'))
colors = torch.randn(100, 3, device='cuda')
viewmat = torch.eye(4, device='cuda').unsqueeze(0)
K = torch.tensor([[500,0,960],[0,500,540],[0,0,1]], device='cuda').unsqueeze(0).float()
img = gsplat.rasterization(
    means, quats, scales, opacities, colors,
    viewmat, K, 1920, 1080, tile_size=16
)[0]
print('Render OK, shape:', img.shape, 'mean:', img.mean().item())
"
}

forward_check() {
    echo "=== Step 2: Forward correctness ==="
    echo "  Run: python3 scripts/epic05/phase8b_fwdbwd_snapshot.py --scene $1 --tile-size $2"
}

backward_check() {
    echo "=== Step 3: Backward/gradient correctness ==="
    echo "  Run: python3 scripts/epic05/gradient_check_modules.py --scene $1 --tile-size $2"
}

quality_check() {
    echo "=== Step 4: GT quality audit ==="
    echo "  Run: python3 scripts/epic05/evaluate_official_quality.py ..."
}

run_500step() {
    scene="$1"
    tile="$2"
    echo "=== 500-step: $scene tile_size=$tile ==="
    conda run -n anysplat python3 scripts/epic05/phase7/run_full.py \
        --scene "$scene" --tile-size "$tile" --steps 500 \
        --label "a100_500step_${scene}_t${tile}"
}

run_30k() {
    scene="$1"
    tile="$2"
    echo "=== 30K: $scene tile_size=$tile ==="
    echo "  WARNING: This takes 2-8 hours per run"
    echo "  conda run -n anysplat python3 scripts/epic05/phase7/run_full.py \\"
    echo "      --scene \"$scene\" --tile-size \"$tile\" --steps 30000 \\"
    echo "      --label \"a100_30k_${scene}_t${tile}\""
}

case "$step" in
    0) prepare_datasets ;;
    1) smoke_test ;;
    2) forward_check "${2:-room}" "${3:-16}" ;;
    3) backward_check "${2:-room}" "${3:-16}" ;;
    5)
        shift
        for scene in room bicycle garden; do
            for tile in 16 20 32; do
                run_500step "$scene" "$tile"
            done
        done
        ;;
    7)
        for tile in 16 32; do
            run_30k "bicycle" "$tile"
            run_30k "garden" "$tile"
        done
        ;;
    all)
        prepare_datasets
        smoke_test
        ;;
    *)
        echo "Usage: bash run_a100.sh {0|1|2|3|4|5|7|all}"
        echo "  0: prepare datasets"
        echo "  1: smoke test"
        echo "  2: forward check (scene, tile)"
        echo "  3: backward check (scene, tile)"
        echo "  5: 500-step all (room, bicycle, garden ? 16,20,32)"
        echo "  7: 30K bicycle/garden ? 16,32"
        echo "  all: datasets + smoke"
        ;;
esac
