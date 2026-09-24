#!/bin/bash
# Quick check + Stage 2 (C44) run on mx
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_systems
export CUDA_CACHE_PATH=/tmp/cuda_cache_systems
export CUDA_VISIBLE_DEVICES=1

cd /tmp/gsplat_systems_revalidation
mkdir -p results/reference_v1/systems_revalidation

echo "=== Stage 1: Historical Definition Recovery ==="
python3 -c '
import json, sys, os

stage1 = {
    "C42": {
        "mechanism": "Downsample pred/target images via F.interpolate(scale_factor, mode=area) before SSIM loss",
        "source_files": ["scripts/phase-c42/track_a_scale_sweep.py"],
        "loss_definition": "loss = (1-0.2)*L1_full_res + 0.2*D_SSIM_downsampled",
        "semantics": "APPROXIMATE",
        "historical_benchmark": "scale=0.50: 3.54x SSIM speedup, +60.7% E2E, grad cos min=0.9854",
        "historical_environment": "A100-PCIE-40GB, gsplat 1.5.3, CUDA 11.8, PyTorch 2.7.1"
    },
    "C44": {
        "mechanism": "Separable SSIM using 2x 1D conv (1x11 + 11x1) with groups=15 on batch-stacked [5,3,H,W] input",
        "source_files": ["baseline/reference_v1/trainer.py (SepSSIM class)"],
        "loss_definition": "Mathematically identical to standard SSIM; 2x1D conv replaces 2D 11x11 conv",
        "semantics": "EXACT (within FP32 tolerance)",
        "historical_benchmark": "21.9x SSIM speedup, 70.2% E2E, loss diff 3.6e-4",
        "historical_environment": "A100-PCIE-40GB, gsplat 1.5.3, CUDA 11.8, PyTorch 2.7.1"
    },
    "AbsGrad-off": {
        "mechanism": "After densify_until_iter=15000, means2d.absgrad is computed but never consumed",
        "source_files": ["baseline/reference_v1/trainer.py:125,338"],
        "loss_definition": "EXACT: output gradients identical, absgrad only used for densification stats",
        "semantics": "EXACT",
        "historical_benchmark": "~5-15% raster backward saving, ~0.8 MB GPU memory",
        "historical_environment": "A100-PCIE-40GB, gsplat 1.5.3, CUDA 11.8, PyTorch 2.7.1"
    }
}
with open("results/reference_v1/systems_revalidation/stage1_definitions.json", "w") as f:
    json.dump(stage1, f, indent=2)
print("Stage 1 saved")
'

echo "=== Stage 2: C44 Validation ==="
python3 /tmp/gsplat_systems_revalidation/tmp_systems_validation.py --stage c44

echo "=== Check Results ==="
ls -la results/reference_v1/systems_revalidation/
cat results/reference_v1/systems_revalidation/stage1_definitions.json
