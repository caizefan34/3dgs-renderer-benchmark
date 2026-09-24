#!/bin/bash
# Freeze Reference V1 baseline: git commit + tag
cd /home/liaoyuanjun/3dgs-renderer-benchmark

# Configure git if needed
git config user.email "liaoyuanjun@reference-v1" 2>/dev/null || true
git config user.name "Reference V1 Freeze" 2>/dev/null || true

# Add all reference v1 files
git add baseline/reference_v1/
git add tests/reference_v1/
git add configs/reference_v1/
git add scripts/phase-c0/run_room_30k.sh
git add reports/reference-v1-baseline-lock.md

# Also add supporting files
git add src/benchmark_framework/scene.py 2>/dev/null || true
git add src/benchmark_framework/cameras.py 2>/dev/null || true
git add scripts/epic05/phase7/dataset.py 2>/dev/null || true
git add scripts/phase-c0/ 2>/dev/null || true
git add reference/graphdeco/ 2>/dev/null || true

# Commit
git commit -m "freeze: REFERENCE_V1_ABSGRAD baseline lock

Canonical Room 30K trajectory completed. All 11 Graphdeco semantic deviations corrected.
12/12 unit tests passing. PSNR=32.30, SSIM=0.9263, N=952353.

gsplat adaptations documented:
- absgrad=True (gradient flow)
- Pixel-space gradient scaling (width/2, height/2)
- grow_grad2d=0.0008 (AbsGS-style, not literal original 0.0002)
- 1D opacity [N] (gsplat internal squeeze)
- SfM init from COLMAP points3D.bin

Semantic label: REFERENCE_V1_ABSGRAD
Pinned source: graphdeco-inria/gaussian-splatting@54c035f"

# Create tag
git tag -a baseline/reference-v1-absgrad -m "Reference V1 AbsGrad baseline lock"

# Show commit SHA
echo "=== COMMIT SHA ==="
git rev-parse HEAD
echo "=== TAG ==="
git tag -l baseline/reference-v1-absgrad
echo "=== STATUS ==="
git status --short | head -5
echo "=== DIRTY CHECK ==="
if git diff --quiet && git diff --cached --quiet; then
    echo "git_dirty=false"
else
    echo "git_dirty=true"
fi
