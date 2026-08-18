# EPIC-05 Optimization Protocol

## Overview

This protocol defines a rigorous methodology for identifying bottlenecks and
validating optimizations in the 3DGS rendering pipeline on EPIC-05 (8x A100 80GB).

## Pipeline Decomposition

The full rendering pipeline consists of:

1. **Projection** (project_gaussians): 3D 鈫?2D screen-space projection
2. **Tile Intersection** (isect_tiles): Assign Gaussians to tiles
3. **Sorting** (sort): Radix sort by tile/depth keys
4. **Rasterization Forward** (rasterize_to_pixels): Blend Gaussians per pixel
5. **Rasterization Backward** (rasterize_to_pixels_bwd): Gradient computation

## Optimization Modules

### M0: Baseline
gsplat default with tile_size=16, packed=True, SH degree 3.

### M1: Tile Size Tuning
- M1a: tile_size=8 (finer tiles, more tile work, less overdraw)
- M1b: tile_size=16 (default - baseline)
- M1c: tile_size=32 (coarser tiles, less scheduling, more overdraw)

### M2: Packed vs Dense
- packed=True: Skip invisible Gaussians per camera
- packed=False: Process all Gaussians (dense)

### M3: SH Degree
- degree=0: Only DC component (fastest, lowest quality)
- degree=1: 4 coefficients
- degree=3: 16 coefficients (full quality)

### M4: Radius Clipping
Controls near/far clipping of projected Gaussians.

### M5: Epsilon 2D
Adds blur to 2D covariance to prevent degeneracy.

### M6: Block Size
Number of threads per block in rasterization kernel.

## Ablation Matrix

Each module tested independently against baseline:

| ID | Module | Baseline | +Module |
|----|--------|----------|---------|
| M0 | baseline | 鉁?| - |
| M1 | tile_size | 鉁?| M1 effect |
| M2 | packed | 鉁?| M2 effect |
| M3 | SH degree | 鉁?| M3 effect |
| M4 | radius_clip | 鉁?| M4 effect |
| M5 | eps2d | 鉁?| M5 effect |

## Interaction Matrix

| ID | Combination | Rationale |
|----|------------|-----------|
| M1+M2 | tile_size + packed | Tile+packing interaction |
| M1+M3 | tile_size + SH | Tile+shader interaction |
| M2+M3 | packed + SH | Memory+computation interaction |
| M1+M2+M3 | All three | Full combined effect |

## Metrics

For each configuration:
- Mean/median/P99 latency (ms)
- FPS
- Peak VRAM (MB)
- GPU utilization (via nvidia-smi)
- Kernel breakdown
- PSNR (quality)

## Statistical Protocol

- Warmup: 30 frames
- Measurement: 100 frames per repeat, 3 repeats
- Reporting: mean, std, median, CI
- Quality: fixed PLY, fixed camera path

## Phase 3: Official Dataset Validation

The synthetic workload results are extended to official real-scene validation
using Mip-NeRF 360 pretrained checkpoints. See:

- [Validation matrix](../configs/epic05/official_validation_matrix.json)
- [Run script](../scripts/epic05/run_official_validation.py)
- [Aggregation](../scripts/epic05/aggregate_official_results.py)
- [Quality evaluation](../scripts/epic05/evaluate_official_quality.py)
- [Synthetic vs official analysis](../scripts/epic05/analyze_synthetic_vs_official.py)
- [Charts](../scripts/epic05/generate_official_charts.py)
- [Report](../reports/epic05/official-dataset-validation-2026-07-19.md)

### Validation Protocol Additions

| Parameter | Synthetic | Official |
|-----------|-----------|----------|
| Scene source | Generated PLY | Official checkpoint |
| Gaussians | Controlled (50K/200K/400K) | Natural population |
| Camera | Generated orbit | Evaluation split |
| Resolution | All profiles | 1080p canonical |
| Quality metrics | N/A | PSNR, SSIM, LPIPS |
| Pixel equivalence | N/A | tile16 vs tile32 |
| Repeats | 3 | 5 |
| Steady-state check | N/A | CV < 0.15
