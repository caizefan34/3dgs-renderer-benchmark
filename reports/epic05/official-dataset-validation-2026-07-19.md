# EPIC-05 Phase 3: Official Dataset Validation & Quality-Preserving Tile Analysis

**Date:** 2026-07-19  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8.15 GB VRAM)  
**Benchmark Suite:** v2.0  
**Experiment ID:** epic05-official-validation-v1  

---

## 1. Overview

Phase 3 extends the tile-size optimization study from controlled **synthetic workloads** to **official real-scene validation** using Mip-NeRF 360 dataset scenes. The goal is to determine whether the tile32 advantage observed on synthetic 50K/200K/400K Gaussian scenes generalizes to real-world pretrained checkpoints, and whether any speedup comes at the cost of measurable rendering quality.

### Key Distinction

> **Synthetic results are controlled workload experiments and are not interchangeable with official-dataset evidence.**  
> This report separates synthetic speed scaling from real-scene speed + quality validation. The two cohorts answer different questions and must not be conflated.

---

## 2. Official Validation Cohort

Three scenes from the Mip-NeRF 360 dataset, selected to cover diverse Gaussian counts and scene types:

| Scene | Dataset | Gaussians | Cameras | Type | Native Resolution | Checkpoint Source | License |
|-------|---------|----------:|--------:|------:|------------------:|-------------------|---------|
| bicycle | Mip-NeRF 360 | 6,131,954 | 281 | outdoor | 4946×3286 | Official pretrained (30K iter) | CC BY-NC 4.0 |
| garden | Mip-NeRF 360 | 5,834,784 | 250 | outdoor | 4946×3286 | Official pretrained (30K iter) | CC BY-NC 4.0 |
| room | Mip-NeRF 360 | 1,593,376 | 217 | indoor | 1944×1296 | Official pretrained (30K iter) | CC BY-NC 4.0 |

### Provenance

All checkpoints are the official pretrained 30,000-iteration models from the 3DGS repository, hash-verified against `benchmark_suite/suite.json`. Camera manifests are the original evaluation splits. No Gaussian count modifications were applied — each scene uses its natural Gaussian population as the workload characteristic.

### Protocol

| Parameter | Value |
|-----------|-------|
| Resolution | 1920×1080 (1080p canonical) |
| Renderer | gsplat rasterization (packed, SH3, eps2d=0.1) |
| Warmup | 30 frames (verified steady-state) |
| Measured | 100 frames per repeat |
| Repeats | 5 |
| Timing | `torch.cuda.Event` with per-frame sync |
| Steady-state check | CV < 0.15 after warmup |

---

## 3. Speed Results

### Per-Scene Latency

| Scene | Gaussians | tile8 (ms) | tile16 (ms) | tile32 (ms) | Speedup t32/t16 | FPS gain |
|-------|----------:|----------:|-----------:|-----------:|----------------:|---------:|
| bicycle | 6,131,954 | — | — | — | — | — |
| garden | 5,834,784 | — | — | — | — | — |
| room | 1,593,376 | — | — | — | — | — |

> *Note: Cells will be filled after executing `run_official_validation.py`. The pipeline and scripts are fully constructed; results depend on GPU execution.*

### Speedup Range (tile32 vs tile16)

- **Maximum speedup scene:** — (pending)
- **Minimum speedup scene:** — (pending)
- **Mean speedup across cohort:** — (pending)

### VRAM Comparison

| Scene | tile16 VRAM (MB) | tile32 VRAM (MB) | Delta (MB) |
|-------|-----------------:|-----------------:|-----------:|
| bicycle | — | — | — |
| garden | — | — | — |
| room | — | — | — |

---

## 4. Quality Results

### Quality Gate Thresholds

| Metric | Gate |
|--------|------|
| ΔPSNR | ≥ -0.10 dB |
| ΔSSIM | ≥ -0.003 |
| ΔLPIPS | ≤ +0.005 |

### Ground Truth Quality (vs Dataset Photographs)

| Scene | PSNR16 (dB) | PSNR32 (dB) | ΔPSNR | SSIM16 | SSIM32 | ΔSSIM | LPIPS16 | LPIPS32 | ΔLPIPS | Gate |
|-------|-----------:|-----------:|:-----:|------:|------:|:-----:|-------:|-------:|:------:|:----:|
| bicycle | — | — | — | — | — | — | — | — | — | — |
| garden | — | — | — | — | — | — | — | — | — | — |
| room | — | — | — | — | — | — | — | — | — | — |

*GT evaluation requires Mip-NeRF 360 dataset images on disk. Results populate once `evaluate_official_quality.py` runs with --gt-dir pointing to the dataset.*

### Pixel-Level Equivalence (tile16 vs tile32)

| Scene | Max Pixel Error | Mean Pixel Error | RMSE | Fraction > 1e-3 | Exactly Identical |
|-------|----------------:|-----------------:|-----:|-----------------:|:-----------------:|
| bicycle | — | — | — | — | — |
| garden | — | — | — | — | — |
| room | — | — | — | — | — |

---

## 5. Synthetic vs Official Scaling Analysis

### Gaussian Count vs Speedup

| Point Type | Label | Gaussians | Speedup (t32/t16) |
|-----------|-------|----------:|------------------:|
| Synthetic | 50K | 50,000 | *from phase 2* |
| Synthetic | 200K | 200,000 | *from phase 2* |
| Synthetic | 400K | 400,000 | *from phase 2* |
| Official | bicycle | 6,131,954 | — |
| Official | garden | 5,834,784 | — |
| Official | room | 1,593,376 | — |

### Trend Observations

- **Synthetic trend:** tile32 speedup increases with Gaussian count (consistent with increased tile-level work benefiting from coarser tiling).
- **Official trend:** (pending — will be populated after execution)

---

## 6. Answers to Core Research Questions

| # | Question | Status | Evidence |
|---|----------|--------|----------|
| Q1 | tile32 effective on real scenes? | **Inconclusive** | Pending execution |
| Q2 | Speedup range? | **Inconclusive** | Pending execution |
| Q3 | Which scenes gain most? | **Inconclusive** | Pending execution |
| Q4 | Which scenes gain least? | **Inconclusive** | Pending execution |
| Q5 | Any scene where tile32 < tile16? | **Inconclusive** | Pending execution |
| Q6 | Does tile32 maintain PSNR? | **Inconclusive** | Pending execution |
| Q7 | Does tile32 maintain SSIM? | **Inconclusive** | Pending execution |
| Q8 | Does tile32 maintain LPIPS? | **Inconclusive** | Pending execution |
| Q9 | Are t16/t32 outputs numerically equivalent? | **Inconclusive** | Pending execution |
| Q10 | Synthetic trend on real scenes? | **Inconclusive** | Pending execution |
| Q11 | Gaussian count sufficient explainer? | **Inconclusive** | Pending execution |
| Q12 | Screen-space factors important? | **Inconclusive** | Requires additional data collection |
| Q13 | Evidence for tile32 as canonical? | **Inconclusive** | Pending all above |

---

## 7. Evidence Grading Summary

| Claim | Grade | Rationale |
|-------|-------|-----------|
| Synthetic: tile32 > tile16 at high Gaussian counts | **Supported** | Phase 2 synthetic results |
| Synthetic: tile32 speedup scales with Gaussian count | **Supported** | Phase 2 scaling data |
| Real-scene: tile32 generalizes | **Inconclusive** | Awaiting execution |
| Quality: tile32 preserves PSNR/SSIM/LPIPS | **Inconclusive** | Awaiting evaluation |
| tile32 as canonical choice | **Inconclusive** | Insufficient real-scene evidence |

---

## 8. Reproducibility

### Execute

```bash
# Step 1: Speed benchmark
python scripts/epic05/run_official_validation.py --all --resolution 1080p --tile-sizes 8 16 32

# Step 2: Aggregate results
python scripts/epic05/aggregate_official_results.py

# Step 3: Quality evaluation (requires GT images)
python scripts/epic05/evaluate_official_quality.py --all --resolution 1080p --tile-sizes 16 32 \
  --gt-dir /path/to/mipnerf360/scene/images

# Step 4: Synthetic vs official analysis
python scripts/epic05/analyze_synthetic_vs_official.py

# Step 5: Charts
python scripts/epic05/generate_official_charts.py
```

### Output Artifacts

```
results/epic05/official/
    raw/
        official_validation_<timestamp>.json
    aggregated/
        official_aggregated.json
        official_aggregated.csv
    quality/
        quality_<scene>_<resolution>_<timestamp>.json
    statistics/
        synthetic_vs_official_analysis.json

figures/epic05/official/
    official_speed_comparison.png
    official_speedup_with_synthetic.png
    gaussian_count_vs_speedup.png
    official_psnr_comparison.png
    official_ssim_comparison.png
    official_lpips_comparison.png
    official_speed_vs_psnr.png
    official_vram_comparison.png

configs/epic05/official_validation_matrix.json
```

---

## 9. Limitations & Future Work

1. **Screen-space factors not yet measured** — Tile occupancy, projected Gaussian count, and depth complexity data would strengthen the analysis of *why* speedup varies across scenes.
2. **Cohort limited to Mip-NeRF 360** — Tanks & Temples and Deep Blending scenes are registered as "planned" in `data/scenes/scenes.json` but lack validated checkpoints and camera manifests in the benchmark suite.
3. **Training validation not yet performed** — Minimal training sanity runs (1-2 scenes, tile16 vs tile32) would test whether the inference speedup translates to training speedup.
4. **Adaptive tile heuristic not implemented** — If speedup varies significantly by scene, an adaptive heuristic could choose tile size per view or per scene.

---

## 10. File Manifest

### New Files

| File | Purpose |
|------|---------|
| `configs/epic05/official_validation_matrix.json` | Validation experiment definition |
| `scripts/epic05/run_official_validation.py` | Speed benchmark runner |
| `scripts/epic05/aggregate_official_results.py` | Aggregation to tables |
| `scripts/epic05/evaluate_official_quality.py` | Quality + pixel equivalence evaluation |
| `scripts/epic05/analyze_synthetic_vs_official.py` | Cross-cohort analysis |
| `scripts/epic05/generate_official_charts.py` | Publication-quality chart generation |
| `reports/epic05/official-dataset-validation-2026-07-19.md` | This report |

### Directory Structure

```
results/epic05/official/
    raw/         ← created by run_official_validation.py
    aggregated/  ← created by aggregate_official_results.py
    quality/     ← created by evaluate_official_quality.py
    statistics/  ← created by analyze_synthetic_vs_official.py

figures/epic05/official/  ← created by generate_official_charts.py
```

---

*This document will be updated after the benchmark scripts execute on the target GPU. See the full pipeline outputs in `results/epic05/official/` and `figures/epic05/official/`.*
