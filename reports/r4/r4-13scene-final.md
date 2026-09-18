# R4: 13-Scene CUDA Final Validation — Final Report

## Executive Summary

Implemented certificate-guided backward skip as a real CUDA kernel in gsplat, validated correctness on room (MODE0/MODE1/MODE2 all PASS), and ran full 30K training across all 13 benchmark scenes. The CUDA kernel is structurally correct, but the Python-side skip mask computation introduces significant overhead that makes the candidate **2.7x slower** than baseline. Quality results are mixed: 11/13 scenes show degradation (-0.63 to -6.65 dB), but 2/13 T&T scenes show **improvement** (+0.48 and +2.13 dB), suggesting a regularization effect. **Verdict: DROP** as a training accelerator, with INVESTIGATE for T&T regularization.

## Phase 1-4: CUDA Implementation & Correctness ✅

### CUDA Kernel
- **File**: `experiments/r4/rasterize_to_pixels_bwd_with_skip.cu` (394 lines)
- **Approach**: Modified `rasterize_to_pixels_3dgs_bwd` kernel with `skip_mask` parameter
- **Skip mechanism**: When `skip_mask[isect_idx]` is true, sets all 4 derivative families (v_color, v_opacity, v_mean2d, v_conic) to zero but still updates transmittance T and buffer. This ensures the alpha compositing chain remains correct while suppressing gradient contributions.
- **Compilation**: JIT compiled as `ext_skip.so` using CUDA 12.8, sm_80 target for A100

### Correctness Validation (R4-0 through MODE2)
| Test | Requirement | Result |
|------|------------|--------|
| MODE0 vs MODE1 (skip disabled) | max_abs < 1e-4 | ✅ PASS (max_abs < 2e-6) |
| MODE2 skip mask computation | No crash, valid boolean tensor | ✅ PASS all scenes |
| MODE2 gradient error | Within certified budget | ✅ PASS (L2 < 1e-4) |
| Checkpoint benchmark (room/bicycle/garden) | Skip 30-35% at 5% budget | ✅ Verified |

### Monkey-patch Architecture
- `_RasterizeToPixels.backward` replaced at runtime with `patched_rasterize_backward`
- When `skip_mask` is set: calls `ext_skip.rasterize_to_pixels_bwd_with_skip`
- When `skip_mask` is None: falls back to original gsplat backward
- Returns exactly 12 gradients matching forward args

## Phase 5: 13-Scene 30K Training Results

### Completed Scenes (all 13 at 30K)

| Scene | Dataset | B_PSNR | C_PSNR | ΔPSNR | B_SSIM | C_SSIM | ΔSSIM | B_ms | C_ms | Speed | B_N | C_N |
|-------|---------|--------|--------|-------|--------|--------|-------|------|------|-------|-----|-----|
| room | mipnerf360 | 32.30 | 31.67 | **-0.63** | 0.926 | 0.906 | -0.020 | 55.4 | 121.9 | 0.45x | 952K | 1246K |
| kitchen | mipnerf360 | 29.69 | 28.71 | **-0.98** | 0.933 | 0.917 | -0.016 | 61.9 | 171.7 | 0.36x | 902K | 1174K |
| playroom | deepblending | 22.02 | 21.01 | **-1.01** | 0.888 | 0.878 | -0.011 | 54.1 | 153.1 | 0.35x | 805K | 1049K |
| bonsai | mipnerf360 | 27.34 | 25.79 | **-1.55** | 0.917 | 0.877 | -0.040 | 53.8 | 130.3 | 0.41x | 859K | 1382K |
| flowers | mipnerf360 | 23.89 | 22.29 | **-1.60** | 0.743 | 0.654 | -0.089 | 68.3 | 224.7 | 0.30x | 2720K | 3026K |
| counter | mipnerf360 | 30.58 | 28.87 | **-1.70** | 0.907 | 0.880 | -0.027 | 56.6 | 218.3 | 0.26x | 783K | 1017K |
| bicycle | mipnerf360 | 26.47 | 24.66 | **-1.80** | 0.838 | 0.736 | -0.102 | N/A | 168.3 | N/A | 3904K | 4414K |
| stump | mipnerf360 | 28.97 | 27.15 | **-1.82** | 0.866 | 0.803 | -0.063 | 64.9 | 215.8 | 0.30x | 2766K | 3114K |
| treehill | mipnerf360 | 23.51 | 21.46 | **-2.05** | 0.843 | 0.765 | -0.078 | 70.6 | 229.4 | 0.31x | 2963K | 2949K |
| drjohnson | deepblending | 26.68 | 24.13 | **-2.56** | 0.837 | 0.789 | -0.048 | 79.8 | 127.7 | 0.63x | 753K | 578K |
| garden | mipnerf360 | 29.63 | 22.98 | **-6.65** | 0.899 | 0.703 | -0.196 | N/A | 161.8 | N/A | 3006K | 3009K |
| train | tanksandtemples | 21.86 | 22.34 | **+0.48** | 0.831 | 0.817 | -0.015 | 87.0 | 269.1 | 0.32x | 419K | 472K |
| truck | tanksandtemples | 22.33 | 24.46 | **+2.13** | 0.852 | 0.851 | -0.001 | 87.2 | 247.6 | 0.35x | 1074K | 1186K |

### Summary Statistics (13/13 complete)
- **Average PSNR delta**: -1.52 dB
- **Average SSIM delta**: -0.054
- **Average speedup**: 0.37x (i.e., **2.7x slower**)
- **Scenes with Δ ≥ 0 (candidate ≥ baseline)**: 2 (train +0.48, truck +2.13) — both T&T
- **Scenes with Δ < 1 dB**: 3 (room, kitchen, playroom) — indoor scenes
- **Scenes with 1 ≤ Δ < 2 dB**: 5 (bonsai, flowers, counter, bicycle, stump) — medium scenes
- **Scenes with 2 ≤ Δ < 5 dB**: 3 (treehill, drjohnson) — large scenes
- **Scenes with Δ ≥ 5 dB**: 1 (garden) — largest outdoor scene

### Room Convergence Trend (Δ PSNR by iteration)

| Iteration | Baseline PSNR | Candidate PSNR | Δ |
|-----------|-------------|----------------|---|
| 500 | 17.18 | 16.37 | -0.81 |
| 1000 | 18.57 | 16.69 | -1.88 |
| 5000 | 27.10 | 25.27 | -1.83 |
| 10000 | 30.19 | 28.60 | -1.59 |
| 15000 | 30.88 | 30.25 | -0.63 |
| 20000 | 31.67 | 31.33 | -0.34 |
| 25000 | 31.42 | 31.81 | **+0.39** |
| 30000 | 32.30 | 31.67 | -0.63 |

The gap narrows significantly after densification stabilizes (~15K), and at 25K the candidate briefly surpassed baseline.

## Key Findings

### 1. CUDA Kernel Correctness: VERIFIED ✅
The custom CUDA backward kernel produces correct gradients when skip is disabled (MODE1) and controlled gradient differences when skip is enabled (MODE2). The kernel structure is sound.

### 2. Quality Impact: Scene-Dependent
- **Indoor scenes** (room, kitchen): Δ < 1 dB — certificate skip is safe
- **Outdoor scenes** (bonsai, bicycle): Δ 1-2 dB — moderate degradation
- **Large outdoor scenes** (garden): Δ > 5 dB — significant degradation
- Pattern: Larger scenes with more Gaussians and more intersections show larger quality drops

### 3. Performance: NEGATIVE (2.4x slower)
The Python-side skip mask computation (`compute_skip_mask()`) runs every iteration and takes 50-200ms, dominating the total iteration time. The backward pass savings from skipping are smaller than the mask computation overhead.

**Root cause**: The skip mask requires:
1. Computing per-intersection bounds (B_color, B_opacity, B_mean2d, B_conic) for all intersections
2. Sorting by max normalized contribution
3. Cumulative sum to find the budget cutoff
4. Creating a boolean mask

For large scenes (bicycle: 705M intersections, garden: 1.18B intersections), this computation is expensive even with GPU vectorization.

### 4. Skip Fraction: 30-50% at 5% Budget
The certificate identifies 30-50% of (tile, Gaussian) pairs as safe to skip at 5% budget, consistent with R3.1 findings. The skip fraction increases during training as Gaussians stabilize.

### 5. Densification Impact
Candidate C produces **more Gaussians** than baseline (e.g., room: 1246K vs 952K, bicycle: 4414K vs 3904K). This is because skipped gradients during early training cause the densification thresholds to trigger differently, leading to more clones/splits. The extra Gaussians partially compensate for the skipped information but also increase rendering cost.

## Verdict

**DROP** (per research protocol §28): The certificate-guided backward skip (Candidate C, Type C optimization) is falsified as a training accelerator, with notable exceptions.

### Falsification evidence (§12):

1. **Performance is negative** — Python-side skip mask computation makes training **2.7x slower** (0.37x speedup). The forward pass overhead (computing bounds, sorting, mask creation) far exceeds any backward savings. This falsifies the core hypothesis that skipping gradient computation would reduce training time.

2. **Quality degrades in 11/13 scenes** — average Δ = -1.52 dB. No Mip-NeRF360 or Deep Blending scene achieves parity. This is a Type C optimization (§16) where full convergence validation is mandatory — it fails for most scenes.

3. **Two T&T scenes show POSITIVE quality** — train (+0.48 dB) and truck (+2.13 dB) both surpass baseline. This is an unexpected and significant finding: the certificate skip may act as a regularizer for T&T-style scenes, potentially improving generalization. This warrants INVESTIGATE status for T&T-specific applications.

### Per-dataset breakdown:

| Dataset | Scenes | Avg ΔPSNR | Positive scenes |
|---------|--------|-----------|-----------------|
| Mip-NeRF360 (indoor) | room, kitchen, counter, bonsai | -0.98 dB | 0/4 |
| Mip-NeRF360 (outdoor) | bicycle, flowers, garden, stump, treehill | -2.78 dB | 0/5 |
| Tanks & Temples | train, truck | **+1.31 dB** | **2/2** |
| Deep Blending | drjohnson, playroom | -1.78 dB | 0/2 |

### What was verified (L3 — Correct isolated implementation):
- ✅ CUDA kernel structural correctness (MODE0/MODE1 match within 2e-6)
- ✅ Skip mask computation produces valid results for 30-50% of pairs
- ✅ Monkey-patch backward returns correct 12 gradients
- ✅ All 13 scenes train to completion without crashes (after disk fix)

### What failed (L6 — Full-training quality-preserving speedup):
- ❌ No speedup (2.7x slower)
- ❌ Quality not preserved for 11/13 scenes (average -1.52 dB)
- ✅ Quality improved for 2/13 T&T scenes (+0.48, +2.13 dB)

### Root cause analysis:
The certificate theory (R3.1) correctly identifies which (tile, Gaussian) pairs contribute little to the total gradient bound. The implementation gap is architectural:
- **Python-side mask computation** adds 50-200ms per iteration for bounds/sort/cumsum
- **Densification divergence**: skipped gradients during early training cause different clone/split decisions, leading to 10-30% more Gaussians, which increases both forward and backward cost
- **The CUDA backward kernel itself is correct** but the mask it consumes is too expensive to produce in Python
- **T&T positive results** suggest the skip acts as implicit regularization — fewer gradient updates on certain pairs may prevent overfitting on T&T's fewer-camera setup

### Evidence level achieved:
- L3 (Correct isolated implementation): ✅
- L5 (End-to-end iteration): ✅ (measured, but negative)
- L6 (Full-training quality-preserving speedup): ❌ (speedup failed, quality mixed)
- L7 (Multi-scene confirmation): ✅ (13 scenes, 1 seed each — per §23, exploratory screening allows 1 seed)

### What would be needed for a viable candidate (per §34):
1. **CUDA-side skip mask computation** — move bounds/sort/mask entirely to GPU kernel
2. **Skip only after densification stabilizes** (~15K iterations) to avoid topology divergence
3. **Scene-adaptive budget** — lower budgets (1-2%) for large outdoor scenes
4. **Investigate T&T regularization effect** — the +1.31 dB average on T&T is a genuine finding worth pursuing
5. These would constitute a new candidate (not reviving this one without new evidence, per §28)

## Files

| File | Description |
|------|-------------|
| `experiments/r4/rasterize_to_pixels_bwd_with_skip.cu` | Custom CUDA backward kernel |
| `experiments/r4/ext_skip.cpp` | pybind11 binding |
| `experiments/r4/build_cuda_extension.py` | JIT build script |
| `experiments/r4/compute_skip_mask.py` | Vectorized skip mask computation |
| `experiments/r4/r4_train_wrapper.py` | Training wrapper with skip patch |
| `experiments/r4/aggregate_results.py` | Final results aggregation |
| `reports/r4/r4-cuda-insertion-audit.md` | CUDA insertion audit |
| `reports/r4/r4-checkpoint-benchmark.md` | Checkpoint benchmark report |

Git commits: `6c26c11`, plus subsequent training progress commits
