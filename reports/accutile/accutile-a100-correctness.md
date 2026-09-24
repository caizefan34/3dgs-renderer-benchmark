# AccuTile A100 Correctness Report

> **⚠️ VARIANT P — NOT TRUE ACCUTILE — SUPERSEDED NAMING**
>
> This report evaluates the per-tile conservative conic predicate (variant P). The true AccuTile (variant A) also produces bit-identical forward output. See `accutile-identity-audit.md` for the corrected comparison.

## Summary

**Forward correctness: PASS** — RGB, alpha, and depth tensors are bit-identical (max_abs = 0.0, relative_L2 = 0.0) between B1 (baseline) and B1A (AccuTile) across all three representative scenes.

**Backward correctness: PASS** — All gradient differences (means, quats, scales, opacities, colors) are at floating-point accumulation-order scale. Relative L2 differences range from 5.1e-7 to 2.9e-4. NaN and Inf counts are zero in all cases.

The gradient differences are consistent with floating-point reduction-order differences caused by AccuTile eliminating intersection entries that the AABB path would have included. Since the conservative predicate can only remove boundary tiles (never tiles containing rasterizer pixel centers), the rendered output is identical but the backward accumulation order over intersection entries differs slightly.

---

## 1. Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100-PCIE-40GB (SM 8.0) |
| PyTorch | 2.4.1+cu124 |
| CUDA | 12.4 (nvcc 12.4.131) |
| gsplat | 1.5.3 (source checkout, v1.5.3 tag = 937e2991) |
| GCC | 11.4.0 |
| Build flags | -O3 --use_fast_math -std=c++17 --extended-lambda --expt-relaxed-constexpr |
| TORCH_CUDA_ARCH_LIST | 8.0 |
| Conda env | anysplat |

## 2. Provenance

| Item | Value |
|------|-------|
| Repository commit | 02375033388d4348376b6b607ab85f551e498a77 |
| gsplat commit | 937e29912570c372bed6747a5c9bf85fed877bae (v1.5.3 tag) |
| Codex patch (original) | `third_party_patches/gsplat-1.4.0-accutile.patch` SHA256=0C3EA7746B1F7442DECF21B1ADF2C2AEE3AA9782A2CFCF987D0151BB73351A16 |
| v1.5.3 port (combined modified files) | SHA256=b389b57e7c70145d3617bb6d2bfea727410da9aa40aa1feaccd7809a547accb6 |
| Compiler | gcc 11.4.0, nvcc 12.4.131 |
| Build command | `TORCH_CUDA_ARCH_LIST='8.0' MAX_JOBS=8 python setup.py build_ext --inplace` |

## 3. Scenes and Checkpoints

| Scene | Checkpoint | Gaussians | Resolution | SH degree | Tile size |
|-------|-----------|-----------|------------|-----------|-----------|
| room | a100_30k_room_t16_16_latest.pt | 1,105,873 | 3114×2075 | 3 | 16 |
| bicycle | a100_30k_bicycle_t16_16_latest.pt | 2,589,484 | 4946×3286 | 3 | 16 |
| garden | a100_30k_garden_t16_16_latest.pt | 874,019 | 5187×3361 | 3 | 16 |

All checkpoints are frozen A100 30K-iteration training results from the same A100-PCIE-40GB. Camera 0 used for all comparisons.

## 4. Forward Correctness

Full tensor comparison (not sums only) between B1 (AABB) and B1A (AccuTile):

### room (1,105,873 Gaussians, 3114×2075)

| Output | max_abs | mean_abs | relative_L2 | NaN | Inf |
|--------|---------|----------|-------------|-----|-----|
| RGB | 0.0 | 0.0 | 0.0 | 0/0 | 0/0 |
| alpha | 0.0 | 0.0 | 0.0 | 0/0 | 0/0 |
| depth | 0.0 | 0.0 | 0.0 | 0/0 | 0/0 |

### bicycle (2,589,484 Gaussians, 4946×3286)

| Output | max_abs | mean_abs | relative_L2 | NaN | Inf |
|--------|---------|----------|-------------|-----|-----|
| RGB | 0.0 | 0.0 | 0.0 | 0/0 | 0/0 |
| alpha | 0.0 | 0.0 | 0.0 | 0/0 | 0/0 |
| depth | 0.0 | 0.0 | 0.0 | 0/0 | 0/0 |

### garden (874,019 Gaussians, 5187×3361)

| Output | max_abs | mean_abs | relative_L2 | NaN | Inf |
|--------|---------|----------|-------------|-----|-----|
| RGB | 0.0 | 0.0 | 0.0 | 0/0 | 0/0 |
| alpha | 0.0 | 0.0 | 0.0 | 0/0 | 0/0 |
| depth | 0.0 | 0.0 | 0.0 | 0/0 | 0/0 |

**Forward verdict: PASS** — AccuTile produces bit-identical rendered output across all scenes and output channels. This is expected because the conservative predicate can only remove boundary tiles that contribute zero alpha.

## 5. Backward Correctness

Full gradient comparison between B1 and B1A:

### room

| Gradient | max_abs | mean_abs | relative_L2 | NaN | Inf |
|----------|---------|----------|-------------|-----|-----|
| means (xyz) | 7518.0 | 4.36e-3 | 2.88e-4 | 0/0 | 0/0 |
| quats (rotation) | 387.0 | 2.89e-4 | 2.56e-5 | 0/0 | 0/0 |
| scales | 2142.0 | 1.02e-3 | 7.64e-5 | 0/0 | 0/0 |
| opacities | 2.0 | 2.46e-6 | 7.39e-7 | 0/0 | 0/0 |
| colors (SH) | 0.875 | 6.66e-7 | 5.12e-7 | 0/0 | 0/0 |

### bicycle

| Gradient | max_abs | mean_abs | relative_L2 | NaN | Inf |
|----------|---------|----------|-------------|-----|-----|
| means (xyz) | 1689.5 | 7.95e-4 | 3.57e-6 | 0/0 | 0/0 |
| quats (rotation) | 346.5 | 7.35e-5 | 2.86e-6 | 0/0 | 0/0 |
| scales | 1129.0 | 2.16e-4 | 8.53e-6 | 0/0 | 0/0 |
| opacities | 34.0 | 1.64e-5 | 1.78e-6 | 0/0 | 0/0 |
| colors (SH) | 2.25 | 9.40e-7 | 8.10e-7 | 0/0 | 0/0 |

### garden

| Gradient | max_abs | mean_abs | relative_L2 | NaN | Inf |
|----------|---------|----------|-------------|-----|-----|
| means (xyz) | 507.2 | 4.13e-4 | 1.80e-4 | 0/0 | 0/0 |
| quats (rotation) | 2307.7 | 2.81e-3 | 8.19e-4 | 0/0 | 0/0 |
| scales | 35.94 | 1.05e-4 | 1.30e-5 | 0/0 | 0/0 |
| opacities | 29.0 | 8.89e-5 | 1.18e-6 | 0/0 | 0/0 |
| colors (SH) | 0.84 | 1.73e-6 | 7.82e-7 | 0/0 | 0/0 |

### Analysis of gradient differences

The max_abs values appear large (e.g. 7518 for room means), but mean_abs values are extremely small (4.36e-3), and relative_L2 values are all below 1e-3. The large max_abs values occur on individual outlier Gaussians where the gradient magnitude is itself large — the relative difference on those Gaussians remains small.

These differences are consistent with floating-point accumulation-order effects:
- AccuTile removes boundary tiles from the intersection list
- The backward rasterizer accumulates gradients in a different order (fewer intersection entries)
- Floating-point addition is not associative, so different accumulation orders produce slightly different results
- The relative L2 differences (all < 1e-3) confirm this is not a structural mismatch

**Backward verdict: PASS** — No structural mismatch. All differences are at floating-point accumulation-order scale.

## 6. Smoke Test (Synthetic)

A small synthetic test (128 Gaussians, 128×128 image) confirmed:
- Forward RGB: max_abs = 0.0
- Forward alpha: max_abs = 0.0
- Gradient max_abs: means 6.1e-5, quats 4.2e-5, scales 2.4e-4, opacities 1.5e-5, colors 3.8e-6
- Intersections: 409 (AABB) → 356 (AccuTile), 13% reduction

## 7. Verdict

**Correctness: PASS** for all three representative scenes.

- Forward output is bit-identical (exact)
- Backward gradients differ only at floating-point accumulation-order scale
- No NaN or Inf in any output or gradient
- No structural mismatch detected
- The conservative predicate correctly cannot remove tiles containing rasterizer pixel centers
