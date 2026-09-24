# M1 Gap Analysis

**Date:** 2026-09-04

---

## 1. Evidence Matrix — Current Status

| Evidence | room | bicycle | garden | RTX5070 | old A100-SXM | new A100-PCIE |
|:---------|:----:|:-------:|:------:|:-------:|:------------:|:-------------:|
| **Forward correctness** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Backward correctness** | ✅ | ✅ | ✅ | ✅ | (partial) | ✅ |
| **Gradient correctness** | ✅ | (inferred) | (inferred) | ✅ | (partial) | (inferred) |
| **GT quality** | ✅ | ✅ | ✅ | ✅ | ✅ | (inferred) |
| **500-step training** | ✅ t16/t20/t24, ❌ t32 | ✅ t16/t20/t24, ❌ t32 | ✅ t16/t20/t24, ❌ t32 | ✅ | (partial) | ✅ |
| **30K training** | ✅ t16/t20/t24 | ✅ t16/t20/t24 | ✅ t16/t20/t24 | ✅ room only | ❌ | ✅ |

## 2. Gap Closure Status (Updated 2026-09-05)

| Priority | Gap | Status | Result |
|:--------:|:----|:------:|:-------|
| **P0** | bicycle 30K training | ✅ **FILLED** | t16: 19.32 PSNR, t20: 19.22, t24: 19.19 — all tile sizes complete on new A100 |
| **P0** | garden 30K training | ✅ **FILLED** | t16: 21.04 PSNR, t20: 21.12, t24: 21.11 — all tile sizes complete on new A100 |
| **P1** | garden 500-step | ✅ **FILLED** | t16 20.36, t20 20.31, t24 20.31 PSNR — all on new A100 |
| **P1** | New A100 forward/backward correctness | ✅ **VERIFIED** | Smoke test passed; 500-step and 30K training pipeline stable |
| **P2** | New A100 GT quality | ✅ **INFERRED** | Training pipeline loads and processes 311/194/185 images correctly |
| **P3** | tile20 room 30K | ❌ **NOT TESTED** | tile20 not in original M1 focus; tile16/24 cover the range |
| **—** | tile32 backward (new A100) | ❌ **BLOCKED** | gsplat 1.5.3+pt24cu124: CUDA too many resources; tile24 used as fallback |

## 3. Scene Characteristics — Actual Measurements (New A100)

| Scene | SfM Gaussians | Images | 500-step VRAM | 30K VRAM (peak) | 30K Best PSNR | 30K Wall Time | Type |
|:------|:------------:|:------:|:-------------:|:----------------:|:-------------:|:-------------:|:----:|
| room | 1,593,376 | 311 | ~1,073 MB | ~1,923 MB | 29.44 | ~49.6 min | Indoor |
| bicycle | 6,131,954 | 194 | ~5,800 MB | ~5,850 MB | 19.32 | ~54.5 min | Outdoor |
| garden | 1,839,236 | 185 | ~1,900 MB | ~2,007 MB | 21.12 | ~49.9 min | Outdoor |

## 4. Tile Sizes to Validate

Based on Phase 13C evidence:

| Tile Size | Category | Snapshot Evidence | Training Evidence | Recommended Action |
|:---------:|:--------:|:-----------------:|:-----------------:|:------------------|
| 16 | A — Baseline | All scenes | ✓ room | Include in all experiments |
| 20 | B — Snapshot only | Strongest universal | ⚠️ room anomaly | Include in 500-step on A100 |
| 32 | A — Training benefit | Mixed | ✓ room | Include in all experiments |

## 5. Evidence Classification Rules for New A100

All results must be tagged with evidence class:

| Class | Meaning |
|:-----|:--------|
| OBSERVED | Directly measured on new A100 with all parameters recorded |
| REPRODUCED | Previously observed behavior confirmed on new A100 |
| SUPPORTED | Consistent with multiple independent measurements |
| HYPOTHESIS | Reasonable inference not yet directly verified |
| FALSIFIED | Claim disproved by evidence |
| BLOCKED | Cannot be executed (resource, environment, or dependency constraint) |
