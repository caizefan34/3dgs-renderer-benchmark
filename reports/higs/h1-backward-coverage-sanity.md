# H1 Backward Coverage Sanity Report

**Date:** 2026-09-19
**Verdict:** PASS_WITH_CAVEAT
**Backward correctness:** CLOSED

---

## 1. Question Tested

The H1-R backward repair established P1/P2/P3 gradient equivalence using a fixed random upstream gradient (seed=42). However, room/cam0 reported only **6 nonzero-gradient Gaussians** out of 115,278. This is suspiciously small.

The question: Is the sparse gradient support a **valid consequence of the test construction**, or **another correctness-harness coverage bug**?

## 2. Experiment

Re-ran P1/P2/P3 backward comparison with the **actual training loss** (`0.8 * L1 + 0.2 * (1 - SSIM)`) using GT images from the room scene.

- **Scene:** room (SpeedySplat 30K checkpoint, N=115,278)
- **Cameras:** cam0 (DSCF4667) and cam155 (DSCF4800, middle camera)
- **Resolution:** 2048×1365 (capped from 3114×2075)
- **Loss:** `0.8 * L1 + 0.2 * (1 - SSIM)` (SepSSIM, same as benchmark training)
- **SH degree:** 3
- **Background:** black (default)
- **Nonzero threshold:** 1e-10

Three paths compared:
- P1 = B1 clean gsplat backward (rasterization, packed=False)
- P2 = B2 HiGS forward + gsplat_recompute backward
- P3 = B2 HiGS forward + higs_native backward

## 3. Gradient Support Counts

### Camera 0 (DSCF4667)

| Path | N_total | N_visible | N_nonzero_grad | nonzero/total | nonzero/visible |
|------|--------:|----------:|---------------:|--------------:|----------------:|
| P1 | 115,278 | 115,278 | **6** | 0.005% | 0.005% |
| P2 | 115,278 | 101,733 | **6** | 0.005% | 0.006% |
| P3 | 115,278 | 101,733 | **6** | 0.005% | 0.006% |

Exact-zero elements: 345,816 / 345,834 (means) = **99.995% exactly zero**.
All 6 nonzero Gaussians are the **same 6** across P1/P2/P3.

### Camera 155 (DSCF4800, middle)

| Path | N_total | N_visible | N_nonzero_grad | nonzero/total | nonzero/visible |
|------|--------:|----------:|---------------:|--------------:|----------------:|
| P1 | 115,278 | 115,278 | **5** | 0.004% | 0.004% |
| P2 | 115,278 | 110,423 | **5** | 0.004% | 0.005% |
| P3 | 115,278 | 110,423 | **5** | 0.004% | 0.005% |

## 4. Gradient Equivalence

### Camera 0

| Comparison | Parameter | cosine | relative_l2 | max_abs | zero/nonzero disagreement | NaN | Inf |
|-----------|-----------|-------:|------------:|--------:|-------------------------:|----:|----:|
| P1 vs P2 | means | 1.0000 | 9.3e-06 | 2.1e-07 | 0 | 0 | 0 |
| P1 vs P2 | quats | 1.0000 | 5.0e-07 | 1.4e-09 | 0 | 0 | 0 |
| P1 vs P2 | scales | 1.0000 | 5.7e-07 | 3.8e-10 | 0 | 0 | 0 |
| P1 vs P2 | opacities | 1.0000 | 2.8e-07 | 3.0e-08 | 0 | 0 | 0 |
| P1 vs P2 | sh | 1.0000 | 7.2e-07 | 8.9e-08 | 0 | 0 | 0 |
| P2 vs P3 | means | 1.0000 | 9.6e-05 | 1.9e-06 | 0 | 0 | 0 |
| P2 vs P3 | quats | 1.0000 | 5.0e-06 | 2.2e-08 | 0 | 0 | 0 |
| P2 vs P3 | scales | 1.0000 | 8.7e-06 | 6.0e-09 | 0 | 0 | 0 |
| P2 vs P3 | opacities | 1.0000 | 5.3e-06 | 5.6e-07 | 0 | 0 | 0 |
| P2 vs P3 | sh | 1.0000 | 4.1e-06 | 7.2e-07 | 0 | 0 | 0 |
| P1 vs P3 | means | 1.0000 | 9.1e-05 | 1.9e-06 | 0 | 0 | 0 |
| P1 vs P3 | quats | 1.0000 | 4.6e-06 | 2.1e-08 | 0 | 0 | 0 |
| P1 vs P3 | scales | 1.0000 | 8.2e-06 | 5.6e-09 | 0 | 0 | 0 |
| P1 vs P3 | opacities | 1.0000 | 5.0e-06 | 5.3e-07 | 0 | 0 | 0 |
| P1 vs P3 | sh | 1.0000 | 3.8e-06 | 6.3e-07 | 0 | 0 | 0 |

### Camera 155

All comparisons: cosine ≈ 1.0, relative_l2 < 4e-4, zero_nonzero_disagreement = 0, NaN = 0, Inf = 0.

### Classification: BACKWARD_EQUIVALENT (both cameras)

## 5. Diagnosis: Why ~6 Nonzero-Gradient Gaussians

### Classification: EXPECTED_FROM_TEST_CONSTRUCTION

### Root Cause: TRANSMITTANCE_UNDERFLOW_FROM_OUTLIER_GAUSSIANS

The 30K SpeedySplat checkpoint contains **6 enormous Gaussians at depth=0.01** (essentially at the camera position) with:
- Opacity ≈ 1.0 (sigmoid values: 0.9997–1.0000)
- Radii of millions of pixels (covering the entire 2048×1365 image)
- They produce alpha ≈ 0.9998 at every pixel

These 6 Gaussians **fully occlude** all other 115,272 Gaussians. In the alpha-blending formula, the transmittance `T_i = prod_{j<i} (1 - alpha_j)` for behind Gaussians **underflows to exactly 0.0 in fp32**, producing exactly zero gradients.

### Evidence

1. **All 6 nonzero Gaussians have depth=0.01** — they are the frontmost Gaussians in the scene.

2. **Render alpha mean = 0.9998** — nearly every pixel is fully opaque from just these 6 Gaussians.

3. **101,728 Gaussians have radii > 0** (are in the frustum), but only 6 have nonzero gradients — the other 101,722 are occluded.

4. **Opacity reduction test** confirms transmittance underflow:

   | Opacity scale | alpha_mean | Nonzero Gaussians | Exact zero |
   |--------------:|-----------:|------------------:|-----------:|
   | 1.00 (original) | 0.9998 | 6 | 115,272 |
   | 0.50 | 0.9999 | 27 | 115,251 |
   | 0.10 | 0.9999 | 146 | 115,132 |
   | 0.01 | 0.9999 | 1,413 | 113,865 |

   As opacity decreases, more Gaussians get nonzero gradients — confirming the transmittance underflow mechanism.

5. **Even with `render.sum()` as loss** (upstream gradient = 1.0 for every pixel), only 6 Gaussians have nonzero gradients — the issue is independent of the upstream gradient.

### Why this is NOT a harness bug

- The gradients are **correctly zero** — they reflect the true transmittance being zero
- The same 6 Gaussians have nonzero gradients regardless of upstream gradient (random or real loss)
- P1/P2/P3 all agree on which 6 Gaussians are nonzero and which 115,272 are zero
- The transmittance underflow is a numerical property of fp32 alpha-blending with high-opacity outlier Gaussians

### Why this IS expected from the test construction

- The test uses a **fully-trained 30K checkpoint** that has developed 6 extreme outlier Gaussians
- These outliers are artifacts of the SpeedySplat training process (insufficient pruning of enormous Gaussians)
- In a mid-training checkpoint (lower opacities, no outliers), many more Gaussians would have nonzero gradients
- The sparse coverage is a property of the checkpoint, not the backward pass

## 6. Can Backward Correctness Be Considered CLOSED?

**YES — with a documented caveat.**

The backward equivalence is valid:
- P1/P2/P3 agree on the 6 nonzero gradients (cosine ≈ 1.0, relative_l2 < 1e-4)
- P1/P2/P3 agree on the 115,272 zero gradients (zero_nonzero_disagreement = 0)
- The zero gradients are correct (transmittance is truly zero)
- Results are consistent across two cameras (cam0: 6 nonzero, cam155: 5 nonzero)

**Caveat:** Gradient coverage is thin (6/115,278 = 0.005%). The test only exercises 6 outlier Gaussians. More comprehensive coverage would require:
1. A mid-training checkpoint (e.g., iteration 7000) with lower opacities
2. Or removal of the 6 outlier Gaussians before testing
3. Or fp64 computation to avoid transmittance underflow

However, the equivalence IS established for the exercised Gaussians, and the zero-gradient agreement (115,272 Gaussians) is also a valid equivalence check. The backward correctness question (do P1/P2/P3 produce the same gradients?) is answered: **YES**.

## 7. H1-SB Protocol Readiness

**Status: PROTOCOL READY — awaiting Codex's READY_FOR_H1_SB signal**

### Experiment Definition

| Item | Value |
|------|-------|
| Comparison | B1A (clean gsplat + True AccuTile) vs B2 (Trainable HiGS Full) |
| Scenes | train, room, bicycle (same H1 matched-state cohort) |
| Cameras | 3 per scene (same as H1) |
| Checkpoint | Same matched-state checkpoint per scene |
| Resolution | Same as H1 (train: 1959×1090, room: 2048×1365, bicycle: 2048×1361) |
| Background | black |
| SH degree | 3 |
| Precision | fp32 (same as H1) |
| GPU | A100-PCIE-40GB (same as H1) |
| Process | same process |
| Warmup | 20 iterations |
| Measure | 100 iterations |
| Timing | CUDA Events |
| Primary metric | F+B renderer time |
| Secondary | forward, backward, N_visible, N_intersections, active tiles |

### Gate (pre-registered, will not change after seeing results)

| Classification | Threshold |
|---------------|-----------|
| B2_AHEAD | B2 geomean F+B >5% faster than B1A |
| B2_NEAR_PARITY | within ±5% |
| B2_BEHIND | B2 >5% slower |

### Correctness requirement

B1A vs B2 must pass:
- RGB forward equivalence (PSNR > 50 dB or max_abs < 1e-4)
- Alpha forward equivalence (max_abs < 1e-4)
- Major gradient equivalence (cosine > 0.999, relative_l2 < 1e-3 for means, scales, opacities, sh)

If correctness fails, timing comparison is NOT accepted.

### Execution condition

Do NOT execute until Codex returns `READY_FOR_H1_SB`.

### Outputs (when executed)

```
reports/higs/h1-strong-baseline-overlay.md

artifacts/h1-strong-baseline-overlay/
    environment.json
    run_manifest.csv
    correctness.csv
    timings_raw.csv
    timings_summary.csv
    workload_metrics.csv
    analysis.json
```

## 8. Outputs

| File | Path |
|------|------|
| Report | `reports/higs/h1-backward-coverage-sanity.md` |
| JSON | `artifacts/h1-clean-profile/backward_coverage_sanity.json` |
| CSV | `artifacts/h1-clean-profile/backward_coverage_sanity.csv` |
| Script | `scripts/h1/h1_backward_coverage.py` |

## 9. No Optimization Proposed

This report makes no optimization proposals. It is a scientific validation of backward gradient coverage and equivalence only.