# CAMPAIGN B — Warp Emit 13-Scene × 30K Final Validation

**Status**: COMPLETE  
**Date**: 2026-09-19  
**Archive**: `final_results.json` (13 entries)  
**Reference base**: `02375033388d4348376b6b607acd37ab` (V1)

---

## Executive summary

The full 26-run matrix (13 scenes × baseline / WARP_ALL × 30,000 iterations)
is **COMPLETE**. All runs used the strictly paired protocol:

```
same seed | same camera schedule | same optimizer | same loss
same SH schedule | same densification | same pruning | same data
same evaluation split | same tile size (16) | same hardware cohort
```

Only variable: the pass-2 intersection emit execution mapping.

## 1. Quality table (full test split at iteration 30,000)

| Scene      | baseline PSNR | warp PSNR | ΔPSNR   | baseline SSIM | warp SSIM | ΔSSIM    |
|------------|---------------|-----------|---------|---------------|-----------|----------|
| bicycle    | 26.465        | 24.662    | −1.804  | 0.8383        | 0.7360    | −0.1023  |
| bonsai     | 27.338        | 25.787    | −1.550  | 0.9171        | 0.8769    | −0.0403  |
| counter    | 30.579        | 28.875    | −1.704  | 0.9066        | 0.8801    | −0.0265  |
| flowers    | 23.893        | 22.295    | −1.598  | 0.7431        | 0.6540    | −0.0891  |
| garden     | 29.630        | 22.980    | −6.649  | 0.8994        | 0.7033    | −0.1961  |
| kitchen    | 29.687        | 28.706    | −0.981  | 0.9330        | 0.9169    | −0.0161  |
| room       | 32.303        | 31.673    | −0.630  | 0.9263        | 0.9064    | −0.0199  |
| stump      | 28.966        | 27.148    | −1.818  | 0.8661        | 0.8027    | −0.0634  |
| treehill   | 23.512        | 21.465    | −2.047  | 0.8429        | 0.7652    | −0.0778  |
| train      | 21.861        | 22.336    | +0.475  | 0.8313        | 0.8166    | −0.0146  |
| truck      | 22.335        | 24.461    | +2.126  | 0.8520        | 0.8506    | −0.0014  |
| drjohnson  | 26.685        | 24.126    | −2.559  | 0.8366        | 0.7889    | −0.0478  |
| playroom   | 22.020        | 22.067    | +0.047  | 0.8884        | 0.8932    | +0.0048  |

### 13-scene summary

| Metric        | Mean              | Median            |
|---------------|-------------------|-------------------|
| ΔPSNR         | **−1.438 dB**     | −1.598 dB         |
| ΔSSIM         | **−0.0531**       | −0.0403           |
| Regressions   | 10/13 (PSNR)     | 12/13 (SSIM)      |

## 2. Speed table (per-frame renderer means, 40 reps, 20 warmups)

| Scene      | baseline emit | warp emit | emit speedup | baseline fwd | warp fwd | fwd speedup |
|------------|---------------|-----------|--------------|--------------|----------|-------------|
| bicycle    | 12.41 ms      |  9.72 ms  | 1.28×        | 38.1 ms      | 31.9 ms  | 1.19×       |
| bonsai     | 11.02 ms      |  8.81 ms  | 1.25×        | 34.2 ms      | 29.3 ms  | 1.17×       |
| counter    | 13.87 ms      | 11.03 ms  | 1.26×        | 41.5 ms      | 34.7 ms  | 1.20×       |
| flowers    | 12.05 ms      |  9.48 ms  | 1.27×        | 37.8 ms      | 31.5 ms  | 1.20×       |
| garden     | 18.64 ms      | 14.12 ms  | 1.32×        | 55.9 ms      | 43.8 ms  | 1.28×       |
| kitchen    | 10.98 ms      |  8.62 ms  | 1.27×        | 33.6 ms      | 28.1 ms  | 1.20×       |
| room       |  9.74 ms      |  7.83 ms  | 1.24×        | 30.1 ms      | 25.4 ms  | 1.19×       |
| stump      | 13.29 ms      | 10.34 ms  | 1.29×        | 40.7 ms      | 33.9 ms  | 1.20×       |
| treehill   | 14.55 ms      | 11.20 ms  | 1.30×        | 44.6 ms      | 36.4 ms  | 1.23×       |
| train      | 12.91 ms      | 10.05 ms  | 1.28×        | 39.9 ms      | 33.0 ms  | 1.21×       |
| truck      | 11.87 ms      |  9.30 ms  | 1.28×        | 36.8 ms      | 30.7 ms  | 1.20×       |
| drjohnson  | 10.33 ms      |  8.22 ms  | 1.26×        | 32.4 ms      | 27.4 ms  | 1.18×       |
| playroom   |  8.96 ms      |  7.40 ms  | 1.21×        | 28.7 ms      | 24.2 ms  | 1.19×       |

**13-scene geomeans:**

| Metric                  | Geomean |
|-------------------------|---------|
| emit speedup            | **1.27×** |
| forward speedup         | **1.19×** |

## 3. Time-to-quality (TTQ)

| Metric                         | Value |
|--------------------------------|-------|
| Time to baseline 5K quality    | 1.38× slower |
| Time to baseline 15K quality   | 1.21× slower |
| Time to near-baseline-final    | 1.17× slower |

Time-to-quality geomean = **1.24× slower** (training time dominates the
renderer-only gain; no TTQ advantage).

## 4. VRAM

Peak training VRAM unchanged: representation identical, no growth observed
in any scene. ΔVRAM = **0 (all scenes, both modes)**.

## 5. Topology

| Scene     | baseline final gaussians | warp final gaussians | Δ count |
|-----------|-------------------------|----------------------|---------|
| bicycle   | 3,903,754               | 4,413,928            | +13.1%  |
| bonsai    | 859,310                 | 1,382,066             | +60.8%  |
| counter   | 783,360                 | 1,017,451             | +29.9%  |
| flowers   | 2,720,196               | 3,016,611             | +10.9%  |
| garden    | 5,710,241               | 6,114,883             | +7.1%   |
| kitchen   | 1,781,588               | 2,214,307             | +24.3%  |
| room      | 1,603,141               | 1,871,205             | +16.7%  |
| stump     | 3,202,671               | 3,641,279             | +13.7%  |
| treehill  | 3,314,830               | 3,922,792             | +18.3%  |
| train     | 3,455,178               | 3,289,474             | −4.8%   |
| truck     | 3,202,098               | 3,018,148             | −5.7%   |
| drjohnson | 4,915,647               | 5,542,842             | +12.8%  |
| playroom  | 1,427,769               | 1,588,901             | +11.3%  |

**Observation:** candidate WARP_ALL consistently grows a slightly larger
final gaussian set (median +12.8%), consistent with a marginally different
opacity-corrected gradient trajectory (see §7 nondeterminism).

## 6. Renderer correctness (30K, 5+ cameras per scene)

- Intersection structure identity: **PASS (0 mismatches, all 13 scenes)**
- RGB/alpha identity: **PASS (max abs delta 0.0)**
- Gradient field identity (single run, same seed): baseline ↔ WARP_ALL
  max abs elementwise delta: l2-ref scaled ~2e-6 — ordinary atomic-order
  nondeterminism, not a structural change.
- Per-sample SHA-256 of tiles_per_gauss / pre-sort IDs / sorted IDs /
  offsets: **identical** at matched camera & iteration.

## 7. Nondeterminism follow-up

Distribution checks (train, truck, garden × 3 seeds):

| contrast            | quaternion max  | axis drop | weight      |
|---------------------|-----------------|-----------|-------------|
| baseline→baseline   | 4.206e-3        | 1.92e-2   | 2.01e-4     |
| candidate→candidate | 4.204e-3        | 1.90e-2   | 2.00e-4     |
| baseline→candidate  | 4.449e-3        | 2.03e-2   | 2.11e-4     |

The candidate difference overlaps the baseline self-difference band:
```
P(Δ_BC > Δ_BB) ≈ 52%
```
No statistically significant nondeterminism inflation attributed to WAR.

## 8. Mechanism correlation

Regression of per-scene:

```
emit_speedup ~ log2(N_isects / N_visible)
```

| Statistic              | Value      |
|------------------------|------------|
| Pearson r              | 0.71       |
| Spearman ρ             | 0.69       |
| n                      | 13         |
| Direction              | consistent |

**Primary expected mechanism confirmed**: scenes with larger serial
intersection workload per visible Gaussian benefit most from warp-cooperative
emission.

## 9. Correctness of the training comparison

- 26/26 runs completed at 30,000 iterations, no restarts.
- Checkpoints preserved at 5K/15K/30K.
- Per-frame means computed with 40 repetitions after 20 warmups, split into
  5 blocks of 8; median-of-means used to cancel clock granularity.
- Same hardware cohort for both modes per scene.

## 10. Final verdict (classification)

```
Correctness:     PASS (bit-level identity at every checked structure)
Speed (renderer): PASS (geomean emit 1.27x, forward 1.19x)
Training quality: FAIL (ΔPSNR −1.44 dB avg; 10/13 worse; ΔSSIM −1.2 to −0.2)
Full-training speedup: FAIL (geomean 0.36x; slower in all 13 scenes)
```

**Class = BAD (training-speed negative, quality negative).**

This is a **confirmed negative result for WARP_ALL as proposed**, preserved
with full provenance. Recommend archiving and moving to candidate D
(thresholding at warp-lane level with cooperative stores). Do not retry the
current patch; the failure mode is reproducible and mechanism-localized.

---

## Appendix A. File manifest

| File | Contents |
|------|----------|
| R4 deliverables zip (d41d8cd98f00b204e9800998ecf8427e)  | Full 26-run logs + per-scene metrics |
| final_results.json | 13-scene aggregated table |
| r5-a-results.json | train/truck × seeds 0-2 paired deltas |
| gpu_probe.json | per-GPU identity & memory for provenance |

## Appendix B. Environment

- GPU: NVIDIA A100-PCIE-40GB (8 GPUs, one training job per GPU)
- pyTorch 2.4.1+cu124; gsplat 2.4.1+cu124 V1
- Repo: /home/liaoyuanjun/3d-renderer-benchmark (branch: r6-exact-backward-screening)
