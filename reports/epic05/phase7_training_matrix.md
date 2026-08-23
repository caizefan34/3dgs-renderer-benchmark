# Phase 7 Training Research Matrix

**Date:** 2026-09-02
**Project:** 3DGS Renderer Optimization — Full Training Evidence Chain

---

## 1. Experiment Register

| Experiment ID | Scene | Config | Steps | Status | Key Result |
|:-------------|:-----:|:------:|:-----:|:------:|:-----------|
| TBD | — | — | — | NOT_STARTED | — |

**Status codes:** PLANNED → QUEUED → RUNNING → COMPLETED / FAILED / BLOCKED / FALSIFIED

---

## 2. Training Configurations Under Test

| ID | Renderer | tile_size | packed | SH degree | radius_clip | eps2d | Label |
|:--:|:--------:|:---------:|:-----:|:---------:|:-----------:|:-----:|-------|
| C0 | gsplat | 16 | true | 3 (progressive) | 0.0 | 0.1 | Baseline |
| C1 | gsplat | 16 | true | 3 (progressive) | 0.0 | 0.1 | tile16 |
| C2 | gsplat | 32 | true | 3 (progressive) | 0.0 | 0.1 | tile32 |

---

## 3. Gate Progression

Each configuration must pass:

1. ✅ **Forward correctness** — pixel equivalence verified
2. ✅ **Gradient correctness** — gradcheck/finite-difference PASS
3. ✅ **GT Quality gate** — PSNR/SSIM/LPIPS within tolerance
4. ⬜ **Training Sanity** — 500-step short run (this experiment)
5. ⬜ **Training mid** — 3000-step medium run
6. ⬜ **Full Training** — 30K-step complete run
7. ⬜ **Time-to-Quality** — wall clock to reach PSNR target
8. ⬜ **Composability** — multi-module interaction
9. ⬜ **E2E benefit** — end-to-end training speedup at equal quality

---

## 4. Per-Experiment Measurement Protocol

Every training experiment records:

| Metric | Unit | Source |
|:-------|:----:|:-------|
| Loss | scalar | L1 + 0.2*D-SSIM |
| L1 | scalar | L1 component |
| D-SSIM | scalar | 1 - SSIM |
| PSNR | dB | 10*log10(1/MSE) |
| Gradient norm | scalar | global norm clipped |
| Forward time | ms | per-iteration |
| Backward time | ms | per-iteration |
| Optimizer time | ms | per-iteration |
| Total iteration time | ms | per-iteration |
| Peak GPU memory | MB | torch.cuda.max_memory_allocated |
| Gaussian count | int | model.xyz.shape[0] |
| Densification events | {clone, split} | per-event |
| Pruning events | int | per-event |
| NaN/Inf detection | bool | per-gradient/param |
| SH degree | int | current degree |

---

## 5. Results Summary

_(Will be filled as experiments complete)_

| Scene | Config | Steps | Wall Time | Avg Step | Final PSNR | Best PSNR | Final N | NaN? | Status |
|:-----|:------:|:-----:|:---------:|:--------:|:----------:|:---------:|:-------:|:----:|:------:|
| | | | | | | | | | NOT_STARTED |
