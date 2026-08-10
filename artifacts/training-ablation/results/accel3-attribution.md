# HiGS Accel3 Attribution (vs in-matrix gsplat_27k, same 27k steps)

Phase-split tile sampling (sampling starts at 15k, full-frame SSIM + GT imputation kept) is a
**per-step net loss** relative to the in-matrix gsplat_27k control with identical step budget:

| config | per-step ratio mean | min scene | max scene |
|---|---|---|---|
| higs_tilesamp_phase_27k_r04 | 0.978 | 0.901 (truck) | 1.030 (bicycle) |
| higs_tilesamp_phase_27k_r05 | 0.982 | 0.905 (train) | 1.027 (bicycle) |
| higs_tilesamp_phase_27k_r05_polish | 0.979 | 0.891 (train) | 1.024 (bicycle) |
| higs_tilesamp_phase_27k_r06 | 0.981 | 0.904 (train) | 1.025 (bicycle) |

- The apparent 1.08-1.16x speedup vs the frozen 30k control is fully explained by the
  27k-vs-30k step budget (early-stop effect), not by the sampling algorithm.
- The sampling overheads (mask bookkeeping, GT imputation, full-frame SSIM forward/backward,
  per-step Python) exceed the rasterization savings of sampling 40-60% tiles at 27k steps.
- Quality is additionally degraded (PSNR -0.38..-0.69, SSIM below -0.003, LPIPS above +0.005),
  with deep_blending/playroom collapsing 2-4 dB in every variant.

Conclusion: full-frame-loss tile sampling is a dead end. accel4 (error-guided importance sampling
+ gsplat sparse rasterization that only launches active tiles + sparse sampled-window loss that
skips GT imputation and full-frame SSIM) is the pre-registered next test.
