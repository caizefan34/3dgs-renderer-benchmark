# INVALID FIRST ATTEMPT — `eval=False`

This first pair is preserved for audit only. The launcher omitted `--eval`, so FastGS trained on all 311 room cameras, including the 39 views later measured. Its reported metrics do not answer the unseen-view question and must not be used for a quality conclusion. The corrected paired run uses new `_eval` output paths and passes `--eval`.

# FastGS 75% sparse adjoint veto pilot (invalid run)

**Verdict:** reject this candidate. On the 10K native FastGS PLYs, the 75% stratified image-gradient mask regressed mean PSNR, SSIM and LPIPS relative to the paired unmasked run. It also left the CUDA raster backward intact, so the run cannot establish a training speedup.

Both arms used author commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`, the same 4099-step checkpoint (`d91ba09a…f7e927b`), the original full-resolution `0.8 L1 + 0.2 (1 - fused SSIM)` loss and the same author configuration through iteration 10000. Training input stayed at 1600×1066. The mask arm sampled three of four checkerboard parity classes independently per 16×16 tile, scaled retained VJP entries by 4/3 and used a separate CUDA generator seeded 20260927. The p=1 arm installed no image-gradient hook. The author’s `train.py` was not modified; the local generator produced a separate `train_sparse_adj.py`.

| Arm | GPU | Mean PSNR ↑ | Mean SSIM ↑ | Mean LPIPS ↓ | Native PLY points @ 4612 / 10000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| p=1 control | 3 | 31.704181 | 0.922928 | 0.213054 | 674,957 / 693,955 |
| p=0.75 mask | 4 | 31.693483 | 0.922279 | 0.213975 | 672,571 / 689,582 |
| p=0.75 minus p=1 | — | −0.010698 dB | −0.000648 | +0.000921 | — |

Across the same 39 room development-retain views, p=0.75 was worse in 22 views by PSNR, 33 by SSIM and 31 by LPIPS. Descriptive paired 95% t intervals were [−0.07146, +0.05006] dB for PSNR, [−0.000920, −0.000376] for SSIM, and [+0.000412, +0.001430] for LPIPS. The views are correlated and were already consulted during development; this is a single-scene, single-pair veto screen, not untouched confirmation.

The complete process walls were 128.98 s for p=1 and 135.32 s for p=0.75. Author-reported training times were 60.4515 s and 65.1424 s. These short-run timing differences include startup, data loading and file writes; the Python mask does not skip CUDA raster backward work, so they are not evidence of algorithmic acceleration.

The author training loop saves its 10K PLY before the 10K densification event. The added Adam checkpoint is saved at process end, after that event. As a result, terminal counts in the training logs/checkpoints are 694,419 (p=1) and 690,230 (p=0.75), respectively 464 and 648 above the native PLY counts used for evaluation. Both PLYs and final checkpoints are preserved remotely.

The author RNG initialization and camera-selection code remained unchanged in both fresh runs, while the mask used its own generator. That matches the original RNG setup as closely as possible, but model/topology divergence can still change data-dependent random operations. No formal test views were used, no tuning was done, and no SOTA or cross-scene claim is made.

The audit-only summary is [`summary_invalid_eval_false.json`](summary_invalid_eval_false.json); the raw [`metrics_39.json`](metrics_39.json), training logs and wall-time files are beside it. The corrected report and valid artifacts are in the parent folder's [`REPORT.md`](../REPORT.md) and [`eval_true/`](../eval_true/). The patch generator is [fastgs_sparse_adjoint_training_patch.py](../../fastgs_sparse_adjoint_training_patch.py), the generated author-source variant is [train_sparse_adj.py](../../train_sparse_adj.py), and the evaluator is [fastgs_sparse_adjoint_eval.py](../../fastgs_sparse_adjoint_eval.py). The invalid run's remote logs remain at `/root/fastgs_sparse_adjoint_pilot_20260927`; its outputs remain under `/root/codex_fastgs_audit_20260927/output/room_sparse_adj_{p1,p75}_10k`.
