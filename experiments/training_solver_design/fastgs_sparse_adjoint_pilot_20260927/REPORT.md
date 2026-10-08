# FastGS 75% sparse-adjoint pilot

The corrected run found **no clear immediate quality damage**, but it also did not demonstrate the requested combination of faster full training and higher novel-view quality. On the 10K PLYs, p=0.75 raised mean PSNR and SSIM slightly while worsening mean LPIPS; all paired-view descriptive intervals cross zero. The mask leaves the CUDA raster backward intact, and measured training times were effectively equal. This Python-mask candidate should not advance as an acceleration method.

Both valid arms used author commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f` and the same 4099-step model plus Adam checkpoint (`d91ba09a…f7e927b`). The corrected output `cfg_args` files record `eval=True`. FastGS reads all 311 COLMAP camera records to construct its split, then `eval=True` assigns 272 cameras to training and 39 to development evaluation; the evaluator asserted those counts. Training used 1600×1066 full-resolution forward passes and the original `0.8 L1 + 0.2 (1 - fused SSIM)` loss. The p=0.75 arm applied an independently seeded, tile-stratified 75% mask with 4/3 scaling only to the image VJP. The p=1 arm used no image hook. The author `train.py` remained unchanged.

| Arm | GPU | Mean PSNR ↑ | Mean SSIM ↑ | Mean LPIPS ↓ | PLY points @ 4612 / 10000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| p=1 control | 3 | 30.273385 | 0.913998 | 0.221100 | 671,455 / 689,336 |
| p=0.75 mask | 4 | 30.386346 | 0.914130 | 0.221554 | 671,647 / 691,612 |
| p=0.75 minus p=1 | — | +0.112961 dB | +0.000132 | +0.000454 | — |

On the paired 39 development views, p=0.75 was worse in 17 views by PSNR, 30 by SSIM and 29 by LPIPS. The descriptive paired 95% t intervals were [−0.23095, +0.45687] dB for PSNR, [−0.001780, +0.002044] for SSIM, and [−0.000176, +0.001084] for LPIPS. These views are correlated, had already been consulted during development, and represent one scene and one paired run; the intervals do not establish a general effect.

The PSNR mean is highly sensitive to one view: DSCF4771 improved by +6.26484 dB, more than the +4.40549 dB sum of all 39 paired PSNR changes. Excluding it as a descriptive sensitivity check, the other 38 views average −0.04893 dB PSNR, −0.000781 SSIM and +0.000697 LPIPS. The primary 39-view metrics above retain every view; this check shows that the positive PSNR mean is not a broad per-view improvement.

The complete process walls were 127.20 s for p=1 and 126.81 s for p=0.75. Author-reported training times were 61.2065 s and 61.2021 s. The roughly 0.4-second wall difference is not meaningful evidence of faster training: the Python mask does not skip raster backward work, and the process measurements include startup, loading and file writes.

The source uses the same author RNG initialization and camera-selection code in both fresh processes, while the candidate mask has its own CUDA generator. That matches the original random setup as closely as possible; topology divergence can still affect later data-dependent random operations. No formal test views were used.

The author loop writes `point_cloud/iteration_10000/point_cloud.ply` before the final-iteration densification event, while the added compatible Adam checkpoint is serialized after the event. Thus checkpoint/log terminal counts are 689,237 (p=1) and 691,618 (p=0.75), versus evaluated native PLY counts 689,336 and 691,612. Both artifacts are preserved, and evaluation uses the author-native PLY outputs.

The first pair is **invalid for quality conclusions** because its launcher omitted `--eval`, putting all 311 cameras in training. Its logs and outputs remain for audit in [`initial_invalid_eval_false/`](initial_invalid_eval_false/). The corrected raw logs, wall times, `cfg_args` and per-view metrics are in [`eval_true/`](eval_true/); artifact hashes and remote paths are in [`summary.json`](summary.json). Remote corrected outputs remain at `/root/codex_fastgs_audit_20260927/output/room_sparse_adj_eval_{p1,p75}_10k`, with original logs at `/root/fastgs_sparse_adjoint_pilot_eval_20260927`.
