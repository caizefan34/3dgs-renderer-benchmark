# DSCF4771 outlier diagnostic

This was a read-only render check of the corrected `eval=True` room 10K PLY pair. The evaluator pairing appears valid: the script asserts 272 training and 39 held-out cameras, uses the same held-out camera and `original_image` for both PLYs, and computes PSNR from clamped RGB MSE. A separate `render_fastgs(..., 0.5)` pass reproduced the recorded PSNRs to under `0.000002 dB`. The two remote PLY hashes match the corrected-run manifest. No evaluator bug was found.

| Held-out view | p=1 PSNR | p=0.75 PSNR | p=0.75 − p=1 |
| --- | ---: | ---: | ---: |
| DSCF4763 | 33.256641 | 33.214533 | −0.042108 dB |
| DSCF4771 | 24.414514 | 30.679356 | +6.264841 dB |
| DSCF4779 | 29.360898 | 29.655271 | +0.294373 dB |

For DSCF4771, p=1 has mean RGB L1 `0.034130` and MSE `0.00361867`; p=0.75 has L1 `0.019090` and MSE `0.00085519`. The p=1 residual is concentrated in lower-right 64×64 pixel tiles over foreground furniture: `[1280,960,1344,1024]` (mean absolute RGB error `0.23940`), `[1536,896,1600,960]` (`0.23095`), and `[1216,896,1280,960]` (`0.22498`). For p=0.75, the largest mean-error tiles move to the top edge. The largest p1/p0.75 image gap by tile remains in the lower-right furniture area (`[1280,960,1344,1024]`, `0.23751`); its largest single-pixel mean RGB gap is at `(1013,69)`. Both renders' largest individual GT residual is at `(942,933)`, so the improvement is spatially broad rather than removal of every local error.

The adjacent held-out views do not repeat the six-decibel jump: DSCF4763 is nearly unchanged and DSCF4779 differs by about `0.29 dB`. This supports describing the recorded outlier as view-specific in this three-view neighborhood; it does not establish a general quality effect or an algorithmic win.

## In-memory opacity probe

To test whether a single p=1 Gaussian drives the 4771 artifact, I selected the eight largest positive projected radii in its baseline render and set each raw opacity tensor entry to `-20` one at a time, restoring it immediately after each render. The largest-radius point, PLY index `689298` (radius `1334 px`, opacity `0.619515`), accounts for almost the entire outlier under this frozen-view intervention: PSNR rises from `24.414514` to `30.677618 dB` (`+6.263104 dB`), MSE drops `76.36%`, and L1 drops `0.014988`. This is within `0.00174 dB` of the p=0.75 render's `30.679356 dB`.

The other seven candidates each changed PSNR by at most `0.00355 dB`; none reduced MSE by 1%. The same point's opacity intervention changed DSCF4763 and DSCF4779 metrics by exactly zero at reported precision. All intervention tensors were restored, and the p=1 PLY hash still matches the expected hash. This isolates a high-radius p=1 point as the direct cause of this frame's excess error; it does not establish why the two trained PLYs differ or show that the sparse-adjoint method generally improves quality. The view was already selected after its development-set metric was observed, so this is post-hoc, view-specific evidence.

The probe made 13 renders (one focal baseline, eight single-point interventions, then a baseline and intervention on each adjacent view). Their synchronized render time summed to `0.0831 s`; no PLY was written. The probe output includes each candidate index, radius, opacity, before/after image metrics, and restoration status.

## Training-view visibility gate

I scanned the same p=1 10K PLY over all 272 training cameras. Point `689298` had positive projected radius in 13 views; among those, radius quantiles (p50/p90/p95/p99/max) were `51/62/639/1330/1503 px`. It appears in the neighboring training frame DSCF4770 at `1503 px` radius, but not in DSCF4772 (radius `0`). A fixed selection rule always included those two filenames and then chose the six other training views with the largest radius, without using their GT metrics.

Opacity suppression on DSCF4770 improves its own full-image metrics (`ΔPSNR +0.622551 dB`, `ΔfusedSSIM +0.005412`, `ΔL1 −0.001815`). DSCF4772 has no change. Across the six radius-selected views (radii `51–63 px`), each PSNR change is at most `0.002478 dB` and each fused-SSIM change is below `0.000002`. This means at least one training image, DSCF4770, contains a measurable residual signal for the point; the scan does not establish why training retained it.

The existing `get_flag=True` / `accum_metric_counts` path can count per-Gaussian pixel contributions when `metric_map` is all ones. Point `689298` contributed to `403,968` pixels (`23.68%`) in DSCF4770 and `658,112` pixels (`38.59%`) in DSCF4771. This count increments after the rasterizer's alpha and transmittance checks, so it measures accepted pixel contributions. Exact per-point tile-instance or bucket-list counts are not returned by the current Python API; obtaining them would need additional rasterizer instrumentation. These counts describe potential work to avoid, not a measured speedup.

The 10K PLY has standard position, opacity, scale, rotation, and SH fields, with no birth-iteration metadata. The exact creation iteration of point `689298` is therefore unverifiable from this artifact. Across the radius scan and interventions, 289 renders took `0.902 s` by the synchronized per-call timer (which includes metric computation); camera loading is excluded. The PLY hash remained unchanged.

`render_fastgs` exposes `render`, visibility, radii, and accumulated metric counts, but no alpha or depth. On DSCF4771 it reported 247,354 / 247,003 visible Gaussians for p=1 / p=0.75; 10,231 / 10,327 had radii above 64 px. Maximum projected radii were 1,334 px (p=1, PLY index 689298) and 846 px (p=0.75, index 620470). These are large-radius candidates, not evidence that any one Gaussian caused the residual pattern.

The three panel PNGs include GT, both renders, both error maps, and the inter-model gap. Separate full-resolution GT/render/error PNGs and per-view statistics are in this directory. Error-map panels share a per-view scale. The full `render_diagnostics.json` records source-image hashes, camera geometry hashes, model hashes, pixel residual summaries, spatial tiles, radius candidates, and output keys.


## Post-event checkpoint identity check

The pre-event 10K PLY candidate at index `689298` survives in the post-event `chkpnt10000.pth`: checkpoint row `681593` has identical xyz, raw scale, and raw rotation values, with raw-opacity absolute difference `2.68e-7` (within `1e-6`). The PLY contains `689336` Gaussians and the checkpoint contains `689237`, a net decrease of `99`; row reordering explains why the candidate's index differs. This rules out deletion of this specific Gaussian by the final densify/prune event, but does not establish its birth iteration. The PLY hash remains `4f0b7032dd97da9da1807a2047ff12a4d8c871ffe7e465aa212397aedae9cff3`; the checkpoint SHA-256 is `a09abf982a0f97dfe4ae22dee0ed2f824b790bcd3cb138a5ae661368e2e86040`. The raw result is in `check_final_checkpoint_match.json` and is embedded in `outlier_summary.json`.

Source commit: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`  
`train.py` SHA-256: `1fd700c341d109f13f95ff3a3cdf1328445f27909119e7eacadb31913e34c963`  
Evaluator SHA-256: `cf216e5ec335d9376f1d13f9530a3fd5d1b3f67c09839820d7e99ae036ecb84d`  
p=1 PLY SHA-256: `4f0b7032dd97da9da1807a2047ff12a4d8c871ffe7e465aa212397aedae9cff3`  
p=0.75 PLY SHA-256: `a19a098df82f2a4f27ecba84432ab87c5b91d7197234c77171ac2a9d6b77b12b`

The diagnostic ran on previously idle EPIC-05 GPU 2 with `CUDA_VISIBLE_DEVICES=2`; it rendered only DSCF4763, DSCF4771, and DSCF4779. The command was:

```sh
cd /root/codex_fastgs_audit_20260927 && CUDA_VISIBLE_DEVICES=2 FASTGS_REPO=/root/codex_fastgs_audit_20260927 FASTGS_DATASET=/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room FASTGS_OUT=/root/codex_fastgs_audit_20260927/outlier_diagnostic_dscf4771_20260927 FASTGS_P1=/root/codex_fastgs_audit_20260927/output/room_sparse_adj_eval_p1_10k FASTGS_P75=/root/codex_fastgs_audit_20260927/output/room_sparse_adj_eval_p75_10k .venv/bin/python /root/codex_fastgs_audit_20260927/outlier_diagnostic_dscf4771_20260927/render_outlier_diagnostic.py
```

The full raw results remain on EPIC-05 under `/root/codex_fastgs_audit_20260927/outlier_diagnostic_dscf4771_20260927/`.
