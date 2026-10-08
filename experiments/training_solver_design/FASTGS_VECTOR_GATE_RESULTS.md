# Full-resolution vector split gate: frozen-state opportunity screen

## Verdict (2026-09-27)

The continuously learnable split gate has **not** shown a joint speed and
quality improvement. At one FastGS room@4100 event, optimizing 9,345 family
gates on ten fixed full-resolution training views reduced their mean loss from
0.0329433 before the split to 0.0327590. Yet on 39 room development views,
the frozen continuous gates were slightly worse than leaving the event
unsplit: PSNR −0.002344 dB, SSIM −0.000165, LPIPS +0.000470. Thresholding at
0.5 kept 1,922 splits and was weaker still. The gates did avoid most of the
**instantaneous** image damage of a native all-family split, but that
comparison alone is not evidence of a better trained reconstruction.

This was a frozen-state diagnostic. Geometry, color, opacity, optimizer state,
and camera input were not trained after gate selection. The 39 views were
already used during development, and the view bootstrap intervals are only
descriptive. No complete training wall time, final reconstruction gain,
untouched-scene confirmation, or originality claim follows from this test.

## Protocol and correctness checks

- Pinned author FastGS commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`;
  room 4,099-step pre-event state, 9,345 native selected parents, child seed
  20260927, effective image resolution 1600×1066.
- Ten predetermined training cameras supplied the original weighted
  L1 + fused-SSIM objective. Only one scalar gate per family was optimized
  for 12 Adam updates; a gate of zero renders the unsplit parent, and one
  renders the native split children. Camera, geometry, color, and all other
  model parameters remained fixed.
- Across those ten cameras, the gate-zero and gate-one diagnostic images
  differed from the author native pre/post images by at most `2.384e-7` RGB.
  One selected pixel-family gate VJP differed from an independent FP64
  compositor by `9.14e-9`. These checks do not establish native parameter
  gradient parity or end-to-end training correctness.
- The saved gate bundle SHA256 is
  `77396ec289ba65f5924b0940431d88e21504511a23a3d5c3e274ada1f5d9a3bb`.

| Frozen state | Ten train-view mean loss ↓ | 39 development-view PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| Native before split | 0.0329433 | 28.739716 | .889377 | .267235 |
| Native all-family split | 0.0344714 | 28.492960 | .884946 | .273439 |
| Continuous gates | **0.0327590** | 28.737372 | .889212 | .267705 |
| Gates thresholded at 0.5 | 0.0328032 | 28.733602 | .889113 | .267881 |

Relative to the unsplit state, the continuous gate improved PSNR on 18/39
development views, SSIM on 9/39, and LPIPS on only 2/39. Its descriptive
paired-view 95% intervals were [−0.01069, +0.00605] dB PSNR,
[−0.000251, −0.000079] SSIM, and [+0.000347, +0.000598] LPIPS. The 39
views are correlated and from one previously inspected scene, so these
intervals must not be interpreted as cross-scene or cross-seed uncertainty.

The recorded 1.2913 seconds covers 12 gate updates and four final train-view
scores after cached projected buffers were built. It excludes scene loading
and other full-run costs. The recorded 0.1928 seconds of cache setup also
excludes scene loading. Neither is a deployable training overhead or a net
speed measurement.

## Decision

Deprioritize a fused native implementation on the current evidence. The
engineering cost case in `FASTGS_FUSED_GATE_OPPORTUNITY.md` only shows that
fusion might make gate gradients affordable; this fixed-state test supplies
no clear quality advantage over doing nothing at the event. A matched
full-resolution long-horizon continuation could test whether the frozen
binary mask later recovers and reduces work, but that would not test the
distinct method of continuously learned gates. Pursue either only if a
specific, cost-accounted long-term mechanism justifies it; do not spend a
30K run or claim success from this instantaneous result.

Reproduction artifacts in this directory: `fastgs_vector_gate_cuda.cu`,
`fastgs_vector_gate_train_probe.py`, `fastgs_vector_gate_train_20260927.json`,
`fastgs_vector_gate_train_20260927.pt`, `fastgs_vector_gate_dev_eval.py`,
`fastgs_vector_gate_dev_eval.json`, and
`fastgs_vector_gate_dev_analysis.json`.
