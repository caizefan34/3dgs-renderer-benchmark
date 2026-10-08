# Faster-GS room baseline on EPIC-05

2026-09-27. This is an independent strong-method reference, not a new method
or a demonstrated winner. One complete 30K room run per method and the
`FasterGSFused` variant is available; untouched-scene replication is pending.

## Pinned sources and environment

- NeRFICG: `e458ac6e41810ad6d9405687c1e698fadd28dac5`, isolated at
  `/root/codex_nerficg_benchmark_20260927` on EPIC-05.
- Author Faster-GS `main`: `117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847`
  at `src/Methods/FasterGS`. Author `FasterGSFused` is separately pinned at
  `ae665af2f91a4973bb92c40fdc4cba8d12fe3c96` in
  `src/Methods/FasterGSFused`.
- Dedicated Conda environment `codex_nerficg`: Python 3.11.16,
  PyTorch 2.8.0+cu128, CUDA toolkit 12.8. The official method installer
  completed, including MortonEncoding, FasterGSCudaBackend, simple-knn,
  PPISP and FusedSSIM. A first invocation used the global `pip` for
  MortonEncoding; that unintended global Python 3.12 package was uninstalled,
  and the method was installed with the dedicated environment's `pip`.
- The 100-step smoke run succeeded on GPU 2 with 272 training views and
  112,627 starting Gaussians; it is not a quality or speed comparison.

## Input and comparison boundary

- Dataset: the same source room directory and COLMAP calibration used for
  the FastGS-Big audit, with 311 images. Both source loaders sort image names
  and hold out every eighth image, yielding 272 train / 39 test views.
- Config: `fastergs_room_1600_s0.yaml`, SHA256
  `b77e1edee68f4f29b5bb9e2e1c263f2901c0e94086c791a9dbe5bff3212d7fed`.
  The fused config is `fastergsfused_room_1600_s0.yaml`, SHA256
  `f17bd720363bcbf15a77843e3bf33bc0d09f259d60f48875e362dbe2c0a4f9f7`;
  it differs in method type and model name while keeping the same dataset,
  seed, resolution, 30K updates, and comparison boundary.
  `IMAGE_SCALE_FACTOR=1600/3114` maps the 3114x2075 room originals to
  1600x1066. The method uses its own 30K algorithm and author defaults,
  with GPU 2, headless mode, no validation and no test rendering included
  in the training process wall. PCA alignment is disabled to match the
  FastGS coordinate frame.
- **Remaining protocol difference:** FastGS uses Pillow `Image.resize` for
  the 1600x1066 target, while NeRFICG decodes with torchvision and downsizes
  with PyTorch area interpolation. Three sampled originals yield target-to-
  target PSNR 42.71/44.76/45.28 dB and mean absolute RGB difference
  .00389/.00311/.00310. This is small relative to image scale but not
  identical; any close quality ranking needs a shared training-input rerun.
  The diagnostic is `fastgs_fastergs_resize_audit.py` and local
  `fastgs_evidence/fastergs_resize_audit.json`.
- Evaluate both renders with the **same** FastGS 39-view PNG reference and
  metric implementation. Native NeRFICG metrics are useful for its own
  reproduction but not numerically interchangeable with FastGS's metrics.
  Count complete training process wall including data load and final save;
  render/evaluation time is reported separately. GPU/software differences
  from the FastGS run remain explicit (both A100-SXM4 on EPIC-05, but
  Torch 2.8 vs 2.9.1).

## Single-run result on the same host

| 30K system | Complete train process wall | Trainer loop | Final Gaussians | PSNR / SSIM / LPIPS on FastGS 39 PNG targets |
|---|---:|---:|---:|---|
| Author FastGS-Big, newly timed from step 0 | 287.27 s | 228.418 s | 575,264 | 32.20546 / .930243 / .188115 |
| Author Faster-GS `main` | 251.31 s | 228.932 s | 1,191,971 | 31.72786 / .926164 / .200405 |
| Author FasterGSFused | 252.98 s | 230.671 s | 1,179,916 | 31.70539 / .925952 / .200744 |

The Faster-GS process finished 35.96 s sooner (1.143x FastGS/Faster-GS
wall ratio) while its 39-view PSNR/SSIM/LPIPS were worse by
0.47760 dB/.004079/.012290 respectively; 11/39, 13/39, and 4/39 views
favored Faster-GS in those metrics. The trainer loops themselves were close
in this run; the complete-process gap mostly lies outside them. This is a
**speed-quality tradeoff**, not evidence that either system jointly dominates.
FastGS-Big's previous room run gave 32.04247 dB rather than the newly timed
32.20546 dB, so one run does not establish a stable 0.48 dB method effect.

`FasterGSFused` took 1.67 s longer than its author `main` run and its
39-view PSNR/SSIM/LPIPS were lower by .02247 dB/.000212 and higher by .000339.
It favored fused on 20/39, 13/39, and 16/39 views respectively. These small
single-run gaps cannot establish a stable difference between the two Faster-GS
branches. Relative to FastGS-Big, fused was 34.29 s faster but worse by
.50007 dB/.004291/.012629 on the common FastGS targets. Thus the added
author fused baseline does not make the observed room speed-quality frontier
jointly better. Neither Faster-GS branch is a new method from this project.

As a resize sensitivity check, the same FastGS evaluator was also applied
using Faster-GS's own 39 PNG targets. Faster-GS obtained
32.02953/.937588/.170217; the **new timed** FastGS render obtained
32.43275/.939730/.160732. The quality ranking persists under both target
sets, though the magnitude changes. The FastGS targets from its old and new
room runs are byte-identical across all 39 PNGs. Native NeRFICG 8-bit
TorchMetrics report 32.03/.937/.240 for the Faster-GS model; its LPIPS
definition is not numerically interchangeable with the FastGS evaluator.

Evidence is in `fastgs_evidence/`: `fastergs_room_main_full30k_train.log`,
`fastergs_room_main_timings.txt`, `fastergs_room_main_common_metrics.json`,
`fastergs_room_main_common_per_view.json`,
`fastergs_room_main_own_gt_common_metrics.json`,
`room_big_timed_full30k_train.log`, `room_big_timed_full30k_results.json`,
`room_big_timed_full30k_per_view.json`, and
`fastgs_timed_on_fastergs_gt_metrics.json`, plus
`fastergsfused_room_full30k_train.log`, `fastergsfused_room_timings.txt`,
`fastergsfused_room_inference.log`, `fastergsfused_room_common_metrics.log`,
`fastergsfused_room_common_metrics.json`, and
`fastergsfused_room_common_per_view.json`. Render and metric originals remain
in the two isolated remote checkouts. The comparison has one scene, one
seed/run each, two slightly different training-target resize paths, and
different Torch versions. Do not call it a cross-method SOTA result.

A separate isolated FastGS input-pipeline trial reduced room 30K complete
wall from 287.27 s to 248.50 s while retaining all 311 full-resolution
resized image tensors bitwise; its single-run quality is close to the author
FastGS result. See `PARALLEL_CAMERA_LOAD_RESULTS.md`. This is a systems
improvement, not an additional author baseline or a new 3DGS algorithm.
