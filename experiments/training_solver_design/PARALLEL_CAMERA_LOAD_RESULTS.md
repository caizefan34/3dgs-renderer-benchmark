# FastGS-Big parallel camera preparation: full-resolution systems result

2026-09-27. This is a measured improvement to the author's input pipeline, not
a new 3DGS representation, an original reconstruction algorithm, or a
cross-scene SOTA result. Every training step retains the author's 1600x1066
effective image resolution, loss, rasterizer, optimizer, and topology rule.

## Source and protocol

- Author FastGS source: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f` in
  `/root/codex_fastgs_audit_20260927` on EPIC-05. The trial is an isolated
  clone at `/root/codex_fastgs_loader_trial_20260927`; the author source file
  remains unchanged.
- Trial changes only `utils/camera_utils.py` (19 inserted, 5 removed lines),
  SHA256 `a399a6cd9441c9b1986e45f7d467e5cf5f90efb6e5326c18beb0b084c0a7c1f2`.
  Its complete source is `fastgs_loader_camera_utils.py`. It prepares batches
  of up to 32 Pillow images on 16 CPU threads, returns results in original
  order, then constructs `Camera` objects serially on the original thread.
  No training or renderer code changed.
- Same room source directory, seed 0, `train_big.sh` room options, GPU 2
  (A100-SXM4-80GB), and `.venv` as the newly timed author FastGS-Big baseline.
  Complete process wall includes imports, loading, full 30K training and final
  save; rendering and evaluation are separate.
- The full 311-image CPU audit uses the author's unchanged `PILtoTorch` for
  both serial and 16-thread paths, in 32-image batches. Every output tensor
  passes `torch.equal`. Serial image preparation took 45.633 s versus 4.669 s
  threaded (9.77x for this component). This is a read/decode/resize/convert
  microbenchmark, not full training acceleration; repeated file-cache effects
  may influence these times. Raw output:
  `fastgs_evidence/fastgs_image_prep_311.json`; script:
  `fastgs_camera_load_probe.py`.
- Author and trial `cameras.json` have identical SHA256
  `f01df344ca14773bb318fb92a2509bd46f6220fd7e021e97cc32e3e6691cb9a6`;
  `input.ply` have identical SHA256
  `2a1071fc49a8e7bda185bb22e13cc57dc64e3a72fd2ddc7173840eb29d50367a`.
  All 39 saved GT PNGs have identical names and byte hashes. This does not
  prove the stochastic CUDA training trajectory is bitwise identical: the
  first logged losses already differ at roughly 1e-5 and final point counts
  differ.

## One complete room result

| 30K system | Complete wall | Built-in train timer | Final Gaussians | PSNR / SSIM / LPIPS on identical FastGS 39 PNG targets |
|---|---:|---:|---:|---|
| Author FastGS-Big | 287.27 s | 228.418 s | 575,264 | 32.20546 / .930243 / .188115 |
| FastGS-Big + parallel camera preparation | 248.50 s | 231.294 s | 574,458 | 32.22626 / .930709 / .188195 |
| Author Faster-GS `main` | 251.31 s | 228.932 s | 1,191,971 | 31.72786 / .926164 / .200405 |

The trial's complete wall is 38.77 s lower than the author FastGS run
(1.156x ratio), although its built-in training timer is 2.88 s higher.
Log timestamps place training-camera loading at about 42 s in the author run
and 6 s in the trial; test-camera loading falls from about 6 s to 1 s.
The observed saving is therefore in input preparation, not a faster
rasterizer or optimizer. Against the same author FastGS run, PSNR and SSIM
are slightly higher while LPIPS is worse by .000080. This is one seed and
within the variation already observed between FastGS room runs; it does not
establish a genuine reconstruction-quality gain. The trial was 2.81 s faster
than Faster-GS `main` in this one run and better on all three quality means,
but the small wall-time gap, differing software and training image resize
paths, and single scene/run preclude a robust cross-method superiority claim.

The 39 room views are now development evidence. A new quality-positive
mechanism still needs independent design and tests; generic multithreaded
image loading cannot be claimed as a pioneering 3DGS algorithm. Next test
on untouched scenes and several seeds with matching image preparation before
asserting a stable systems Pareto improvement, and audit a distinct quality
mechanism against current strong literature.

Local raw evidence: `fastgs_evidence/room_big_parallel_load_full30k_train.log`,
`room_big_parallel_load_render.log`, `room_big_parallel_load_metrics.log`,
`room_big_parallel_load_results.json`, `room_big_parallel_load_per_view.json`,
and `fastgs_image_prep_311.json`. Original render PNGs, the checkpoint and the
working source remain in the isolated remote trial checkout.
