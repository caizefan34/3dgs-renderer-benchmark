# SAD-GS room 3K author run result (2026-09-27)

## Result

The full author indoor Config 2 training, held-out rendering, and author metrics scripts completed. All three stage processes returned 0. A manual output check passed after the supervisor's own post-check used the wrong filename assumption: author `render.py` writes `00000.png`…`00038.png`; its view order is the `Scene(..., shuffle=False)` order, matching the frozen sorted COLMAP holdout list. The supervisor therefore exited 1 after all three commands had already succeeded; its raw report is retained as `run.meta.supervisor_raw.json`.

The official author `metrics.py` result over 39 held-out views is PSNR 31.4351273 dB, SSIM 0.9303483, LPIPS 0.1813089. Per-view JSON has 39 entries for each metric and is mapped back to source test-image names by `render_index_mapping.json`. This is the author's indoor Config 2 objective with default `lambda_dssim=0.2` and `lambda_l2=2.0`; it is not original-loss SAD-GS or a same-protocol comparison.

## Timing and memory

- Training process wall: 147.618 s, including camera loading, structure-tensor precompute, training, and saving.
- Author CUDA-event training sum: 76.751473 s (render/loss/backward and densification/optimizer intervals only).
- Author render process wall: 116.934 s for 39 test views; author render loop reports 0.26 s summed render intervals.
- Author metrics process wall: 49.479 s.
- Full train→render→metrics supervisor wall: 314.032 s.
- Sampled peak physical GPU use: 31729 MiB on A100 UUID `GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72`; sampled peak training-process VmHWM: 24255316 KiB. One-second samples are in `gpu_samples.csv`.
- Final model: 434880729 bytes, SHA256 `63b6c0b36e5f19c768778013a641777df4a814a11af603a3dd72988aec1617d6`; 1,725,711 Gaussians.
- Last reported training loss: 0.01995.

## Frozen protocol and identity

- Source HEAD `029814fb5ed773e4a794cec5115951a64f610ab8`, clean before/after; tracked source archive SHA256 `6f67c7928b20e7eecab2f7102c318fad999ac923e56b00ec6d335a7d9ec9efde`.
- 311 room cameras: 272 train, 39 held out every eighth sorted camera. All image checksums match `images.sha256`; COLMAP and exact split manifests are retained.
- Runtime: Python 3.11.16 / Torch 2.8.0+cu128 / CUDA 12.8 on physical GPU index 3. No allocator cap.
- Exact commands, defaults, and stop gates: `FROZEN_PROTOCOL.md`; script and inputs are hashed alongside.

## Preserved startup abort

Attempt 001 terminated after 6.056 s and did not complete an iteration. Its false stop was caused by mismatched host/container PID namespaces during CUDA process attribution at ~423 MiB, not memory pressure. Evidence is retained in `attempt_001.md` and related attempt files. Attempt 002 used a clean, separate output path.
