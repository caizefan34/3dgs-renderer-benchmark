# SAD-GS room 3K original-loss arm (2026-09-27)

## Result and scope

The pinned SAD-GS author code completed indoor Config 2 room training, all 39 held-out test renders, and author metrics with `--lambda_l2 0.0`; all three processes and the supervisor returned 0. The resulting scalar image objective is the FastGS expression `0.8 L1 + 0.2(1−fused_ssim)`, verified in `LOSS_PARITY.md` by source and fused-SSIM comparison. This is a **single-scene, single-run baseline calibration**, not an original method, multiscene validation, or a claim that the two rendering/optimization systems are otherwise identical.

| Room 3K arm | Train process wall s | Author CUDA-event training s | Full train→render→metrics wall s | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | Final Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| Author `lambda_l2=2.0` | 147.618 | 76.751473 | 314.032 | 31.435127 | .930348 | .181309 | 1,725,711 |
| Original loss `lambda_l2=0.0` | 140.937 | 72.084830 | 307.133 | 31.360905 | .930564 | .180577 | 1,655,304 |
| Zero minus author | −6.681 | −4.666643 | −6.899 | −.074223 | +.000216 | −.000732 | −70,407 |

The three metric differences are paired by the same 39 test cameras: 13/39 PSNR, 27/39 SSIM and 28/39 LPIPS views improved under `lambda_l2=0` (LPIPS is better when lower). This is one run per configuration; the time and metric deltas are descriptive, not a variance estimate or a causal decomposition of individual topology decisions. Removing the author's L2 did **not** repair its PSNR gap while preserving its small SSIM/LPIPS advantage within this room comparison. It cannot establish cross-scene noninferiority to FastGS-Big or ACE-GS.

## Frozen identities and verification

`FROZEN_PROTOCOL.md` fixed the sole training-flag change, data, effective loader width 1600, GPU and output before launch. Both arms used source HEAD `029814fb5ed773e4a794cec5115951a64f610ab8` and tracked source archive SHA256 `6f67c7928b20e7eecab2f7102c318fad999ac923e56b00ec6d335a7d9ec9efde`. `images.sha256`, `split.json`, `source.sha256`, and `frozen_inputs.json` match the prior author arm byte-for-byte. The second `colmap.sha256` file accidentally repeats the same four entries; its actual four named binary/PLY file hashes match the first arm. The rendered 39 GT PNGs also have the same name/content manifest hash `8e5f70657a1b52a8f3caaa008598d2cb53b8c2fa93836ae74f1ad0f1796b788a` in both arms. Per-view JSON has the same 39 numeric test names and all three metrics. The numeric-index-to-sorted-test-name rule was checked in the prior author arm.

Both arms used the same physical A100 GPU3 UUID `GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72`; the new preflight was idle, peak sampled memory 31,729 MiB, and post-run GPU3 was 0 MiB/0%. The author checkout remained clean and at the same HEAD. The original-loss run's final PLY is 417,138,165 bytes, SHA256 `ac55a30eb06011b40534a0df75e8288b2b7ff3dd48c1d54ef72d3f1fe5d5c3d6`. The local copied `run.meta.json`, protocol, loss audit, two metric JSONs and three stage logs matched EPIC-05 SHA256; raw GPU samples, hashes, scripts and logs are retained in this directory. Remote model and full renders remain under `/root/sadgs_room_author_3k_l2zero_20260927/model`.

The SAD and FastGS `metrics.py` files differ by method-folder sorting only in the inspected evaluation loop; both call the same named PSNR, SSIM and VGG LPIPS routines. The SAD and FastGS `utils/image_utils.py` and `lpipsPyTorch/__init__.py` hashes match; SAD's `utils/loss_utils.py` adds other functions after the original `ssim` implementation. Cross-method numerical metric parity by rerunning one common evaluator has not been separately established; ACE-GS author code remains unavailable for the required full strong-method comparison.

## Existing FastGS-Big room reference and remaining gap

The already completed author FastGS-Big from-zero timed room 30K run in `FASTERGS_BASELINE_PROTOCOL.md` reports **287.27 s complete training-process wall**, 32.205460 dB PSNR, .930243 SSIM and .188115 LPIPS on 39 test views. Its 39 GT PNG name/content manifest hash is exactly the same `8e5f70657a1b52a8f3caaa008598d2cb53b8c2fa93836ae74f1ad0f1796b788a` as both SAD arms. `utils/camera_utils.py` is byte-identical between pinned FastGS and SAD, and the inspected metric implementations differ only in method-folder sorting; LPIPS module Python files match by SHA256. This makes the room test-image comparison informative, although PyTorch versions and trainer/rasterizer details differ and no multi-run uncertainty is available.

Against that **single** FastGS-Big run, original-loss SAD 3K uses 146.333 s less training wall (2.038× FastGS/SAD), gains .000321 SSIM and lowers LPIPS by .007538, but loses **.844555 dB PSNR**. Thus speed and two metrics are promising on this scene, while the explicit triple-quality requirement fails. The next bounded baseline question is whether one 5K SAD original-loss run closes the PSNR deficit before its complete training wall reaches FastGS's 287.27 s; this is a quality–time frontier check on an existing method, not an original algorithm claim.
