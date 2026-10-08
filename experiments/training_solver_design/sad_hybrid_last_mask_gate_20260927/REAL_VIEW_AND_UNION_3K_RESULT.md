# SAD hybrid SH mask: real-view incidence and bounded union-mask training

2026-09-28. This is an existing-method correctness investigation, **not an original 3DGS algorithm**. The conditional gradient loss established in `FROZEN_CONDITIONS.md` occurs in four fixed-state room training-camera pairs, but a single 3K correction arm loses all three official held-out metrics against the independent unmodified SAD-GS 3K arm. Stop treating this correction alone as a quality-and-speed candidate.

## Fixed-state occurrence

`real_view_incidence.py` loaded the frozen original-loss SAD-GS room 5K PLY (SHA256 `3a088855de73bcd4e123d105812791ab75dd621bd19b0dc6cc89789b4c552894`) with 1,665,316 Gaussians. Four pairs were sampled without replacement from the 272 sorted training cameras with `Random(20260928)`. Each full 1600×1066 camera used the original per-camera `0.8 L1 + 0.2(1−fusedSSIM)` and training `mult=.25`; both backward calls accumulated into `_features_rest.grad`. A row counted as skipped only if its accumulated higher-order SH gradient had nonzero squared norm and the final camera's `radii>0` mask was false. No optimizer step or parameter edit occurred.

| Pair, sorted indices | First-only visible | First-only nonzero SH gradient | All nonzero SH-gradient rows | Skipped gradient squared-norm fraction |
|---|---:|---:|---:|---:|
| 143,16 | 103,652 | 92,101 | 776,015 | 36.54% |
| 109,48 | 425,405 | 358,625 | 802,533 | 6.65% |
| 227,180 | 268,640 | 219,502 | 402,378 | 85.45% |
| 100,171 | 497,941 | 390,078 | 956,585 | 53.80% |

Across these four pairs only, 1,060,306/2,937,511 nonzero SH-gradient row occurrences are skipped; the squared-norm-weighted fraction is 56.16%. This is not an estimate of the fraction over a complete training trajectory, lost optimizer displacement, or final image-quality impact. The script, output and remote copies had matching SHA256 `ec1a4e89544f6ae6898a47e67b820a66b15fb1ec02fe981f88db9cf01c41ddb2` and `9df75c76b6516f351b733432552d0a66fb9be79da63aeaa623edf40088d42d4f`. GPU3 returned idle.

## Same-protocol correction arm

An isolated `train_union_mask.py` was generated from SAD author commit `029814fb5ed773e4a794cec5115951a64f610ab8`; the three functional edits initialize a batch visibility accumulator, OR each camera's `radii>0`, and pass that union to **hybrid higher-order SH** sparse Adam. Dense geometry/DC optimizer, renderer, training loss, effective full resolution, batch size 2, topology schedule and all evaluation commands stayed the same. The author checkout was clean before/after. The trial code and wrapper are retained here, with remote generated hashes `ed467177c9379bfd7c0fff5cc51a32a567fd524351d8f111d1aff0d1efa32983` and `7ced90f9407d1d0f7bd3cc318cc8350f822e9d5b8205165192b1707ae23175e1`.

| Room, independently started 3K | Train process wall s ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Final Gaussians |
|---|---:|---:|---:|---:|---:|
| Unmodified SAD-GS, original loss | 140.937 | 31.360905 | .930564 | .180577 | 1,655,304 |
| SH batch-visible union | 140.158 | 31.175938 | .929890 | .180953 | 1,641,517 |

The correction arm's deltas are −.184967 dB PSNR, −.000674 SSIM, and +.000375 LPIPS (worse in all three). The 0.779 s wall difference is a single-run observation, not an established speedup. All train/render/metrics commands exited 0; 39/39 official test images and per-view metric entries were produced. Source image manifest, split and frozen-input JSON match the unmodified 3K run byte-for-byte; all 39 rendered GT PNGs also match byte-for-byte. The trial's training process wall excludes rendering and metric calculation, exactly as in the reference. The full train→render→metrics wall was 308.613 s. Final PLY SHA256 is `a34405273057bd058e6a58fcdf12876054d68cb5102e555f44eb80c044d25a36`.

The first preflight attempt saw an unrelated transient GPU3 process and stopped before creating the model output or training; the next preflight found GPU3 idle and the single arm above ran. The stored remote model and logs are under `/root/sadgs_room_unionmask_3k_l2zero_20260928/`; local `union_3k_run.meta.json`, `union_3k_results.json` and `union_3k_per_view.json` are copied raw evidence. GPU3 returned to 4 MiB/0% after completion. The trial does not establish that preserving SH gradients is generally harmful: it is one developed scene/seed, and changing SH updates changes the full stochastic topology trajectory. It does reject a direct correction-only claim of closing the FastGS-Big PSNR gap or preserving all three metrics on this gate. Do not promote it to a new algorithm or spend a 5K/30K arm on this mechanism alone.
