# Fast 3DGS Renderer Survey

Updated: 2026-07-20. Paper-reported speedups are not directly comparable
across GPUs, scenes, resolutions, quality targets, or forward/backward modes.
Only rows marked **locally verified** belong in the benchmark leaderboard.

## Reproducible open-source candidates

| Renderer | Upstream commit | Main idea | Upstream claim | Local status |
|---|---|---|---|---|
| original 3DGS | `54c035f7834b564019656c3e3fcc3646292f727d` (rasterizer `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`) | Official reference rasterizer; Thrust-based tile sorting | Original real-time reference implementation | **Tier A complete**; five-case EPIC-05 reference baseline |
| gsplat | `77ab983ffe43420b2131669cb35776b883ca4c3c` | Packed/dense CUDA rasterization, AccuTile, current CUDA support | General-purpose optimized rasterizer | **Tier A complete**; 1.966x speed index and lowest full-suite peak VRAM |
| gsplat HiGS | same gsplat commit | Macro-tile partitioning, fine-tile rasterization, fp16 packed inference scene, persistent workspace | Up to 15.8x over original 3DGS | **Tier A complete**; fastest in all five cases and 5.671x speed index |
| Speedy-Splat | `34c45c6d9b8bd6110231864f2f358b6d3abbf73d` | Exact Gaussian localization plus primitive pruning | 6.71x with 10.6x fewer primitives | **Tier A complete**; generated best-balance recommendation at 2.385x |
| TC-GS (Speedy-Splat integration) | `0bb82f88fde211c34b42e1497f0fc7265461592b` | FP16 Tensor-Core alpha evaluation with `mma.sync` | Authors report about 2x on A800 with nearly unchanged aggregate quality | **Tier A complete**; best aggregate PSNR/LPIPS and 2.048x speed index |
| FlashGS | `cdfc4e4002318423eda356eed02df8e01fa32cb6` | Redundancy elimination, pipelining, scheduling, memory-access optimization | Average 4x on tested GPUs | Source acquired; build/adapter pending |
| Local-GS / TiCoGS | `0c6d9e4a2cc458de90d3dc40753187d6d03ea514` | Tile-local warp coherence, parameter hoisting, warp culling, branch reduction | 1.4-1.6x on Ada; up to 7.76x on Deep Blending | Source acquired; CUDA extension pending |
| GEMM-GS | `aca61f897f58964ff7204e1e3c6485995b5f212c` | Reformulates blending for Tensor Cores, double-buffered kernel pipeline | 1.42x over vanilla; 1.47x additional with other accelerators | Source acquired; build/adapter pending |
| StopThePop | `859b11bde9195e19a1c40536e1ab16765b64b365` | Hierarchical sorting to reduce popping/view inconsistency | Same-checkpoint path is about 4% slower; faster reduced models require retraining | Quality/view-consistency control candidate, not a renderer-only speed claim |
| fast-gaussian-rasterization | `bbe7196f8bc0708cb24cdaea2c264a7b6942d980` | Global CUDA sort plus OpenGL geometry/fragment rasterization | 5-10x direct framebuffer; 2-3x offline | Blocked locally by EGL DLL/application-control policy |

## Paper tracking: not locally reproducible yet

| Work | Public result | Why it is not on the measured leaderboard |
|---|---|---|
| TemporalGS (arXiv:2607.03390) | Temporal culling and selective tile rendering; up to 1.48x | No official source located; sequence-dependent quality/latency |
| Local-GS paper (arXiv:2606.16566) | Up to 7.76x | Source exists, but local build and quality validation are still required |
| Accelerating 3DGS using Tensor Cores (arXiv:2605.17855) | Tensor-Core acceleration | No reproducible local package located |

## TC-GS author-reported quality (not local leaderboard data)

The TC-GS repository reports the following aggregate GT-relative metrics on an
NVIDIA A800. They are retained as upstream context and are not mixed with the
EPIC-05 Tier A measurements.

| Dataset | Path | PSNR vs GT | SSIM vs GT | LPIPS vs GT | Reported speedup |
|---|---|---:|---:|---:|---:|
| Tanks & Temples | 3DGS | 23.687 | 0.851 | 0.169 | 1.00x |
| Tanks & Temples | 3DGS + TC-GS | 23.682 | 0.851 | 0.169 | 2.13x |
| Deep Blending | 3DGS | 29.803 | 0.907 | 0.238 | 1.00x |
| Deep Blending | 3DGS + TC-GS | 29.803 | 0.906 | 0.236 | 2.185x |
| Mip-NeRF360 | 3DGS | 26.546 | 0.785 | 0.250 | 1.00x |
| Mip-NeRF360 | 3DGS + TC-GS | 26.544 | 0.785 | 0.250 | 2.01x |

## Inclusion rules

1. A renderer must execute its own upstream implementation. A wrapper that
   calls another backend is listed as that backend, not as a separate result.
2. Static scene activation and renderer workspace construction happen before
   timing. Per-camera work remains inside the timed frame.
3. GPU latency uses CUDA events. Wall-clock or GUI-present FPS must be reported
   separately and never mixed into the same ranking.
4. Identical PLY tensors, cameras, resolution, SH degree, background, warmup,
   frame order, and repeats are required.
5. A result is valid only with finite output, changing images across cameras,
   recorded source/package version, and PSNR/SSIM/LPIPS against the paired
   original image. A held-out claim additionally requires verified training
   split provenance. Renderer-to-renderer comparisons are diagnostics only.
6. Approximate, pruned, LOD, temporal, fp16, or hardware-rasterized paths must
   declare their quality constraint and cannot be labeled lossless without data.

## Sources

- <https://github.com/nerfstudio-project/gsplat>
- <https://github.com/graphdeco-inria/gaussian-splatting>
- <https://github.com/DeepLink-org/3DGSTensorCore>
- <https://github.com/j-alex-hanson/speedy-splat>
- <https://github.com/InternLandMark/FlashGS>
- <https://github.com/tilaba/Local-GS>
- <https://github.com/shieldforever/GEMM-GS>
- <https://github.com/dendenxu/fast-gaussian-rasterization>
- <https://github.com/r4dl/StopThePop>
- <https://arxiv.org/abs/2505.24796>
- <https://arxiv.org/abs/2607.03390>
- <https://arxiv.org/abs/2606.00352>
- <https://arxiv.org/abs/2606.16566>
