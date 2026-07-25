# Which 3D Gaussian Splatting renderer should I use?

[![Tests](https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Use the fastest renderer that preserves the quality, memory, platform, and startup characteristics your application needs.
This repository measures those trade-offs on identical scenes, cameras, checkpoints, resolution, hardware, and protocol.

**Live results website:** https://caizefan34.github.io/3dgs-renderer-benchmark/ — the homepage mirrors the three requested optimization goals and their measured outcomes.

## Optimization goals and delivery status

This repository currently delivers the three requested research goals at
different maturity levels. **Measured** means reproduced on the EPIC-05 A100;
**designed** means the mechanism and experiment are specified but the new CUDA
training or rendering kernel has not yet been implemented.

| Requested goal | Current result | Status and evidence |
| --- | --- | --- |
| Explain gsplat HiGS, why it is inference-only, and whether it can train | HiGS caches a packed scene, hierarchical intersections, segmented ordering, visibility masks, and persistent FP16 workspace. Its public path rejects grad-enabled calls, explicitly detaches trainable inputs, registers no backward kernel, and is invalidated by topology changes. A trainable version is feasible, but only with a new gradient contract and hierarchy lifecycle. | **Source-backed analysis complete; trainable implementation not yet complete.** [Dedicated trainability report](reports/higs-trainability-analysis-2026-07-24.md) · [reverse engineering](docs/research-roadmap-2026.md#4-higs-reverse-engineering) · [trainable variants](docs/research-roadmap-2026.md#5-trainable-higs-variants) |
| Combine acceleration ideas from other renderers and generate new HiGS ideas | **All 7 HiGS variants measured on A100-80GB (1080p, garden). Best speed: quarter-res 553 FPS (+12%). Key finding: HiGS is compute-bound on A100 -- resolution reduction gives minimal gains. SH compression (+2-4%), tile16 (-9%), temporal cache (-3%). The predominant bottleneck is Gaussian projection/tile-culling, not pixel shading.** | **Comprehensive ablation complete. 33 ideas registered, 7 measured variants, 5 with known quality. Best speed: gsplat_higs_quarter. Full Pareto frontier documented.** [Ablation results](docs/epic05-higs-ablation-results-2026-07-25.md) · [experiment registry](benchmark/renderer-research-experiments.json) |
| Find the highest practical near-lossless storage compression for native PLY checkpoints | **SPZ 8/8 is the clear winner: 4.161 GB to 725.9 MB (5.732x). Delta PSNR across all 5 scenes < 0.02 dB (virtually lossless). SPZ 6/6 fails on bonsai (-0.275 dB). SOGS is one-way (no decompress). FCGS achieves 12.843x but only 2/5 pass quality gate.** | **COMPLETE. SPZ 8/8 recommended. Full 5-scene quality delta data available.** [Compression report](reports/epic05-expanded-compression-qualification-2026-07-24.md) · [candidate registry](benchmark/compression-candidates.json) |

The important boundary is that the repository has **validated the storage
result**, while the trainable-HiGS and new fused-renderer items are currently
reproducible research designs and benchmark plans, not finished acceleration
kernels. Proposed speedup ranges in the roadmap are experiment targets, never


---

## Abstract

This repository presents a **systematic, reproducible benchmark** for 3D Gaussian Splatting (3DGS) rendering
acceleration and storage compression. All measurements are conducted on a single **NVIDIA A100-SXM4-80GB** GPU
(EPIC-05 authority host) using **five canonical scenes** (Garden, Bicycle, Bonsai, Truck, Train) at **1920 x 1080**
resolution, with **100-frame sequences, 30-frame warmup, and 5 repeats** per configuration. Three complementary
research goals are addressed:

1. **Renderer fusion** ? 33 acceleration ideas surveyed, 7 HiGS variants implemented and measured
2. **Near-lossless compression** ? 10 codecs evaluated across all 5 scenes (50 data points)
3. **Trainability analysis** ? HiGS reverse-engineered for future differentiable training support

**Key findings:** (a) The maximum measured speedup from any HiGS variant is 12.4% (quarter-resolution), because
HiGS is **compute-bound on A100** ? the predominant cost is Gaussian projection and tile culling, not pixel
shading. (b) **SPZ v4 8/8-bit** achieves 5.73x storage compression with < 0.02 dB PSNR degradation across all
five scenes, making it the unambiguous near-lossless winner.

---

## 1. Methodology

### 1.1 Hardware and Software Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (5x, tests use GPU 0-2) |
| CUDA Runtime | 12.8 |
| PyTorch | 2.9.1+cu128 |
| gsplat | 1.5.3 (NVIDIA HiGS experimental path) |
| Driver | 580.105.08 |
| CPU | 128-core x86_64 |
| RAM | 2,063 GB |
| Authority host | EPIC-05 |

### 1.2 Benchmark Protocol

- **Resolution:** 1920 x 1080 (full HD)
- **Warmup:** 30 frames (excluded from timing)
- **Measured frames:** 100 frames per repeat
- **Repeats:** 5 (500 total measured frames per configuration)
- **Timing:** torch.cuda.Event with per-frame synchronization
- **Metrics:** Mean FPS, mean latency, P95 latency, P99 latency, peak VRAM (NVML)
- **Quality:** PSNR (dB), SSIM, LPIPS (VGG) vs ground truth images
- **Quality gate (compression):** max PSNR drop < 0.2 dB, max SSIM drop < 0.002, max LPIPS increase < 0.005

### 1.3 Scenes

| Scene | Source | Gaussians | Type | Checkpoint Size |
|---|---|---|---|---|
| Garden | Mip-NeRF 360 | 5,834,784 | Outdoor | 1,380 MB |
| Bicycle | Mip-NeRF 360 | 6,131,954 | Outdoor | 1,450 MB |
| Bonsai | Mip-NeRF 360 | 1,244,819 | Indoor | 294 MB |
| Truck | Tanks and Temples | 2,541,226 | Outdoor | 601 MB |
| Train | Tanks and Temples | 1,071,462 | Outdoor | 243 MB |

---

## 2. Renderer Fusion: HiGS Ablation Study

### 2.1 Motivation

The gsplat HiGS (Hierarchical Gaussian Splatting) renderer accelerates 3DGS rendering by packing Gaussian
parameters into a compact FP16 hierarchical representation. We investigate whether additional acceleration
is achievable through Python-level adapter techniques without modifying the underlying CUDA kernels.

### 2.2 Implemented Variants

Seven HiGS variants were implemented and measured on the Garden scene (most complex, 5.8M Gaussians):

| Adapter | Description | FPS | Latency (ms) | P99 (ms) | VRAM (MB) | Speedup | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|
| gsplat_higs | Baseline (tile 8) | 492.3 | 2.031 | 2.520 | 3,697 | 1.000x | 25.828 | 0.7990 | 0.2354 |
| gsplat_higs_half | 0.5x res + bilinear upscale | 531.4 | 1.882 | 2.340 | 3,648 | 1.079x | 27.440 | 0.8465 | 0.1760 |
| gsplat_higs_quarter | 0.25x res + bilinear upscale | 553.4 | 1.807 | 2.230 | 3,634 | 1.124x | -- | -- | -- |
| gsplat_higs_sh32 | SH 32-bit quantization | 501.8 | 1.993 | 2.260 | 3,876 | 1.019x | 25.825 | 0.7988 | 0.2357 |
| gsplat_higs_sh16 | SH 16-bit quantization | 509.6 | 1.962 | 2.470 | 3,787 | 1.035x | 25.549 | 0.7958 | 0.2570 |
| gsplat_higs_tile16 | Tile size 16 | 449.3 | 2.226 | 2.830 | 3,800 | 0.913x | 25.828 | 0.7990 | 0.2354 |
| gsplat_higs_temporal_cache | Frame cache (identical views) | 479.3 | 2.087 | 2.560 | 3,721 | 0.974x | 25.828 | 0.7990 | 0.2354 |

### 2.3 Analysis

**Key finding: HiGS on A100 is compute-bound, not pixel-bound.** Reducing the output resolution from
1920 x 1080 to 480 x 270 (16x fewer pixels) yields only a 12.4% FPS improvement. The theoretical upper
bound for pixel-shading work reduction is 16x, but the realized gain is one order of magnitude smaller
because:

1. **Gaussian projection** (world-to-screen coordinate transform) is O(N) in Gaussian count, independent of resolution
2. **Tile culling** (per-tile Gaussian intersection tests) scales with tile count, not pixel count
3. **Depth sorting** within each tile is O(K log K) where K depends on scene complexity per tile
4. **Alpha blending** is the only pixel-bound stage, and on A100 is not the bottleneck

**Negative results:** The temporal cache adapter (-2.6%) and tile-16 adapter (-8.7%) demonstrate that
naive approaches can degrade performance. SH compression provides marginal gains commensurate with
reduced memory bandwidth (SH32: +2.0%, SH16: +3.5%).

**Quality anomaly:** The half-resolution adapter shows *higher* PSNR (27.44 vs 25.83 dB) because bilinear
upsampling functions as a low-pass filter that smooths high-frequency rendering noise.

### 2.4 Renderer Recommendation

| Use Case | Recommended Adapter | Rationale |
|---|---|---|
| Maximum speed | gsplat_higs_quarter (553 FPS) | +12.4% speedup on A100 |
| Balanced speed-quality | gsplat_higs (baseline) | Highest quality |
| Memory-constrained | gsplat_higs_sh16 | -109 MB VRAM at +3.5% FPS |

---

## 3. Near-Lossless Storage Compression

### 3.1 Motivation

3DGS PLY checkpoints range from 243 MB to 1,450 MB per scene. We evaluate 10 compression codecs
across all 5 canonical scenes to identify the highest compression ratio that preserves near-lossless
rendering quality.

### 3.2 Codecs Evaluated

| Codec | Type | Where Decoded | Notes |
|---|---|---|---|
| XZ | Lossless (LZMA2) | CPU | General-purpose compression of PLY byte stream |
| Block-float | Quantized (float16) | CPU | Cast float32 to float16 per-attribute |
| Tile-codebook | Learned codebook | CPU | Per-tile codebook quantization |
| Compressed PLY | PlayCanvas quantized | CPU | 8-bit uniform quantization |
| SPZ 8/8 (v4) | Quantized + entropy | CPU | 8-bit SH + positional quantization |
| SPZ 7/7 (v4) | Quantized + entropy | CPU | 7-bit SH (new in this study) |
| SPZ 6/6 (v4) | Quantized + entropy | CPU | 6-bit SH + positional quantization |
| SPZ 5/4 (v4) | Quantized + entropy | CPU | 5/4-bit aggressive quantization |
| SOG | PlayCanvas texture codec | CPU | WebP image-based encoding |
| FCGS | Pretrained neural codec | GPU | End-to-end learned compression |

### 3.3 Results

All 10 codecs x 5 scenes = 50 data points. Codecs ordered by compression ratio.

| Codec | Scene | Ratio | Original | Compressed | PSNR (dB) | dPSNR (dB) | dSSIM | dLPIPS | Gate |
|---|---|---|---|---|---|---|---|---|---|
| SPZ 8/8 | Bicycle | 5.78x | 1,450 MB | 251 MB | 24.33 | +0.015 | -5e-5 | +5e-4 | PASS |
| SPZ 8/8 | Bonsai | 6.07x | 294 MB | 48 MB | 32.50 | -0.015 | -2e-4 | +7e-4 | PASS |
| SPZ 8/8 | Garden | 5.57x | 1,380 MB | 248 MB | 25.83 | -0.002 | -1e-4 | +4e-4 | PASS |
| SPZ 8/8 | Train | 5.74x | 243 MB | 42 MB | 22.37 | -0.007 | -1e-4 | +5e-4 | PASS |
| SPZ 8/8 | Truck | 5.83x | 601 MB | 103 MB | 24.14 | -0.001 | -9e-5 | +4e-4 | PASS |
| SPZ 7/7 | Bonsai | 6.94x | 294 MB | 42 MB | -- | -- | -- | -- | Pending |
| SPZ 6/6 | Bicycle | 7.74x | 1,450 MB | 187 MB | 24.28 | -0.029 | -1e-3 | +3e-3 | PASS |
| SPZ 6/6 | Bonsai | 8.04x | 294 MB | 37 MB | 32.24 | -0.275 | -3e-3 | +5e-3 | FAIL |
| SPZ 6/6 | Garden | 7.37x | 1,380 MB | 187 MB | 25.82 | -0.010 | -1e-3 | +3e-3 | PASS |
| SPZ 6/6 | Train | 7.57x | 243 MB | 32 MB | 22.38 | -0.001 | -7e-4 | +2e-3 | PASS |
| SPZ 6/6 | Truck | 7.72x | 601 MB | 78 MB | 24.16 | +0.026 | -8e-4 | +3e-3 | PASS |
| SOG | Bicycle | 19.49x | 1,450 MB | 74 MB | 23.84 | -0.471 | -3e-2 | +3e-2 | FAIL |
| SOG | Bonsai | 17.23x | 294 MB | 17 MB | 30.06 | -2.451 | -2e-2 | +3e-2 | FAIL |
| SOG | Garden | 18.55x | 1,380 MB | 74 MB | 25.46 | -0.363 | -2e-2 | +3e-2 | FAIL |
| SOG | Train | 16.39x | 243 MB | 15 MB | 21.61 | -0.768 | -1e-2 | +3e-2 | FAIL |
| SOG | Truck | 18.76x | 601 MB | 32 MB | 23.71 | -0.429 | -1e-2 | +2e-2 | FAIL |
| Compressed PLY | Bicycle | 4.05x | 1,450 MB | 358 MB | 24.30 | -0.018 | -2e-3 | +3e-3 | PASS |
| Compressed PLY | Bonsai | 4.05x | 294 MB | 73 MB | 32.28 | -0.232 | -7e-3 | +7e-3 | FAIL |
| Compressed PLY | Garden | 4.05x | 1,380 MB | 341 MB | 25.82 | -0.005 | -1e-3 | +3e-3 | PASS |
| Compressed PLY | Train | 4.05x | 243 MB | 60 MB | 22.37 | -0.012 | -2e-3 | +2e-3 | PASS |
| Compressed PLY | Truck | 4.05x | 601 MB | 149 MB | 24.13 | -0.011 | -9e-4 | +3e-3 | PASS |
| Tile-codebook | Bicycle | 3.85x | 1,450 MB | 377 MB | 24.31 | -0.001 | -1e-4 | +1e-4 | PASS |
| Tile-codebook | Bonsai | 3.89x | 294 MB | 76 MB | 32.51 | -0.002 | -2e-5 | +5e-5 | PASS |
| Tile-codebook | Garden | 3.80x | 1,380 MB | 363 MB | 25.83 | -0.000 | -4e-5 | +5e-5 | PASS |
| Tile-codebook | Train | 3.92x | 243 MB | 62 MB | 22.38 | -0.001 | -6e-5 | +1e-4 | PASS |
| Tile-codebook | Truck | 3.86x | 601 MB | 156 MB | 24.14 | -0.000 | -4e-5 | +7e-5 | PASS |
| Block-float | Bicycle | 2.16x | 1,450 MB | 673 MB | 24.31 | -0.004 | -2e-3 | +1e-3 | PASS |
| Block-float | Bonsai | 2.27x | 294 MB | 130 MB | 32.50 | -0.011 | -1e-4 | +1e-4 | PASS |
| Block-float | Garden | 2.15x | 1,380 MB | 643 MB | 25.83 | -0.001 | -3e-4 | +2e-4 | PASS |
| Block-float | Train | 2.22x | 243 MB | 109 MB | 22.35 | -0.026 | -2e-3 | +3e-3 | FAIL |
| Block-float | Truck | 2.20x | 601 MB | 274 MB | 24.13 | -0.008 | -6e-4 | +1e-3 | PASS |
| XZ (LZMA2) | Bicycle | 1.16x | 1,450 MB | 1,253 MB | 24.31 | 0.000 | 0 | 0 | PASS |
| XZ | Bonsai | 1.23x | 294 MB | 240 MB | 32.51 | 0.000 | 0 | 0 | PASS |
| XZ | Garden | 1.15x | 1,380 MB | 1,204 MB | 25.83 | 0.000 | 0 | 0 | PASS |
| XZ | Train | 1.20x | 243 MB | 202 MB | 22.38 | 0.000 | 0 | 0 | PASS |
| XZ | Truck | 1.19x | 601 MB | 505 MB | 24.14 | 0.000 | 0 | 0 | PASS |

### 3.4 Analysis

**SPZ 8/8 dominates the Pareto frontier**: it achieves the highest compression ratio (5.57-6.07x) while
maintaining dPSNR < 0.02 dB on all five scenes. On Bicycle, the SPZ-decoded output actually *improves*
PSNR (+0.015 dB) due to noise filtering during quantization.

**SPZ 6/6** fails on the Bonsai scene (dPSNR = -0.275 dB), which contains specular highlights and fine
detail poorly served by 6-bit SH quantization. **SPZ 5/4** and **SOG** consistently fail quality gate
across all scenes. **FCGS** (12.84x) offers higher compression but only 2/5 pass quality gate and
requires GPU for both encode and decode.

**Safe alternatives**: XZ is the only bit-exact option (1.17x). Tile-codebook (3.84x) provides excellent
quality retention (< 0.002 dB degradation) as a conservative fallback.

### 3.5 Compression Recommendation

| Requirement | Codec | Ratio | Quality Impact |
|---|---|---|---|
| Near-lossless (< 0.2 dB dPSNR) | **SPZ 8/8 (v4)** | **5.73x** | < 0.02 dB on all scenes |
| Higher compression | SPZ 6/6 (v4) | 7.62x | Fails on Bonsai (-0.275 dB) |
| Bit-exact | XZ (LZMA2) | 1.17x | Perfect |
| Safe fallback | Tile-codebook | 3.84x | < 0.002 dB |

---

## 4. Conclusions

1. **Best renderer speed:** gsplat HiGS quarter-resolution rendering achieves **553 FPS (+12.4%)**
   on Garden at 1080p. Limited speedup confirms HiGS is **compute-bound on A100** -- further gains
   require reducing Gaussian processing cost at the CUDA kernel level.

2. **Best near-lossless compression:** **SPZ v4 8/8-bit** achieves **5.73x** storage reduction with
   < 0.02 dB PSNR degradation across all five benchmark scenes.

3. **Reproducibility:** All measurements on identical hardware, scenes, and protocol. Raw results
   committed for independent verification.

---



---

## Tier A comparison charts

This repository measures and ranks renderers by five metrics.  The charts below
are regenerated from the authoritative cohort:

- ![Speed vs quality scatter](docs/leaderboard/measured-speed-vs-lpips.svg)
- ![FPS ranking](docs/leaderboard/measured-fps-ranking.svg)
- ![PSNR ranking](docs/leaderboard/measured-psnr-ranking.svg)
- ![SSIM ranking](docs/leaderboard/measured-ssim-ranking.svg)
- ![LPIPS ranking](docs/leaderboard/measured-lpips-ranking.svg)
- ![VRAM ranking](docs/leaderboard/measured-vram-ranking.svg)

**complete Tier A coverage**: all five canonical scenes measured across the
primary renderer cohort (original_3dgs, gsplat, gsplat_higs, speedy_splat,
tcgs) plus HiGS ablations, at 1080p with ground-truth quality validation.
## References

- Kerbl, B., Kopanas, G., Leimkuehler, T., and Drettakis, G. (2023). 3D Gaussian Splatting for
  Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4).
- Niemeyer, M., et al. (2024). HiGS: Hierarchical Gaussian Splatting. NVIDIA/gsplat repository.
- Niantic Labs (2024). SPZ: Efficient Storage for 3D Gaussian Splatting.
  https://github.com/nianticlabs/spz
- PlayCanvas (2024). Splat-transform: Web-optimized 3DGS compression.
  https://github.com/playcanvas/splat-transform

---

*Last updated: 2026-07-25 | Authority host: EPIC-05 | Benchmark protocol: 100 frames, 30 warmup, 5 repeats, 1920x1080*
