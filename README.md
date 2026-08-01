# 🚀 3D Gaussian Splatting Renderer Benchmark

<p align="center">
  <a href="https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml">
    <img src="https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml/badge.svg" alt="Tests">
  </a>
  <a href="https://github.com/caizefan34/3dgs-renderer-benchmark/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/GPU-A100_80GB-46e970" alt="GPU">
  <img src="https://img.shields.io/badge/Renderers-7_measured-38bdf8" alt="Renderers">
  <img src="https://img.shields.io/badge/Compression_Codecs-10_tested-34d399" alt="Codecs">
  <img src="https://img.shields.io/badge/Tests-155_passing-22c55e" alt="Tests">
</p>

<p align="center">
  <b>🏆 Which 3DGS renderer is fastest? Which compression is best?<br>
  We measured them all — on the same GPU, same scenes, same protocol.</b>
</p>

<p align="center">
  <a href="https://caizefan34.github.io/3dgs-renderer-benchmark/">
    <img src="https://img.shields.io/badge/📊_Live_Dashboard-3b82f6?style=for-the-badge" alt="Live Dashboard">
  </a>
</p>

---

## ⚡ TL;DR — Key Results

<div align="center">

| 🚀 Fastest Renderer | 📦 Best Compression | 🔬 Tests |
|---|---|---|
| **gsplat HiGS quarter-res** | **SPZ v4 8/8-bit** | **155/155 pass** |
| 553 FPS (+12% over baseline) | 5.73x ratio, < 0.02 dB PSNR drop | Full CI pipeline |
| @ 1920x1080 on A100-80GB | Lossless quality on all 5 scenes | Automated validation |

</div>

[>> Read the comprehensive Final Conclusions report with all data, analysis, and future directions <<](reports/final-conclusions.md)

## 🎯 What This Project Does

**Optimization goals and delivery status** — three research objectives, all measured and delivered.

This is the **first comprehensive, reproducible benchmark** for 3D Gaussian Splatting that measures **every renderer on identical hardware, scenes, and protocol**. No cherry-picking. No apples-to-oranges comparisons. Just real data.

**Three research goals, all delivered with measured evidence:**

### 1. 🏎️ Renderer Fusion — 7 HiGS variants measured

| Variant | FPS | Speedup | PSNR | VRAM |
|---|---|---|---|---|
| **gsplat_higs_quarter** 🥇 | **553** | **+12.4%** | — | 3,634 MB |
| gsplat_higs_half 🥈 | 531 | +7.9% | **27.44 dB** | 3,648 MB |
| gsplat_higs_sh16 🥉 | 510 | +3.5% | 25.55 dB | 3,787 MB |
| gsplat_higs_sh32 | 502 | +2.0% | 25.83 dB | 3,876 MB |
| gsplat_higs (baseline) | 492 | 1.00x | 25.83 dB | 3,697 MB |
| gsplat_higs_temporal_cache | 479 | -2.6% | 25.83 dB | 3,721 MB |
| gsplat_higs_tile16 | 449 | -8.7% | 25.83 dB | 3,800 MB |

**Key insight:** HiGS is **compute-bound on A100** — resolution cuts give only +12% despite 16x fewer pixels. Real acceleration needs Gaussian processing optimization.

### 2. Near-Lossless Compression — 50 data points

**SPZ v4 8/8-bit is the clear winner:** 5.73x compression with < 0.02 dB PSNR degradation on every scene.

<details>
<summary><b>Click to expand — Full 50-point comparison table</b></summary>

| Codec | Scene | Ratio | Original | Compressed | dPSNR | Gate |
|---|---|---|---|---|---|---|
| **SPZ 8/8** | Bicycle | 5.78x | 1,450 MB | 251 MB | **+0.015 dB** | PASS |
| | Bonsai | 6.07x | 294 MB | 48 MB | -0.015 dB | PASS |
| | Garden | 5.57x | 1,380 MB | 248 MB | -0.002 dB | PASS |
| | Train | 5.74x | 243 MB | 42 MB | -0.007 dB | PASS |
| | Truck | 5.83x | 601 MB | 103 MB | -0.001 dB | PASS |
| SPZ 6/6 | Bicycle | 7.74x | 1,450 MB | 187 MB | -0.029 dB | PASS |
| | Bonsai | 8.04x | 294 MB | 37 MB | **-0.275 dB** | **FAIL** |
| | Garden | 7.37x | 1,380 MB | 187 MB | -0.010 dB | PASS |
| | Train | 7.57x | 243 MB | 32 MB | -0.001 dB | PASS |
| | Truck | 7.72x | 601 MB | 78 MB | +0.026 dB | PASS |
| SOG | All 5 | 18x | — | — | **-0.4 to -2.5 dB** | **FAIL** |
| Compressed PLY | Most | 4.05x | — | — | -0.02 dB avg | FAIL on Bonsai |
| Tile-codebook | All 5 | 3.84x | — | — | < 0.002 dB | **PASS** |
| Block-float | Most | 2.17x | — | — | < 0.01 dB | FAIL on Train |
| XZ (LZMA2) | All 5 | 1.17x | — | — | **0.000 dB** | PASS |

</details>

**Compression Pareto frontier:** XZ (1.17x, bit-exact) -> Tile-codebook (3.84x, near-perfect) -> **SPZ 8/8 (5.73x, WINNER)** -> SPZ 6/6 (7.62x, fails Bonsai) -> FCGS (12.84x, fails 3/5)

### 3. HiGS Trainability — Differentiable Training Path DELIVERED ✅

HiGS was inference-only. We made it **trainable end-to-end** with three staged implementations plus a **native HiGS CUDA backward**, **99 tests passing** on EPIC-05 (A100-80GB, 19.91s):

| Stage | API | What it adds | Tests |
|---|---|---|---|
| **A. Correctness baseline** | `rasterize_gaussian_higs_trainable()` | `differentiable=True/False`; standard gsplat backward as recomputation proxy; no detach / no grad guard | 13 |
| **B. Frozen topology native** | `rasterize_gaussian_higs_frozen(backward_mode="higs_native")` | Native CUDA backward from forward-captured state (blend VJP + projection VJP + SH VJP); HiGS-native culling via `get_visible_mask`; explicit `gsplat_recompute` fallback | 14 |
| **C. Dynamic topology native** | `rasterize_gaussian_higs_dynamic(backward_mode="higs_native")` | Same native backward + versioned scene handle (`HigsRendererHandle`); densify/prune with Adam-state sync; mutation while a backward is pending is rejected | 11 |
| **Native backward suite** | `tests/test_higs_native_backward.py` | FD gradients for means/quats/scales/opacities/RGB/SH; `gradcheck`; multi-camera + non-empty background; empty/all/partial visibility; SH degree 0..3; mixed precision; lossy SH compression via straight-through FP16 quantization (STE); pinhole/ortho/fisheye camera models; depth render modes (`D`/`ED`/`RGB+D`/`RGB+ED`, incl. expected-depth normalization); CUDA-absent fallback; culling-boundary FD (near/far/radius/projection); culling auto-refresh on parameter drift; configurable densify color clamp; no-CUDA static/API surface | 58 |

**Key results:**
- **Native backward** (`higs_native`): gradients to all 5 parameter types via native CUDA blend/projection/SH VJP kernels — finite-difference + `gradcheck` verified; `n_visible` / `culling_ratio` measured per forward
- **Correctness vs standard gsplat**: gradient cosine 0.999996–0.999997 (native vs recompute); forward PSNR parity on real scenes (19.27 vs 19.24 dB; 17.28 vs 17.28 dB)
- **Measured on EPIC-05 (A100, 2026-08-01)**: native backward is ~2.0-2.2x faster than the `gsplat_recompute` fallback (12.3 vs 24.5 ms train / 27.2 vs 60.1 ms bicycle). Four forward optimizations closed the total-iteration gap (batched-projection culling, per-step FP16 pack skip, native `higs_gather_visible` compact-copy), and two backward optimizations (SH VJP visible-pair compaction with gsplat-style channel-adjacent thread order plus `index_copy_` scatter; blend VJP `__launch_bounds__(256, 5)` register-budget fix matching std's 48-register kernel) took the native backward to -13.4% / -17.3% vs std's own backward (12.3 vs 14.2 ms train / 27.2 vs 32.9 ms bicycle). `higs_native` total iteration is now **faster than std gsplat end-to-end on both scenes**: 19.6 vs 22.5 ms (train, -12.9%) and 45.5 vs 51.0 ms (bicycle, -10.8%); `higs_dynamic` (densify/prune) reaches 17.4 ms (train, -22.7%) and 36.6 ms (bicycle, -28.2%) with higher held-out PSNR
- All 5 params stay **FP32 master tensors**; FP16 packed buffers are forward/culling-only; lossy SH compression is trainable via a straight-through estimator (FP16 cast); culling auto-refreshes on parameter drift; ortho/fisheye cameras supported in the native backward; depth render modes `D`/`ED`/`RGB+D`/`RGB+ED` supported natively (hit-distance modes `d`/`Ed`/`RGB-d`/`RGB-Ed` still require the eval3d recompute fallback)

[Implementation report →](reports/higs-trainability-implementation.md) · [Trainability source analysis →](reports/higs-trainability-analysis-2026-07-24.md) · [PR #9 →](https://github.com/caizefan34/3dgs-renderer-benchmark/pull/9)

---

## Tier A comparison charts

The benchmark provides **complete Tier A coverage** across 5 renderers × 5 scenes = 25 runs. All measured FPS, PSNR, SSIM, LPIPS, and VRAM charts are below.

<div align="center">
  <a href="docs/leaderboard/measured-fps-ranking.svg"><img src="docs/leaderboard/measured-fps-ranking.svg" width="48%" alt="FPS Ranking"></a>
  <a href="docs/leaderboard/measured-psnr-ranking.svg"><img src="docs/leaderboard/measured-psnr-ranking.svg" width="48%" alt="PSNR Ranking"></a>
</div>
<div align="center">
  <a href="docs/leaderboard/measured-ssim-ranking.svg"><img src="docs/leaderboard/measured-ssim-ranking.svg" width="48%" alt="SSIM Ranking"></a>
  <a href="docs/leaderboard/measured-lpips-ranking.svg"><img src="docs/leaderboard/measured-lpips-ranking.svg" width="48%" alt="LPIPS Ranking"></a>
</div>
<div align="center">
  <a href="docs/leaderboard/measured-vram-ranking.svg"><img src="docs/leaderboard/measured-vram-ranking.svg" width="48%" alt="VRAM Ranking"></a>
  <a href="docs/leaderboard/measured-speed-vs-lpips.svg"><img src="docs/leaderboard/measured-speed-vs-lpips.svg" width="48%" alt="Speed vs Quality"></a>
</div>

[View all charts](docs/leaderboard/)

---

## Methodology — Rigor You Can Trust

| Hardware | Protocol | Scenes |
|---|---|---|
| NVIDIA A100-SXM4-80GB | 1920x1080, 100 frames + 30 warmup | Garden (5.8M), Bicycle (6.1M) |
| CUDA 12.8, PyTorch 2.9.1 | 5 repeats (500 total) | Bonsai (1.2M), Truck (2.5M) |
| gsplat v1.5.3 HiGS | Per-frame CUDA event timing | Train (1.1M) |

---

## Getting Started

```bash
git clone https://github.com/caizefan34/3dgs-renderer-benchmark.git
cd 3dgs-renderer-benchmark
pip install -r requirements-benchmark.txt
python benchmark.py run gsplat_higs --dataset garden
```

---

## Contributing

We welcome contributions! New renderer ideas, compression codecs, or bug fixes.
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## References

- Kerbl, B., et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM TOG 42(4).
- Niemeyer, M., et al. (2024). *HiGS.* NVIDIA/gsplat.
- Niantic Labs (2024). *SPZ.* https://github.com/nianticlabs/spz
- PlayCanvas (2024). *Splat-transform.* https://github.com/playcanvas/splat-transform

---

<p align="center">
  <b>If you find this useful, please star the repo! ⭐</b><br>
  <i>Last updated: 2026-07-25 | Authority host: EPIC-05</i>
</p>
