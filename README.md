# 🚀 3D Gaussian Splatting Renderer Benchmark

<p align="center">
  <a href="https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml">
    <img src="https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml/badge.svg" alt="Tests">
  </a>
  <a href="https://github.com/caizefan34/3dgs-renderer-benchmark/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <img src="https://img.shields.io/github/stars/caizefan34/3dgs-renderer-benchmark?style=flat&label=Stars&color=gold" alt="Stars">
  <img src="https://img.shields.io/badge/GPU-A100_80GB-46e970" alt="GPU">
  <img src="https://img.shields.io/badge/Renderers-7_measured-38bdf8" alt="Renderers">
  <img src="https://img.shields.io/badge/Compression_Codecs-10_tested-34d399" alt="Codecs">
  <img src="https://img.shields.io/badge/Tests-155_passing-22c55e" alt="Tests">
</p>

<p align="center">
  <b>Which 3DGS renderer is fastest? Which compression is best?<br>
  We measured them all — on the same GPU, same scenes, same protocol.</b>
</p>

<p align="center">
  <a href="https://caizefan34.github.io/3dgs-renderer-benchmark/">
    <img src="https://img.shields.io/badge/📊_Live_Dashboard-3b82f6?style=for-the-badge" alt="Live Dashboard">
  </a>
</p>

<p align="center">
  <img src="docs/assets/social-preview.png" alt="3DGS Renderer Benchmark — every renderer measured on the same GPU, scenes and protocol" width="100%">
</p>

**A reproducible benchmark of 3D Gaussian Splatting renderers and compression codecs — measured on identical hardware, scenes, and protocol, so the speed-vs-quality trade-offs are real and directly comparable.**

- **What** — 5 renderer families × 5 scenes (7 HiGS variants + Speedy-Splat + TC-GS + Original 3DGS) and 10 compression codecs (SPZ, FCGS, …), all on one A100-80GB with the same checkpoints and camera trajectories.
- **Why** — papers report numbers on different GPUs and datasets, so you can never trust a head-to-head. This repo normalizes everything (evidence tiers A/B/C) and publishes raw data + charts.
- **Who** — researchers picking a renderer or compression method, engineers deploying 3DGS, and anyone who wants to **train HiGS end-to-end** (native CUDA backward, see below).

## Quickstart

**Prerequisites:** Python 3.10+, a CUDA-capable GPU (validated on NVIDIA A100-80GB). Everything except the CUDA extensions is CPU-safe and tested in CI.

```bash
git clone https://github.com/caizefan34/3dgs-renderer-benchmark.git
cd 3dgs-renderer-benchmark
pip install -r requirements-benchmark.txt
python benchmark.py run gsplat_higs --dataset garden   # any renderer in benchmark/renderers.json
```

That's it — the CLI downloads the scene, runs the suite, and writes results under `results/`. Want the full tour first? Browse the [live dashboard](https://caizefan34.github.io/3dgs-renderer-benchmark/) or start with [docs/README.md](docs/README.md).

> **Run the test suite:** `python -m unittest discover -s tests -v` (155 tests, CPU-safe). For the full suite including the pytest-style HiGS tests, run `python -m pytest tests -q` (HiGS tests skip cleanly when the CUDA extension is unavailable).


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

This is the **independent, reproducible benchmark** for 3D Gaussian Splatting: every renderer is measured on identical hardware, scenes, and protocol. No cherry-picking. No apples-to-oranges comparisons. Just real data.

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

HiGS was inference-only. We made it **trainable end-to-end** with three staged implementations plus a **native HiGS CUDA backward** (100 tests passing on EPIC-05, A100-80GB):

| Stage | API | What it adds | Tests |
|---|---|---|---|
| **A. Correctness baseline** | `rasterize_gaussian_higs_trainable()` | `differentiable=True/False`; standard gsplat backward as recomputation proxy; no detach / no grad guard | 13 |
| **B. Frozen topology native** | `rasterize_gaussian_higs_frozen(backward_mode="higs_native")` | Native CUDA backward from forward-captured state (blend + projection + SH VJP); HiGS-native culling; explicit `gsplat_recompute` fallback | 14 |
| **C. Dynamic topology native** | `rasterize_gaussian_higs_dynamic(backward_mode="higs_native")` | Same native backward + versioned scene handle; densify/prune with Adam-state sync; topology mutation while a backward is pending is rejected | 11 |
| **Native backward suite** | `tests/test_higs_native_backward.py` | FD + `gradcheck` for means/quats/scales/opacities/colors-SH; multi-camera + non-empty background; SH degree 0..3; mixed precision; SH compression STE; pinhole/ortho/fisheye; depth modes; CUDA-absent fallback | 61 |

**Key results**
- **Correctness**: gradient cosine 0.999996–1.0 vs standard gsplat; forward PSNR parity on real scenes (19.27 vs 19.24 dB); all 5 parameter types stay FP32 master tensors (FP16 packed buffers are forward/culling-only).
- **Speed (EPIC-05 A100)**: native backward is **~2× faster** than the `gsplat_recompute` fallback; end-to-end total iteration vs std gsplat — `higs_native` **-9.2% / -10.8%** (train / bicycle), `higs_dynamic` **-19.5% / -23.2%**; `radius_clip=3.0` adds another **-18% ~ -26%** with equal-or-better PSNR/SSIM/LPIPS.
- **Tile-sampled training (M4, merged [PR #18](https://github.com/caizefan34/3dgs-renderer-benchmark/pull/18))**: each step renders/blends only a selected tile subset, so forward isect/radix and the backward blend grid scale with the realized tile fraction (nominal r=0.35 ≈ 27-31% of tiles). EPIC-05 A100, 3000-step recipe, 1920x1080 × 4 train cams:

| Scene | Config | Speedup | PSNR (Δ vs full) | SSIM | LPIPS (Δ) | Seeds |
|---|---|---|---|---|---|---|
| train | full r=1.0 (reference) | 1.00x | 16.673 | 0.6267 | 0.3678 | 3 |
| train | error_guided r=0.35 + λ=0.7 | **1.82x** | 17.074 (+0.40) | 0.6295 (+0.003) | 0.3870 (+0.019) | 3 |
| train | error_guided r=0.35 + λ=0.7 + `--lpips-full-res` | 1.80x | 17.190 (+0.52) | 0.6269 | 0.3838 (+0.016) | 1 |
| bicycle | full r=1.0 (reference) | 1.00x | 16.024 | 0.3908 | 0.4795 | 3 |
| bicycle | error_guided r=0.35 + λ=0.7 + `--lpips-full-res` | **1.98x** | 15.965 (-0.06, parity) | 0.3891 | 0.5298 (+0.050) | 3 |
| bicycle | error_guided r=0.35 + λ=0.7 + `--lpips-full-res` + `--anchor-densify` | **1.87x** | 15.926 (-0.10, parity) | 0.3918 (+0.001) | 0.5179 (+0.038) | 3 |
| bicycle | error_guided r=0.35 + λ=0.7 + `--lpips-full-res` + `--anchor-densify` + `--anchor-densify-every 2` | **1.92x** | 15.940 (-0.08, parity) | 0.3916 | 0.5257 (+0.046) | 3 |
| garden | full r=1.0 (reference) | 1.00x | 18.733 | 0.5007 | 0.3987 | 3 |
| garden | error_guided r=0.35 + λ=0.7 + `--lpips-full-res` + `--anchor-densify` | **1.98x** | 18.192 (-0.54) | 0.4784 (+0.009) | 0.4361 (+0.037) | 3 |
| garden | error_guided r=0.35 + λ=0.7 + `--lpips-full-res` + `--anchor-densify` + `--anchor-densify-every 2` | **2.06x** | 18.068 (-0.66) | 0.4731 | 0.4427 (+0.044) | 3 |

  Recommended operating point: `--tile-sampling-ratio 0.35 --sampling-mode error_guided --error-lambda 0.7 --lpips-full-res` (realized sr≈0.27-0.31; with lr-decay + densify-window + LPIPS regularization, see [run script](scripts/higs/run_m4_a100_retest.sh)). **Honest bound:** the high-N scene gap at ≥1.8x — bicycle LPIPS +0.050±0.002 and garden PSNR -0.76 dB / LPIPS +0.050 (M5 matrix, [results](results/higs-round42/m5-summary.json)) — was robust across λ, full-res LPIPS, a 6000-step convergence probe, and four closed levers (rounds 43-46: alpha sweep, densify grad-accum, densify cadence, LPIPS weight) until `--anchor-densify` (round-47) narrowed it to bicycle LPIPS +0.038 / garden PSNR -0.54 dB / LPIPS +0.037 at 1.98x/1.87x (still ≥1.8x; full-res densify steps give dup/clone the true full-frame gradient, ~6% cost); round-50 `--anchor-densify-every 2` (3-seed) keeps ~half that gain at ~half the cost (garden +0.10 dB / LPIPS +0.044 at 2.06x, bicycle LPIPS +0.046 at 1.92x) — every 1 stays the quality-max opt-in (LPIPS +0.037), every 4 collapses garden back to eg (single-seed screening, not recommended). low/mid-N scenes (train, truck, bonsai) hold quality parity. Round-51 (M6 baseline 1/3) shows the error-guided sampling advantage is scene-dependent: vs the ICCV random-tile baseline (`--sampling-mode uniform`) at **matched realized sampling ratio**, error-guided dominates low-N train (PSNR +0.32 dB, LPIPS -0.002, and -5.6% wall time) but high-N garden/bicycle are not worse under random tiles (PSNR parity, LPIPS 0.004-0.009 *better* for random tiles at +0.7-4.7% cost; at the same nominal r=0.35 random tiles render ~15-20% more tiles and cost +7-12%, so nominal-ratio comparisons are not apples-to-apples). `--anchor-densify` is an opt-in for high-N scenes only: a train single-seed probe shows +6% time (1.82x → ~1.72x, below the 1.8x bar) with no quality benefit. All conclusions are multi-seed (bicycle same-seed reruns vary ±0.1-0.3 dB from CUDA-atom non-determinism amplified by densify/prune). Round-52 (M6 baseline 2/3, Turbo-GS-style progressive resolution, 3-seed) adds a high-speed arm: `--res-schedule 0.5:0,1.0:1500` trains half-res for the first 1500 steps with eval always full-res — train 16.94 (-0.15 dB) @ **2.07x**, garden 18.07 (parity) @ **2.49x**, bicycle 15.59 (-0.35 dB) @ **2.47x** vs full, with LPIPS +0.0017 / +0.0062 / -0.0053 vs the eg/every2 baselines (garden LPIPS falls back to the eg bound because the full-res LPIPS signal is lost in the half-res stage); the `--res-schedule-full-signal` variant (coarse-stage LPIPS + anchor-densify steps at full res) is negative — train parity but garden -0.35 dB / LPIPS +0.027 and bicycle -0.98 dB / +0.011, slower (20.3/20.9 ms) — because densify runs every 5 steps and alternating full/half-res densify events destabilize high-N scenes (more, worse-placed Gaussians). The quality frontier at >=1.8x remains the round-50 `--anchor-densify-every 2` opt-in. Round-53 (M6 baseline 3/3) adds the Speedy-Splat-style sparse-pixel training-signal baseline (`--sampling-mode sparse_pixel --pixel-sampling-ratio 0.35`, full-frame render, 3-seed): at matched ~35% pixel coverage quality returns to near full — garden 18.57 (-0.16 dB vs full, LPIPS 0.4081 vs 0.4427 tile-every2), bicycle 16.12 (+0.10 vs full, LPIPS 0.4871 vs 0.5257), train 16.58 (-0.09, LPIPS 0.3755) — with ~1.0x wall speed (no pixel-sparse rasterizer in frozen gsplat; signal-only baseline). This isolates the high-N tile-sampling bound as sampling-correlation noise (16x16 macro-tile granularity) rather than pixel count. Round-54 (3-seed) tested the in-harness de-correlation lever (stratified tiles): quality-equivalent to uniform tiles at matched sr (garden LPIPS 0.4346 vs 0.4335; bicycle 0.5103 vs 0.5106), i.e. the loss is tile-granularity correlation that stratified selection does not fix — closing the sampling levers (error_guided / uniform / stratified) and pointing to finer-than-tile sampling in the rasterizer as the only remaining quality lever. Round-57 (3-seed, 18 runs) implemented that renderer-level lever with upstream gsplat's sparse-pixel kernels (new `higs_sparse_px` backend + `--pixel-raster-ratio`): at ~40% pixel coverage quality recovers to near full (bicycle PSNR +0.51 dB / LPIPS 0.4846 vs its dense baseline 0.4845, garden LPIPS +0.011 vs +0.044 for tile every-2) but the wall-clock lever is structurally bounded at 1.06-1.09x — iid pixel masks keep nearly every tile active, so intersection cost is unchanged and only the per-pixel blend loop scales with pixel count ([r57-summary](results/higs-round57/r57-summary.json)). Round-58 (3-seed, 9 runs) closed the last frontier cell — **720p × progressive-resolution is a clean negative**: vs plain 720p eg it is slower/equal (0.89-1.09x; train 10.5 vs 9.3 ms) with consistent quality loss (train -0.20 dB / LPIPS +0.007, garden -0.37 / +0.033, bicycle -0.46 / +0.011) because at 720p the per-step cost is already at the resolution-invariant per-Gaussian floor (a 360p-vs-720p probe measures only ~0-9% per-step saving, [probe script](scripts/higs/run_round58_res_scaling_probe.sh)), so the progressive coarse stage has nothing to harvest and its coarse-stage densify dynamics cost quality; the speed-max cell is plain 720p + eg r=0.35 (R56: 1.56-1.88x vs same-res full, 2.4-2.7x wall vs 1080p full) and progressive-res stays a 1080p-only lever (R52; [r58-summary](results/higs-round58/r58-summary.json)). Full analysis: [research report](reports/higs-training-speedup-research-2026-08-03.md) · [M4 results](results/higs-round41d/m4-summary.json). Round-59 (kernel-profile-motivated, 36-run sweep + 12-run bicycle variance audit) adds the first per-Gaussian-floor speed lever: a camera-set-keyed cull-mask cache (`--cull-interval 4` opt-in; also fixes the pre-existing eval/train mask contamination and the never-wired `--cull-interval` flag, plus `--cull-interval-schedule "K:start,..."` for phase gating) cuts 1.8-3.9% wall on the 720p eg recipe with LPIPS/SSIM unchanged everywhere; the apparent bicycle PSNR loss is the scene's inherent run variance (the K=1 baseline itself collapsed to 14.57 dB on seed 4; at n=6 per arm all arms are within noise), so K4 is a quality-neutral speed opt-in ([r59-summary](results/higs-round59/r59-summary.json)).

[Implementation report →](reports/higs-trainability-implementation.md) · [Trainability source analysis →](reports/higs-trainability-analysis-2026-07-24.md) · [PR #9](https://github.com/caizefan34/3dgs-renderer-benchmark/pull/9)

---

## 🗂️ Repository Layout

```text
benchmark.py              CLI entry point (`benchmark run|report|prepare|...`); implemented by src/benchmark_cli.py
benchmark/                Matrix v2 configuration: suite.json, protocol.json, datasets/, schemas/ (see benchmark/README.md)
benchmark_suite/          Legacy compatibility copy of the suite config (deprecated -> benchmark/)
data/                     Curated inputs committed to git: camera_presets/, examples/, official training datasets
datasets/                 Local download/processing cache (mostly git-ignored), see datasets/README.md
src/                      Python package: CLI, framework, adapters, analysis, workers, see src/README.md
scripts/                  Linux/EPIC-05 shell scripts and environment setup, see scripts/README.md
tests/                    unittest + pytest suites (CPU-safe), see tests/README.md
docs/                     Human documentation + generated leaderboard (GitHub Pages), see docs/README.md
reports/                  Run reports + HiGS research/round logs, see reports/README.md
results/                  Measured/reproduced/paper evidence artifacts, see results/README.md
plots/                    Generated publication plots, see plots/README.md
patches/                  HiGS differentiable patch (gsplat source), see patches/README.md
artifacts/                Environment setup + training configs; renderer sources git-ignored, see artifacts/README.md
baselines/                Regression threshold baselines
community/                Submission template for external contributors
third_party_patches/      Patches applied to third-party renderer code
schemas/                  Top-level JSON schemas (benchmark_result, leaderboard, regression_report)
docker/ + docker-compose.yml  Reproducible container definition
.github/                  CI workflows: ci.yml, benchmark-regression.yml, deploy-pages.yml
```

Deep dive: [docs/repository-architecture.md](docs/repository-architecture.md).

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

## 🗺️ Project status & roadmap

- ✅ **Benchmark (Tier A)** — 5 renderers × 5 scenes measured on EPIC-05 A100; charts and raw data published ([leaderboard](docs/leaderboard/)).
- ✅ **Compression qualification** — SPZ 8/8 wins at 5.73x with < 0.02 dB PSNR drop; full Pareto frontier in the [compression reports](reports/README.md).
- ✅ **HiGS trainable end-to-end** — native CUDA backward, 100 tests, -9% ~ -24% training speedup vs std gsplat ([implementation report](reports/higs-trainability-implementation.md)).
- ✅ **Tile-sampled training (M4)** — **1.8-2.0x end-to-end training speedup** at quality parity on train + bicycle (3-seed; bicycle LPIPS +0.05 the sole honest bound) ([research report](reports/higs-training-speedup-research-2026-08-03.md) · [PR #18](https://github.com/caizefan34/3dgs-renderer-benchmark/pull/18)).
- ✅ **M5 multi-scene + multi-resolution matrix** — recommended op point on garden / bonsai / truck (3-seed): **1.74-2.12x**; truck quality beats full, bonsai within seed noise, garden shows the same high-N bound as bicycle ([M5 results](results/higs-round42/m5-summary.json)). Round-56 completed the resolution axis (540p/720p/1080p x train/garden/bicycle x 3-seed, 36 runs): speedup scales with resolution (540p 1.39-1.76x -> 720p 1.56-1.88x -> 1080p 1.80-2.06x), low/mid-N train stays a clear quality win at every resolution (PSNR +0.45..+1.03 dB), the high-N LPIPS bound is resolution-robust (garden +0.044..+0.053, bicycle +0.046..+0.050) but bicycle PSNR degrades at 540p (-1.93 dB) ([r56-summary](results/higs-round56/r56-summary.json)).
- 🚧 **Next** — close the high-N scene (garden/bicycle) quality bound at ≥1.8x; sampling-side levers are closed (alpha sweep and densify grad-accum both negative, [round-43/44](results/higs-round43/r43-summary.json)), the high-N quality bound is now narrowed by `--anchor-densify` (round-47, 3-seed: LPIPS +0.050 → +0.037/+0.038 on garden/bicycle at 1.98x/1.87x, garden PSNR gap halved) and the round-50 cost-quality frontier is explicit (3-seed: `--anchor-densify-every 2` keeps ~half the gain at ~half the cost — garden +0.10 dB / LPIPS +0.044 at 2.06x, bicycle LPIPS +0.046 at 1.92x — as the recommended default opt-in; every 1 = quality-max, every 4 collapses, not recommended); all other levers are closed as negatives (rounds 43-46); density-axis levers under anchor are also closed (round-48/49: densify-threshold and prune-threshold sweeps are flat or slightly negative, single-seed screening), so the high-N envelope is final at this recipe; M6 baselines 1/3 and 2/3 are done (round-51, ICCV random-tile loss: error-guided advantage limited to low/mid-N scenes at matched realized sr; high-N quality is tile-count-driven; round-52, Turbo-GS-style progressive resolution `--res-schedule 0.5:0,1.0:1500`: ~2.1-2.5x wall speed at 3-seed with PSNR parity on train/garden and a modest bicycle PSNR loss, garden/bicycle LPIPS cost explicit, and the coarse-stage full-res-signal variant a clear negative — alternating-resolution densify destabilizes high-N scenes; round-55 isolated the perceptual side with `--res-schedule-full-lpips` (full-res LPIPS steps only during the coarse stage, anchor densify at stage scale): also negative, 3-seed — garden/bicycle/train LPIPS 0.4545/0.5225/0.3920 vs plain-progressive 0.4489/0.5204/0.3888 at +4..+22% wall, so any full-res signal during the coarse stage is closed; plain progressive-resolution stays the best speed arm); M6 comparisons are now 3/3 done (round-53, Speedy-Splat-style sparse-pixel training signal: at matched ~35% pixel coverage with full-frame rendering, quality returns to near full — garden LPIPS 0.408 vs 0.443 (tile eg) / 0.399 (full), bicycle 0.487 vs 0.526 (tile eg) / 0.480 (full) — isolating the high-N tile-sampling bound as sampling-correlation noise, not pixel count; the arm is signal-only, ~1.0x wall speed because frozen gsplat has no pixel-sparse rasterizer); round-54 closed the in-harness sampling levers: stratified tiles are quality-equivalent to uniform tiles at matched sr (garden LPIPS 0.4346 vs 0.4335, bicycle 0.5103 vs 0.5106, 3-seed) — the high-N loss is tile-granularity correlation; round-57 (3-seed, 18 runs) then implemented the renderer-level finer-than-tile lever itself (new `higs_sparse_px` backend, upstream gsplat sparse-pixel kernels): quality recovers to near full at ~40% pixel coverage (bicycle PSNR +0.51 dB / LPIPS parity vs its own dense baseline, garden LPIPS +0.011), but wall speed is only 1.06-1.09x because iid pixel masks keep all tiles active and projection/SH/intersection/backward costs do not scale with pixel count — the pixel-sparse lever is a quality-recovery lever, not a speed lever; the >=1.8x operating point remains tile-level sampling (error_guided r=0.35); R58 closes the 720p×progressive cell as negative (no speed, quality cost — the per-step cost at 720p is already at the resolution-invariant per-Gaussian floor, probe-verified), so the frontier is complete: speed-max = 720p+eg (2.4-2.7x wall vs 1080p full), quality-max ≥1.8x = 1080p eg + anchor-densify-every-2; round-59 adds the cull-mask cache (camera-set-keyed, K4 opt-in) as the first per-Gaussian-floor speed increment (2-4% wall, LPIPS/SSIM neutral); round-60 adds the cull-masked Adam step (--masked-adam, benchmark/higs_masked_adam.py): a fused CUDA kernel runs the exact torch fused-Adam math only on rows whose train-forward union-visibility mask is True, skipping culled Gaussians (isolated probe: fused 4.2 ms -> masked 2.4 ms at 42% visible / 2.8 ms at 58%; correctness probe: params within 1-2 float32 ulps, frozen rows bit-identical); 3-seed x 3-scene sweep at 720p-eg: end-to-end train_ms -8% train / -34% garden / -41% bicycle, quality mostly improved (garden PSNR +1.90 dB / LPIPS -0.105 / SSIM +0.085, train neutral, bicycle PSNR/SSIM +0.05/+0.027 with LPIPS +0.017) - freezing out-of-view rows stops the zero-grad momentum drift that decays eval-relevant Gaussians, keeps total N higher while the rendered visible set shrinks ~half, making this the first quality-positive per-Gaussian-floor speed increment; round-61 closes the remaining per-Gaussian-floor speed candidates as quality-gated negatives (3-seed evidence): union-mask prune collapses PSNR (>=0.2 dB even with grace windows; union-invisible set is mid-migration geometry that later becomes needed); LPIPS train-loss work-size 256 (--lpips-work-size, [r61-summary](results/higs-round61/r61-summary.json)) = 1.03-1.05x train_ms but LPIPS +0.006..+0.013 / PSNR -0.03..-0.14 - the downscaled surrogate loss measurably degrades full-res eval quality; K4 x masked-Adam stack = 1.03-1.04x train_ms but PSNR -0.08..-0.23 (1-seed screen, same direction as round-59 K4 3-seed bicycle -0.34) - the round-60 k1 masked-Adam op point is the final per-Gaussian-floor frontier round-62 tests the mask/prune-decay decoupling ([--masked-adam-union-decay](benchmark/run_higs_train_benchmark.py), [r62 evidence](results/higs-round62)): per-step opacity decay on rows invisible in both train+eval masks lets stale geometry retire via the normal opacity prune - quality-positive in-wave (garden PSNR +0.14 / LPIPS -0.010 at 0.99, retires 1.66M rows) but fails the speed gate - the quality-valid config requires a fresh eval mask every densify (~1 ms/step full-res 3-cam eval forward), train_ms +6.3-7.2%, cross-scene non-robust, and stale masks collapse decay quality (PSNR 13.88 at 0.99) - closed as a quality-gated negative; round-63 revives the lever with projection-only eval-mask refresh (--masked-adam-union-decay-eval-proj, [r63 evidence](results/higs-round63/exp2)): the decay mask is projection-based and the projection cull mask is bitwise-identical to the full eval-forward mask (probe: 1.35 ms vs 13.5 ms, low-res rendering is slower not faster), so in-wave 3000-step 3-seed garden shows the same quality at +2.3% train_ms instead of +6.8% (PSNR +0.21 / LPIPS -0.009 vs R60) - exp3 cross-scene 3-seed confirms the lever is quality-safe everywhere (bicycle PSNR +0.22 / LPIPS -0.009 at -0.7% train_ms; train +0.03 / +0.001 at +4.9%; R62's 1-seed train -0.23 was seed noise, not reproduced at 3-seed) - the lever stays opt-in, now recommended as a quality opt-in for high-N scenes (garden/bicycle), not for low-N train, and the round-60 op point remains final; the recommended refresh for any decay config is now the projection path

---

## FAQ

**Does this run on my GPU?**
Measured on an NVIDIA A100-80GB (EPIC-05), but any CUDA GPU can run it. Numbers differ per machine — which is exactly why every result here is reported per-cohort with full protocol disclosure (see `docs/hardware.md`).

**What is HiGS?**
HiGS (Hierarchically Tiled Gaussian Splatting) is a macro-tile renderer inside gsplat: it sorts Gaussians by coarse macro-tiles, then rasterizes fine render tiles, preserving exact front-to-back compositing. It is the fastest renderer in this benchmark — and we made it **trainable end-to-end** with a native CUDA backward (see the HiGS section above and `reports/higs-trainability-implementation.md`).

**How is this different from other 3DGS benchmarks?**
Every renderer gets the same GPU, checkpoints, 100-camera trajectory, and measurement protocol. Evidence tiers (measured / reproduced / paper) never mix into one ranking. Details: [docs/methodology.md](docs/methodology.md).

**Can I add my own renderer or codec?**
Yes. Follow [docs/adding-a-renderer.md](docs/adding-a-renderer.md) and use `community/submission_template.json` for new submissions.

**Where is the raw data?**
`results/` holds the evidence artifacts (measured / reproduced / paper); the generated leaderboard is in `docs/leaderboard/` and published to GitHub Pages.

---

## Contributing

We welcome contributions! New renderer ideas, compression codecs, or bug fixes.
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🖋️ Citation

If this benchmark helps your research, please cite it:

```bibtex
@misc{cai2026gsrendererbenchmark,
  title        = {{3DGS Renderer Benchmark}: A Reproducible Benchmark Suite for 3D Gaussian Splatting Renderers},
  author       = {Cai, Zefan},
  year         = {2026},
  howpublished = {\url{https://github.com/caizefan34/3dgs-renderer-benchmark}},
  note         = {MIT License}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

---

## References

- Kerbl, B., et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM TOG 42(4).
- *HiGS: A Hierarchical Rendering Architecture for Real-Time 3D Gaussian Splatting.* NVIDIA Research / gsplat. arXiv:2606.00352 (2026).
- Niantic Labs (2024). *SPZ.* https://github.com/nianticlabs/spz
- PlayCanvas (2024). *Splat-transform.* https://github.com/playcanvas/splat-transform

---

<p align="center">
  <b>If you find this useful, please star the repo! ⭐</b><br>
  <i>Last updated: 2026-08-03 | Authority host: EPIC-05</i>
</p>
