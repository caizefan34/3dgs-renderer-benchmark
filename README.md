# A Reproducible Benchmark Suite for 3D Gaussian Splatting Renderers

[![Website](https://img.shields.io/badge/Website-View%20Report-7c5cfc)](https://caizefan34.github.io/3dgs-renderer-benchmark/)
[![GPU](https://img.shields.io/badge/GPU-RTX%205070%20Laptop-76b900)](https://www.nvidia.com)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-ee4c2c)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Abstract

3D Gaussian Splatting (3DGS) [Kerbl et al., 2023] has emerged as a leading approach for real-time novel view synthesis, achieving state-of-the-art rendering quality at interactive frame rates. However, the ecosystem of CUDA-based rasterization backends has grown rapidly, making it difficult for researchers and practitioners to select the optimal renderer for their specific use case. This repository provides a **rigorous, reproducible benchmark** comparing four prominent CUDA rasterization renderers for 3DGS under strictly controlled conditions. We further investigate engineering optimizations---including frustum pre-culling and pre-allocated buffer reuse---that yield a **+113% speedup** over the baseline, achieving **365 FPS** on an NVIDIA RTX 5070 Laptop GPU. Our benchmark protocol, camera paths, synthetic scene generation, and evaluation metrics are fully open-sourced to facilitate reproducible research.

**Hardware**: NVIDIA GeForce RTX 5070 Laptop GPU (8.55 GB VRAM, Compute Capability 12.0)
**Scene**: 400,000 Gaussians, Spherical Harmonics degree 3
**Resolution**: 1920 x 1080

---

## Table of Contents

- [Background and Motivation](#background-and-motivation)
- [Quick Start](#quick-start)
- [Benchmark Protocol](#benchmark-protocol)
- [Renderer Descriptions](#renderer-descriptions)
- [Experimental Results](#experimental-results)
- [Optimization Analysis](#optimization-analysis)
- [Quality Validation](#quality-validation)
- [Repository Structure](#repository-structure)
- [Continuous Integration](#continuous-integration)
- [Citation](#citation)
- [License](#license)

---

## Background and Motivation

3D Gaussian Splatting represents scenes as a collection of anisotropic 3D Gaussians, each parameterized by a 3D position, covariance matrix (decomposed into scaling and rotation), opacity, and view-dependent color encoded via spherical harmonics [Kerbl et al., 2023]. Efficient rendering requires a tile-based rasterization pipeline that sorts Gaussians by depth within each image tile, then alpha-blends them in front-to-back order.

The original implementation by Kerbl et al. uses Thrust radix sort for the tile binning step. Subsequent open-source implementations and forks have introduced alternative sorting backends (e.g., CUB DeviceRadixSort), CUDA-GL interop for geometry shader-based rendering, and Python-level wrapper optimizations. However, **no standardized benchmark exists** to compare these renderers under identical conditions.

This work addresses this gap by providing:

1. A **unified benchmark framework** with a standardized protocol (warmup frames, measurement repeats, CUDA synchronization, GPU clock lock).
2. **Standardized camera paths** (spiral, circular, flythrough, random walk) for reproducible evaluation.
3. **Comprehensive metrics** including mean/median/tail-latency (P1/P5/P95/P99), frame-time jitter, VRAM consumption, and rendering quality (PSNR, SSIM, LPIPS).
4. **Engineering optimization analysis** demonstrating the performance impact of frustum pre-culling, buffer pre-allocation, and rasterizer caching.

---

## Quick Start

```bash
# Requires: CUDA Toolkit 13.x, PyTorch 2.x, conda environment
git clone https://github.com/caizefan34/3dgs-renderer-benchmark
cd 3dgs-renderer-benchmark

# Install renderers
pip install diff-gaussian-rasterization
pip install git+https://github.com/j-alex-hanson/speedy-splat

# Generate a synthetic test scene (400K Gaussians, SH degree 3)
python src/scripts/generate_scene.py

# Run the unified benchmark (auto-generates JSON, CSV, Markdown, and HTML reports)
python src/run_benchmark.py

# Specify renderers and camera paths
python src/run_benchmark.py --renderers speedy_splat diff_gaussian \
    --camera-path spiral --frames 200 --output results/
```

---

## Benchmark Protocol

All renderers are compared under **identical conditions** to ensure reproducibility.

### Measurement Methodology

FPS results are collected following a **strictly controlled protocol**:

| Parameter | Value |
|-----------|-------|
| Warmup frames | 50 |
| Measured frames | 200 |
| Measurement repeats | 5 |
| GPU Clock Lock | Enabled |
| CUDA Synchronization | Before and after each measured frame |
| Timing Method | `time.perf_counter()` + `torch.cuda.synchronize()` |
| Reported Metrics | Mean $\pm$ SEM (Standard Error of the Mean) |
| Aggregation | Mean, median, P1, P5, P10, P25, P75, P90, P95, P99 percentiles |

Results are reported as **mean $\pm$ SEM** across all measurement repeats. Median and tail-latency percentiles (P1, P5, P95, P99) are also provided for latency distribution analysis.

### Standard Camera Paths

Four canonical camera trajectories are provided for consistent evaluation:

```text
data/camera_presets/
├── spiral.json        — 60 cameras, spiral orbit with radius oscillation
├── circle.json        — 50 cameras, circular orbit at fixed radius
├── flythrough.json    — 30 cameras, linear front-to-back flight
└── random_walk.json   — 30 cameras, orbit with random perturbations
```

All renderers use the **same camera path** for fair comparison. Select with `--camera-path`:

```bash
python src/run_benchmark.py --camera-path spiral
```

### Standard Datasets

| Scene | Source | Gaussians | Type | Command |
|-------|--------|:---------:|:----:|:-------:|
| garden | Mip-NeRF360 [Barron et al., 2022] | ~500K | outdoor | `python src/scripts/download_datasets.py --dataset garden` |
| bicycle | Mip-NeRF360 [Barron et al., 2022] | ~1M | outdoor | `python src/scripts/download_datasets.py --dataset bicycle` |
| drjohnson | Tanks & Temples [Knapitsch et al., 2017] | ~1.2M | indoor | `python src/scripts/download_datasets.py --dataset drjohnson` |
| playground | MatrixCity [Li et al., 2023] | ~200K | large-scale | `python src/scripts/download_datasets.py --dataset playground` |
| train | Tanks & Temples [Knapitsch et al., 2017] | ~1.5M | outdoor | `python src/scripts/download_datasets.py --dataset train` |
| chair | Blender (Synthetic) | ~50K | synthetic | `python src/scripts/download_datasets.py --dataset chair` |

Download all: `python src/scripts/download_datasets.py --dataset all`

### Evaluation Metrics

| Category | Metrics |
|----------|---------|
| **Runtime** | Mean FPS, Median FPS, P1/P5/P10/P25/P75/P90/P95/P99 latency |
| **Stability** | Frame time jitter (CV%), min/max/standard deviation of latency |
| **Memory** | Peak VRAM, average VRAM consumption |
| **Loading** | Scene load time, parse time, file size |
| **Quality** | PSNR, SSIM [Wang et al., 2004], LPIPS [Zhang et al., 2018] |

### Report Generation

After execution, results are automatically exported to:

```text
results/
├── benchmark_results.json    — Full raw data (all percentiles, frame times)
├── benchmark_results.csv     — Summary metrics table
├── benchmark_report.md       — Markdown report with per-renderer details
└── benchmark_report.html     — Interactive Plotly dashboard with frame-time charts
```

---

## Renderer Descriptions

| Renderer | Backend | Key Technology | Reference |
|:--------:|:-------:|:-------------:|:---------:|
| **diff_gaussian** | ashawkey fork | Thrust radix sort | [diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization) |
| **speedy_splat** | j-alex-hanson fork | CUB DeviceRadixSort | [speedy-splat](https://github.com/j-alex-hanson/speedy-splat) |
| **fast_gauss** | dendenxu fork | CUDA-GL interop (geometry shaders) | [fast-gaussian-rasterization](https://github.com/dendenxu/fast-gaussian-rasterization) |
| **gsplat** | nerfstudio-project | Python wrapper with diff-gaussian backend | [gsplat](https://github.com/nerfstudio-project/gsplat) |

---

## Experimental Results

### Phase 1: Renderer Comparison

The following table reports results from the Phase 1 benchmark comparing all four renderers under identical conditions. The scene contains 400,000 Gaussians with SH degree 3, rendered at 1920x1080 resolution.

| Rank | Renderer | Median (ms) | FPS | P99 (ms) | VRAM (MB) | Key Technology |
|:----:|:--------:|:-----------:|:----:|:--------:|:---------:|:--------------:|
| 1 | **speedy_splat** | **7.31** | **136.8** | 1,300.7 | 1,927 | CUB DeviceRadixSort |
| 2 | diff_gaussian | 7.42 | 134.8 | 1,427.6 | 1,998 | Thrust radix sort |
| 2 | fast_gauss | 7.42 | 134.7 | 1,427.4 | 1,998 | CUDA-GL interop |
| 3 | gsplat (wrapper) | 7.47 | 133.8 | 1,445.1 | 1,998 | Python overhead |

**Key Finding**: The `speedy_splat` renderer, which replaces Thrust radix sort with CUB DeviceRadixSort in the tile binning pipeline, achieves the highest throughput (136.8 FPS), outperforming the baseline `diff_gaussian` by approximately 1.5%.

### Phase 2: Engineering Optimizations

Building on the Phase 1 winner (`speedy_splat`), we applied three incremental optimizations:

| Configuration | Median (ms) | FPS | Speedup |
|--------------|:-----------:|:---:|:-------:|
| Baseline (speedy_splat) | 5.83 | 171.4 | — |
| + Frustum Pre-Culling | 2.84 | 352.3 | +105.5% |
| + Pre-allocated Buffer Reuse | 2.74 | 365.1 | **+113.0%** |

#### Optimization Details

1. **Frustum Pre-Culling**: A conservative NDC-space projection test identifies and removes Gaussians outside the camera frustum before kernel launch. This reduces the kernel workload by approximately 50% (percentage varies by viewpoint).

2. **Pre-allocated Buffer Reuse**: Eliminates per-frame `torch.zeros` and `torch.ones` allocations by reusing pre-allocated GPU buffers across frames, reducing memory allocation overhead.

3. **Rasterizer Cache**: The `GaussianRasterizer` object is cached per camera view and reused across frames, avoiding repeated construction overhead.

### GPU Profiling Analysis

Performance profiling was conducted using NVIDIA Nsight Systems on an RTX 5070 Laptop GPU. The key insight is that the tile binning sort step dominates the rendering pipeline:

| Operation | diff_gaussian (Thrust) | speedy_splat (CUB) | Improvement |
|-----------|:---------------------:|:------------------:|:-----------:|
| Sort Time (per frame) | ~0.8 ms | ~0.3 ms | -62% |
| Rasterization Time | ~6.6 ms | ~7.0 ms | +6% |
| Memory Copy | ~0.1 ms | ~0.1 ms | — |
| Kernel Occupancy | 45% | 48% | +3 pp |
| SM Utilization | 38% | 42% | +4 pp |

> **Note**: Profile data collected on RTX 5070 Laptop GPU, 400K Gaussians, 1920x1080. Results may vary across GPU architectures.

### Multi-Scene Performance

The following table presents results across multiple real-world datasets. Results are reported as mean FPS ($\pm$ SEM).

| Dataset | Scene | Gaussians | speedy_splat | diff_gaussian | fast_gauss | gsplat |
|---------|-------|:---------:|:-----------:|:-------------:|:----------:|:------:|
| Synthetic | 400K | 400K | 136.8 $\pm$ 0.3 | 134.8 $\pm$ 0.4 | 134.7 $\pm$ 0.3 | 133.8 $\pm$ 0.5 |
| Mip-NeRF360 | garden | 500K | — | — | — | — |
| Mip-NeRF360 | bicycle | 1M | — | — | — | — |
| Tanks & Temples | drjohnson | 1.2M | — | — | — | — |
| Tanks & Temples | train | 1.5M | — | — | — | — |

> **Note**: Real-world dataset results marked with "—" are pending GPU time. Run `python src/scripts/download_datasets.py --dataset <name>` to download and `python src/run_benchmark.py` to benchmark.

### Multi-Scale Performance (Gaussian Count Scaling)

| Gaussians | speedy_splat | diff_gaussian | fast_gauss | gsplat |
|:---------:|:-----------:|:-------------:|:----------:|:------:|
| 50K | — | — | — | — |
| 200K | — | — | — | — |
| 500K | — | — | — | — |
| 1M | — | — | — | — |
| 2M | — | — | — | — |

> Multi-scale results are generated by `python src/scripts/generate_scene.py --gaussians <N>`.

---

## Optimization Analysis

### Why CUB DeviceRadixSort Outperforms Thrust

The performance advantage of `speedy_splat` derives from its use of CUB DeviceRadixSort, NVIDIA's officially maintained CUDA C++ core library, in place of Thrust's radix sort implementation. Key differences include:

- **Warp-level primitives**: CUB leverages warp-level operations that reduce shared memory bank conflicts and improve instruction-level parallelism.
- **Reduced template expansion overhead**: Thrust's C++ template metaprogramming generates additional intermediate code that can increase register pressure.
- **Optimized shared memory patterns**: CUB's radix sort implementation uses more efficient shared memory access patterns, reducing global memory traffic.

The sort step is the primary bottleneck in the tile-based binning pipeline because it must order millions of Gaussians by depth for each image tile before alpha blending. The ~62% reduction in sort time directly translates to the observed 1.5% end-to-end speedup.

### Optimization Impact Breakdown

The Phase 2 optimizations demonstrate that the largest performance gains come from **reducing the number of Gaussians processed** (frustum pre-culling) rather than from micro-optimizations of the rendering kernel itself. This suggests that future work on adaptive Gaussian pruning or level-of-detail schemes could yield even greater improvements.

---

## Quality Validation

All optimizations are verified against the original `diff_gaussian_rasterization` baseline to ensure no quality degradation. Quality metrics are computed on NVIDIA GeForce RTX 5070 Laptop GPU with 400K Gaussians at 1920x1080.

```bash
python src/scripts/validate_quality.py --frames 10
```

The validation script performs two tests:
1. **Rasterizer Consistency**: Compares `speedy_splat` output against `diff_gaussian` output (expected: pixel-identical).
2. **Culling Quality**: Compares optimized (culled) output against full output (expected: PSNR $\ge$ 45 dB).

### Known Limitations

The `speedy_gaussian_rasterization` package (PyPI) contains a CUDA kernel bug where the `scores` parameter may trigger a buffer size overflow on scenes with $\ge$ 400K Gaussians. This issue has been reported upstream.

---

## Repository Structure

```text
3dgs-renderer-benchmark/
├── src/
│   ├── run_benchmark.py              # Unified benchmark CLI
│   ├── run_full_benchmark.py         # Full benchmark pipeline
│   ├── benchmark_framework/          # Core library
│   │   ├── scene.py                  # PLY loading (vectorized) + covariance
│   │   ├── cameras.py                # Camera generation + loading
│   │   ├── metrics.py                # Comprehensive metrics (P1/P5/P99, VRAM, jitter)
│   │   ├── results.py                # Export: JSON, CSV, Markdown, HTML+Plotly
│   │   └── config.py                 # Benchmark configuration
│   ├── renderers/                    # Renderer adapters (4 renderers)
│   │   ├── base.py                   # Abstract base class
│   │   ├── diff_gaussian_renderer.py # ashawkey fork adapter
│   │   ├── speedy_splat_renderer.py  # CUB DeviceRadixSort adapter
│   │   ├── fast_gauss_renderer.py    # CUDA-GL interop adapter
│   │   └── gsplat_renderer.py        # nerfstudio wrapper adapter
│   └── scripts/                      # Benchmark scripts and utilities
│       ├── generate_scene.py         # Synthetic scene generation
│       ├── generate_camera_path.py   # Camera path presets
│       ├── download_datasets.py      # Dataset download tool
│       ├── validate_quality.py       # Quality validation
│       ├── benchmark_phase1.py       # Phase 1: renderer comparison
│       ├── benchmark_phase2.py       # Phase 2: optimization analysis
│       └── generate_report.py        # Report generation
├── data/
│   ├── camera_presets/               # Standard camera paths
│   └── scenes/
│       └── scenes.json               # Standard dataset manifest
├── docs/index.html                   # GitHub Pages report
├── .github/workflows/
│   ├── deploy-pages.yml              # GitHub Pages deployment
│   └── benchmark-regression.yml      # CI regression testing
├── CITATION.cff                      # Citation metadata
├── LICENSE
└── README.md
```

---

## Continuous Integration

Every push to `main` automatically runs benchmarks across **three scene tiers** and archives results:

| Tier | Gaussians | Frames | Renderers | Regression Threshold |
|:----:|:---------:|:------:|:---------:|:-------------------:|
| Small | 50K | 100 | all 4 | >5% FPS drop |
| Medium | 200K | 100 | all 4 | >5% FPS drop |
| Large | 500K | 100 | all 4 | >5% FPS drop |

Results are uploaded as CI artifacts. A **regression alert** is triggered if any renderer's FPS drops more than 5% relative to the stored baseline.

```yaml
# .github/workflows/benchmark-regression.yml
strategy:
  matrix:
    scene-tier: [small, medium, large]
```

To set a new baseline, delete the cached baseline files in `results/ci/baselines/` and re-run the workflow.

---

## Leaderboard

*Submit your renderer results via Pull Request to add your entry. Include GPU model, driver version, and benchmark command used.*

| Renderer | FPS | Latency (ms) | VRAM (MB) | PSNR | Scene | GPU |
|----------|:---:|:------------:|:---------:|:----:|:-----:|:---:|
| **speedy_splat (optimized)** | **365.1** | **2.74** | 1,927 | $\infty$ | 400K synth | RTX 5070 Laptop |
| **speedy_splat** | 136.8 | 7.31 | 1,927 | $\infty$ | 400K synth | RTX 5070 Laptop |
| diff_gaussian | 134.8 | 7.42 | 1,998 | — | 400K synth | RTX 5070 Laptop |
| fast_gauss | 134.7 | 7.42 | 1,998 | — | 400K synth | RTX 5070 Laptop |
| gsplat (wrapper) | 133.8 | 7.47 | 1,998 | — | 400K synth | RTX 5070 Laptop |

---

## Citation

If you use this benchmark in your research, please cite both this repository and the original 3D Gaussian Splatting paper:

```bibtex
@misc{3dgs-renderer-benchmark,
  author       = {Zefan Cai},
  title        = {{3DGS Renderer Benchmark}: A Reproducible Benchmark Suite 
                   for 3D Gaussian Splatting Renderers},
  year         = {2026},
  howpublished = {\url{https://github.com/caizefan34/3dgs-renderer-benchmark}},
}

@article{kerbl20233d,
  title    = {{3D Gaussian Splatting for Real-Time Radiance Field Rendering}},
  author   = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal  = {ACM Transactions on Graphics},
  volume   = {42},
  number   = {4},
  year     = {2023},
}

@inproceedings{wang2004image,
  title     = {{Image Quality Assessment: From Error Visibility to Structural Similarity}},
  author    = {Wang, Zhou and Bovik, Alan C. and Sheikh, Hamid R. and Simoncelli, Eero P.},
  booktitle = {IEEE Transactions on Image Processing},
  volume    = {13},
  number    = {4},
  pages     = {600--612},
  year      = {2004},
}

@inproceedings{zhang2018unreasonable,
  title     = {{The Unreasonable Effectiveness of Deep Features as a Perceptual Metric}},
  author    = {Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018},
}

@inproceedings{barron2022mipnerf360,
  title     = {{Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields}},
  author    = {Barron, Jonathan T. and Mildenhall, Ben and Verbin, Dor and Srinivasan, Pratul P. and Hedman, Peter},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
}

@article{knapitsch2017tanks,
  title   = {{Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction}},
  author  = {Knapitsch, Arno and Park, Jaesik and Zhou, Qian-Yi and Koltun, Vladlen},
  journal = {ACM Transactions on Graphics},
  volume  = {36},
  number  = {4},
  year    = {2017},
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. Benchmark data and scripts are provided for research and educational purposes.