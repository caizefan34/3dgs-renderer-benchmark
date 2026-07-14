# Reproducible 3DGS Renderer Benchmark

[![Tests](https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml)
[![Pages](https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/deploy-pages.yml/badge.svg)](https://caizefan34.github.io/3dgs-renderer-benchmark/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/caizefan34/3dgs-renderer-benchmark?style=social)](https://github.com/caizefan34/3dgs-renderer-benchmark/stargazers)

A quality-gated benchmark for CUDA 3D Gaussian Splatting renderers. It answers
a deceptively hard question: **which renderer is actually faster when scene
tensors, cameras, resolution, timing, and image quality are held constant?**

[Explore the results](https://caizefan34.github.io/3dgs-renderer-benchmark/)
| [Review the protocol](#methodology)
| [Add a renderer](CONTRIBUTING.md#adding-a-renderer)

## Why this benchmark

Cross-paper FPS numbers are rarely comparable. This project provides:

- real renderer adapters instead of renamed fallbacks;
- identical scene tensors and fixed camera trajectories for every renderer;
- synchronized CUDA-event latency plus separate end-to-end latency;
- GT-relative PSNR, SSIM, and LPIPS gates before a real-scene performance
  result is called quality-verified;
- machine-readable result summaries with runtime and hardware metadata;
- explicit separation of reproduced measurements from upstream claims.

## Quick start

The CPU test suite validates camera conventions, metrics, scene parsing, and
adapter contracts without requiring any optional renderer backend:

```text
git clone https://github.com/caizefan34/3dgs-renderer-benchmark.git
cd 3dgs-renderer-benchmark
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

For a GPU run, first install a CUDA-enabled PyTorch build that matches your
system, then install at least one backend such as `gsplat`:

```text
python -m pip install numpy gsplat
python src/scripts/generate_scene.py --gaussians 50000 --output data/scene.ply
python src/run_benchmark.py --list-renderers
python src/run_benchmark.py --scene data/scene.ply --camera-path circle --renderers gsplat --frames 100 --warmup 30 --repeats 3 --output results/quickstart
```

> [!NOTE]
> CUDA extensions are sensitive to PyTorch, CUDA toolkit, compiler, and driver
> versions. Record all four when publishing a result. Windows + CUDA 13 users
> should also read [the recorded build notes](#windows--cuda-13-build).

![Latency scaling for verified renderers on three synthetic scene sizes](docs/assets/latency-scaling.svg)

## Verified headline

On an RTX 5070 Laptop at 1920x1080, the fastest locally timed path on the
synthetic heavy-overlap stress test is the inference-only HiGS renderer at
gsplat commit
[`77ab983`](https://github.com/nerfstudio-project/gsplat/commit/77ab983ffe43420b2131669cb35776b883ca4c3c).
These historical scenes have no source photographs, so their GT-relative
quality is **not measured**. This timing result alone does not prove that HiGS
preserves trained-scene quality.

> [!WARNING]
> A follow-up 38-view held-out GT audit on the official 3DGS Train checkpoint
> found a measurable HiGS regression: -0.626 dB PSNR, -0.00712 SSIM, and
> +0.00233 LPIPS on average versus original 3DGS, with a worst per-view PSNR
> delta of -5.46 dB. HiGS remains the fastest synthetic path, but is **not
> quality-equivalent** on the tested real checkpoint.

| Scene | Renderer | GPU mean | GPU median | P99 | Peak VRAM | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 50K | HiGS tile16 | 1.99 ms | 1.9 ms | 2.45 ms | 147 MB | N/A | N/A | N/A |
| 50K | Speedy-Splat | 12.56 ms | 12.5 ms | 13.82 ms | 584 MB | N/A | N/A | N/A |
| 50K | gsplat dense | 12.25 ms | 12.2 ms | 13.76 ms | 368 MB | N/A | N/A | N/A |
| 200K | HiGS tile16 | 6.34 ms | 6.3 ms | 7.23 ms | 391 MB | N/A | N/A | N/A |
| 200K | Speedy-Splat | 145.75 ms | 38.8 ms | 1934.10 ms | 2183 MB | N/A | N/A | N/A |
| 200K | gsplat dense | 383.10 ms | 50.0 ms | 876.45 ms | 1450 MB | N/A | N/A | N/A |
| 400K | HiGS tile8 | 15.96 ms | 15.8 ms | 23.22 ms | 1057 MB | N/A | N/A | N/A |
| 400K | Speedy-Splat | 1705.36 ms | 1608.9 ms | 4776.47 ms | 4276 MB | N/A | N/A | N/A |

The included generated scenes intentionally create heavy overlap. The large
Speedy/standard-gsplat tails are tied to specific camera views with very long
tile lists, rather than random timer noise.

## Ground-truth quality gate

The leaderboard quality reference is always a held-out original image, never
another renderer. A valid real-scene row therefore has the form:

| Renderer | Reference | PSNR | SSIM | LPIPS | Status |
|---|---|---:|---:|---:|---|
| original 3DGS | original test images | 24.9319 | 0.864349 | 0.223592 | measured |
| HiGS | original test images | 24.3057 | 0.857229 | 0.225921 | measured; quality regression |
| gsplat dense | original test images | pending | pending | pending | local extension rebuild failed |
| Speedy-Splat | original test images | 24.9311 | 0.864339 | 0.223610 | measured; matches original |
| TC-GS | original test images | pending | pending | pending | adapter/build pending |

The measured rows use the official pretrained Train checkpoint (1,071,462
Gaussians), 38 held-out views, black background, and the released 980x545 GT
images. They are a quality audit, not a 1920x1080 speed table. See the
[machine-readable result](data/results/rtx5070_train_gt_quality_2026-07-14.json).

Install the quality dependencies and evaluate the same trained PLY with every
renderer. The camera file can be the `cameras.json` exported by original 3DGS;
only camera names present in the held-out GT directory are selected.

```powershell
python -m pip install -r requirements-quality.txt
python src/scripts/validate_quality.py `
  --renderers original_3dgs gsplat_higs gsplat_dense speedy_splat `
  --scene path/to/model/point_cloud/iteration_30000/point_cloud.ply `
  --cameras path/to/model/cameras.json `
  --ground-truth-dir path/to/held_out_test_images `
  --split-label test `
  --baseline-renderer original_3dgs `
  --max-psnr-drop 0.1 --max-ssim-drop 0.001 --max-lpips-increase 0.001
```

The JSON report stores per-view and aggregate metrics, the camera-manifest
SHA-256, renderer versions, metric definitions, background, and thresholds.
PSNR is the mean of per-view PSNR values; SSIM uses the original 3DGS 11x11
valid Gaussian window; LPIPS defaults to the official evaluation's VGG net.

### Renderer consistency diagnostics

Historical renderer-to-renderer checks remain useful for finding rasterizer
regressions, but are not reconstruction-quality evidence: Speedy-Splat vs
gsplat dense reached minimum 111.96 dB / 1.0000 SSIM at 50K; HiGS vs gsplat
dense reached minimum 59.37 / 58.80 / 59.45 dB and 0.9997 SSIM at
50K / 200K / 400K. They must not be placed in the GT quality columns above.

## Implemented adapters

- `speedy_splat`: Speedy-Splat with static activation and fixed buffers.
- `speedy_splat_raw`: uncached wrapper ablation.
- `original_3dgs`: original graphdeco-inria diff-gaussian rasterizer (also
  available under the legacy name `diff_gaussian`).
- `gsplat`: real `gsplat.rasterization(..., packed=True)`.
- `gsplat_dense`: real `gsplat.rasterization(..., packed=False)`.
- `gsplat_higs`: HiGS inference, tile 8, uncompressed SH.
- `gsplat_higs_tile16`: HiGS inference, tile 16.
- `gsplat_higs_sh32` / `gsplat_higs_sh16`: SH packing ablations.
- `gsplat_higs_auto`: locally calibrated scale-aware configuration.
- `fast_gauss`: registered but unavailable locally because EGL loading is
  blocked by the current Windows environment/application policy.

TC-GS is tracked from the now-located official source at
[`DeepLink-org/3DGSTensorCore`](https://github.com/DeepLink-org/3DGSTensorCore/commit/0bb82f88fde211c34b42e1497f0fc7265461592b).
It uses a conflicting package name and must run in an isolated environment;
the benchmark adapter and RTX 5070 measurement are still pending. The public
code demonstrates TC-GS applied to Speedy-Splat, so future rows will use the
label **TC-GS (Speedy-Splat integration)** rather than implying an unrelated
rasterizer.

See [the renderer survey](docs/renderer_survey.md) for upstream commits, paper
claims, and reproducibility status.

## Correctness fixes

The previous repository results are not comparable with the verified table.
Before measurement, this work corrected:

- fake gsplat and TC-GS comparisons that actually called diff-gaussian;
- log-scales passed without `exp` activation;
- camera paths facing away from the scene;
- reversed full projection matrix multiplication;
- non-standard SH PLY property naming, while preserving legacy loading;
- a documented but unimplemented GPU clock-lock claim;
- mixed CPU/GPU timing without separate end-to-end metrics;
- missing renderer runtime version and source metadata.

## Methodology

- Corrected fixed camera paths use +Z facing the scene.
- Camera validation rejects paths placing the scene center behind the camera.
- Static scene packing and activation occur before timing.
- Every measured frame uses CUDA start/end events and synchronizes after the
  end event. Synchronization is outside the GPU event interval and avoids deep
  WDDM queues.
- GPU and end-to-end latency are exported separately with percentiles and VRAM.
- A real-scene result is quality-verified only after finite-output,
  camera-change, and GT-relative PSNR/SSIM/LPIPS checks pass. Synthetic timing
  results are labeled separately.

Example:

```powershell
python src/run_benchmark.py `
  --scene data/scene.ply `
  --camera-path circle `
  --renderers gsplat_higs gsplat_higs_tile16 speedy_splat gsplat_dense `
  --frames 100 --warmup 30 --repeats 3 `
  --output results/verified/example
```

## Optimization findings

HiGS separates coarse macro-tile partitioning from fine rasterization and uses
a persistent packed inference scene. It attacks the measured bottleneck: long,
imbalanced tile-Gaussian lists and redundant work in dense views.

Tile size is workload-dependent:

- 50K: tile16 is about 15% faster than tile8.
- 200K: tile16 is about 19% faster than tile8.
- 400K: tile8 is about 21% faster than tile16.

Larger tiles reduce partition/scheduling overhead at low density. Finer tiles
reduce overdraw and load imbalance at high density. `gsplat_higs_auto` uses the
local rule `<300K -> tile16`, otherwise tile8 + SH32. This is a local heuristic,
not a universal threshold.

SH packing trades decode work for bandwidth:

- SH32 preserves high quality (minimum 64.85 dB against uncompressed HiGS).
- SH16 has a larger error (minimum 49.79 dB).
- At 50K, SH decode overhead hurts performance.
- At 400K, SH32 leaves mean latency roughly unchanged but improved P99 in one
  controlled run from 24.31 ms to 17.31 ms.

The previous claim that Python buffer reuse alone adds 7.5% was not reproduced.
Wrapper ordering varied by several percent, so it is not presented as a stable
speedup.

## Windows + CUDA 13 build

Validated gsplat commit:
`77ab983ffe43420b2131669cb35776b883ca4c3c`.

Apply the recorded Windows/CUDA13 fixes:

```powershell
git -C path/to/gsplat apply `
  path/to/3dgs-renderer-benchmark/third_party_patches/gsplat-windows-cuda13.patch
```

For an RGB-only benchmark, these upstream-supported build variables avoid
unrelated template instantiations:

```text
NUM_CHANNELS=3
BUILD_3DGS=1
BUILD_2DGS=0
BUILD_3DGUT=0
BUILD_ADAM=0
BUILD_RELOC=0
BUILD_LOSSES=0
```

## Limitations

- Current 1920x1080 speed numbers use generated scenes. The Train checkpoint
  has a held-out GT quality audit at 980x545, but matched real-scene speed and
  quality measurements at one resolution remain pending.
- GPU clocks are not locked on this WDDM laptop.
- HiGS is inference-only and uses packed/fp16 internals.
- FlashGS, Local-GS, GEMM-GS, and fast-gaussian remain candidates until they
  pass the same local quality/timing protocol.
- Cross-paper speedup claims are not leaderboard results.

The headline measurements and environment metadata are also available as a
[machine-readable JSON summary](data/results/rtx5070_laptop_2026-07-13.json).

## Tests

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md), especially
the evidence required for new renderer results. See [CITATION.cff](CITATION.cff)
for citation metadata. The benchmark code is MIT licensed; upstream renderers
retain their own licenses.
