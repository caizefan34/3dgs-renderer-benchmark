# Reproducible 3DGS Renderer Benchmark

A local, quality-gated benchmark for CUDA 3D Gaussian Splatting renderers.
It separates upstream paper claims from results reproduced with identical scene
tensors, cameras, resolution, timing, and quality gates.

## Verified headline

On an RTX 5070 Laptop at 1920x1080, the fastest locally verified path is the
inference-only HiGS renderer in current `gsplat` main.

| Scene | Renderer | GPU mean | GPU median | P99 | End-to-end mean | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| 50K | HiGS tile16 | 1.99 ms | 1.9 ms | 2.45 ms | 2.03 ms | 147 MB |
| 50K | Speedy-Splat | 12.56 ms | 12.5 ms | 13.82 ms | 12.61 ms | 584 MB |
| 50K | gsplat dense | 12.25 ms | 12.2 ms | 13.76 ms | 12.33 ms | 368 MB |
| 200K | HiGS tile16 | 6.34 ms | 6.3 ms | 7.23 ms | 6.39 ms | 391 MB |
| 200K | Speedy-Splat | 145.75 ms | 38.8 ms | 1934.10 ms | not recorded | 2183 MB |
| 200K | gsplat dense | 383.10 ms | 50.0 ms | 876.45 ms | not recorded | 1450 MB |
| 400K | HiGS tile8 | 15.96 ms | 15.8 ms | 23.22 ms | 16.03 ms | 1057 MB |
| 400K | Speedy-Splat | 1705.36 ms | 1608.9 ms | 4776.47 ms | 1705.4 ms | 4276 MB |

The included generated scenes intentionally create heavy overlap. The large
Speedy/standard-gsplat tails are tied to specific camera views with very long
tile lists, rather than random timer noise.

## Quality gate

| Comparison | Scene | Minimum PSNR | Minimum SSIM | Result |
|---|---:|---:|---:|---|
| Speedy-Splat vs gsplat dense | 50K | 111.96 dB | 1.0000 | pass |
| HiGS vs gsplat dense | 50K | 59.37 dB | 0.9997 | pass |
| HiGS vs gsplat dense | 200K | 58.80 dB | 0.9997 | pass |
| HiGS vs gsplat dense | 400K | 59.45 dB | 0.9997 | pass |
| HiGS SH32 vs uncompressed HiGS | 50K | 64.85 dB | 0.9999 | pass |
| HiGS SH16 vs uncompressed HiGS | 50K | 49.79 dB | 0.9997 | pass |

Run a quality comparison:

```powershell
python src/scripts/validate_quality.py `
  --reference gsplat_dense `
  --test gsplat_higs `
  --scene src/data/scene_50k.ply `
  --cameras data/camera_presets/circle.json `
  --frames 10
```

## Implemented adapters

- `speedy_splat`: Speedy-Splat with static activation and fixed buffers.
- `speedy_splat_raw`: uncached wrapper ablation.
- `gsplat`: real `gsplat.rasterization(..., packed=True)`.
- `gsplat_dense`: real `gsplat.rasterization(..., packed=False)`.
- `gsplat_higs`: HiGS inference, tile 8, uncompressed SH.
- `gsplat_higs_tile16`: HiGS inference, tile 16.
- `gsplat_higs_sh32` / `gsplat_higs_sh16`: SH packing ablations.
- `gsplat_higs_auto`: locally calibrated scale-aware configuration.
- `fast_gauss`: registered but unavailable locally because EGL loading is
  blocked by the current Windows environment/application policy.

TC-GS is tracked as a paper result, not a measured renderer: no official source
was located, so the repository no longer aliases it to diff-gaussian.

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
- A renderer is verified only after finite-output, camera-change, and quality
  checks pass.

Example:

```powershell
python src/run_benchmark.py `
  --scene src/data/scene_200k.ply `
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

- Current verified numbers use generated scenes, not trained Mip-NeRF360 or
  Tanks & Temples PLYs. Real-scene validation remains the next dataset step.
- GPU clocks are not locked on this WDDM laptop.
- HiGS is inference-only and uses packed/fp16 internals.
- FlashGS, Local-GS, GEMM-GS, and fast-gaussian remain candidates until they
  pass the same local quality/timing protocol.
- Cross-paper speedup claims are not leaderboard results.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

See [CITATION.cff](CITATION.cff). The benchmark code is MIT licensed; upstream
renderers retain their own licenses.
