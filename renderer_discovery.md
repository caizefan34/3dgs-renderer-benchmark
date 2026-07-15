# Renderer Discovery and Inclusion

The platform tracks globally relevant 3DGS renderer candidates in
[`data/renderers/renderer_candidates.json`](../data/renderers/renderer_candidates.json).

Candidate tracking is not leaderboard inclusion. A renderer enters a measured
leaderboard only after:

1. its own upstream implementation runs locally;
2. the scene is trained from an official dataset or is clearly labeled
   synthetic stress;
3. camera path, resolution, warmup, repeats, and timing protocol match the
   comparison cohort;
4. GT-relative PSNR, SSIM, and LPIPS are measured for quality-bearing claims;
5. renderer version, source URL, commit hash, hardware, driver, CUDA, and
   benchmark commit are recorded.

## Current Candidate Classes

- Locally measured: original 3DGS, gsplat, TC-GS.
- Synthetic speed measured but still needs official-scene quality coupling:
  gsplat HiGS, Speedy-Splat.
- Adapter or environment pending: FlashGS, Local-GS/TiCoGS, GEMM-GS,
  fast-gaussian-rasterization.
- Quality/view-consistency candidate: StopThePop.

## Validation

```text
python src/scripts/validate_official_training.py
```

This validates both official dataset policy and renderer candidate registry.

## Local Batch Testing

```text
python src/scripts/run_local_renderer_suite.py \
  --scene data/official/mipnerf360/garden/point_cloud.ply \
  --cameras data/official/mipnerf360/garden/cameras.json \
  --ground-truth-dir data/official/mipnerf360/garden/images \
  --renderers all \
  --output-dir results/local_renderer_suite
```

The suite records renderer availability first. Missing candidate libraries such
as FlashGS, Local-GS, GEMM-GS, and StopThePop are registered as interface
stubs and are skipped until a real backend adapter is installed.
