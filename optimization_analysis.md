# Optimization Analysis

## Where the time goes

A tile-based 3DGS forward pass has four relevant stages:

1. Project each 3D Gaussian and compute its screen-space ellipse.
2. Enumerate intersected tiles and emit tile/depth keys.
3. Sort or group the keys.
4. Blend each tile front-to-back until transmittance is exhausted.

The corrected synthetic trajectory shows modest, stable latency at 50K but
extreme view-dependent tails at 200K/400K in standard gsplat and Speedy-Splat.
The slow views contain much longer tile lists and more pixel-Gaussian tests.
This makes tile work generation, load balance, and blending more important than
Python allocation or rasterizer-object construction.

## Baseline mechanisms

### Speedy-Splat

Speedy-Splat improves Gaussian localization and combines it with primitive
pruning during training. Its full paper speedup includes fewer Gaussians, so a
rasterizer-only comparison on an identical PLY cannot reproduce the complete
6.71x claim. On the same unpruned PLY, it is close to gsplat at 50K and faster
in typical 200K frames, but it retains severe high-overdraw tails.

Static activation and fixed buffers were moved outside the timed loop. The
wrapper ablation did not show a stable positive speedup, so buffer reuse is not
treated as the main optimization.

### gsplat packed vs dense

Packed mode removes invalid camera/Gaussian pairs and can reduce intermediate
memory, but it also adds compaction and index handling. Dense mode was faster
at 50K in the local run. The best choice depends on visibility sparsity; the
registry exposes both rather than labeling one universally faster.

### TC-GS

TC-GS maps conditional alpha computation to Tensor Core matrix operations and
reports an additional 2.18x over other accelerated pipelines. This targets the
blend stage and is conceptually complementary to HiGS. Official source is now
available at `DeepLink-org/3DGSTensorCore` commit
`0bb82f88fde211c34b42e1497f0fc7265461592b`; its public example integrates
TC-GS with Speedy-Splat. An isolated RTX 5070 build and GT audit remain pending.

## Why HiGS wins

HiGS decouples the granularity used for partitioning from that used for
rasterization:

- coarse macro-tiles reduce global partition/sort overhead;
- fine render tiles reduce unnecessary pixel/Gaussian work;
- work is issued in proportion to macro-tile occupancy, improving load balance;
- a stateful renderer reuses packed scene data and persistent workspace;
- fp16-oriented layouts reduce memory traffic.

This combination directly addresses the fixed-camera long tails. At 200K,
HiGS reduced P99 from 1934 ms (Speedy) and 876 ms (gsplat dense) to 7.23 ms.
At 400K, it reduced Speedy's 1609 ms median to about 16 ms. The earlier roughly
59 dB PSNR was only a synthetic renderer-consistency check against gsplat
dense. A later 38-view held-out GT audit measured -0.626 dB PSNR, -0.00712
SSIM, and +0.00233 LPIPS versus original 3DGS, so HiGS is not
quality-equivalent on that checkpoint.

## Further optimization performed

### Scale-aware tile selection

Tile 16 is faster at 50K/200K because there is less partition and scheduling
work. Tile 8 is faster at 400K because tighter tiles reduce overdraw and long
lists. The local auto adapter uses:

```text
N < 300K  -> tile 16
N >= 300K -> tile 8
```

Gaussian count is only a proxy. A stronger future selector should use a short
calibration pass and choose from measured tile duplication, maximum tile-list
length, or GPU time on representative cameras.

### SH packing

SH32 and SH16 reduce appearance bandwidth but add decode work:

- At 50K, decode overhead dominates and SH32 is slower.
- At 400K, SH32 has similar mean time and improved the observed P99.
- SH32 is the safer quality choice: minimum 64.85 dB against uncompressed HiGS,
  compared with 49.79 dB for SH16.

The auto adapter enables SH32 only at the high-count side of the local rule.

## Best next combinations

1. Integrate the official TC-GS source in an isolated environment and apply the
   same GT-relative quality protocol before comparing its speed.
2. Add Local-GS warp-coherent blending to dense HiGS tiles to reduce divergence
   and hoist tile-shared parameters.
3. Add TemporalGS selective tile updates for coherent camera motion. This
   changes the benchmark from independent frames to a sequence benchmark and
   requires temporal error metrics.
4. Calibrate tile size from measured occupancy rather than Gaussian count.
5. Validate on trained real scenes before adopting thresholds or compression
   settings in production.
