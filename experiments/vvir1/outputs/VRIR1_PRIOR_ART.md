# VRIR1-P0 Prior-Art Check (G2)

Overlap review for INCREMENTAL RASTER UPDATE (refit a fraction of raster structure
between training checkpoints), performed immediately before verdict.

## Closest prior art
- Classic incremental scene maintenance: incremental BVH refit/rebuild (Karras,
  Hashable BVH 2013; kinetic data structures / Basch-Guibas-Hershberger 1997).
  Certificate-based sorted/structural maintenance is a known concept. Hazard to
  claiming the certificate machinery itself as novel.
- Dynamic-scene 3DGS training: DSD-GS (2026) caches a static-region render after
  decomposing a dynamic scene. Closest in *training-time* reuse, but it partitions
  by scene dynamics (static vs dynamic regions), not by per-Gaussian tile-membership
  drift under frozen-topology canonical training.
- Inference-time temporal reuse (fixed trained scene, adjacent frames): Lumina/S2
  (ISCA 2025), Neo (ASPLOS 2026), CaT-GS (CVPR 2026), GSReuse (TVCG 2026),
  TemporalGS (arXiv 2026). All assume a *fixed* trained scene; none measures
  cross-checkpoint structural drift during differentiable training.
- Parameter/gradient caching: 3DGS-LM (ICCV 2025) caches JVPs; TideGS (2026) caches
  parameters/SSD. Data-movement, not raster intersection structure.

## Gap this work fills
No prior work (to our knowledge) measures, under canonical frozen-topology 3DGS
training, the fraction of (Gaussian, tile) raster intersections that remain
UNCHANGED between training checkpoints, using a bit-matched tile-membership model
to compute the exact unchanged/added/removed split. VRIR0 showed EXACT cross-visit
reuse fails (DROP); VRIR1 shows the complementary INCREMENTAL result: a large
majority of intersections survive a modest checkpoint gap, so only the changed
fraction must be rebuilt.

## Status
This is a measurement + certification memo. No CUDA incremental kernel is
implemented or claimed (prereg: "No CUDA before all oracle gates pass"). The
result characterizes headroom; implementation is out of scope for P0.
