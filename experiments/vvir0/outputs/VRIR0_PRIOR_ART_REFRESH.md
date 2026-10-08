# VRIR0 Prior-Art Refresh

Refresh performed immediately before verdict (pack freeze date 2026-10-01).

## Overlap hazards (unchanged; all are inference-time or dynamic-scene, not evolving-full-state training structural certification)
- Lumina / S2 (ISCA 2025): adjacent-frame sorting reuse, fixed trained scene. Hazard to "first sorting reuse".
- Neo (ASPLOS 2026): reuse-and-update depth sorting, fixed scene, inference. Hazard to incremental sorting reuse.
- CaT-GS (CVPR 2026): inter-frame visible-set/tile caching, fixed scene, inference. Hazard to cached tile/list skip.
- GSReuse (TVCG 2026): warped screen-space reuse, approximate. Hazard to broad temporal reuse.
- TemporalGS (arXiv 2026): temporal priors/culling, approximate.
- DSD-GS (2026): static-region render cache after decomposition (dynamic-scene training). Hazard to "first training-time cache".
- 3DGS-LM (ICCV 2025): cached gradients for JVPs, different object/timescale.
- TideGS (2026): SSD/CPU/GPU parameter cache, data-movement not raster structure.
- Kinetic Data Structures (classic): certificate-based sorted maintenance is a known concept. Hazard to claiming certificates themselves as novel.

## Safe novelty wording (only if VRIR0 had succeeded)
> Fail-closed structural memoization for differentiable 3DGS training: certifying that Gaussian->tile membership and tile-local depth order remain exactly compatible with cached traversal structure before reuse.

## Status for this run
VRIR0 did NOT succeed (structure not preserved; see VRIR0_REPORT.md). No novelty claim is made. The negative result stands on its own: exact cross-visit structural reuse is not feasible in the canonical 3DGS training baseline's 15k-30k window because continuous parameter evolution (radius/center/depth drift) changes tile membership and within-tile order faster than the ~311-iter revisit gap preserves them.
