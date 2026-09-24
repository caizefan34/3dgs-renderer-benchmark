# AccuTile A100 Intersection Evidence Report

> **⚠️ VARIANT P — NOT TRUE ACCUTILE — SUPERSEDED NAMING**
>
> This report evaluates the per-tile conservative conic predicate (variant P). The true AccuTile (variant A) achieves a similar 46-53% intersection reduction. See `accutile-identity-audit.md`.

## Summary

AccuTile reduces total Gaussian-tile intersections by **46–53%** across the three representative A100 workloads. The reduction is consistent across scenes with very different Gaussian counts and resolutions.

AccuTile primarily reduces **average** tile work, not heavy-tail per-Gaussian work. The per-Gaussian maximum intersection count is unchanged (the largest Gaussians still cover all tiles in both modes). The reduction comes from the vast majority of Gaussians that cover only a few tiles — the conservative conic predicate eliminates false-positive boundary tiles.

---

## 1. Intersection Counts

| Scene | Gaussians | Resolution | N_tiles | N_AABB | N_AccuTile | Reduction |
|-------|-----------|------------|---------|--------|------------|-----------|
| room | 1,105,873 | 3114×2075 | 25,350 | 29,174,416 | 14,260,680 | **51.12%** |
| bicycle | 2,589,484 | 4946×3286 | 63,860 | 26,008,111 | 12,127,959 | **53.37%** |
| garden | 874,019 | 5187×3361 | 68,575 | 17,517,701 | 9,400,611 | **46.34%** |

**Reduction = 1 - N_AccuTile / N_AABB**

All three scenes show substantial intersection reduction (46–53%). Bicycle has the largest reduction (53.37%), likely because its many small outdoor Gaussians have tight conic footprints that the AABB overestimates. Garden has the smallest reduction (46.34%) but still eliminates nearly half the intersections.

## 2. Per-Tile Work Distribution

| Scene | avg/tile AABB | avg/tile AccuTile | Δ avg/tile |
|-------|---------------|-------------------|------------|
| room | 1,150.9 | 562.6 | -588.3 (-51.1%) |
| bicycle | 407.3 | 189.9 | -217.4 (-53.4%) |
| garden | 255.5 | 137.1 | -118.4 (-46.3%) |

The average intersections per tile drops proportionally with the total intersection count, confirming that the reduction is distributed across tiles rather than concentrated in a few.

## 3. Per-Gaussian Work Distribution

| Scene | p50 AABB | p50 AccuTile | p95 AABB | p95 AccuTile | p99 AABB | p99 AccuTile | max AABB | max AccuTile |
|-------|----------|--------------|----------|--------------|----------|--------------|----------|--------------|
| room | 0 | 0 | 0 | 0 | 42 | 23 | 25,350 | 25,350 |
| bicycle | 0 | 0 | 0 | 0 | 2 | 2 | 63,860 | 63,860 |
| garden | 0 | 0 | 9 | 6 | 195 | 106 | 68,575 | 68,575 |

### Interpretation

- **p50 = 0**: The median Gaussian covers zero tiles in both modes — most Gaussians are behind the camera or outside the viewport. AccuTile does not change this.
- **p95 = 0 (room/bicycle)**: Even the 95th percentile Gaussian covers very few tiles. AccuTile does not significantly change the p95 for room/bicycle because most Gaussians are already near-zero.
- **p99 reduction**: Room p99 drops from 42 to 23 tiles (-45%). Garden p99 drops from 195 to 106 (-46%). This shows AccuTile does reduce heavy-tail per-Gaussian work for the top 1%.
- **max unchanged**: The maximum per-Gaussian tile count is identical (it equals the total tile count) because the largest Gaussians cover the entire image in both modes. The conservative predicate cannot eliminate any tiles for a Gaussian whose conic footprint covers the whole image — this is correct behavior.

**Conclusion**: AccuTile primarily reduces average work. It also reduces p99 per-Gaussian work by ~45%, but does not affect the absolute maximum (giant Gaussians). The reduction is broadly distributed across the Gaussian population, not concentrated in the heavy tail.

## 4. Tile Grid Statistics

| Scene | Resolution | Tile size | Tile width | Tile height | Total tiles |
|-------|------------|-----------|------------|-------------|-------------|
| room | 3114×2075 | 16 | 195 | 130 | 25,350 |
| bicycle | 4946×3286 | 16 | 310 | 206 | 63,860 |
| garden | 5187×3361 | 16 | 325 | 211 | 68,575 |

## 5. Comparison with Codex RTX 5070 Room Probe

The Codex local probe on RTX 5070 measured:
- baseline intersections = 42,229,706
- AccuTile intersections = 16,998,835
- reduction = 59.75%

The A100 measurement for room (same scene, different checkpoint):
- baseline intersections = 29,174,416
- AccuTile intersections = 14,260,680
- reduction = 51.12%

The difference is because the Codex probe used a different checkpoint (`phase7_room_30k_16` from the local laptop build, 1,000,684 Gaussians) while the A100 uses the A100-trained checkpoint (`a100_30k_room_t16_16`, 1,105,873 Gaussians). The intersection counts differ because the Gaussian populations differ, but both show substantial reduction (>50%).

## 6. Method

Intersections are measured from `meta["tiles_per_gauss"]` returned by `gsplat.rasterization()`. This tensor records the exact number of tiles each Gaussian contributes to, and its sum is the total intersection count. The same camera (index 0), same tile size (16), same tensors, and same process conditions are used for OFF vs ON.
