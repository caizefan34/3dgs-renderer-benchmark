# VRIR2-P0 Report — Minimal Incremental Raster Tile-Membership Update

## Verdict
`DROP_NO_E2E_HEADROOM` — the incremental tile-membership repair is **100% bit-exact** against the frozen gsplat baseline, but it delivers **no end-to-end performance headroom** (membership speedup `0.099x` vs required `>=1.5x`; E2E `0.099x` vs required `>=1.1x`). Gate G3 (performance) fails decisively; G1 (correctness) and G2 (change set) pass.

## Question
Can the Gaussian→tile membership structure of a 3DGS rasterizer be **incrementally repaired** (`M_old + Delta_G -> M_new`) instead of fully rebuilt at every refresh, while remaining bit-exact and faster?

## Scope freeze
Only the Gaussian→tile membership update path is optimized. No change to sorting semantics, blending, backward, optimizer, densification, or the training schedule. Ground truth is the real frozen gsplat 1.5.3 path (`isect_tiles(sort=True)` + `isect_offset_encode`), not a re-implementation.

## Experiment setup
- Host: `dsw-933622-64cc845c7d-lqswc`, 8×A100-SXM4-80GB (GPUs 0/1/2 used, one scene each), torch 2.9.1+cu128, CUDA 12.8, gsplat 1.5.3, python 3.10.21. Repo SHA `abfa6c3e4b23caf5a927bf3722a65cba37b6fab3`.
- Scenes (Mip-NeRF 360, densification OFF, window iter 15001–30000, gap=500): `room` (N=1,593,376; 311 cams), `bicycle` (N=6,131,954; 194), `garden` (N=5,834,784; 184). 8 tracked cameras per scene.
- Cadences evaluated (fixed, preregistered, not tuned): 500, 1000, 2000, 5000. Records per scene: 416 (8 cams × 52 refreshes), 1248 total across scenes.

## Membership model
Membership is fully determined by the per-Gaussian descriptor `D = (b_minus[N,2], b_plus[N,2], active[N], depth_bits[N])`:
- The gsplat intersection key is `(tile_id << 32) | float32_bits(depth)`, and `flatten_ids` is the dense Gaussian index.
- Within a tile, the stable CUB radix sort orders by `depth_bits` ascending, ties broken by emission order = Gaussian index ascending.
- Tile bounds use the VRIR0-validated AABB port (rate 1.0): `tile_radius = radius/tile_size`, clamped `floor/ceil` to `[0, dim]`; per-axis cull when `radius <= 0`.

## Incremental algorithm (`experiments/vvir2/oracle/incremental.py`)
1. `changed_gaussians = AABB OR active OR depth_bits changed` (the change set `Delta_G`).
2. Affected tiles = union of old/new AABB tiles of changed Gaussians; **unaffected tiles are copied verbatim** (no global rebuild).
3. Rebuild only affected tiles: keep unchanged entries in place, emit new entries for changed Gaussians, then two stable argsorts (by gid, then by `(tile<<32)|depth_bits`) to reproduce gsplat ordering.
4. Rebuild per-tile offsets via cumsum; scatter unaffected (verbatim) and rebuilt entries into the new layout.
5. `exact_match()` verifies bit-exact equality of `flatten_ids`, `isect_ids`, and `offsets` vs the baseline full rebuild.

## Results

### Gate G1 — Correctness: PASS
All 1248 records across 3 scenes and all 4 cadences are **100% bit-exact** (0 mismatches). `flatten_ids`, `isect_ids`, and `offsets` all match the frozen gsplat baseline exactly.

### Gate G2 — Change set non-empty: PASS
`Delta_G` is non-empty every refresh. Per-scene mean changed-gaussian ratio `r_g`: room 0.839, bicycle 0.880, garden 0.927. But **all tiles are affected every refresh** (`r_t = 1.0` for all scenes; 8160/8160 tiles).

### Gate G3 — Performance: FAIL
Per-scene median timing at operational cadence 500:

| Scene | baseline_full_ms | baseline_emit_ms | baseline_sort_ms | incremental_ms | membership_speedup | r_g | r_t |
|-------|-----------------|------------------|------------------|----------------|--------------------|-----|-----|
| room    | 0.975 | 0.373 | 0.602 | 8.69  | 0.115 | 0.828 | 1.0 |
| bicycle | 1.399 | 0.581 | 0.818 | 14.85 | 0.096 | 0.875 | 1.0 |
| garden  | 1.445 | 0.535 | 0.909 | 17.00 | 0.086 | 0.923 | 1.0 |

Aggregated (mean over scenes) `membership_speedup_500 = e2e_speedup_500 = 0.0988` — **~10x slower than the baseline**, versus the required `>=1.5x` (membership) and `>=1.1x` (E2E). This is stable across all four cadences (0.086–0.118x).

## Root cause
With ~195 Gaussians per tile, even modest per-Gaussian drift (depth_bits is highly sensitive: a 1-ULP depth change flips the sort key) touches **every tile** (`r_t = 1.0`). The incremental path then does **two full argsorts + O(M) gather/scatter/repeat_interleave** passes, whereas the baseline is a **single fused emit + one CUB radix sort**. Because the incremental rebuild ends up reconstructing essentially the entire structure, it is strictly *more* work than the fused baseline — there is no headroom to recover.

Note: VRIR1 showed high per-gap intersection-level persistence (unaffected fraction room 0.935 / bicycle 0.945 / garden 0.959), but that intersection-level persistence does **not** translate into incremental membership headroom, because a single depth-bit change forces a full re-sort of any tile it touches, and drift touches all tiles.

## Decision rule (fail-closed, preregistered order)
`DROP_CORRECTNESS` if any membership mismatch → elif empty change set `DROP_NO_CHANGESET` → elif membership < 1.5x or E2E < 1.1x `DROP_NO_E2E_HEADROOM` → else `KEEP_VRIR2`.

## Next step
Incremental membership repair is not worthwhile for dense 3DGS at this tile density. Any future attempt must (a) reduce affected-tile fraction (e.g., spatial tiling that bounds drift locality) or (b) avoid a full per-tile re-sort (e.g., incremental insertion sort exploiting near-order), and (c) beat the fused emit+radix-sort baseline in wall-clock — none of which the current descriptor model permits. Recommended: keep the full rebuild; revisit only if a membership update becomes a proven bottleneck in a larger-scope pipeline.

## Provenance
Deliverables in `experiments/vvir2/outputs/`: `VRIR2_VERDICT.json`, `VRIR2_SOURCE_LOCK.json`, `remote_identity.json`, `VRIR2_CORRECTNESS.json`, `VRIR2_CHANGESET.json`, `VRIR2_TIMING.json`, `VRIR2_REFRESH_ANALYSIS.json`, `VRIR2_REPORT.md`, `MANIFEST.sha256`.