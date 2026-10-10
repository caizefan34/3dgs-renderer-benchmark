# CODE-LEVEL OVERLAP — TFC-P0-MX

Direct source-function/line comparison between TFC and the nearest prior art. Purpose (Stage 0C): show, at the code level, that TFC's footprint primitive is already present upstream. **No GPU, no benchmarking, no source modification.**

## 1. TFC frozen core
- File: `_tfc1_oracle/src/tfc1_rowwise.cu`
- SHA-256 (live, this task): `a44e4ee63c7ba9d713a4489693d189cb3fd6e7ba95764e491027c0639e34e68e` — **matches** the freeze lock in `_tfc_fc0_pack/deliverables/TFC_FC0_SOURCE_LOCK.json` and `_tfc_prior_art_audit/TFC_VS_ACCUTILE_MEMBERSHIP.json` (`tfc_freeze_verified_intact: true`). TFC implementation, thresholds, geometry **not modified after seeing results**.
- Mode dispatch (from `_tfc_fc0_pack/bwd_test.py`, `parity_packed.py`, `parity_packed2.py`): `footprint_intersect(means2d, conics, opacity, depths, near, TILE, W, H, RC, mode)`; `mode=0` per-axis AABB, `mode=1` ellipse-exact row-wise.
- The mode-1 inner loop solves, per screen row `y` in `[tminy, tmaxy]`, the conic for the horizontal tile span and emits the contiguous tile range. Conic order `(A,B,C)` in both projection output and raster kernel (matches gsplat).

## 2. Upstream AccuTile (gsplat post-v1.5.3, PR #927)
- File (local): `public_baselines/gsplat/gsplat/cuda/csrc/IntersectTile.cu`
- SHA-256 (live, this task): `1f331b6dc73cf16f1e496f4a525f3abec3ef1bc132b126408f04a46f2599e712`
- Matches `TFC_CURRENT_GSPLAT_BASELINE_AUDIT.json` "current_main" = gsplat v1.6.0 `IntersectTile.cu` commit `512d366` (PR-merge snapshot `3d4f9027` archived; current main adds sparse `tile_mask`/packed path but the core `accutile_process_tiles` primitive is unchanged).
- Key functions:
  - `accutile_ellipse_intersection(float A, float B, float C, float disc, float t, float2 p, bool isY, float coord)` — analytic ellipse/tile-boundary intersection on the shorter-selected axis (`isY ? p.y : p.x`, `isY ? A : C`).
  - `accutile_process_tiles(...)` — strip traversal: SnugBox + `isY`-selected shorter dimension, one ellipse cross-range per strip boundary (`~2 sqrt + 1 div`), contiguous range emission.
  - Dispatch: `bool isY = y_span < x_span;` (line ~337/342).

## 3. Line-level overlap summary

| Aspect | TFC mode-1 (`tfc1_rowwise.cu`) | AccuTile (`IntersectTile.cu`) | Overlap |
|---|---|---|---|
| emitted tile set | exact ellipse per-row span | ellipse-exact on shorter axis + SnugBox rounding | **IDENTICAL up to ≤1 boundary tile** (`tfc_only_total=0`, `max_acc_only=1`) |
| conic formulation | `A·dx²+2B·dx·dy+C·dy²=κ`, roots `(−b·dy ± sqrt(κa−det·dy²))/a` | same conic, `accutile_ellipse_intersection` | **MATHEMATICALLY_EQUIVALENT** (same roots) |
| strip axis | always rows, `O(rows)` | `isY=y_span<x_span`, `O(min(rows,cols))` | **AccuTile strictly cheaper**; TFC omits the lever |
| per-strip cost | `~5 sqrt + 2 div` | `~2 sqrt + 1 div` | TFC heavier |
| emit | contiguous range, no per-tile test | contiguous range, no per-tile test | **IDENTICAL** |
| raster consumer | `rasterize_to_pixels` (unchanged) | gsplat rasterizer (unchanged) | **IDENTICAL** |
| opacity thresholding | sub-alpha-threshold boundary tiles removed | same (ellipse from opacity-threshold conic) | **MATHEMATICALLY_EQUIVALENT** |

## 4. Verdict
TFC mode-1 is a **rows-only special case of AccuTile that omits the `isY` shorter-axis heuristic** and uses exact per-row min/max instead of conservative SnugBox rounding. At the code level it is **contained within** AccuTile's output (strict subset, zero TFC-only tiles). It is **not** a re-implementation of a brand-new primitive, **not** a new backward/sorting-aware interface, and removes **no** work that AccuTile does not already remove. ⇒ **footprint-only, no novel execution-level mechanism.**

## 5. Additional prior-art notes (context, not deltas)
- FlashGS: per-tile ellipse/rectangle edge test — a different exact-overlap primitive, also covering ellipse-tile overlap; not new.
- QuadBox / StopThePop: tile-bounding / AABB tile membership; bounding-box tightening only.
- In-repo AccuTile integration attempts (`reports/final_sprint/accutile_correctness.md`, `accutile_forensics.md`): a *per-tile conservative conic predicate* variant was `DROP_BEFORE_BACKPORT` (26 MATERIAL pixel-support false negatives, max alpha/threshold 16.72). That is a **different, failed** backport attempt and does not change the conclusion about TFC mode-1 (which is the strip-based exact form, equivalent to upstream AccuTile).
