# R6-A — Evidence Repair: Dimensional Inconsistency

## Problem A1 — Warp statistic dimensional inconsistency

### The bug

The original `r6_4_warp_duplicate.py` reported two statistics side by side:

```
tiles_per_gaussian = 25.1   (for room 5K)
warps_per_gaussian = 6.6    (for room 5K)
```

If `warps_per_gaussian = tiles_per_gaussian × warps_per_tile`, then
`warps_per_gaussian` should be ≥ `tiles_per_gaussian`. But 6.6 < 25.1 —
the relationship is impossible. **The labels are wrong.**

### Root cause

The script computed:
- `gaussian_tile_counts_mean` = mean tiles per visible Gaussian = 25.1
  - This is CORRECT: average number of tiles each Gaussian intersects
- `n_warps_per_gaussian_mean` = mean of `ceil(footprint_height / 2)` clamped to [1,8] = 6.6
  - This is the **within-tile** warp count (warps per Gaussian **per tile**)
  - NOT the total warps per Gaussian across all tiles
  - The label `warps_per_gaussian` implies total, but the value is per-tile

### Corrected definitions

| Quantity | Symbol | Definition | room 5K value |
|----------|--------|------------|---------------|
| tiles per Gaussian | T/G | tiles each visible Gaussian intersects | 24.4 (direct) |
| within-tile warps | W/tile/G | warps per Gaussian within each tile | 2.86 (direct) |
| total warps per Gaussian | T/G × W/tile/G | total warp-Gaussian pairs per Gaussian | 69.7 |
| R_atomic | W/tile/G (weighted) | block-aggregation reduction factor | 2.86 |

The original `R_atomic_potential_est = total_warp_gaussian_pairs / total_tile_gaussian_pairs`
was a **tile-count-weighted** average of the within-tile warp estimate. This is
mathematically correct as a weighted average, but the footprint-based estimate
itself was wrong (see Problem A2).

### What was wrong in the report

The r6-4-warp-duplicate-analysis.md report computed:

```
warps_per_gaussian / tiles_per_gaussian = 6.6 / 25.1 = 0.26
→ "3.8 warps per Gaussian per tile → 74% cross-tile"
```

This is **nonsensical**. Dividing a per-tile quantity by a total quantity
produces a dimensionally meaningless ratio. The "cross-tile fraction" of 48-74%
reported in the original analysis is invalid.

### The fix

1. Renamed `warps_per_gauss_mean` → `within_tile_warps_per_gauss_mean` in the
   aggregate output of `r6_4_warp_duplicate.py`
2. Added `total_warps_per_gauss_est = tiles_per_gauss_mean × R_atomic_mean`
3. Added SUPERSEDED warning to the script docstring
4. The direct measurement script `r6_a_direct_atomic.py` uses correct labels
   from the start: `cross_tile_dup`, `within_tile_warps`, `total_warps_per_gaussian`

## Problem A3 — Conflicting oracle numbers

### The bug

The R6-A conservative E2E oracle in `r6-3-atomic-profile.md` reported:
- bicycle 30K: 7.8%
- garden 30K: 10.0%

But the `r6-profile-results.json` (v1) computed different values because the
hardware throughput model parameters were embedded in `collect_results.py` and
not in the report. The two sources disagreed.

### The fix

The `r6-profile-results-v2.json` is now the **single authoritative source**.
All report tables are generated from this file. The v1 `r6-profile-results.json`
is marked SUPERSEDED but retained for provenance.

The v2 file contains:
- `r6_a_direct_atomic`: direct pixel-level measurements (corrected R_atomic)
- `r6_a_original_estimated`: original footprint estimates (SUPERSEDED)
- `gate_verdicts_v2`: gate verdicts computed from the same file
- `summary`: aggregate pass/fail counts

No manual computation is needed — all values flow from the same JSON.

## Summary of corrections

| Metric | Original (v1) | Repaired (v2) | Error factor |
|--------|---------------|---------------|--------------|
| R_atomic (room 5K) | 7.61 | 2.86 | 2.7× overestimate |
| R_atomic (bicycle 30K) | 5.50 | 2.11 | 2.6× overestimate |
| R_atomic (garden 30K) | 6.06 | 2.37 | 2.6× overestimate |
| A-GATE-3 pass count | 2/9 | 0/9 | All were false positives |
| "warps_per_gaussian" label | "warps per Gaussian" | "within-tile warps per Gaussian" | Dimensional error |
| "cross-tile fraction" | 48-74% | N/A (meaningless) | Removed |
