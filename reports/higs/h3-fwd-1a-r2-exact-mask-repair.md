# H3-FWD-1A-R2: Exact AccuTile-to-HiGS mask repair

The B2 reference is `IntersectTile.cu` AccuTile, not the earlier Python structural oracle. Macro-F4 now uses an `accutile_fine_mask_8x4` device helper that preserves B2’s `__logf`, `sqrtf`, FP32 expression order, integer conversions, line-sweep coverage rule, and boundary comparisons. It tests only candidate macros from the B2 coarse rectangle, stores one `uint32` mask per surviving macro entry, and does not materialize a conventional fine list outside validation.

## Failing-pair root cause

The old generic `hit_rect` test and B2 AccuTile are not semantically identical.

- Visible Gaussian 90325, bicycle fine tile `(95, 6)`, macro `(11, 1)`: B2 emits the tile through its AccuTile line-sweep envelope. The old closed-rectangle quadratic test has `qmin=11.080848694` and `t=11.080818176`, therefore rejects it. The first divergence is the geometric coverage rule, not `logf`.
- Visible Gaussian 133259, bicycle fine tile `(24, 62)`, macro `(3, 15)`: B2 obtains `bbox_min.y=1008.0` and `rect_min.y=(int)(1008/16)=63`; fine row 62 is never enumerated. The old generic rectangle test treats its closed upper edge as intersecting and accepts it. The first divergence is B2’s integer lower-bound conversion at an exact tile boundary.

## Exact structural validation

| Scene | B2 pairs | Macro entries | Reconstructed pairs | missing / extra / duplicate / wrong |
| --- | ---: | ---: | ---: | --- |
| room/cam0 | 953,144 | 124,017 | 953,144 | 0 / 0 / 0 / 0 |
| bicycle/cam0 | 1,412,189 | 311,610 | 1,412,189 | 0 / 0 / 0 / 0 |
| garden/cam0 | 533,928 | 66,682 | 533,928 | 0 / 0 / 0 / 0 |

Semantic non-tie inversions are zero. Equal-depth permutations remain: room 72, bicycle 67, garden 6 raw inversions.

## Timing and memory

The retained masks are included in Macro-F4 timing and production memory. CUDA Event median timing: room B2/Macro `0.546816/0.808960 ms`, bicycle `0.669696/0.817152 ms`, garden `0.417792/0.458752 ms`; each Macro-F4 result is slower than B2 by the requested 5% band. Exact coverage count plus fill dominate the repaired stage breakdown.

Macro persistent / temporary / peak bytes are room `994,960 / 1,988,496 / 2,488,320`, bicycle `2,495,704 / 4,989,984 / 6,241,280`, and garden `536,152 / 1,070,944 / 1,341,440`. Retained masks are respectively 496,068, 1,246,440, and 266,728 bytes.

The structural gate is `PROMOTE_TO_MACRO_RASTER`; timing is independently `MACRO_F4_SLOWER`. No macro rasterizer or backward change was made.

Research-record corrections: bicycle wrong-pair rate is `2 / 1,412,189 = 0.000141624%`; theoretical scalar-adjoint occupancy is `28/64=43.75%` baseline and `36/64=56.25%` scalar.
