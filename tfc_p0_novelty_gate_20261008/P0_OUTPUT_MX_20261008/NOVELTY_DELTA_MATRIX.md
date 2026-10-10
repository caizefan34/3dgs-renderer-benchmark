# NOVELTY DELTA MATRIX — TFC-P0-MX

Per `PRIOR_ART_AND_DELTA.md` the required form is an 11-field `DELTA_ID` block per proposed difference. Per `DSH_EXECUTION_PROMPT.md` Stage 0C and `DECISION_PROTOCOL.md` Gate N0: **if there is no demonstrable additional operation, do NOT fill the form speculatively — output `TFC_FOOTPRINT_ONLY_NO_NOVELTY`.**

The table below enumerates **every concrete TFC operation** and shows that each falls into `IDENTICAL` or `MATHEMATICALLY_EQUIVALENT` to existing prior art. **No row yields a completable `DELTA_ID`.** Therefore no novelty delta is claimed and the direction is stopped.

Legend for `overlap_class`: IDENTICAL = same emitted tile set / same primitive; MATHEMATICALLY_EQUIVALENT = same output up to floating-boundary rounding; PRIOR_ART_ONLY = covered by an existing method.

| # | TFC operation (mode-1) | code path | nearest prior-art function/equation | overlap_class | new execution-level mechanism? |
|---|---|---|---|---|---|
| 1 | per-axis AABB rectangle (mode-0, baseline) | `tfc1_rowwise.cu` radius_x/radius_y AABB emit | gsplat v1.5.3 `IntersectTile.cu` `isect_tiles` AABB | IDENTICAL | no (this is the baseline, not TFC novelty) |
| 2 | ellipse-exact per-row analytic tile span: `dx = (−b·dy ± sqrt(κa − det·dy²))/a` | `tfc1_rowwise.cu` row_xrange loop | AccuTile `accutile_ellipse_intersection` (gsplat main, `isY`-selected strip); FlashGS per-tile ellipse-rectangle test | MATHEMATICALLY_EQUIVALENT (same roots, up to SnugBox boundary rounding) | **no** |
| 3 | shorter-dimension strip iteration | **omitted** by TFC (always rows) | AccuTile `bool isY = y_span < x_span` → `O(min(rows,cols))` | PRIOR_ART_ONLY (TFC omits the very lever that makes AccuTile cheaper) | **no — TFC does strictly MORE work** |
| 4 | contiguous tile-range emission (no per-tile predicate) | `tfc1_rowwise.cu` emit | AccuTile `accutile_process_tiles` contiguous emit; StopThePop AABB emit | IDENTICAL | no |
| 5 | pre-binning tile-membership reduction (removes sub-alpha-threshold boundary tiles) | mode-1 vs mode-0 | Speedy-Splat SnugBox/AccuTile training-compatible tile intersection | PRIOR_ART_ONLY | no |
| 6 | packed/unpacked rasterization consumer of the reduced tile set | `rasterize_to_pixels` (unchanged) | gsplat rasterizer (unchanged) | IDENTICAL | no |

## 11-field DELTA_ID evaluation

A completable `DELTA_ID` requires `PROPOSED_ADDITIONAL_OPERATION` + `WHY_PRIOR_ART_CANNOT_PERFORM_IT` + `PER_ITERATION_EXPENSIVE_WORK_SAVED_AFTER_ACCU_TILE` + `MEASURABLE_PREREG_FALSIFIER`, with `EVIDENCE_STATUS ∈ {OBSERVED/RECOMPUTED}`.

- **`PROPOSED_ADDITIONAL_OPERATION`**: none. Every TFC operation (#1–#6) is `IDENTICAL`/`MATHEMATICALLY_EQUIVALENT`/`PRIOR_ART_ONLY`. TFC adds **no** new operation beyond AccuTile.
- **`PER_ITERATION_EXPENSIVE_WORK_SAVED_AFTER_ACCU_TILE`**: ~0. `TFC_VS_ACCUTILE_MEMBERSHIP.json`: `tfc_only_total = 0` on 10 500 real cases (TFC never emits a tile AccuTile omits); `TFC_CURRENT_GSPLAT_BASELINE_AUDIT.json`: TFC-vs-current-main(AccuTile) intersection advantage ~0; `TFC_COMPLEXITY_OPERATION_AUDIT.json`: `O(rows) ≥ O(min(rows,cols))`.
- **`MEASURABLE_PREREG_FALSIFIER`**: none available (no delta to falsify).

**Conclusion:** no `DELTA_ID` can be honestly completed. Output: **`TFC_FOOTPRINT_ONLY_NO_NOVELTY`**. No `CONDITIONAL_RESEARCH` is declared because there is no source-verified non-overlapping mechanism.

## Prior-art references (primary)
- Speedy-Splat (CVPR 2025): SnugBox + AccuTile exact ellipse–tile assignment; project https://speedysplat.github.io/ ; paper https://openaccess.thecvf.com/content/CVPR2025/html/Hanson_Speedy-Splat_Fast_3D_Gaussian_Splatting_with_Sparse_Pixels_and_Sparse_CVPR_2025_paper.html (also in-repo: `reports/accutile/accutile_source_audit.md` attributes SNUGBOX+AccuTile to Speedy-Splat; arXiv 2412.00578).
- FlashGS (CVPR 2025): fast ellipse–rectangle / tile overlap test; https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_FlashGS_Efficient_3D_Gaussian_Splatting_for_Large-scale_and_High-resolution_Rendering_CVPR_2025_paper.pdf
- gsplat upstream AccuTile, PR #927 (Apr 2026): https://github.com/nerfstudio-project/gsplat/pull/927 ; commit `28e794ca44a4c25ffc39175370c5ee7b38bfcc36` ; merge `3d4f9027`.
- FastGS (CVPR 2026) Compact Bounding Box: bounding-box tightening only (not an execution-level mechanism).
