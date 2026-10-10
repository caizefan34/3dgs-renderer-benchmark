# Prior art / originality audit (2026-10-08)

| Existing work | Verified concept | What TFC **cannot** claim as new |
|---|---|---|
| Speedy-Splat (CVPR 2025) | SnugBox (tight ellipse AABB), AccuTile (exact ellipse–tile overlap), training-compatible tile intersection scheduling | Tight ellipse, exact tile overlap, eliminating unnecessary tile IDs, before-raster binning |
| FlashGS (CVPR 2025) | Fast ellipse–rectangle / tile overlap testing for splatting | Ellipse–tile overlap test as the first algorithm |
| gsplat upstream (Apr 2026) | Conservative AccuTile ellipse-based tile intersection in 3DGS (PR #927; main after v1.5.3) | Implementation of AccuTile as a brand-new gsplat contribution |
| FastGS (CVPR 2026) | Compact Bounding Box and other rendering/training acceleration | Ordinary footprint tightening / bounding-box simplification |
| TFC archival package | Pooled -38.2% tile intersection in 11 scenes (current suite) | Originality, because work reduction may reproduce prior art |

## Required novelty gap form

For each proposed difference:

```
DELTA_ID:
ACTUAL_TFC_CODE_PATH_AND_LINES:
PRIOR_ART_SOURCE_PAPER_SECTION_AND_CODE_PATH:
PRIOR_ART_OPERATION:
PROPOSED_ADDITIONAL_OPERATION:
WHY_PRIOR_ART_CANNOT_PERFORM_IT:
WHY_IT_IS_NOT_A_REBRANDED_VARIANT:
PER_ITERATION_EXPENSIVE_WORK_SAVED_AFTER_ACCU_TILE:
NUMERICAL_SEMANTICS_AND_GRADIENT_INVARIANTS:
MEASURABLE_PREREG_FALSIFIER:
EVIDENCE_STATUS: OBSERVED / RECOMPUTED / DERIVED / HYPOTHESIS / BLOCKED
```

No paragraph of mere motivation counts as a code-level algorithmic delta. If there is no demonstrable additional operation, stop instead of filling the form speculatively.

## Primary literature URLs
- Speedy-Splat CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Hanson_Speedy-Splat_Fast_3D_Gaussian_Splatting_with_Sparse_Pixels_and_Sparse_CVPR_2025_paper.html
- Speedy-Splat method: https://speedysplat.github.io/
- FlashGS CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_FlashGS_Efficient_3D_Gaussian_Splatting_for_Large-scale_and_High-resolution_Rendering_CVPR_2025_paper.pdf
- gsplat PR #927: https://github.com/nerfstudio-project/gsplat/pull/927
- gsplat README upstream: https://github.com/nerfstudio-project/gsplat/blob/main/README.md
