# COLMAP track coverage preflight (2026-09-27)

This CPU-only, read-only check asks whether the room COLMAP tracks could support a **limited** geometry diagnostic for the broad SAD-vs-FastGS residual. It uses the same eight training cameras selected before the fog/work diagnostic, scales their original 3114x2075 COLMAP 2D observations to the effective 1600x1066 image, and counts valid linked 3D tracks in each 64px block. No learned model, renderer, GT depth or GPU was used.

| Measure across eight registered training views | Range | Median |
| --- | ---: | ---: |
| Valid COLMAP-linked observations per image | 472–5151 | 3380 |
| Of 425 image blocks with >=1 track | 124–334 | 275.5 (64.8%) |
| Blocks with >=3 tracks | 53–288 | 216 (50.8%) |
| Blocks with >=10 tracks | 8–193 | 122.5 (28.8%) |

The lower-coverage views have sparse constraints across most broad-scale blocks. Even in covered blocks, a few SfM features are **not dense depth truth**. More importantly, the pinned SAD-GS and FastGS `Scene` implementations both initialize Gaussians from `scene_info.point_cloud`, which is derived from this same COLMAP reconstruction. The tracks are independent of the final optimized parameters but not of the methods' initialization or input pipeline. An apparent discrepancy with rendered depth could reflect SfM error, Gaussian surface thickness, visibility or the learned scene; it would not identify a causal 3D parameter correction. Track occupancy is also not native accepted-work coverage. Therefore this preflight does not authorize a geometry/occlusion training intervention or a full-view depth probe as a route to the original quality-and-cost claim.

Exact per-view counts and source are in `result.json` and `analyze.py`. Local and EPIC-05 SHA256 match: script `ec1adf859f20eeea026f001fc3f112a1543c714ea45a4511dbd749f597163624`; result `ed9df17d46602aa4fcce128ce36482cca487e69c91de9249ae5d3c4f323b400e`.
