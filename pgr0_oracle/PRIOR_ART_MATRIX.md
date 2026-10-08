# PGR0 Prior-Art Exclusion Matrix (freeze date: 2026-10-01)

| Work | Setting / changing axis | Reuse object | Training? | Exact redundancy cert? | PGR0 overlap / hazard |
|---|---|---|---|---|---|
| gsplat fused projection (this repo) | per-call forward vs backward | R, covar, mean_c, covar_c, J, cov2d | yes | n/a (same call) | Direct subject: the backward recomputes forward intermediates |
| 3DGS-LM (ICCV 2025) | optimization / LM inner solves | cached intermediate gradients for JVP | yes | no | Hazard to "first cache in 3DGS optimization"; PGR0 is projection-fwd/bwd, not LM |
| DSD-GS (2026) | dynamic-scene training, static/dynamic split | static-region compute/render caching | yes | frozen subset | Hazard to "first training-time cache"; PGR0 reuses within one projection call |
| TideGS (2026) | billion-scale training | SSD/CPU/GPU working-set parameter cache | yes | data-movement cache | Hazard to generic "training cache/system" claim; PGR0 is not a data-movement cache |
| Neo (ASPLOS 2026) | fixed scene, inference frames | depth-sort reuse-and-update | no | update logic | Hazard to "first temporal sorting reuse"; PGR0 is not temporal |
| CaT-GS (CVPR 2026) | fixed scene, nearby frames | visible-set/frustum cache + cached list | no | no exact proof | Hazard to "first cached rasterization list"; PGR0 is not a list cache |
| Lumina / S² (ISCA 2025) | fixed trained scene, moving camera | previous sort + radiance cache | no | no | Hazard to "first temporal reuse"; PGR0 is intra-call redundancy |
| Kinetic Data Structures (classic) | continuously changing keys | combinatorial structure by certificates | generic | certificate concept | Hazard to claiming certificates themselves as novel |

## Safe novelty wording if PGR0 ever advances to a CUDA pack
Candidate wording (only if a future memory-aware microbench confirms net speedup):

> We identify and certify intra-call redundancy in the differentiable 3DGS projection backward: the backward recomputes continuous intermediates (rotation, world/camera covariance, camera-space mean, perspective Jacobian, 2D covariance) that the forward already computed and discarded, and we show a cached-reuse backward is bit-gradient-equivalent to autograd.

## Forbidden novelty wording
Do not write "first temporal reuse", "first sorting reuse", "first training cache",
"first certificate-based reuse", or any cross-architecture generalization claim.

## Primary references recorded for DSH refresh
- gsplat (Ye et al., JMLR 2025), `projection_ewa_3dgs_fused` kernel + `Utils.cuh`.
- 3DGS-LM, ICCV 2025.
- DSD-GS, arXiv:2605.30863.
- TideGS, arXiv:2605.20150.
- Neo, ASPLOS 2026, DOI 10.1145/3779212.3790192.
- CaT-GS, CVPR 2026.
- Lumina, ISCA 2025 / arXiv:2506.05682.
- CGAL Kinetic Data Structures / Kinetic Sort (classic certificate precedent).
