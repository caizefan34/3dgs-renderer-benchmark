# PGR0 Prior-Art Refresh (2026-10-01)

Refresh performed immediately before the PGR0 verdict. Scope: intra-call
redundancy in the differentiable 3DGS projection backward.

## What PGR0 actually claims (narrow)
PGR0 identifies that the gsplat fused projection backward recomputes continuous
intermediates (R, world covar, mean_c, covar_c, J, covar2d) that the forward
already computed and discarded, and that a cached-reuse backward is
bit-gradient-equivalent to autograd. PGR0 makes NO speedup claim (P0 runs no
CUDA) and NO cross-architecture claim.

## Overlap / hazard assessment
| Work | Relationship to PGR0 | Hazard |
|---|---|---|
| gsplat fused projection (subject) | Direct subject of the redundancy finding | none (this is the analyzed code) |
| 3DGS-LM (ICCV 2025) | caches intermediate gradients for LM JVP solves | hazard to "first cache in 3DGS optimization"; PGR0 is projection fwd/bwd, not LM |
| DSD-GS (2026) | static-region training cache after decomposition | hazard to "first training cache"; PGR0 reuses within one projection call |
| TideGS (2026) | working-set parameter cache for billion-scale training | hazard to generic "training cache/system"; PGR0 is not a data-movement cache |
| Neo (ASPLOS 2026) | inference-frame sort reuse-and-update | hazard to "first temporal sorting reuse"; PGR0 is not temporal |
| CaT-GS (CVPR 2026) | inter-frame visible-set/list cache | hazard to "first cached raster list"; PGR0 is not a list cache |
| Lumina/S2 (ISCA 2025) | temporal sort + radiance cache | hazard to "first temporal reuse"; PGR0 is intra-call |
| Kinetic Data Structures (classic) | certificate-based structure maintenance | hazard to claiming certificates as novel |

## Boundary
PGR0's differentiator, if ever advanced to a CUDA pack, is: certifying
intra-call forward/backward redundancy in the differentiable projection backward
and showing a reuse path is gradient-equivalent to autograd. This is distinct
from temporal caching, sorting reuse, list caching, and data-movement caching.

## Forbidden wording (still in force)
No "first temporal reuse", "first sorting reuse", "first training cache",
"first certificate-based reuse", or cross-architecture generalization.

## Primary references
- gsplat (Ye et al., JMLR 2025) - projection_ewa_3dgs_fused, Utils.cuh.
- 3DGS-LM, ICCV 2025. DSD-GS, arXiv:2605.30863. TideGS, arXiv:2605.20150.
- Neo, ASPLOS 2026, DOI 10.1145/3779212.3790192. CaT-GS, CVPR 2026.
- Lumina, ISCA 2025 / arXiv:2506.05682. CGAL Kinetic Data Structures.
