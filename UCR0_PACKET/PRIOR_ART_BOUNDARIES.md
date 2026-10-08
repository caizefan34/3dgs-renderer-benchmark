# PRIOR-ART BOUNDARIES — frozen 2026-09-30

## UCR0
StochasticSplats (ICCV 2025) already provides unbiased Monte-Carlo forward and reverse-mode
gradient estimators for stochastic, sorting-free splat rasterization. Eulerian Splatting (CVPR 2026)
already uses an unbiased control-variate estimator for gradients of a probabilistic Gaussian-center
density.

UCR0 is therefore NOT allowed to claim:
- first unbiased stochastic gradient for 3DGS;
- first control variate in Gaussian splatting;
- first stochastic/sorting-free splat training.

Candidate boundary being tested:
multi-fidelity control variates for the *standard sorted C0 full-resolution objective*:
a cheap deterministic full-image proxy plus sparse exact residual correction, with the fine C0
renderer/objective remaining the reference target.

Mandatory comparator: exact-unit Horvitz-Thompson sampling with NO proxy at the same sampling rate.
If the proxy does not provide >=2x variance reduction at equal q, the distinct mechanism is not useful.

References:
- StochasticSplats, ICCV 2025:
  https://openaccess.thecvf.com/content/ICCV2025/html/Kheradmand_StochasticSplats_Stochastic_Rasterization_for_Sorting-Free_3D_Gaussian_Splatting_ICCV_2025_paper.html
- Eulerian Splatting, CVPR 2026:
  https://euleriansplatting.github.io/

## VB0
Efficient Multi-View Training for 3DGS (2025) already establishes that multi-view batches reduce
gradient variance and implements thread-efficient partial rendering across views. GSReuse (TVCG 2026)
already warps screen-space content between neighboring views for inference acceleration.

VB0 is therefore NOT allowed to claim:
- first multi-view 3DGS training;
- first partial multi-view rasterization;
- first screen-space reuse between 3DGS views.

Candidate boundary:
one exact differentiable anchor render is reused through differentiable RGB/depth/contribution warps
to create gradients from 3/7 neighboring training cameras, without independently rasterizing those
neighbor views.

References:
- Efficient Multi-View Training:
  https://arxiv.org/abs/2506.12727
- GSReuse:
  https://pubmed.ncbi.nlm.nih.gov/41921174/

## MR0
Scaffold-GS (CVPR 2024) already uses sparse anchors that generate local view-adaptive neural Gaussians.
FastGS/SteepGS/EDGS and numerous compression works already reduce Gaussian count; compression works
also merge/prune Gaussians.

MR0 is therefore NOT allowed to claim:
- first structured/anchor Gaussian representation;
- first hierarchical/compact Gaussian representation;
- first Gaussian merging;
- first training with fewer Gaussians.

Candidate boundary:
a strict capacity upper bound asking whether K fine C0 Gaussians can be replaced by ONE directly
rendered, view-independent macro Gaussian while preserving multi-view contribution, transmittance,
and collective first-order response. No anchor MLP, no generated children, no view-conditioned decoder.

Reference:
- Scaffold-GS:
  https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Scaffold-GS_Structured_3D_Gaussians_for_View-Adaptive_Rendering_CVPR_2024_paper.html
