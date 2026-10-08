# PRIOR-ART / POSITIONING BOUNDARIES — 2026-09-30

## VJS0
3DGS-LM (ICCV 2025) already replaces Adam with LM and uses a cached gradient/Jacobian data structure
plus custom J/J^T kernels and PCG. 3DGS^2-TR (ICML 2026) already uses second-order curvature and a
Hellinger trust region.

Therefore VJS0 is NOT a novelty claim. It is a feasibility upper-bound:
can one exact C0 renderer linearization be amortized across >=4 cheap inner parameter updates in
early/mid training without frequent exact re-rasterization?

References:
- 3DGS-LM:
  https://openaccess.thecvf.com/content/ICCV2025/html/Hollein_3DGS-LM_Faster_Gaussian-Splatting_Optimization_with_Levenberg-Marquardt_ICCV_2025_paper.html
- 3DGS^2-TR:
  https://proceedings.mlr.press/v306/hsiao26a.html

## SFO0
The official 3DGS repository already exposes SparseGaussianAdam when the accelerated rasterizer is
installed and steps it using the visible mask. Therefore visible-only sparse Adam is NOT novel.

SFO0 is a systems lane only:
measure whether optimizer time in frozen C0 can be reduced enough that UCR/VB-style raster or
information improvements are not capped by the HM0 optimizer floor.

Reference:
- official 3DGS train.py:
  https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py

## MPI0
Efficient Multi-View Training (2025) already motivates multi-view batches by single-view gradient
variance and modifies rasterization to reduce multi-view overhead.

MPI0 is therefore NOT a novelty claim. It is an exact 8xA100 upper bound:
how much useful optimization progress per wall-clock second can exact 8-view gradient aggregation
deliver when view renders are physically parallel?

Reference:
- Efficient Multi-View Training:
  https://arxiv.org/abs/2506.12727
