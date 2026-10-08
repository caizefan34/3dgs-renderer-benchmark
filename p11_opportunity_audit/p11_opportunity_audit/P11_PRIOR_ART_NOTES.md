# P11 Prior-Art / Ownership Notes

## Existing fused SSIM
The public fused-ssim implementation already provides CUDA forward/backward and supports
both `padding="same"` and `padding="valid"`. Current gsplat can use fused-ssim, and the
current Graphdeco training acceleration path also integrates fused SSIM. Therefore:
- "CUDA SSIM" is not novel.
- "separable SSIM" is not novel.
- any P11 loss contribution must be narrower than replacing PyTorch convs with a CUDA loss.

## Public training systems
Current Graphdeco also exposes an accelerated rasterizer and SparseGaussianAdam. Faster-GS
and other engines include fused optimizer code. Generic fused Adam or "update only visible
Gaussians" is therefore not a safe novelty direction, and sparse Adam can differ from dense
Adam semantics.

## Loss/raster boundary
The audit found public systems that optimize/fuse **loss operators**, but did not yet identify
a public CUDA 3DGS implementation whose exact photometric-loss adjoint is produced and consumed
inside the raster backward so that the standalone full-image `dL/dimage` boundary disappears.
This is only a scoped audit result, not a priority claim.

A prototype may use the working name:
`LARF — Loss-Adjoint Raster Fusion`

Only use this name if P11-L passes the headroom gate.

## Historical project evidence
Do not reuse superseded opportunity numbers:
- R6-B zero-init old 5.5–11.6% backward attribution was false; direct measurement reduced it
  to 0.14–0.36% backward.
- R6-A atomic aggregation old estimates were corrected; direct-counter median E2E opportunity
  is ~1.38%, best ~2.57%.
- R6-C fusion old 4.8–17.2% estimate was repaired to ~0.48–2.21% E2E.
- C25/C28 sparse-tail history must be read through the final C28 closure, not early KEEP reports.
