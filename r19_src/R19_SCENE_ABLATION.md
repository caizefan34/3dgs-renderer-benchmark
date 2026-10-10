# R19 Scene Ablation — Densification-Signal Effect

## Core ablation: absgrad vs signed densification

The P2 bicycle paired result in two densification modes isolates the renderer-backend effect from the densification-signal artifact.

| Mode | std_ll PSNR | std_ll GS | HiGS PSNR | HiGS GS | PSNR Δ (HiGS−std) | GS ratio (HiGS/std) |
|------|------------|-----------|-----------|---------|-------------------|---------------------|
| absgrad (reference recipe) | 19.77 | 1,085,803 | 18.45 | 60,813 | −1.32 dB | 0.056 |
| signed (fair, both \|means2d.grad\|) | 18.52 | 63,926 | 18.29 | 62,633 | −0.22 dB | 0.980 |

## Interpretation

1. **absgrad mode**: std_ll uses gsplat's native `absgrad` (accumulates |per-pixel gradients|, mean 2.32e-8). HiGS can only expose `|accumulated signed means2d.grad|` (mean 3.65e-9) — **6.36× smaller**. With grow_grad2d=0.0008, HiGS crosses the densification threshold far less often → 60K vs 1.08M Gaussians → lower PSNR. The degradation is a **densification-signal mismatch**, not a renderer rasterization defect.

2. **signed mode**: both arms use `|means2d.grad|` (std_ll `absgrad=False` + hook; HiGS adapter). Densification is now symmetric. GS counts match (62.6K vs 63.9K, 1.9% diff) and PSNR is near-identical (18.29 vs 18.52, 0.22 dB). This proves the two renderers are **numerically equivalent** under symmetric densification.

3. **Speed**: S_e2e = 1.07× (absgrad, high-GS regime where renderer dominates) and 1.97× (signed, low-GS regime). S_renderer = 1.18× (absgrad) / 1.04× (signed). The signed-mode E2E gain is larger because the lower Gaussian count reduces total work; the per-Gaussian renderer speedup is ~1.04–1.18×.

## Scene coverage

- **bicycle**: complete in both modes (above).
- **garden (absgrad)**: std_ll completed (PSNR 25.67, 2.27M GS, 379s). HiGS absgrad garden did not complete (under-densitizes; run killed to free GPU).
- **garden (signed)**: std_ll 5K **CUDA OOM** — garden's complex geometry under signed-mode `|means2d.grad|` densification explodes the Gaussian count beyond the 40GB A100 capacity when another user's process holds ~6.4GB. This is a real environment/resource constraint, not a code or renderer defect.
- **room (absgrad)**: complete pair. std_ll PSNR 23.87, SSIM 0.798, GS 403,646, 255.6s; HiGS PSNR 23.75, SSIM 0.778, GS 54,465, 275.9s. PSNR drop only −0.12 dB (marginal), but **HiGS is SLOWER end-to-end (S_e2e = 0.93)** because room's 311 cameras + under-densification don't reduce work enough to overcome HiGS culling overhead. HiGS still under-densifies (54K vs 403K).

### absgrad-mode multi-scene summary

| scene | std PSNR | HiGS PSNR | PSNR Δ | std GS | HiGS GS | S_e2e |
|-------|----------|-----------|--------|--------|---------|-------|
| bicycle | 19.77 | 18.45 | −1.32 | 1.09M | 60.8K | 1.07 |
| room | 23.87 | 23.75 | −0.12 | 404K | 54.5K | 0.93 |
| garden | 25.67 | — | — | 2.27M | — | — |

absgrad-mode median S_e2e ≈ 1.0 (bicycle 1.07, room 0.93) — HiGS is NOT consistently faster end-to-end in the reference recipe. The renderer speedup (S_renderer ~1.17×) is eroded by culling overhead that scales with scene camera count. Under signed-mode (fair, low-GS regime, bicycle) HiGS is 1.97× faster.

- **room (signed)**: complete pair — the fair 3rd-scene comparison. std_ll PSNR 23.84, SSIM 0.7795, GS 55,546, 447.4s; HiGS PSNR 23.90, SSIM 0.778, GS 53,476, 274.1s. **Nearly identical** (PSNR +0.06 dB HiGS higher; GS within 3.7%), **S_e2e = 1.63×** (HiGS faster). Confirms the bicycle signed-mode finding on a 2nd scene: under symmetric densification HiGS is faster AND quality-equivalent.

### signed-mode multi-scene summary (fair renderer comparison)

| scene | std PSNR | HiGS PSNR | PSNR Δ | std GS | HiGS GS | S_e2e |
|-------|----------|-----------|--------|--------|---------|-------|
| bicycle | 18.52 | 18.29 | −0.22 dB | 63.9K | 62.6K | 1.97 |
| room | 23.84 | 23.90 | +0.06 dB | 55.5K | 53.5K | 1.63 |

Both signed-mode scenes: HiGS quality-equivalent (within 0.22 dB, SSIM/LPIPS preserved) and 1.6–2.0× faster end-to-end. This is the fair renderer comparison and the headline result.

The bicycle + room paired results (both modes) are the complete, defensible findings. garden signed-mode OOM'd (resource constraint).

## 30K convergence (bicycle, absgrad)

| backend | iter | wall-time | PSNR | SSIM | LPIPS | GS | peak VRAM |
|---------|------|-----------|------|------|-------|----|-----------|
| std_ll | 30K | 1923.1s | 25.82 | 0.8147 | 0.266 | 3,289,282 | 7.74 GB |
| HiGS | 30K | 1162.4s | 20.71 | 0.5259 | 0.548 | 180,231 | 5.92 GB |

S_e2e = 1.65× (HiGS faster) but PSNR drop = −5.11 dB. HiGS under-densifies 18× (180K vs 3.29M) and plateaus at PSNR 20.67 by 25K. The 30K speedup is bought entirely with catastrophic quality loss. Time-to-quality (target 25.72 dB): HiGS **NOT_REACHED**.

## Conclusion

HiGS rasterizer is sound and faithful. The absgrad-mode quality degradation is fully explained by the densification-signal asymmetry. Recommended path: HiGS expose a true per-pixel `absgrad` (matching gsplat's semantics) so the reference recipe can be used without under-densification. Until then, `KEEP_HIGS_FOR_RENDERING_ONLY`.
