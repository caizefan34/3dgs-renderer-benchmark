# C51 Sparse-Backward Patch Notes — Historical Reference (Round-2 Mechanism Evidence)

## 1. Context

C51 was the first mechanism-level attempt to exploit per-Gaussian gating in the fused backward kernel. It introduced an `importance_mask` (per-Gaussian boolean vector) to skip gradient computation and atomic-writes for gated-off Gaussians. This patch is preserved as historical reference: it is the direct ancestor of the R2.1 three-branch gating and shares the same kernel body.

**Files preserved here:**

| File | Purpose |
|------|---------|
| `IntersectTile.c1.cu` | Minimal C51 gating kernel — tile intersection with mask-based skip |
| `patch_cuda.py` | Patch script that applies C51 modifications to gsplat CUDA sources |
| `canonical_training.py` | C51 training harness (profiling-only) |
| `measure_recall.py` | Recall metric for gated gradient maps |
| `run_experiment.py` | C51 experiment runner |

## 2. What C51 Changed

1. **New input tensor**: `importance_mask` — `[N]` or `[nnz]` per-Gaussian boolean, where `1` means compute and accumulate gradient, `0` means skip.
2. **Kernel body**: The per-Gaussian inner loop checks `mask_batch[t]` before computing:
   - `sigma` (spatial exponent)
   - `alpha` (opacity-based alpha)
   - `vis` / `fac` (visibility times alpha)
   - per-channel gradient contributions
3. **Atomic writes**: masked Gaussians skip `atomicAdd` entirely — their gradient map blocks remain zero.
4. **B1/B3 vs B2 distinction**: `compute_densify_grad` flag preserved B2-only mode (compute `v_means2d` for gated-off Gaussians so densification still sees their view-space gradient).

## 3. Result Summary (from Round-2 evidence, see `reports/phase-r2-attribute-decoupled-backward-gate.md`)

The C51 gating achieved correct masking (recall of gated maps > 0.95 with >50% gating) but did **not** translate to meaningful wall-clock speedup: the fused kernel's traversal + `T` recompute + `buffer` update remain unconditional, and masking only skips the arithmetic portion. The R2 analysis confirmed the per-branch arithmetic cost is substantially below the traversal cost, which is the core difficulty this Candidate C audit must address.

## 4. Caveats

- The C51 patches were applied to a **custom fork** of gsplat for measurement; they are not part of the canonical baseline.
- The kernel shown is the minimal round-2 version — it is not the same file shipped with the repository's installed gsplat-1.4.0, but retains the same loop structure (verified line-for-line in the captured `RasterizeToPixels3DGSBwd.cu`).
