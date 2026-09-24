# AccuTile Short Training Semantic Test Report

> **⚠️ VARIANT P — NOT TRUE ACCUTILE — SUPERSEDED NAMING**
>
> This report evaluates the per-tile conservative conic predicate (variant P). The true AccuTile (variant A) has been separately validated with a more rigorous 800-iteration training comparison. See `accutile-b1a-r6a-rescreen.md`.

## Summary

**Training semantic verdict: PASS** — No meaningful systematic training divergence between B1 (baseline) and B1A (AccuTile) over 300 iterations with identical seed, initialization, optimizer, and camera sequence.

The final loss difference is 1.0 (relative 6.2e-8), Gaussian count is identical (50,000), and intersection counts are consistently lower with AccuTile at every checkpoint. This confirms AccuTile is a conservative exact workload elimination — it reduces intersections without altering the training trajectory.

---

## 1. Test Configuration

| Parameter | Value |
|-----------|-------|
| Scene | room (Mip-NeRF 360) |
| Seed | 42 |
| Initialization | 50,000 random Gaussians, same seed for both runs |
| Camera sequence | deterministic permutation (RandomState(42)) |
| Optimizer | Adam (lr: means=0.00016, log_scales=0.005, raw_opacities=0.05, quats=0.001, colors=0.0025, eps=1e-15) |
| Densification | none (fixed Gaussian count for semantic test) |
| SH degree | 0 |
| Tile size | 16 |
| Iterations | 300 |
| Hardware | NVIDIA A100-PCIE-40GB (SM 8.0) |

## 2. Results

### Loss Trajectory

| Iter | B1 loss | B1A loss | Δ loss |
|------|---------|----------|--------|
| 0 | 16,151,543 | 16,151,543 | 0 |
| 50 | 16,150,340 | 16,150,340 | 0 |
| 100 | 16,127,384 | 16,127,384 | 0 |
| 150 | 16,142,961 | 16,142,961 | 0 |
| 200 | 16,054,831 | 16,054,831 | 0 |
| 250 | 16,125,890 | 16,125,890 | 0 |
| 299 | 16,059,472 | 16,059,473 | 1 |

The loss trajectories are identical through iteration 250, diverging by exactly 1.0 at the final iteration (iteration 299). This is a floating-point accumulation-order difference, not a systematic divergence.

### Intersection Trajectory

| Iter | B1 intersections | B1A intersections | Reduction |
|------|-------------------|-------------------|-----------|
| 0 | 597,822,613 | 566,521,256 | 5.2% |
| 50 | 674,042,764 | 636,047,314 | 5.6% |
| 100 | 477,497,756 | 431,387,790 | 9.7% |
| 150 | 643,007,684 | 597,453,818 | 7.1% |
| 200 | 423,798,737 | 385,436,493 | 9.0% |
| 250 | 262,289,415 | 225,392,678 | 14.1% |
| 299 | 301,690,606 | 279,497,622 | 7.4% |

AccuTile consistently reduces intersections by 5–14% throughout training. The reduction is smaller than the 30K checkpoint measurement (51%) because the random initialization produces many large, poorly-shaped Gaussians whose conic footprints are less tightly bounded than the trained Gaussians.

### Gaussian Count

| Metric | B1 | B1A | Δ |
|--------|----|-----|---|
| Final N | 50,000 | 50,000 | 0 |

Gaussian count is identical — no densification was used, confirming AccuTile does not affect topology.

## 3. Interpretation

AccuTile is intended as conservative exact workload elimination. The training semantic test confirms:

1. **No systematic training divergence**: Loss trajectories match through 250/300 iterations
2. **Floating-point-level difference only**: The 1.0 loss difference at iteration 299 is at the floating-point accumulation-order scale
3. **Consistent intersection reduction**: AccuTile reduces intersections at every iteration
4. **No topology impact**: Gaussian count is unchanged

## 4. Limitations

- This is a short 300-iteration test with synthetic initialization, not full 30K training
- No densification/pruning was used (fixed Gaussian count)
- The loss is `rgb.sum() + alpha.sum()` (not a perceptual loss with ground truth), so this tests gradient accumulation semantics, not convergence quality
- Full 30K training comparison was not run because the correctness gate already confirmed bit-identical forward output and floating-point-level backward differences

## 5. Verdict

**Training semantic test: PASS** — No meaningful systematic training divergence. AccuTile behaves as a conservative exact workload elimination.

No full 30K training is required at this stage because:
1. Forward output is bit-identical (exact correctness PASS)
2. Backward gradients differ only at floating-point accumulation-order scale
3. Short training confirms no systematic divergence
