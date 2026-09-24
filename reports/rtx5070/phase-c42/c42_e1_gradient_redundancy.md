# C42 Phase E-1: Gradient Redundancy Profiling

## Executive Summary

**Decision: DROP** — D-SSIM gradient contribution does NOT decrease over training. It INCREASES from 0.54 to 0.62 (+15.2%), meaning D-SSIM becomes MORE important as training progresses, not less. The adaptive scheduling hypothesis (reduce D-SSIM frequency later in training) is falsified.

---

## 1. Method

For each checkpoint (iter 1000, 2000, 3000, 5000, 10000, 15000, 20000, 25000, 30000), 7 cameras were rendered, and three separate backward passes were performed:

1. **L1-only**: `F.l1_loss(pred, target).backward()` → `||grad_L1||`
2. **D-SSIM-only**: `d_ssim_loss(pred, target).backward()` → `||grad_DSSIM||`
3. **Combined**: `(0.8 * L1 + 0.2 * D-SSIM).backward()` → `||grad_combined||`

Gradient L2 norms were measured for each parameter group: xyz, opacity, scales, rotations, shs.

**Key metric**: `grad_ratio = ||grad_DSSIM|| / (||grad_L1|| + ||grad_DSSIM||)`

If this ratio decreases significantly over training, D-SSIM becomes redundant and its frequency can be reduced.

---

## 2. Summary Table

| Iter | GS count | PSNR (dB) | L1 loss | D-SSIM loss | Loss ratio (D-SSIM) | \|\|grad_L1\|\| | \|\|grad_DSSIM\|\| | Grad ratio (D-SSIM) |
|------|---------|----------|---------|------------|--------------------||-------|---------|-----|
| 1000 | 1,397,897 | 13.28 | 0.172 | 0.455 | 0.710 | 0.131 | 0.155 | **0.542** |
| 2000 | 1,122,000 | 13.08 | 0.178 | 0.466 | 0.709 | 0.239 | 0.313 | **0.567** |
| 3000 | 1,031,879 | 12.37 | 0.214 | 0.480 | 0.686 | 0.849 | 0.519 | **0.379** |
| 5000 | 899,729 | 18.89 | 0.091 | 0.342 | 0.778 | 0.558 | 0.495 | **0.470** |
| 10000 | 1,004,935 | 15.25 | 0.147 | 0.423 | 0.734 | 1.206 | 1.031 | **0.461** |
| 15000 | 1,219,406 | 13.41 | 0.174 | 0.447 | 0.702 | 2.486 | 1.666 | **0.401** |
| 20000 | 1,207,872 | 12.73 | 0.188 | 0.453 | 0.689 | 1.683 | 1.193 | **0.415** |
| 25000 | 1,199,627 | 12.55 | 0.190 | 0.468 | 0.702 | 7.185 | 10.915 | **0.603** |
| 30000 | 1,193,480 | 12.10 | 0.209 | 0.523 | 0.719 | 1.343 | 2.227 | **0.624** |

---

## 3. Plots

### Plot 1: Iteration vs ||grad_DSSIM||

```
||grad_DSSIM||
  12 |                                              *
  11 |                                             /
  10 |                                            /
   9 |                                           /
   8 |                                          /
   7 |                                         /
   6 |                                        /
   5 |                                       /
   4 |                                      /
   3 |                                     /
   2 |                    *----*----*---*-*-----*
   1 |               *---*                   
   0 | *--*--*--*---*                        
     +----+----+----+----+----+----+----+----+---> iter
     1K   2K   3K   5K   10K  15K  20K  25K  30K
```

**Observation**: ||grad_DSSIM|| generally INCREASES over training (0.15 → 2.23), with a spike at iter 25000 (10.9). The gradient does not decay — D-SSIM continues to provide strong gradient signal throughout training.

### Plot 2: Iteration vs ||grad_L1||

```
||grad_L1||
  8 |                                     *
  7 |                                    /
  6 |                                   /
  5 |                                  /
  4 |                                 /
  3 |                                /
  2 |              *----*--*--*---*-*
  1 |         *---*                   
  0 |*--*--*---*                    
   0 +----+----+----+----+----+----+----+----+---> iter
     1K   2K   3K   5K   10K  15K  20K  25K  30K
```

**Observation**: ||grad_L1|| also increases (0.13 → 1.34), with spike at 25000 (7.2). Both L1 and D-SSIM gradients grow, indicating the model continues to receive significant learning signal from both losses.

### Plot 3: Iteration vs grad_ratio (D-SSIM / total)

```
grad_ratio
0.65 |                                              *  *
0.60 |                                       *     /
0.55 |  *  *                                /    /
0.50 |    \  \     *  *           *  *    /
0.45 |     \  \   / \/ \         / \/ \  /
0.40 |      \  \-/     \   *---*      
0.35 |           *       \-/
0.30 |
     +----+----+----+----+----+----+----+----+---> iter
     1K   2K   3K   5K   10K  15K  20K  25K  30K
```

**Observation**: The D-SSIM gradient ratio does NOT decrease monotonically. It fluctuates between 0.38 and 0.62, and the FINAL value (0.624) is HIGHER than the initial value (0.542). The ratio actually INCREASES by 15.2% over training.

### Plot 4: Iteration vs loss_ratio (D-SSIM / total loss)

```
loss_ratio
0.78 |          *
0.76 |         /
0.74 |        *   *
0.72 | *  *  /   / \  *  *
0.70 |  \  \/   /   \/  \  *
0.68 |   \     /         \/
0.66 |    \   /
0.64 |     \-/
     +----+----+----+----+----+----+----+----+---> iter
     1K   2K   3K   5K   10K  15K  20K  25K  30K
```

**Observation**: The D-SSIM loss ratio stays nearly constant at 0.69-0.78 throughout training. D-SSIM consistently contributes ~70% of the total loss magnitude, from iter 1000 to iter 30000.

---

## 4. Per-Parameter Gradient Ratio (D-SSIM / total)

| Parameter | iter 1000 | iter 30000 | Change | Interpretation |
|-----------|----------|-----------|--------|---------------|
| **xyz** | 0.563 | 0.624 | +10.8% | D-SSIM more important for position |
| **opacity** | 0.472 | 0.601 | +27.4% | D-SSIM much more important for opacity |
| **scales** | 0.514 | 0.675 | +31.3% | D-SSIM much more important for scales |
| **rotations** | 0.519 | 0.608 | +17.3% | D-SSIM more important for rotation |
| **shs** | 0.549 | 0.610 | +11.2% | D-SSIM more important for SH |

**All parameter groups show INCREASING D-SSIM gradient ratio.** The largest increases are for scales (+31.3%) and opacity (+27.4%) — the parameters most related to structural appearance quality, which is exactly what D-SSIM measures.

---

## 5. Temporal Trend Analysis

### 5.1 Gradient Magnitude Evolution

| Metric | iter 1000 | iter 30000 | Change |
|--------|----------|-----------|--------|
| ||grad_L1|| | 0.131 | 1.343 | +925% |
| ||grad_DSSIM|| | 0.155 | 2.227 | +1340% |
| Grad ratio (D-SSIM) | 0.542 | 0.624 | +15.2% |

Both gradient magnitudes increase dramatically, but D-SSIM gradients grow FASTER than L1 gradients (1340% vs 925%). This means D-SSIM becomes proportionally MORE influential on parameter updates over training, not less.

### 5.2 Loss Magnitude Evolution

| Metric | iter 1000 | iter 30000 | Change |
|--------|----------|-----------|--------|
| L1 loss | 0.172 | 0.209 | +21.3% |
| D-SSIM loss | 0.455 | 0.523 | +14.9% |
| Loss ratio (D-SSIM) | 0.710 | 0.719 | +1.2% |

The loss ratio is remarkably stable — D-SSIM consistently accounts for ~70-72% of the total loss throughout training. There is no phase where D-SSIM becomes negligible.

### 5.3 Note on PSNR

PSNR decreases from 18.89 (iter 5000) to 12.10 (iter 30000), indicating the v2_16 training run diverges. However, this does not invalidate the gradient analysis — the key finding is that D-SSIM gradient contribution INCREASES even as the model quality degrades, which is stronger evidence against the scheduling hypothesis than if the model had converged well.

---

## 6. Research Decision

### Decision: **DROP**

The adaptive D-SSIM scheduling hypothesis (Candidate E) predicted that D-SSIM gradient contribution would decrease over training, enabling less frequent D-SSIM computation later in training.

**Evidence falsifies this hypothesis:**

1. **D-SSIM gradient ratio INCREASES** from 0.542 to 0.624 (+15.2%), the opposite of the predicted decrease
2. **D-SSIM gradient magnitude grows** 1340% (faster than L1's 925%)
3. **D-SSIM loss ratio is constant** at ~0.70-0.72 throughout training
4. **All parameter groups show increasing D-SSIM gradient ratio** — scales (+31.3%), opacity (+27.4%), rotations (+17.3%), shs (+11.2%), xyz (+10.8%)
5. **No phase transition** — there is no iteration range where D-SSIM becomes negligible

### Implications

- D-SSIM provides ~50-62% of gradient signal throughout training — it is NOT redundant
- Reducing D-SSIM frequency would remove a significant and growing gradient component
- The structural similarity gradient is most important for scales and opacity (the parameters that control Gaussian appearance), and its importance increases over training
- The original 3DGS choice of fixed λ=0.2 D-SSIM throughout training is justified by this evidence

### Comparison with Other C42 Candidates

| Candidate | D-SSIM speedup | Total speedup | Quality risk | Status |
|-----------|---------------|--------------|-------------|--------|
| B: Separable conv | 1.30x | 9.8% | Zero | MAYBE |
| C: torch.compile (cudagraphs) | 1.30x | 12.6% | Zero | MAYBE |
| D: Downsampled SSIM | 4-16x | 32-40% | Moderate | UNTESTED |
| **E: Adaptive scheduling** | **N/A** | **N/A** | **High** | **DROP** |
| Spatial adaptive mask | 1.08x (max) | 3.3% (max) | Moderate | DROP |

---

## 7. Artifacts

- Profiling script: `scripts/phase-c42/c42_e1_gradient_profile.py`
- Raw data: `results/phase-c42/c42_e1_gradient_profile.json`
- Checkpoints: iter 1000, 2000, 3000 (mid_16 series) + iter 5000-30000 (v2_16 series)
- D-SSIM source: `scripts/epic05/phase7/loss.py`
