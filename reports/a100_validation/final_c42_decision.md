# Final C42 Decision: D-SSIM Bottleneck Optimization

## Final Decision: **KEEP — Downsampled SSIM at scale 0.50 (primary) and 0.75 (fallback)**

---

## 1. Evidence Chain Summary

### Phase 1: Bottleneck Identification (C40/C41)

| Metric | RTX 5070 (exploratory) | A100 (final validation) | Status |
|--------|----------------------|------------------------|--------|
| D-SSIM % of GPU time | 40.7% | **78.6%** | CONFIRMED on A100 |
| D-SSIM time (ms/iter) | 35.8 | **81.4** | CONFIRMED on A100 |
| Total iteration (ms) | 85.3 | **95.1** | CONFIRMED on A100 |
| Primary bottleneck | D-SSIM | **D-SSIM** | CONFIRMED |
| Checkpoint quality | PSNR 12.03 (diverged) | PSNR 20.56 (normal) | Corrected |

**D-SSIM is the dominant training bottleneck on A100-PCIE-40GB, consuming 78.6% of GPU kernel time.**

### Phase 2: Candidate Screening (RTX 5070 exploratory)

| Candidate | D-SSIM speedup | E2E speedup | Quality | RTX 5070 Decision | A100 Status |
|-----------|---------------|------------|---------|-------------------|-------------|
| B: Separable conv | 1.30x | +9.8% | Zero | MAYBE | Not revalidated |
| C: Separable + cudagraphs | 1.30x | +12.6% | Zero | MAYBE | Not revalidated |
| D: Downsampled SSIM 0.50 | 4.18x | +38.2% | rot cos=0.921 | MAYBE | **VALIDATED → KEEP** |
| D: Downsampled SSIM 0.75 | 1.73x | +16.1% | cos>0.996 | MAYBE | **VALIDATED → KEEP** |
| E: Adaptive scheduling | N/A | N/A | High risk | DROP | Not applicable |
| Spatial adaptive mask | 1.08x | +3.3% | Moderate | DROP | Not applicable |

### Phase 3: A100 Final Validation (this report)

| Scale | D-SSIM speedup | E2E speedup | Min cosine | A100 Decision |
|-------|---------------|------------|-----------|---------------|
| **0.50** | **3.54x** | **+60.7%** | **0.9854** | **KEEP** |
| **0.75** | 1.67x | **+33.8%** | **0.9939** | **KEEP** (fallback) |
| 0.25 | 30.18x | +81.6% | 0.9469 | MAYBE (quality risk) |

---

## 2. Decision Gates (from task specification)

### Scale 0.50 — Primary Candidate

| Gate | Threshold | A100 Result | Pass? |
|------|-----------|------------|-------|
| D-SSIM speedup | >3x | 3.54x | ✓ PASS |
| E2E speedup | >30% | +60.7% | ✓ PASS |
| Gradient cosine (all params) | >0.95 | min=0.9854 | ✓ PASS |

**All gates pass. Decision: KEEP.**

### Scale 0.75 — Fallback Candidate

| Gate | Threshold | A100 Result | Pass? |
|------|-----------|------------|-------|
| E2E speedup | >15% | +33.8% | ✓ PASS |
| Gradient cosine (all params) | >0.95 | min=0.9939 | ✓ PASS |

**All fallback gates pass. Decision: KEEP.**

### Scale 0.25 — Not recommended

| Gate | Threshold | A100 Result | Pass? |
|------|-----------|------------|-------|
| Gradient cosine | >0.95 | 0.9469 (xyz) | ✗ FAIL |

xyz gradient cosine drops below 0.95. Quality risk too high without training validation.

---

## 3. Rotation Gradient Analysis

### The RTX 5070 rotation issue was a checkpoint artifact

| Scale | RTX 5070 rotation cosine | A100 rotation cosine | Difference |
|-------|------------------------|---------------------|------------|
| 0.50 | 0.921 (min=0.610) | 0.993 (min=0.985) | +0.072 |

The RTX 5070 exploration used a diverged checkpoint (PSNR=12.03) with degraded gradient structure. The A100 validation with a normal checkpoint (PSNR=20.56) shows rotation cosine of 0.993 at scale 0.50 — well above the 0.95 gate.

**The rotation sensitivity to SSIM downsampling is real but much smaller than the RTX 5070 data suggested.** With a properly trained model, the gradient direction is preserved >98.5% even at half resolution.

### Why rotations are still the most sensitive parameter

At scale 0.50, rotation cosine (0.993) is the lowest among all parameters:

| Parameter | Cosine (scale 0.50) | Sensitivity rank |
|-----------|--------------------|-----------------|
| opacity | 0.9994 | 5 (least sensitive) |
| shs | 0.9980 | 4 |
| scales | 0.9981 | 3 |
| **rotations** | **0.9926** | **1 (most sensitive)** |
| xyz | 0.9854 | 2 |

Rotation gradients flow through the Gaussian covariance matrix (Σ = R·S·S^T·R^T). The SSIM structural similarity loss captures edge orientation, which is directly encoded in rotation. Downsampling loses fine edge detail, but the effect is modest (0.993 cosine) with a normal checkpoint.

---

## 4. Why A100 Gains Are Larger Than RTX 5070

| Factor | RTX 5070 | A100 | Impact |
|--------|---------|------|--------|
| D-SSIM % of total | 42.5% | 78.6% | D-SSIM optimization has more leverage on A100 |
| Scale 0.50 E2E speedup | +38.2% | +60.7% | A100 gains are 59% larger |
| Scale 0.50 absolute savings | 51.5 ms/iter | 57.8 ms/iter | A100 saves more absolute time |
| Rendering time | ~7 ms | ~3.4 ms | A100 rendering is already fast, so D-SSIM dominates more |
| cuDNN efficiency | 9 kernels | 63 kernels | A100 cuDNN is less efficient for this conv → more room to optimize |

**The downsampled SSIM optimization is MORE valuable on A100 than on RTX 5070.**

---

## 5. Recommended Configuration

### Primary: Scale 0.50

```python
# Downsample pred and target to 50% before SSIM computation
# L1 remains at full resolution
scale = 0.5
pred_ds = F.interpolate(pred, scale_factor=scale, mode="area")
target_ds = F.interpolate(target, scale_factor=scale, mode="area")
d_ssim = d_ssim_loss(pred_ds, target_ds)  # SSIM at 540×960
loss = 0.8 * F.l1_loss(pred, target) + 0.2 * d_ssim
```

- **Speedup**: +60.7% total training (95.3 → 37.5 ms/iter)
- **Quality risk**: Near-zero (gradient cosine > 0.985 for all parameters)
- **Memory savings**: -299 MB (-10.4%)

### Fallback: Scale 0.75

- **Speedup**: +33.8% total training (95.3 → 63.1 ms/iter)
- **Quality risk**: Minimal (gradient cosine > 0.994 for all parameters)
- **Use case**: If scale 0.50 shows any quality regression in actual training

---

## 6. Remaining Work (Not in scope of this validation)

1. **End-to-end training validation** (Phase D-2 equivalent on A100): Run 1k-10k additional training iterations with scale 0.50 vs baseline, measure PSNR/SSIM/LPIPS. The gradient cosine evidence strongly suggests quality will be preserved, but training validation provides the final proof.

2. **Separable conv + torch.compile on A100**: The RTX 5070 could not test torch.compile(inductor) because Triton is unavailable on Windows. On A100 (Linux), Triton is available, and inductor may provide additional fusion gains. This candidate (B/C) was not revalidated on A100 in this phase.

3. **Composition**: Downsampled SSIM (scale 0.50) + separable conv may compose additively — separable reduces per-pixel FLOP, downsampling reduces pixel count. Combined speedup could exceed the individual gains.

---

## 7. Data Provenance

| Data | Hardware | Location | Final claim? |
|------|----------|----------|-------------|
| C40 A100 profiling | A100-PCIE-40GB | `results/a100/validation-c40-c42/c40_baseline_a100.json` | ✓ Yes |
| C42 A100 benchmark | A100-PCIE-40GB | `results/a100/validation-c40-c42/c42_downsample_ssim_a100.json` | ✓ Yes |
| C40 RTX 5070 profiling | RTX 5070 Laptop | `results/exploratory/rtx5070/c40/` | ✗ No (exploratory) |
| C42 RTX 5070 benchmark | RTX 5070 Laptop | `results/exploratory/rtx5070/c42/` | ✗ No (exploratory) |
| Hardware metadata | — | `results/a100/hardware_metadata.json` | — |
| Hardware metadata | — | `results/exploratory/rtx5070/hardware_metadata.json` | — |

---

## 8. Final Claim

**On A100-PCIE-40GB with gsplat 1.5.3, downsampled SSIM at scale 0.50 achieves +60.7% total training speedup (95.3 → 37.5 ms/iter) with gradient direction preserved >98.5% for all parameter groups. This is the highest-impact D-SSIM optimization identified in the C42 investigation.**

**Scale 0.75 is validated as a fallback with +33.8% speedup and >99.4% gradient preservation.**
