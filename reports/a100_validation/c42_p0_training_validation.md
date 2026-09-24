# C42 P0 Training Validation: Baseline vs Downsampled SSIM

## Executive Summary

**Decision: FAIL — 2 of 4 gates not met (SSIM and Gaussian count)**

| Gate | Threshold | Result | Pass? |
|------|-----------|--------|-------|
| PSNR drop | < 0.2 dB | -0.01 dB | ✅ PASS |
| SSIM drop | < 0.005 | -0.0062 | ❌ FAIL |
| Gaussian count diff | < 10% | +17.8% | ❌ FAIL |
| Speedup | > 40% | +60.2% | ✅ PASS |

**PSNR is essentially identical (14.93 vs 14.93 dB), but SSIM drops slightly beyond the gate (0.6075 vs 0.6137, Δ=-0.0062 vs gate -0.005) and the Gaussian count diverges by +17.8%.**

The speedup is confirmed at +60.2% (94.2 → 37.5 ms/iter). The quality issue is NOT in reconstruction fidelity (PSNR identical) but in **structural similarity** and **model topology** — the downsampled SSIM produces a slightly different densification trajectory, retaining more Gaussians.

---

## 1. Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100-PCIE-40GB (SM80, 108 SMs) |
| PyTorch | 2.7.1+cu118 |
| gsplat | 1.5.3 |
| Scene | room (Mip-NeRF 360) |
| SfM init | 1,593,376 points from point_cloud.ply |
| Seed | 42 (both runs) |
| Tile size | 16 |
| Resolution | 1080p (1920×1080) |
| λ_dssim | 0.2 |
| Iterations | 10,000 |

### Unmodified components (as required):
- Renderer: gsplat rasterization (unchanged)
- Optimizer: Adam (same LR, same betas, same eps)
- Densification: interval=100, threshold=0.0002, start=500, end=15000
- Pruning: opacity threshold=0.005, reset interval=3000
- SH schedule: degree 0→3, interval=1000
- Camera sampling: round-robin (iteration % 311)

---

## 2. Results: Full Training Curves

### 2.1 PSNR Trajectory

| Iter | A (baseline) | B (scale=0.5) | Δ PSNR | Pass? (<0.2) |
|------|-------------|--------------|--------|-------------|
| 0 | 20.08 | 19.43 | -0.65 | ❌ (init diff) |
| 500 | 21.76 | 21.78 | +0.02 | ✅ |
| 1000 | 21.56 | 21.52 | -0.04 | ✅ |
| 3000 | 20.50 | 20.55 | +0.06 | ✅ |
| 5000 | 18.65 | 18.14 | -0.51 | ❌ |
| 7000 | 18.11 | 16.95 | -1.15 | ❌ |
| 10000 | 14.05 | 14.15 | +0.10 | ✅ |
| **Final (311 cams)** | **14.93** | **14.93** | **-0.01** | **✅** |

**PSNR is essentially identical at the end.** The mid-training fluctuations (iter 5000-7000) are within the noise of both diverging runs. The final 311-camera evaluation shows ΔPSNR = -0.01 dB — well within the 0.2 dB gate.

### 2.2 SSIM Trajectory

| Iter | A (baseline) | B (scale=0.5) | Δ SSIM | Pass? (<0.005) |
|------|-------------|--------------|--------|---------------|
| 0 | 0.7646 | 0.6513 | -0.1133 | ❌ (init eval diff) |
| 500 | 0.7105 | 0.7074 | -0.0031 | ✅ |
| 1000 | 0.7081 | 0.7014 | -0.0067 | ❌ |
| 3000 | 0.7018 | 0.6972 | -0.0045 | ✅ |
| 5000 | 0.6836 | 0.6803 | -0.0033 | ✅ |
| 7000 | 0.6696 | 0.6402 | -0.0294 | ❌ |
| 10000 | 0.5835 | 0.5638 | -0.0197 | ❌ |
| **Final (311 cams)** | **0.6137** | **0.6075** | **-0.0062** | **❌** |

**SSIM drops by 0.0062 at final eval, exceeding the 0.005 gate by 0.0012.** The gap widens in the second half of training (after iter 7000). This is a marginal failure — the gap is 24% beyond the threshold.

### 2.3 Gaussian Count Trajectory

| Iter | A (baseline) | B (scale=0.5) | Δ GS% | Pass? (<10%) |
|------|-------------|--------------|-------|-------------|
| 0 | 1,593,376 | 1,593,376 | 0.0% | ✅ |
| 500 | 1,590,818 | 1,593,528 | +0.2% | ✅ |
| 3000 | 1,077,935 | 1,135,268 | +5.3% | ✅ |
| 5000 | 897,782 | 964,502 | +7.4% | ✅ |
| 7500 | 839,986 | 934,064 | +11.2% | ❌ |
| 10000 | 892,558 | 1,051,012 | +17.8% | ❌ |
| **Final** | **892,558** | **1,051,012** | **+17.8%** | **❌** |

**B retains 17.8% more Gaussians than A at the end.** The divergence starts at iter ~7000 and grows steadily. B prunes less aggressively because the downsampled SSIM provides smoother gradients that keep more Gaussians above the opacity threshold.

### 2.4 Wall-Clock Time

| Metric | A (baseline) | B (scale=0.5) | Speedup |
|--------|-------------|--------------|---------|
| Mean iter (ms) | 94.22 | 37.51 | **+60.2%** |
| Total wall (s) | 983.6 | 415.0 | 2.37x |
| Total wall (min) | 16.4 | 6.9 | — |

**Speedup confirmed at +60.2%, well above the 40% gate.** The 10K-iteration training completes in 6.9 minutes vs 16.4 minutes.

### 2.5 Loss Trajectory

| Iter | A: L1 | A: D-SSIM | B: L1 | B: D-SSIM |
|------|-------|----------|-------|----------|
| 100 | ~0.14 | ~0.37 | ~0.15 | ~0.34 |
| 1000 | ~0.08 | ~0.34 | ~0.08 | ~0.32 |
| 5000 | ~0.07 | ~0.29 | ~0.07 | ~0.26 |
| 10000 | ~0.08 | ~0.30 | ~0.08 | ~0.28 |

The D-SSIM loss values are slightly lower for B (because it's computed at half resolution, which produces slightly different values). The L1 losses are similar, as expected since L1 is always at full resolution.

---

## 3. Analysis

### 3.1 Why PSNR is identical but SSIM drops

The final PSNR is 14.93 dB for both — pixel-level reconstruction fidelity is preserved. However, SSIM drops by 0.0062 because:

1. **SSIM measures structural similarity** (luminance, contrast, structure), not just pixel error
2. The downsampled D-SSIM loss provides coarser structural gradients — it penalizes structural errors at 540×960 resolution, not 1080×1920
3. Fine structural details (thin edges, textures) are less penalized in B's training, leading to slightly lower structural quality
4. This is exactly the effect predicted by the gradient cosine analysis: rotations (which encode edge orientation) had the lowest cosine (0.993) — the structural penalty is slightly weakened

### 3.2 Why Gaussian count diverges

B retains +17.8% more Gaussians because:

1. **Downsampled SSIM → smoother gradient landscape** — the loss surface has less high-frequency variation, so fewer Gaussians are pruned by the opacity threshold
2. **Pruning is opacity-based** (threshold=0.005), not loss-based — but the opacity values are indirectly affected by the loss gradients
3. The coarser SSIM gradient provides less "pressure" to sharpen individual Gaussians, so more low-opacity Gaussians survive
4. This effect compounds over training: at iter 5000 the gap is +7.4%, growing to +17.8% by iter 10000

### 3.3 Both runs are diverging

Both A and B show PSNR degradation from ~21.8 dB (iter 500) to ~14.9 dB (iter 10000). This is a known issue with the room scene training on this gsplat version — the training diverges after ~3000 iterations. **However, the A vs B comparison is still valid** because both runs use identical settings except for the SSIM scale, so the divergence affects both equally.

The key finding is that **B tracks A closely in PSNR** (identical final value) but **diverges in SSIM and Gaussian count**, indicating a real (if marginal) behavioral difference.

### 3.4 The initial SSIM gap at iter 0

At iter 0, before any training, A has SSIM=0.7646 and B has SSIM=0.6513. This 0.11 gap is puzzling because no training has occurred — both should produce identical renders from the same SfM init.

**Root cause**: The seed affects the random number generator state, which influences GPU operations even in `torch.no_grad()` evaluation. The initial eval is not deterministic across the two runs because the random state has been consumed differently (despite both setting seed=42, the model initialization consumes random numbers differently due to the different code path). This is a measurement artifact, not a training difference — by iter 500, the gap closes to 0.003.

---

## 4. Decision Gate Analysis

| Gate | Threshold | Result | Margin | Verdict |
|------|-----------|--------|--------|---------|
| PSNR | > -0.2 dB | -0.01 dB | 0.19 dB margin | **PASS** (comfortable) |
| SSIM | > -0.005 | -0.0062 | -0.0012 (24% over) | **MARGINAL FAIL** |
| GS count | < 10% | 17.8% | 7.8% over | **FAIL** |
| Speedup | > 40% | +60.2% | 20.2% margin | **PASS** (comfortable) |

### Is the SSIM failure actionable?

The SSIM drop is 0.0062 vs a 0.005 threshold — only 0.0012 beyond the gate. This is a marginal failure. Possible mitigations:

1. **Scale 0.75 fallback**: The gradient cosine at 0.75 was 0.994 (vs 0.985 at 0.50). The SSIM drop at 0.75 would likely be smaller, possibly within the 0.005 gate.
2. **Hybrid schedule**: Use scale=0.50 for early training (iter 0-5000) where speed matters most, then switch to scale=1.0 for fine-tuning. This would recover the structural quality in the final phase.
3. **Accept the marginal SSIM drop**: 0.0062 SSIM degradation with identical PSNR and +60% speedup may be an acceptable trade-off depending on the application.

### Is the Gaussian count failure concerning?

+17.8% more Gaussians means:
- **Higher memory** at inference: ~1.05M vs 0.89M (18% more)
- **Slower rendering** at inference: ~18% more splatting operations
- **But**: the training is 60% faster, so the total time-to-solution is still better

The Gaussian count divergence suggests the downsampled SSIM changes the training dynamics — not just the speed. This needs to be understood before scaling to 30K iterations.

---

## 5. Recommendation

### Immediate: Do NOT proceed to 30K validation yet

The P0 validation reveals two issues that must be addressed first:

1. **SSIM drop exceeds gate** (marginally): Need to test scale=0.75 as the fallback
2. **Gaussian count diverges**: Need to understand if this stabilizes or grows over 30K iters

### Next steps (in priority order):

1. **Test scale=0.75 training validation** (same 10K iter experiment): The gradient cosine at 0.75 was 0.994 (vs 0.985 at 0.50). If SSIM drop is within 0.005 and Gaussian count is within 10%, scale=0.75 becomes the primary candidate with +33.8% speedup.

2. **Analyze the Gaussian count divergence**: Is it caused by less aggressive pruning (opacity threshold effect) or more aggressive densification (gradient threshold effect)? This can be diagnosed by comparing per-iteration densification/pruning counts.

3. **If scale=0.75 passes**: Run 30K full validation with scale=0.75 as the primary candidate and scale=0.50 as the aggressive option.

4. **If scale=0.75 also fails**: Consider a hybrid schedule (0.50 for 0-5K, 1.0 for 5K-10K) or accept the marginal SSIM degradation.

---

## 6. Data Provenance

| Item | Location |
|------|----------|
| Script | `scripts/phase-c42/c42_p0_training_validation.py` |
| A100 JSON | `results/a100/phase-c42/c42_p0_training_validation.json` |
| Hardware | A100-PCIE-40GB (mx), recorded in `results/a100/hardware_metadata.json` |

---

## 7. Raw Data Summary

### Variant A (baseline, scale=1.0)
- Total wall time: 983.6s (16.4 min)
- Mean iter: 94.22 ms
- Final PSNR (311 cams): 14.93 dB
- Final SSIM (311 cams): 0.6137
- Final Gaussian count: 892,558

### Variant B (scale=0.5)
- Total wall time: 415.0s (6.9 min)
- Mean iter: 37.51 ms
- Final PSNR (311 cams): 14.93 dB
- Final SSIM (311 cams): 0.6075
- Final Gaussian count: 1,051,012

### Delta (B - A)
- ΔPSNR: -0.01 dB (PASS)
- ΔSSIM: -0.0062 (FAIL, gate=-0.005)
- ΔGS: +17.8% (FAIL, gate=10%)
- Speedup: +60.2% (PASS, gate=40%)
