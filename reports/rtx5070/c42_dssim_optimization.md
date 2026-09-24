# C42: D-SSIM Bottleneck Optimization Investigation

## 1. Current D-SSIM Implementation Analysis

### Source Location
`scripts/epic05/phase7/loss.py` — lines 14-62

### Computation DAG

```
Input: pred [H, W, 3] (rendered image, float32, [0,1])
       target [H, W, 3] (GT image, float32, [0,1], CONSTANT across iterations)

Step 0: Permute to [1, 3, H, W]  (lines 27-31)

Step 1: Build Gaussian kernel (lines 34-38)
  coords = [-5, -4, ..., 5]  (11 elements)
  kernel_1d = exp(-coords² / 2σ²), σ=1.5  → normalized 1D Gaussian
  kernel_2d = kernel_1d ⊗ kernel_1d  → [3, 1, 11, 11]  (SEPARABLE by construction!)
  ⚠️ Rebuilt EVERY iteration (kernel is constant)

Step 2: Gaussian blur (5 conv2d calls, each 11×11 depthwise)
  ┌─────────────────────────────────────────────────────┐
  │  blur(pred)      → mu_pred        [1,3,H,W]  (conv) │  ── CHANGES per iter
  │  blur(target)    → mu_target      [1,3,H,W]  (conv) │  ── CONSTANT ⚠️
  │  blur(pred²)     → mu_pred_sq_raw [1,3,H,W]  (conv) │  ── CHANGES per iter
  │  blur(target²)   → mu_tgt_sq_raw  [1,3,H,W]  (conv) │  ── CONSTANT ⚠️
  │  blur(pred*tgt)  → mu_pt_raw      [1,3,H,W]  (conv) │  ── CHANGES per iter
  └─────────────────────────────────────────────────────┘

Step 3: Elementwise operations (lines 48-54)
  mu_pred_sq     = mu_pred²                    (elementwise)
  mu_target_sq   = mu_target²                  (elementwise)
  mu_pred_target = mu_pred * mu_target         (elementwise)
  sigma_pred_sq  = mu_pred_sq_raw - mu_pred_sq (elementwise)
  sigma_tgt_sq   = mu_tgt_sq_raw - mu_tgt_sq   (elementwise)
  sigma_pt       = mu_pt_raw - mu_pred_target  (elementwise)

Step 4: SSIM map (lines 56-60)
  C1 = 0.0001, C2 = 0.0009
  numerator   = (2*mu_pred_target + C1) * (2*sigma_pt + C2)
  denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_tgt_sq + C2)
  ssim_map = numerator / denominator  (elementwise)

Step 5: Loss (line 62)
  loss = 1 - ssim_map.mean()
```

### Kernel Count Breakdown

| Phase | Operation | cuDNN kernel | Count |
|-------|-----------|-------------|-------|
| Forward | conv2d (blur pred) | conv2d_grouped_direct | 1 |
| Forward | conv2d (blur target) | conv2d_grouped_direct | 1 |
| Forward | conv2d (blur pred²) | conv2d_grouped_direct | 1 |
| Forward | conv2d (blur target²) | conv2d_grouped_direct | 1 |
| Forward | conv2d (blur pred*tgt) | conv2d_grouped_direct | 1 |
| Forward | elementwise (squaring, mul, sub, div) | elementwise_kernel | ~15 |
| Backward | dgrad (blur pred) | dgrad2d_grouped_direct | 1 |
| Backward | dgrad (blur pred²) | dgrad2d_grouped_direct | 1 |
| Backward | dgrad (blur pred*tgt) | dgrad2d_grouped_direct | 1 |
| Backward | elementwise (gradient ops) | elementwise_kernel | ~10 |
| **Total** | | | **~33** |

Profiler measured: 8.7 cuDNN kernels/iter + ~15 elementwise = ~24 kernels attributed to D-SSIM

### FLOP Estimation

| Operation | FLOPs | Count | Total |
|-----------|-------|-------|-------|
| 2D conv 11×11, 3 channels, 1920×1080 | 3 × 11² × 1920 × 1080 × 2 = 1.50 GFLOP | 5 fwd + 3 bwd = 8 | **12.0 GFLOP** |
| Elementwise (mul, sq, sub, div, mean) | ~3 × 1920 × 1080 × 2 = 12.4 MFLOP | ~25 | **0.31 GFLOP** |
| **Total** | | | **~12.3 GFLOP** |

### Memory Estimation

| Operation | Data movement | Count | Total |
|-----------|-------------|-------|-------|
| Each conv2d: read input + read kernel + write output | 23.7 + 0.001 + 23.7 = 47.4 MB | 8 | 379 MB |
| Elementwise: read 2 + write 1 per op | ~71 MB | 25 | ~1.78 GB |
| **Total estimated** | | | **~2.16 GB** |

**Note**: Elementwise ops dominate memory movement because PyTorch doesn't fuse them — each reads and writes full [1,3,1080,1920] tensors.

### Redundant Computation Identified

| Redundancy | Impact | Fix |
|------------|--------|-----|
| **blur(target) recomputed every iter** | 1 conv2d fwd + 0 bwd (no grad needed) = 1 wasted kernel | Cache mu_target |
| **blur(target²) recomputed every iter** | 1 conv2d fwd + 0 bwd = 1 wasted kernel | Cache mu_tgt_sq_raw |
| **Gaussian kernel rebuilt every iter** | 3 small ops (exp, sum, outer product) | Cache kernel as buffer |
| **2D conv with separable kernel** | 121 MACs/pixel instead of 22 MACs/pixel (5.5× waste) | Use two 1D convs |
| **Elementwise ops not fused** | 25 separate kernel launches for simple arithmetic | torch.compile or manual fusion |
| **mu_pred_sq = mu_pred² computed separately** | Extra read of mu_pred | Fuse into sigma computation |

**Summary**: Of 5 forward blur calls, **2 are completely redundant** (target terms are constant). Of the remaining 3, all use a **separably-constructed kernel applied as 2D conv** (5.5× FLOP waste).

---

## 2. Prior-Art Audit

> ⚠️ Web search unavailable (authentication failure). Prior-art classification based on built-in knowledge.

| Technique | Known? | Applicable to 3DGS? | Notes |
|-----------|--------|---------------------|-------|
| **Separable Gaussian filtering** | Classic DSP technique | ✅ Direct | 11×11 → 11+11 = 5.5× FLOP reduction. Standard for image processing. |
| **Downsampled SSIM** | Used in video quality assessment | ✅ Direct | Compute SSIM at 1/2 or 1/4 resolution. Common in real-time applications. |
| **Box filter approximation** | Used in fast SSIM variants | ✅ Direct | Replace Gaussian with box (uniform) filter → avg_pool2d is 10× faster than conv2d. |
| **Multi-scale SSIM (MS-SSIM)** | Wang et al. 2003 | ⚠️ Different | MS-SSIM adds more scales (more compute), not a speedup technique. |
| **Frequency-domain SSIM** | DCT-based SSIM | ❌ Not suitable | DCT adds overhead for small kernels; beneficial only for large kernels (>31×31). |
| **torch.compile fusion** | PyTorch 2.x feature | ✅ Direct | Fuses elementwise ops; may also optimize conv dispatch. |
| **Caching constant terms** | Standard optimization | ✅ Direct | Target image terms don't change between iterations. |
| **Gradient checkpointing** | Memory optimization | ❌ Wrong direction | We want speed, not memory savings. |
| **D-SSIM scheduling** | Novel for 3DGS | ✅ Potentially novel | Reduce D-SSIM weight or frequency as training converges. |

**Key finding**: The separable Gaussian filtering technique is textbook DSP — applying an 11×11 kernel as a 2D conv when it was *constructed* as a separable outer product is a clear implementation oversight. The original 3DGS code (graphdeco-inria) also uses 2D conv, so this is an inherited inefficiency.

---

## 3. Candidate Design

### Candidate A: Separable Gaussian Filtering

**Mechanism**: Replace `F.conv2d(x, kernel_2d, ...)` with two 1D convolutions:
```python
# Instead of: F.conv2d(x, kernel_2d, padding=5, groups=3)
# Use: F.conv2d(x, kernel_1d_h, padding=5, groups=3)  # horizontal
#      F.conv2d(x, kernel_1d_v, padding=5, groups=3)  # vertical
```

| Metric | Value |
|--------|-------|
| Expected speedup on D-SSIM | **3-5×** (FLOP reduction 5.5×, but 2 kernel launches instead of 1) |
| Expected total speedup | **~25-30ms saved (30-35% of total)** |
| Expected PSNR impact | **<0.01 dB** (mathematically identical — separable conv produces same result) |
| Expected SSIM impact | **0.000** (bit-exact same computation) |
| Implementation complexity | **LOW** — modify `blur()` function in loss.py, ~10 lines |
| Research novelty | **LOW** — textbook DSP technique |
| Risk | **VERY LOW** — mathematically equivalent |

**Caveat**: cuDNN may not optimize 1D depthwise convolutions as well as 2D. Need to measure actual cuDNN performance for 1D vs 2D.

### Candidate B: Lower Resolution SSIM

**Mechanism**: Downsample pred and target to 1/2 resolution before computing SSIM:
```python
pred_down = F.avg_pool2d(pred, 2)
target_down = F.avg_pool2d(target, 2)
dsim = d_ssim_loss(pred_down, target_down)
```

| Metric | Value |
|--------|-------|
| Expected speedup on D-SSIM | **4×** (4× fewer pixels: 960×540 instead of 1920×1080) |
| Expected total speedup | **~28ms saved (33% of total)** |
| Expected PSNR impact | **0.1-0.5 dB** (SSIM at lower resolution captures less high-freq detail) |
| Expected SSIM impact | **<0.005** (structural similarity is scale-invariant by design) |
| Implementation complexity | **LOW** — add downsampling before d_ssim_loss, ~3 lines |
| Research novelty | **LOW-MEDIUM** — downscaled SSIM is known but not standard in 3DGS |
| Risk | **LOW-MEDIUM** — small quality drop, needs measurement |

### Candidate C: Approximate Gaussian / Box Filter

**Mechanism**: Replace Gaussian blur with box filter (average pooling):
```python
# Instead of F.conv2d(x, gaussian_kernel, padding=5, groups=3)
# Use F.avg_pool2d(x, 11, stride=1, padding=5) or F.blur(x, kernel_size=11)
```

| Metric | Value |
|--------|-------|
| Expected speedup on D-SSIM | **8-10×** (avg_pool2d is highly optimized, no weight multiplication) |
| Expected total speedup | **~34ms saved (40% of total)** |
| Expected PSNR impact | **0.2-1.0 dB** (box filter ≠ Gaussian → different SSIM weights) |
| Expected SSIM impact | **0.005-0.02** (different frequency weighting) |
| Implementation complexity | **LOW** — replace blur function, ~5 lines |
| Research novelty | **MEDIUM** — box-filter SSIM variant for 3DGS training |
| Risk | **MEDIUM** — quality drop depends on scene; needs measurement |

### Candidate D: Adaptive SSIM Scheduling

**Mechanism**: Reduce D-SSIM contribution during later training phases:
```python
# Phase 1 (iter 0-5000): λ_dssim = 0.2 (full D-SSIM)
# Phase 2 (iter 5000-15000): λ_dssim = 0.1 (half D-SSIM)
# Phase 3 (iter 15000-30000): λ_dssim = 0.0 (L1 only)
```
Or: compute D-SSIM every N iterations instead of every iteration.

| Metric | Value |
|--------|-------|
| Expected speedup on D-SSIM | **2-10×** (depends on schedule; every-other-iter = 2×, phase-out = up to 10×) |
| Expected total speedup | **~19-35ms saved (22-41% of total)** |
| Expected PSNR impact | **0.0-0.5 dB** (D-SSIM matters most early in training for structural detail) |
| Expected SSIM impact | **<0.01** (if phased correctly) |
| Implementation complexity | **LOW** — modify training loop, ~5 lines |
| Research novelty | **MEDIUM-HIGH** — D-SSIM scheduling specific to 3DGS training dynamics |
| Risk | **MEDIUM** — needs careful scheduling to avoid quality regression |

### Candidate E: torch.compile / Operator Fusion

**Mechanism**: Apply `torch.compile` to the loss function or the entire training step:
```python
compiled_loss = torch.compile(combined_loss)
```

| Metric | Value |
|--------|-------|
| Expected speedup on D-SSIM | **1.5-2×** (fuses elementwise ops, may optimize conv dispatch) |
| Expected total speedup | **~10-19ms saved (12-22% of total)** — also helps elementwise_misc |
| Expected PSNR impact | **0.0 dB** (mathematically identical) |
| Expected SSIM impact | **0.000** (same computation) |
| Implementation complexity | **VERY LOW** — one-line change |
| Research novelty | **LOW** — standard PyTorch optimization |
| Risk | **LOW** — may have compatibility issues with gsplat custom autograd |

---

## 4. Experiment Protocol

### Setup

| Parameter | Value |
|-----------|-------|
| Scene | room (Mip-NeRF 360) |
| Resolution | 1080p |
| Checkpoint | iter 5000 (1M Gaussians, SH degree 3) |
| Densification | DISABLED (isolate loss cost) |
| Iterations | 500 per variant |
| Optimizer | Fresh Adam (same as C38-C41) |
| Camera | Fixed camera 0 |
| GPU | A100-PCIE-40GB |

### Variants

| ID | Description | Key change |
|----|-------------|------------|
| **V0** | Baseline | Original d_ssim_loss (2D conv, no caching) |
| **V1** | Separable + cache target | 1D convs + cache blur(target), blur(target²) |
| **V2** | Downsampled SSIM | 1/2 resolution before SSIM computation |
| **V3** | Box filter SSIM | Replace Gaussian with avg_pool2d |
| **V4** | Adaptive schedule | D-SSIM every 2 iterations, L1-only otherwise |
| **V5** | torch.compile | `torch.compile(combined_loss)` |
| **V6** | Combined best | Separable + cache + torch.compile |

### Metrics

**Training performance:**
| Metric | How measured |
|--------|-------------|
| Iteration time (ms) | CUDA events around full training step |
| D-SSIM time (ms) | CUDA events around loss computation only |
| GPU kernel count | torch.profiler Chrome trace |
| GPU utilization | Profiler kernel time / wall time |
| Memory bandwidth | Estimated from tensor sizes / kernel time |

**Quality:**
| Metric | How measured |
|--------|-------------|
| PSNR | `10 * log10(1 / MSE(rendered, gt))` — computed every 50 iters |
| SSIM | Standard SSIM at full resolution — computed every 50 iters |
| LPIPS | Using lpips library if available, else skipped |
| Final Gaussian count | model.xyz.shape[0] (should be constant — no densification) |

### Decision Gate

| Condition | Result |
|-----------|--------|
| Speedup > 20% AND PSNR drop < 0.1 dB AND SSIM drop < 0.005 | **KEEP** |
| Speedup 10-20% OR quality borderline | **MAYBE** — investigate further |
| Speedup < 10% OR PSNR drop > 0.1 dB OR SSIM drop > 0.005 | **DROP** |

### Priority Ranking for Implementation

| Priority | Candidate | Rationale |
|----------|-----------|-----------|
| 1 | **V1: Separable + cache** | Mathematically identical (zero quality risk), highest expected speedup among safe options |
| 2 | **V5: torch.compile** | Zero quality risk, one-line change, also helps elementwise |
| 3 | **V6: Combined** | V1 + V5 stacked, maximum safe speedup |
| 4 | **V2: Downsampled** | Small quality risk, high speedup, good research evidence |
| 5 | **V4: Adaptive schedule** | Research-novel, needs careful tuning |
| 6 | **V3: Box filter** | Higher quality risk, needs measurement |

### Implementation Plan (NOT yet — for C42 prototype)

```
Step 1: Implement V1 (separable + cache) → measure speedup + quality
Step 2: Implement V5 (torch.compile) → measure additional speedup
Step 3: Combine V1+V5 → measure total
Step 4: If V1+V5 speedup >20% with <0.1dB PSNR drop → KEEP
Step 5: If not, try V2 (downsampled) as fallback
```

---

## Summary

### D-SSIM Computation DAG Summary

```
5 conv2d (11×11, depthwise) × [1,3,1920,1080]
  ├── 2 are CONSTANT (target terms) → cacheable
  ├── 1 is SEPARABLE but applied as 2D → 5.5× FLOP waste
  └── All rebuild kernel every iter → cacheable

~25 elementwise kernels
  └── None fused → torch.compile candidate

Total: 38ms/iter, 42.5% of GPU time
  ├── ~8ms from redundant target blur (cacheable)
  ├── ~25ms from 2D conv overhead (separable fix)
  └── ~5ms from elementwise fragmentation (torch.compile fix)
```

### Candidate Ranking

| Rank | Candidate | Speedup | Quality Risk | Difficulty | Decision |
|------|-----------|---------|-------------|------------|----------|
| 1 | **V1: Separable + cache** | 30-35% | **ZERO** (identical math) | LOW | **KEEP — implement first** |
| 2 | **V5: torch.compile** | 12-22% | **ZERO** | VERY LOW | **KEEP — implement second** |
| 3 | V2: Downsampled | 33% | LOW (0.1-0.5 dB) | LOW | MAYBE — fallback |
| 4 | V4: Adaptive schedule | 22-41% | MEDIUM | LOW | MAYBE — research angle |
| 5 | V3: Box filter | 40% | MEDIUM (0.2-1.0 dB) | LOW | NEED EVIDENCE |
| 6 | V6: Combined V1+V5 | 35-45% | **ZERO** | LOW | **KEEP — after V1+V5 verified** |

### Key Insight

The D-SSIM bottleneck is caused by an **inherited implementation oversight**: the Gaussian kernel is constructed as a separable outer product (line 37: `kernel_1d[:, None] * kernel_1d[None, :]`) but then applied as a full 2D convolution. This means 121 MACs per pixel are computed when only 22 are needed — a 5.5× waste that has been present since the original 3DGS implementation.

Additionally, 2 of 5 blur operations per iteration use the constant target image, which never changes during training. These are pure waste.

**The combination of separable filtering + target caching can eliminate ~30-35% of total training time with ZERO quality impact.**

---

## Artifacts

- D-SSIM source: `scripts/epic05/phase7/loss.py` (78 lines)
- C40 profiling: `reports/phase-c31/c40_training_breakdown.md`
- C41 GPU utilization: `reports/phase-c31/c41_gpu_utilization.md`
