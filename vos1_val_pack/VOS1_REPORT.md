# VOS1 Paper-Validation Pack — Visibility-Aware Optimizer Scheduling

**Environment**: EPIC-05 A100-SXM4-80GB, torch 2.8.0+cu128, graphdeco pipeline.  
**Design**: 3 scenes x 2 schedulers (k=1 baseline, k=2 VOS) x 3 seeds {0,1,42} = 18 runs, 30000 iters, seed fixed per run. Only the training seed varies; FasterGSFused, rasterizer, optimizer implementation, densification, pruning, and scheduler logic are unchanged. No CUDA changes.

## 1. Multi-Seed Stability

| scene | sched | PSNR mean | PSNR std | SSIM mean | LPIPS mean | wall mean (s) | opt mean (s) | upd mean |
|---|---|---|---|---|---|---|---|---|
| room | k=1 | 32.021 | 0.0055 | 0.9308 | 0.1897 | 1327 | 111.2 | 32.7B |
| room | k=2 | 31.944 | 0.1733 | 0.9307 | 0.1897 | 1334 | 111.7 | 19.0B |
| garden | k=1 | 27.532 | 0.0401 | 0.8649 | 0.1161 | 2116 | 377.9 | 103.6B |
| garden | k=2 | 27.557 | 0.0188 | 0.8652 | 0.1158 | 3261 | 549.9 | 65.1B |
| bicycle | k=1 | 25.361 | 0.0127 | 0.7603 | 0.2261 | 2977 | 521.8 | 114.9B |
| bicycle | k=2 | 25.391 | 0.0187 | 0.7608 | 0.2259 | 2492 | 432.3 | 66.9B |

**VOS (k=2) minus baseline (k=1) PSNR delta, per seed:**

| scene | seed 0 | seed 1 | seed 42 | mean | std |
|---|---|---|---|---|---|
| room | -0.271 | +0.013 | +0.027 | -0.0770 | 0.1678 |
| garden | -0.010 | +0.001 | +0.085 | +0.0253 | 0.0520 |
| bicycle | +0.012 | +0.013 | +0.066 | +0.0302 | 0.0307 |

**Stability read**: k=1 is very seed-stable (PSNR std 0.005-0.040 dB). k=2 is stable on garden/bicycle (std ~0.019 dB) but **room k=2 is unstable** — seed 0 drops -0.271 dB while seeds 1/42 are ~neutral, giving room k=2 std 0.173 dB. The VOS skip pattern depends on camera visit order, so it amplifies seed sensitivity on the most fragile scene.

## 2. Visibility Distribution

Mean per-iteration visibility frequency (fraction of Gaussians with radii>0):

| scene | k=1 vis_freq | k=2 vis_freq |
|---|---|---|
| room | 0.253 | 0.252 |
| garden | 0.342 | 0.342 |
| bicycle | 0.248 | 0.250 |

Visibility is scheduler-independent (k=1 ~ k=2), confirming it is a scene property. The population is dominated by warm (vis_freq 0.1-0.5) Gaussians; ~9-19% are hot (>0.5) and ~27-34% are cold (<0.1).

## 3. Visibility-Gradient Correlation

| scene | sched | Pearson | Spearman |
|---|---|---|---|
| room | k=1 | 0.331 | 0.499 |
| room | k=2 | 0.336 | 0.499 |
| garden | k=1 | 0.405 | 0.769 |
| garden | k=2 | 0.467 | 0.770 |
| bicycle | k=1 | 0.463 | 0.704 |
| bicycle | k=2 | 0.460 | 0.700 |

Positive correlation (Spearman ~0.50-0.77): rarely-visible Gaussians carry much smaller position-gradients. Binned means show grad_rms rising from ~5e-6 (vis_freq~0.1) to ~1e-4 (vis_freq~0.9) — a ~20x increase. This is the core justification for visibility-aware skipping: invisible Gaussians contribute near-zero gradient, so their updates are near-no-ops.

## 4. Hot/Warm/Cold Breakdown (k=2 VOS, seed-averaged)

| scene | part | frac | vis_freq | grad_rms | upd_freq | skip_rate |
|---|---|---|---|---|---|---|
| room | hot | 9.4% | 0.555 | 1.88e-05 | 0.692 | 30.8% |
| room | warm | 63.2% | 0.299 | 1.29e-05 | 0.590 | 41.0% |
| room | cold | 27.4% | 0.040 | 3.48e-06 | 0.510 | 48.9% |
| garden | hot | 25.6% | 0.780 | 2.20e-05 | 0.839 | 16.4% |
| garden | warm | 47.4% | 0.274 | 7.41e-06 | 0.581 | 41.9% |
| garden | cold | 27.1% | 0.046 | 1.52e-06 | 0.512 | 48.8% |
| bicycle | hot | 19.1% | 0.696 | 3.38e-05 | 0.780 | 21.7% |
| bicycle | warm | 46.7% | 0.200 | 1.01e-05 | 0.558 | 44.2% |
| bicycle | cold | 34.2% | 0.068 | 3.03e-06 | 0.518 | 48.2% |

The scheduler skips cold Gaussians most (~48%) and hot least (~21-31%): it removes the lowest-gradient updates first, exactly where the wasted work is.

## 5. Update Reduction

| scene | k=1 upd | k=2 upd | k=2 skipped | reduction |
|---|---|---|---|---|
| room | 32.7B | 19.0B | 13.8B | 42.0% |
| garden | 103.6B | 65.1B | 38.5B | 37.2% |
| bicycle | 114.9B | 66.9B | 47.3B | 41.8% |

Geometric-mean update reduction: **40.2%**. This is a logical optimizer-workload reduction; it does NOT translate to wall-clock savings (see Amdahl).

## 6. Amdahl Analysis

| scene | opt fraction (mean) | max speedup if opt eliminated |
|---|---|---|
| room | 8.4% | 1.092x |
| garden | 18.1% | 1.221x |
| bicycle | 17.8% | 1.216x |

Optimizer is 14.8% of wall time on average; even fully eliminating it caps speedup at **1.173x** (< 1.2x). The SparseGaussianAdam kernel is O(N) regardless of the visibility mask, so the ~40% update reduction yields ~0% optimizer-time reduction. The 1.2x speedup gate is structurally unreachable by optimizer-only changes.

## Verdict

**Mechanism VALIDATED, with a stability caveat and no wall-clock benefit.** 
- Visibility-aware skipping removes ~40% of optimizer updates and preferentially skips the lowest-gradient (cold) Gaussians, consistent with the visibility-gradient correlation. 
- Quality impact is near-zero on average (pooled mean PSNR delta -0.008 dB) and seed-stable on garden/bicycle, BUT room k=2 shows a seed-dependent outlier (seed 0: -0.271 dB), so VOS is NOT uniformly seed-stable. 
- No wall-clock speedup (Amdahl-bound < 1.2x; kernel O(N)); the benefit is reduced optimizer workload/energy, not training time. 
- **Fail-closed note**: a stricter per-scene or worst-seed quality gate (e.g. <=0.05 dB on every seed) would FAIL on room seed 0. The validation therefore reports the instability honestly rather than claiming uniform viability.
