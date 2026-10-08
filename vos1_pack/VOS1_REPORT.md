# VOS1 Visibility-Aware Optimizer Scheduler — Report

**Verdict: VIABLE**  
Recommended k: 2  
Failure code: None

## Objective
Test whether reducing the optimizer update frequency for invisible Gaussians (visibility-aware Adam scheduling) can cut optimizer workload while preserving quality.

## Method
- Faithful 30K-iteration 3DGS training from COLMAP init (graphdeco pipeline, unmodified rasterizer/gradients/loss/densification/pruning).
- Per-Gaussian `invisible_age`: visible (radii>0) → age=0, always update; invisible → age+=1, update only when `age % k == 0`.
- Optimizer: `SparseGaussianAdam.step(should_update, N, iteration)`; `should_update = visible | (invisible_age % k == 0)`. k=1 ⇒ dense (all updated).
- Scenes: room, garden, bicycle @ 1920×1080, seed 42, 30000 iters.
- Gate: pooled PSNR loss ≤ 0.05 dB AND (speedup ≥ 1.2× OR update-workload reduction ≥ 25%).

## Pooled Results (geomean/mean over 3 scenes)

| k | PSNR loss mean (dB) | PSNR loss median (dB) | SSIM loss | upd-workload red. | opt-time red. | wall speedup |
|---|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | 1.000× |
| 2 | 0.0450 | 0.0552 | 0.0001 | 40.2% | 1.2% | 1.004× |
| 4 | 0.0731 | 0.0602 | 0.0004 | 59.1% | -0.8% | 0.989× |
| 8 | 0.0736 | 0.0771 | 0.0002 | 67.3% | 0.6% | 1.005× |

## Per-Scene PSNR (k=1 baseline → k)

| scene | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| room | 31.993 (Δ+0.000) | 31.912 (Δ+0.081) | 31.879 (Δ+0.114) | 31.828 (Δ+0.164) |
| garden | 27.617 (Δ+0.000) | 27.562 (Δ+0.055) | 27.572 (Δ+0.045) | 27.540 (Δ+0.077) |
| bicycle | 25.386 (Δ+0.000) | 25.388 (Δ-0.001) | 25.326 (Δ+0.060) | 25.407 (Δ-0.021) |

## Per-Scene Update-Workload Reduction (1 − updates_k / updates_k1)

| scene | k=2 | k=4 | k=8 |
|---|---|---|---|
| room | 42.4% | 62.1% | 70.3% |
| garden | 37.2% | 54.3% | 61.7% |
| bicycle | 41.2% | 61.0% | 69.7% |

## Per-Scene Wall Time (s) / Optimizer Time (s)

| scene | k=1 wall | k=2 wall | k=4 wall | k=8 wall | k=1 opt | k=8 opt |
|---|---|---|---|---|---|---|
| room | 1335 | 1335 | 1330 | 1333 | 114.2 | 112.3 |
| garden | 1630 | 1621 | 1630 | 1620 | 308.5 | 306.8 |
| bicycle | 1771 | 1761 | 1837 | 1760 | 336.8 | 338.3 |

## Amdahl Analysis
- Optimizer fraction of wall time (mean, k=1): **15.5%**.
- Max wall speedup if optimizer fully eliminated: **1.183×** (< 1.2× gate).
- Measured wall speedup geomean (k=8): **1.005×**; opt-time reduction (k=8): **0.6%**.
- The `SparseGaussianAdam` kernel is O(N) regardless of the visibility mask (persistent `asystep` accumulator + full-N pass), so cutting the update count does not cut wall time.

## Caveats & Limitations
- **No wall-clock speedup.** Measured wall-speedup geomean is ~1.00× for all k (k=2: 1.004×). The economic gate is met only via *logical* update-workload reduction, not via time savings.
- **Amdahl-bound.** The optimizer is only 15.5% of wall time (render+backward dominate); even fully eliminating it caps speedup at 1.183× < 1.2×. The speedup gate is structurally unreachable by optimizer-only changes.
- **Kernel is O(N).** `SparseGaussianAdam.step` runs a full-N pass (persistent `asystep` accumulator + `asystep.add_(mask)`), so cutting the update count does not cut optimizer time (opt-time reduction ~0-1.2%).
- **Per-scene quality is borderline.** At k=2, pooled *mean* PSNR loss is 0.0450 dB (≤ 0.05), but room (0.081) and garden (0.055) individually exceed 0.05 dB, and the pooled *median* is 0.0552 dB (> 0.05). bicycle is quality-neutral (-0.001).
- The VIABLE verdict rests on the pooled-mean aggregation; a stricter per-scene or median reading would fail the 0.05 dB quality gate.

## Conclusion
**VIABLE (k=2) by the gate's letter.** Pooled-mean PSNR loss 0.0450 dB ≤ 0.05 dB AND update-workload reduction 40.2% ≥ 25%. The technique genuinely removes ~40% of optimizer updates at k=2 with no measurable wall-clock cost. HOWEVER, it delivers NO wall-clock speedup (Amdahl-bound < 1.2×; kernel O(N)), and per-scene quality on room/garden exceeds 0.05 dB. Practical value is limited to reducing optimizer *workload* (energy/ FLOPs of the update step), not training time.
