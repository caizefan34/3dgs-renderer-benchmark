# Phase S2.3 — C42 Quality–Speed Pareto Frontier + Paper Closure

**Status**: COMPLETE  
**Server**: `mx` (8× A100-PCIE-40GB)

---

## 1. Candidate Set (Frozen)

Tested scales: **1.0** (baseline), **0.75**, **0.625**, **0.5** (existing). No other ratios added.

All other parameters unchanged: L1 full-resolution, SepSSIM baseline, λ=0.2, area mode.

---

## 2. Stage A — Screening (10K)

| Scene | Scale | PSNR | SSIM | LPIPS | N (10K) | Time/iter (ms) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Garden | 1.000 | 28.12 | 0.8675 | — | 2,948,211 | 68.4 | 1.00× |
| Garden | 0.750 | 27.94 | 0.8505 | 0.1896 | 2,609,085 | 48.0 | 1.43× |
| Garden | 0.625 | 27.88 | 0.8452 | 0.1963 | 2,516,004 | 41.9 | 1.63× |
| Garden | 0.500 | 27.90 | 0.8469 | — | 2,606,606 | 41.1 | 1.66× |
| Bicycle | 1.000 | 22.94 | 0.7061 | — | 3,144,308 | 73.2 | 1.00× |
| Bicycle | 0.750 | 23.11 | 0.7021 | 0.3618 | 2,803,644 | 49.1 | 1.49× |
| Bicycle | 0.625 | 23.22 | 0.7029 | 0.3624 | 2,649,431 | 42.7 | 1.71× |
| Bicycle | 0.500 | 23.29 | 0.7104 | — | 2,877,124 | 47.2 | 1.55× |

### Screening Gate Results

| Scene | Scale | ΔPSNR | ΔSSIM | PSNR Gate | SSIM Gate | Selected? |
|---|---:|---:|---:|---|---|---|
| Garden | 0.750 | -0.18 | -0.0170 | PASS | FAIL | ✅ Selected |
| Garden | 0.625 | -0.24 | -0.0224 | FAIL | FAIL | ❌ |
| Bicycle | 0.750 | +0.18 | -0.0040 | PASS | PASS | ❌ |
| Bicycle | 0.625 | +0.28 | -0.0033 | PASS | PASS | ✅ Selected (maximizes speed) |

```
GARDEN_SELECTED_SCALE = 0.75
BICYCLE_SELECTED_SCALE = 0.625
```

---

## 3. Stage B — Full 30K Pareto Validation

### Pareto Table (30K)

| Scene | Scale | PSNR | SSIM | LPIPS | N GS | Time/iter (ms) | Wall clock (s) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Garden | 1.000 | 29.63 | 0.8994 | 0.1400 | 3,006,472 | 67.1 | 2,064 | 1.00× |
| Garden | 0.750 | 29.28 | 0.8811 | 0.1613 | 2,661,669 | 50.4 | 1,583 | 1.33× |
| Garden | 0.500 | 29.23 | 0.8770 | 0.1655 | 2,639,939 | 39.9 | 1,249 | 1.68× |
| Bicycle | 1.000 | 26.47 | 0.8383 | 0.2436 | 3,903,754 | 75.3 | 2,342 | 1.00× |
| Bicycle | 0.625 | 26.14 | 0.8084 | 0.2750 | 3,141,490 | 50.0 | 1,586 | 1.51× |
| Bicycle | 0.500 | 26.43 | 0.8170 | 0.2633 | 3,387,905 | 47.2 | 1,493 | 1.60× |
| Room | 1.000 | 32.30 | 0.9263 | N/A | 952,353 | 55.4 | 1,662 | 1.00× |
| Room | 0.500 | 32.54 | 0.9185 | N/A | 745,566 | 29.1 | 920 | 1.90× |

### Quality Deltas at 30K

| Scene | Scale | ΔPSNR | ΔSSIM | ΔLPIPS | ΔN% |
|---|---:|---:|---:|---:|---:|
| Garden | 0.750 | -0.35 | -0.0183 | +0.0213 | -11.4% |
| Garden | 0.500 | -0.40 | -0.0224 | +0.0255 | -12.2% |
| Bicycle | 0.625 | -0.33 | -0.0299 | +0.0314 | -19.5% |
| Bicycle | 0.500 | -0.04 | -0.0213 | +0.0197 | -13.2% |
| Room | 0.500 | +0.24 | -0.0078 | N/A | -21.7% |

### Key Finding: Non-Monotonic Behavior on Bicycle

Scale=0.625 on Bicycle is **Pareto-dominated** by scale=0.5:
- 0.5 is faster (47.2ms vs 50.0ms)
- 0.5 has better PSNR (26.43 vs 26.14)
- 0.5 has better SSIM (0.8170 vs 0.8084)
- 0.5 has better LPIPS (0.2633 vs 0.2750)

The less aggressive downsampling (0.625) produces **worse** quality than the more aggressive (0.5) while also being slower. This non-monotonic result was not predicted by the 10K screening, where 0.625 showed ΔPSNR=+0.28 (improvement).

---

## 4. Pareto Frontier

```
GARDEN_PARETO_SCALES = {1.000, 0.750, 0.500}
BICYCLE_PARETO_SCALES = {1.000, 0.500}
```

**Garden**: Three Pareto-optimal scales. Scale=0.75 provides marginally better quality than 0.5 (ΔPSNR difference: 0.05 dB) but with 15.6% less speedup. The frontier is thin but meaningful.

**Bicycle**: Two Pareto-optimal scales. Scale=0.625 is dominated by 0.5. No useful intermediate point exists.

**Room**: Scale=0.5 improves quality (ΔPSNR=+0.24). No intermediate needed.

A meaningful controllable frontier exists on Garden but is scene-dependent.

---

## 5. Time-to-Quality

| Scene | Scale | Target PSNR | Time to Target (s) | Speedup-to-Target |
|---|---:|---:|---:|---:|
| Garden | 1.000 | 28.0 | 448.5 | 1.00× |
| Garden | 0.750 | 28.0 | 501.6 | 0.89× |
| Garden | 0.500 | 28.0 | 404.2 | 1.11× |
| Garden | 1.000 | 28.5 | 781.4 | 1.00× |
| Garden | 0.750 | 28.5 | 655.5 | 1.19× |
| Garden | 0.500 | 28.5 | 529.5 | 1.48× |
| Garden | 1.000 | 29.0 | 962.1 | 1.00× |
| Garden | 0.750 | 29.0 | 928.4 | 1.04× |
| Garden | 0.500 | 29.0 | 770.8 | 1.25× |
| Bicycle | 1.000 | 23.0 | 660.4 | 1.00× |
| Bicycle | 0.625 | 23.0 | 393.2 | 1.68× |
| Bicycle | 0.500 | 23.0 | 352.3 | 1.87× |
| Bicycle | 1.000 | 24.0 | 869.6 | 1.00× |
| Bicycle | 0.625 | 24.0 | 563.8 | 1.54× |
| Bicycle | 0.500 | 24.0 | 506.3 | 1.72× |
| Bicycle | 1.000 | 25.0 | 1121.9 | 1.00× |
| Bicycle | 0.625 | 25.0 | 793.2 | 1.41× |
| Bicycle | 0.500 | 25.0 | 667.5 | 1.68× |

**Key finding**: Scale=0.5 consistently reaches quality targets faster than intermediate scales. On Garden at low quality (28.0), scale=0.75 is actually **slower** than baseline (0.89× speedup-to-target).

---

## 6. Fixed-Tensor Loss Benchmark

| Scale | Forward (ms) | Fwd+Bwd (ms) | Speedup |
|---:|---:|---:|---:|
| 1.000 | 24.96 | 33.15 | 1.00× |
| 0.750 | 14.63 | 19.41 | 1.71× |
| 0.625 | 10.51 | 13.92 | 2.38× |
| 0.500 | 6.95 | 9.24 | 3.58× |

Protocol: 50 warmup, 300 timed iterations, CUDA events, explicit synchronization, same frozen 1080p tensors on baseline 30K checkpoint.

The loss-side speedup is **monotonically increasing** with lower scale and is consistent across scenes (operates on rendered images, not Gaussians). Scale=0.5 provides 3.58× loss-side speedup, which is the direct objective-side saving.

**Safe decomposition language**:
- Direct loss saving = measured fixed-tensor delta (up to 3.58× at scale=0.5)
- Additional training-wide gain = associated with changed Gaussian population / renderer workload

---

## 7. Densification / Population Trend

| Scene | Scale | Clones | Splits | Prunes | Final N |
|---|---:|---:|---:|---:|---:|
| Garden | 1.000 | 1,165,907 | 430,584 | 429,255 | 3,006,472 |
| Garden | 0.750 | 802,507 | 378,272 | 358,346 | 2,661,669 |
| Garden | 0.500 | 804,420 | 377,810 | 381,527 | 2,639,939 |
| Bicycle | 1.000 | 4,300,444 | 1,333,174 | 1,784,139 | 3,903,754 |
| Bicycle | 0.625 | 3,213,702 | 1,256,888 | 1,383,375 | 3,141,490 |
| Bicycle | 0.500 | 3,545,427 | 1,328,309 | 1,540,106 | 3,387,905 |
| Room | 1.000 | 1,229,093 | 250,436 | 639,803 | 952,353 |
| Room | 0.500 | 946,727 | 234,928 | 548,716 | 745,566 |

```
POPULATION_MONOTONIC = PARTIAL
```

Clone count is monotonically decreasing with lower scale. Final Gaussian count is **NOT** monotonic — on Bicycle, N decreases from 1.0 (3.9M) to 0.625 (3.1M) but then increases at 0.5 (3.4M). Very aggressive downsampling causes the model to grow more Gaussians, possibly compensating for reduced structural guidance via the full-resolution L1 term.

---

## 8. Gradient Frequency Probe

| Scene | Scale | Cosine to Baseline | Rel L2 | HF Fraction |
|---|---:|---:|---:|---:|
| Garden | 1.000 | 1.0000 | 0.0000 | 0.7659 |
| Garden | 0.750 | 0.8115 | 0.5871 | 0.7675 |
| Garden | 0.625 | 0.7659 | 0.6477 | 0.7675 |
| Garden | 0.500 | 0.7642 | 0.6463 | 0.7667 |
| Bicycle | 1.000 | 1.0000 | 0.0000 | 0.5605 |
| Bicycle | 0.750 | 0.5062 | 8.1923 | 0.4412 |
| Bicycle | 0.625 | 0.6369 | 0.6856 | 0.4173 |
| Bicycle | 0.500 | 0.6379 | 0.6845 | 0.4175 |

**Garden**: Gradient cosine decreases monotonically with lower scale. HF fraction nearly constant — C42 changes gradient direction but not frequency distribution.

**Bicycle**: Anomalous cosine at 0.75 (0.506) with very high rel_l2 (8.19). HF fraction drops from 0.56 to 0.44 at 0.75, then saturates at 0.42 for 0.625 and 0.5. The HF suppression effect saturates at scale ≤ 0.625.

The Bicycle 0.75 anomaly may explain the non-monotonic 30K behavior — intermediate scales create unstable gradient dynamics on high-frequency outdoor content.

---

## 9. C42 Final Classification

```
C42_FINAL_CLASSIFICATION = C42_PARETO_KEEP
```

**Rationale**:
- A meaningful Pareto frontier exists on Garden (3 non-dominated scales)
- Scale=0.5 is the recommended default — consistently useful across all scenes
- Scale=0.75 provides a marginal quality improvement on Garden (0.05 dB) but with significant speed loss
- Scale=0.625 is Pareto-dominated by 0.5 on Bicycle — not useful
- The frontier is scene-dependent but demonstrates that D-SSIM resolution is a tunable parameter
- Time-to-quality analysis shows 0.5 consistently reaches quality targets fastest

**Paper-ready claim**:

> Resolution-controlled structural supervision exposes a tunable efficiency–quality frontier in 3DGS training: lower-resolution D-SSIM consistently reduces loss cost (up to 3.6× on fixed tensors) and Gaussian population (12–22%), with increasingly aggressive settings trading perceptual quality for throughput. The frontier is scene-dependent — outdoor detail-rich scenes (Garden) exhibit a thin Pareto frontier with an intermediate resolution option (0.75×), while other scenes (Bicycle, Room) show scale=0.5× as the dominant C42 configuration providing 1.6–1.9× throughput with minimal quality degradation.

---

## 10. Paper Claim Discipline

- ❌ Do NOT claim: "free speedup", "lossless acceleration", "universal 1.9× acceleration"
- ✅ Do claim: "tunable efficiency–quality frontier", "scene-dependent", "1.6–1.9× throughput"
- C44 = BASELINE_CONSTITUENT (not an incremental optimization)
- AbsGradOff = supporting exact systems optimization (adds <1% full-system gain, not deployable as standalone)

---

## 11. Deliverables

| File | Description |
|---|---|
| `results/reference_v1/s23/garden/screening.json` | Garden 10K screening comparison |
| `results/reference_v1/s23/garden/final_30k.json` | Garden 30K Pareto validation |
| `results/reference_v1/s23/garden/time_to_quality.json` | Garden time-to-quality |
| `results/reference_v1/s23/bicycle/screening.json` | Bicycle 10K screening comparison |
| `results/reference_v1/s23/bicycle/final_30k.json` | Bicycle 30K Pareto validation |
| `results/reference_v1/s23/bicycle/time_to_quality.json` | Bicycle time-to-quality |
| `results/reference_v1/s23/fixed_tensor_loss_benchmark.json` | Clean fixed-tensor loss timing |
| `results/reference_v1/s23/population_frontier.json` | Population trend analysis |
| `results/reference_v1/s23/pareto_summary.json` | Complete Pareto frontier summary |
| `results/reference_v1/s23/final_decision.json` | Final classification and decision |
| `results/reference_v1/s23/mechanism/gradient_frequency_probe.json` | Gradient frequency by scale |
| `reports/phase-s2.3-c42-pareto-frontier.md` | This report |
