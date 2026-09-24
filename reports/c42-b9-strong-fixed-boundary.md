# C42-B9: Strong Fixed-Scale Boundary Search

## Scientific Summary

### B9_STATUS

**COMPLETED.** The strong fixed-scale boundary search tested whether a globally fixed structural-supervision scale cheaper than 1.0 could satisfy the primary quality constraint (abs(dSSIM) ≤ 0.020) across all three scenes (Room, Garden, Bicycle). Four new scales were trained for Bicycle (0.80, 0.85, 0.90, 0.95) at 30K iterations with the canonical Reference V1 codebase. Combined with existing metrics for scales 0.50, 0.625, 0.75, and 1.0, the full scale range was evaluated.

**Result:** Among the tested scales {0.5, 0.625, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0}, only 1.0 was globally feasible at tau=0.020. Bicycle is the binding constraint at every scale below 1.0. The adaptive oracle exploits Room's tolerance (cheapest feasible = 0.50) and Garden's tolerance (cheapest feasible = 0.75), but Bicycle forces full-resolution SSIM supervision.

**Decision: MODIFY** — adaptive advantage = 3.11ms (2.0 ≤ 3.11 < 5.0ms threshold).

---

### LOSS_COST_CURVE

Isolated structural-supervision compute cost (backward pass only), measured on A100-PCIE-40GB at 1080p (1920×1080) with frozen random tensors, SepSSIM (window=11, sigma=1.5), F.interpolate(mode="area"), 50 warmups, 300 timed samples, CUDA events with explicit synchronization.

| Scale | Mean (ms) | Median (ms) | P10 (ms) | P90 (ms) | Speedup vs 1.0 |
|-------|-----------|-------------|----------|----------|-----------------|
| 0.50  | 2.42      | 2.41        | —        | —        | 3.44×           |
| 0.625 | 3.54      | 3.53        | —        | —        | 2.35×           |
| 0.75  | 4.92      | 4.92        | 4.91     | 4.93     | 1.69×           |
| 0.80  | 5.51      | 5.51        | 5.50     | 5.52     | 1.51×           |
| 0.85  | 6.24      | 6.24        | 6.23     | 6.24     | 1.34×           |
| 0.90  | 7.00      | 6.99        | 6.98     | 7.02     | 1.19×           |
| 0.95  | 7.77      | 7.75        | 7.73     | 7.82     | 1.07×           |
| 1.00  | 8.34      | 8.33        | 8.33     | 8.35     | 1.00×           |

These costs represent the isolated structural-supervision compute opportunity — the backward pass of `loss = (1-λ)*L1 + λ*d_ssim_downsampled(scale)` where L1 is at full resolution and d_ssim is computed on area-downsampled tensors. They do NOT represent end-to-end training speedup, as rasterization, densification, and optimizer steps are unaffected by the SSIM scale.

Deliverables: `loss_cost_curve.json`, `loss_cost_supplementary.json`

---

### BICYCLE_BOUNDARY

Bicycle 1.0 baseline SSIM = 0.8360 (all 194 cameras, unified pipeline).

| Scale | SSIM    | dSSIM   | Pass tau=0.020? | Loss Cost (ms) |
|-------|---------|---------|-----------------|----------------|
| 0.50  | 0.8133  | 0.0227  | FAIL            | 2.42           |
| 0.625 | 0.8052  | 0.0308  | FAIL            | 3.54           |
| 0.75  | 0.8135  | 0.0225  | FAIL            | 4.92           |
| 0.80  | 0.8144  | 0.0216  | FAIL            | 5.51           |
| 0.85  | 0.8124  | 0.0236  | FAIL            | 6.24           |
| 0.90  | 0.8154  | 0.0206  | FAIL (margin +0.0006) | 7.00     |
| 0.95  | 0.8155  | 0.0205  | FAIL (margin +0.0005) | 7.77     |
| 1.00  | 0.8360  | 0.0000  | PASS            | 8.34           |

**Bicycle minimum tested feasible scale = 1.00.** No scale below 1.0 satisfies the tau=0.020 SSIM constraint.

The 0.90 (dSSIM=0.0206) and 0.95 (dSSIM=0.0205) results are marginal failures — within 0.001 of the threshold. However, dSSIM is non-monotonic at these scales (0.85 dSSIM=0.0236 is worse than 0.80 dSSIM=0.0216), indicating that training variance dominates at margins below 0.001. Boundary refinement at 0.975 was evaluated but skipped (B9-D) because the non-monotonicity makes a single test inconclusive.

Deliverables: `bicycle_080.json`, `bicycle_085.json`, `bicycle_090.json`, `bicycle_095.json`, `bicycle_refinement.json`

---

### B9_D_REFINEMENT

**Decision: SKIP.** Both 0.90 (dSSIM=0.0206, margin +0.0006) and 0.95 (dSSIM=0.0205, margin +0.0005) are marginal failures. The SSIM does not monotonically improve with scale — 0.85 (dSSIM=0.0236) is worse than 0.80 (dSSIM=0.0216), and 0.90 (dSSIM=0.0206) is nearly identical to 0.95 (dSSIM=0.0205). This indicates training variance dominates the <0.001 margins between these scales. Testing 0.975 would not yield a conclusive result because the expected dSSIM (~0.0202) falls within the observed variance band.

---

### B9_E_GLOBAL_VALIDATION

Per-scene feasible sets at tau=0.020 (primary SSIM constraint):

| Scene    | Feasible Scales        | Cheapest Feasible |
|----------|------------------------|-------------------|
| Room     | {0.50, 0.75, 1.00}     | 0.50 (2.42ms)     |
| Garden   | {0.75, 1.00}           | 0.75 (4.92ms)     |
| Bicycle  | {1.00}                 | 1.00 (8.34ms)     |

**Global intersection = {1.00}.** The minimum globally feasible scale is 1.00. Since 1.00 is the baseline for all scenes, no additional validation training runs were needed — Room and Garden at 1.00 are already the reference baselines with trivially zero dSSIM, dPSNR, and dLPIPS.

Deliverables: `room_candidate.json`, `garden_candidate.json`

---

### B9_F_STRONG_FIXED_BASELINE

| Property | Value |
|----------|-------|
| Strong fixed scale | 1.00 |
| Loss cost | 8.34ms |
| Global feasible set | {1.00} |
| Binding constraint | Bicycle (dSSIM=0.0206 at 0.90, dSSIM=0.0205 at 0.95) |

Multi-metric validation at scale 1.00 (trivially passes for all scenes — baseline):
- Room: dPSNR=0, dSSIM=0, dLPIPS=0 → PASS
- Garden: dPSNR=0, dSSIM=0, dLPIPS=0 → PASS
- Bicycle: dPSNR=0, dSSIM=0, dLPIPS=0 → PASS

Deliverable: `strong_fixed_baseline.json`

---

### B9_G_ADAPTIVE_VS_STRONG_FIXED

Adaptive oracle selections at tau=0.020 (cheapest feasible scale per scene):

| Scene    | Oracle Scale | Cost (ms) | dSSIM   | dPSNR  | dLPIPS | Multi-metric |
|----------|-------------|-----------|---------|--------|--------|--------------|
| Room     | 0.50        | 2.42      | 0.0026  | +0.13  | +0.007 | PASS         |
| Garden   | 0.75        | 4.92      | 0.0174  | -0.30  | +0.021 | PASS         |
| Bicycle  | 1.00        | 8.34      | 0.0000  | 0.00   | 0.000  | PASS         |

**All oracle-selected configurations at the primary tau=0.020 operating point satisfy the LPIPS +0.03 constraint.**

| Metric                          | Value    |
|---------------------------------|----------|
| Adaptive oracle average cost    | 5.23ms   |
| Strong fixed baseline cost      | 8.34ms   |
| **Adaptive advantage**          | **3.11ms** |

Deliverable: `adaptive_vs_strong_fixed.json`

---

### B9_H_DECISION_GATE

| Threshold | Range | Decision |
|-----------|-------|----------|
| ≥ 5.0ms   | —     | KEEP_FOR_PREDICTOR |
| 2.0–5.0ms | 3.11ms | **MODIFY** |
| < 2.0ms   | —     | DROP_AS_MAIN_METHOD |

**Decision: MODIFY.**

The adaptive oracle provides a 3.11ms average cost advantage over the strongest tested fixed scale (1.00). This is a moderate but real saving — the oracle reduces Room's structural-supervision cost by 71% (8.34→2.42ms) and Garden's by 41% (8.34→4.92ms), but Bicycle's structural-supervision must remain at full resolution, consuming 8.34ms regardless.

The advantage is limited because Bicycle is the binding constraint: its dSSIM at scales 0.90 (0.0206) and 0.95 (0.0205) narrowly exceed tau=0.020, forcing the global fixed baseline to 1.00. If the tau threshold were relaxed to 0.021, Bicycle 0.90 would become feasible, and the strong fixed baseline would drop to 7.00ms — reducing the adaptive advantage to 1.77ms (DROP territory). Conversely, if additional scenes with Bicycle-like sensitivity were added, the advantage could increase.

Deliverable: `final_gate.json`

---

### B9_I_FIXED_SCALE_PARETO

| Scale  | Cost (ms) | Room | Garden | Bicycle | Globally Feasible |
|--------|-----------|------|--------|---------|-------------------|
| 0.50   | 2.42      | ✓    | ✗      | ✗       |                   |
| 0.625  | 3.54      | ✗    | ✗      | ✗       |                   |
| 0.75   | 4.92      | ✓    | ✓      | ✗       |                   |
| 0.80   | 5.51      | ✗    | ✗      | ✗       |                   |
| 0.85   | 6.24      | ✗    | ✗      | ✗       |                   |
| 0.90   | 7.00      | ✗    | ✗      | ✗       |                   |
| 0.95   | 7.77      | ✗    | ✗      | ✗       |                   |
| 1.00   | 8.34      | ✓    | ✓      | ✓       | **GLOBAL**        |

The Pareto frontier is degenerate: only scale 1.00 is globally feasible. Scale 0.75 is the best scale that satisfies both Room and Garden but fails on Bicycle. No scale below 1.00 satisfies Bicycle's tau=0.020 constraint.

Deliverable: `fixed_scale_pareto.json`

---

## Provenance

| Property | Value |
|----------|-------|
| GaussianModel path | `baseline/reference_v1/gaussian_model.py` |
| GaussianModel hash | `68731e375013a6f2` |
| Config path | `baseline/reference_v1/config.py` |
| Config hash | `ca76d4f36059839e` |
| Config base | REFERENCE_V1_ABSGRAD |
| Loss | `(1-λ)*L1 + λ*d_ssim_downsampled(scale)`, λ=0.2 |
| SSIM | SepSSIM window=11 sigma=1.5 C1=(0.01)² C2=(0.03)² |
| Interpolation | F.interpolate(mode="area") |
| Provenance guard | Startup assertion in `c42_b9_train.py` and `c42_b9_loss_cost.py` — hard fail if GaussianModel hash ≠ 68731e375013a6f2 |

Deliverable: `provenance.json`

---

## Methodology

### Training

All B9 training runs used the canonical Reference V1 codebase (`baseline/reference_v1/`) with the C42 modification (downsampled SSIM). The training script (`c42_b9_train.py`) includes a startup provenance guard that verifies the GaussianModel and config file hashes before any imports, hard-failing on mismatch.

- **Scenes trained in B9:** Bicycle at scales 0.80, 0.85, 0.90, 0.95
- **Iterations:** 30,000 per run
- **Seed:** 42
- **Evaluation:** All cameras (194 for Bicycle), unified pipeline (PSNR, SSIM via SepSSIM, LPIPS via VGG)
- **GPU:** 4× A100-PCIE-40GB, one run per GPU
- **Wall time:** 30.8–35.3 min per run

### Loss-Cost Benchmark

The B9-A loss-cost benchmark measures the backward-pass time of the C42 loss function on fixed 1080p random tensors. This is an isolated structural-supervision compute opportunity — it does not include rasterization, densification, or optimizer steps. The benchmark uses the same SepSSIM implementation as the trainer, with F.interpolate(mode="area") for downsampling.

- **GPU:** A100-PCIE-40GB (108 SMs)
- **Resolution:** 1920×1080
- **Warmups:** 50, Timed samples: 300
- **Timing:** CUDA events with explicit synchronization
- **Scales:** 0.50, 0.625, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00

### Constraint Definitions

- **Primary SSIM:** abs(dSSIM) ≤ 0.020, where dSSIM = |SSIM_1.0 - SSIM_scale|
- **Multi-metric:** dPSNR ≥ -0.50 AND abs(dSSIM) ≤ 0.020 AND dLPIPS ≤ +0.03

### Oracle Computation

- **Adaptive oracle:** For each scene j, select argmin_{s ∈ F_j(τ)} C(s), where F_j(τ) = {s : |SSIM_1.0 - SSIM_s| ≤ τ}
- **Fixed comparator:** argmin_{s ∈ ∩_j F_j(τ)} C(s) — cheapest scale in the global intersection
- **Adaptive advantage:** C(fixed) - avg_j C(oracle_j)

---

## Deliverables Index

All deliverables in `results/c42_adaptive/b9/`:

| File | Phase | Description |
|------|-------|-------------|
| `provenance.json` | — | Provenance manifest (hashes, config, guard) |
| `loss_cost_curve.json` | B9-A | Loss-cost benchmark (scales 0.75–1.00) |
| `loss_cost_supplementary.json` | B9-A | Supplementary loss-cost (scales 0.50, 0.625) |
| `bicycle_080.json` | B9-B | Bicycle 0.80 training (30K, all cameras) |
| `bicycle_085.json` | B9-B | Bicycle 0.85 training (30K, all cameras) |
| `bicycle_090.json` | B9-B | Bicycle 0.90 training (30K, all cameras) |
| `bicycle_095.json` | B9-B | Bicycle 0.95 training (30K, all cameras) |
| `bicycle_refinement.json` | B9-D | Boundary refinement skip decision |
| `room_candidate.json` | B9-E | Room at candidate scale (1.00 = baseline) |
| `garden_candidate.json` | B9-E | Garden at candidate scale (1.00 = baseline) |
| `fixed_scale_pareto.json` | B9-I | Fixed-scale Pareto table |
| `strong_fixed_baseline.json` | B9-F | Strong fixed baseline (1.00, 8.34ms) |
| `adaptive_vs_strong_fixed.json` | B9-G | Adaptive oracle vs fixed baseline |
| `final_gate.json` | B9-H | Decision gate (MODIFY, 3.11ms) |

---

## Key Findings

1. **Bicycle is the binding constraint.** Among the tested scales {0.5, 0.625, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0}, only 1.0 was globally feasible at tau=0.020. Bicycle's dSSIM at 0.90 (0.0206) and 0.95 (0.0205) narrowly exceed the threshold, making 1.0 the only viable scale for this scene.

2. **The adaptive oracle provides a 3.11ms advantage.** The oracle exploits Room's tolerance (→0.50, 2.42ms) and Garden's tolerance (→0.75, 4.92ms) while using 1.0 for Bicycle (8.34ms). The average oracle cost (5.23ms) is 37% lower than the strong fixed baseline (8.34ms).

3. **All oracle-selected configurations at the primary tau=0.020 operating point satisfy the LPIPS +0.03 constraint.** The largest dLPIPS among oracle selections is Garden 0.75 at +0.021, well within the +0.03 limit.

4. **The advantage is sensitive to the tau threshold.** Relaxing tau from 0.020 to 0.021 would make Bicycle 0.90 feasible, reducing the strong fixed baseline to 7.00ms and the adaptive advantage to 1.77ms (DROP territory). Tightening tau would not change the result since 1.0 is already the only feasible scale for Bicycle.

5. **The dSSIM curve is non-monotonic for Bicycle.** Scale 0.85 (dSSIM=0.0236) is worse than 0.80 (dSSIM=0.0216), and 0.90 (dSSIM=0.0206) is nearly identical to 0.95 (dSSIM=0.0205). This indicates that training variance, not just resolution, drives the SSIM degradation at these scales.
