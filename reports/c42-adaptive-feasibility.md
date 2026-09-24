# C42++ Adaptive Structural Supervision Feasibility (Corrected)

**Date**: 2026-09-15
**Status**: COMPLETE — Offline analysis only (no GPU, no training, no new experiments)
**Decision**: `C42_ADAPTIVE_MODIFY`
**Correction**: The previous `DROP` decision was based on an invalid comparison: a quality-constrained oracle was compared against fixed-0.5, which *violates* that same quality constraint on Garden and Bicycle. This corrected analysis compares the oracle against the best *globally feasible* fixed policy under the same constraint.

---

## C42_ADAPTIVE_REANALYSIS

```
TAU_005:
  adaptive_cost            = 33.15 ms  (all scenes -> 1.0)
  best_fixed_feasible_cost = 33.15 ms  (policy: 1.0)
  saving                   = 0.00 ms (0.0%)
  fixed_policy             = 1.0

TAU_010:
  adaptive_cost            = 25.18 ms  (room->0.5, garden->1.0, bicycle->1.0)
  best_fixed_feasible_cost = 33.15 ms  (policy: 1.0)
  saving                   = 7.97 ms (24.0%)
  fixed_policy             = 1.0

TAU_015:
  adaptive_cost            = 25.18 ms  (room->0.5, garden->1.0, bicycle->1.0)
  best_fixed_feasible_cost = 33.15 ms  (policy: 1.0)
  saving                   = 7.97 ms (24.0%)
  fixed_policy             = 1.0

TAU_020:
  adaptive_cost            = 20.60 ms  (room->0.5, garden->0.75, bicycle->1.0)
  best_fixed_feasible_cost = 33.15 ms  (policy: 1.0)
  saving                   = 12.55 ms (37.9%)
  fixed_policy             = 1.0

TAU_025:
  adaptive_cost            = 9.24 ms  (room->0.5, garden->0.5, bicycle->0.5)
  best_fixed_feasible_cost = 9.24 ms  (policy: 0.5)
  saving                   = 0.00 ms (0.0%)
  fixed_policy             = 0.5

TAU_030:
  adaptive_cost            = 9.24 ms  (room->0.5, garden->0.5, bicycle->0.5)
  best_fixed_feasible_cost = 9.24 ms  (policy: 0.5)
  saving                   = 0.00 ms (0.0%)
  fixed_policy             = 0.5

SPEED_FIRST:
  best_policy              = fixed_0.5
  quality_tradeoff         = garden: dPSNR=-0.40, dSSIM=-0.0224;
                             bicycle: dPSNR=-0.04, dSSIM=-0.0213;
                             room: dPSNR=+0.24, dSSIM=-0.0078

QUALITY_CONSTRAINED:
  oracle_opportunity       = PASS
  max_oracle_saving        = 12.55 ms (37.9%) at tau=0.020

CURRENT_HF_PREDICTOR       = FAIL
  (constraint violation rate = 33.3% at max-opportunity tau=0.020)

FINAL_DECISION             = MODIFY

NEXT_ACTION                = Search for a better cheap signal that can predict
                             which scenes require scale=1.0 under tight quality
                             constraints. Do NOT launch full adaptive training
                             yet — the offline oracle is from fixed-scale
                             trajectories, not evidence that online scale-
                             switching follows the same path.
```

---

## 1. Corrected Problem Definition

### Constrained optimization

For quality threshold tau, the feasible scale set for scene/checkpoint j is:

```
F_j(tau) = { s : |SSIM_1.0,j - SSIM_s,j| <= tau }
```

The per-scene adaptive oracle selects the cheapest feasible scale:

```
s_j* = argmin_{s in F_j(tau)} C(s)
```

### Correct fixed-policy comparator

The globally fixed feasible set is the intersection across all scenes:

```
F_global(tau) = intersection of all F_j(tau)
```

The best fixed feasible policy is:

```
s_fixed* = argmin_{s in F_global(tau)} C(s)
```

**Key correction**: The previous analysis compared the adaptive oracle against fixed-0.5, but fixed-0.5 violates the quality constraint on Garden (|dSSIM|=0.0224) and Bicycle (|dSSIM|=0.0213) for tau < 0.025. Comparing a constrained oracle against an infeasible fixed policy is invalid. The correct comparison is oracle vs best **globally feasible** fixed policy.

### UNKNOWN handling

Scales with missing SSIM measurements (never tested or only 10K screening) are marked UNKNOWN and excluded from feasible sets. They are neither feasible nor infeasible. This means:
- The computed oracle saving is a **lower bound** — true opportunity could be larger if UNKNOWN scales are feasible.
- The computed best-fixed-feasible may be conservative — more scales might be globally feasible if measurements existed.

---

## 2. Data

### 30K SSIM values (canonical evidence)

| Scene | Scale 1.0 | Scale 0.75 | Scale 0.625 | Scale 0.5 | Source |
|-------|-----------|------------|-------------|-----------|--------|
| Room | 0.9263 | UNKNOWN | UNKNOWN | 0.9185 | S2.2 report |
| Garden | 0.8994 | 0.8811 | UNKNOWN (10K only) | 0.8770 | S2.3 Pareto |
| Bicycle | 0.8383 | UNKNOWN (10K only) | 0.8084 | 0.8170 | S2.3 Pareto |

### |dSSIM| from scale 1.0 at 30K

| Scene | 0.75 | 0.625 | 0.5 |
|-------|------|-------|-----|
| Room | UNKNOWN | UNKNOWN | 0.0078 |
| Garden | 0.0183 | UNKNOWN | 0.0224 |
| Bicycle | UNKNOWN | 0.0299 | 0.0213 |

### Loss costs (historical, fixed-tensor, A100, 1080p)

| Scale | Forward+Backward (ms) | Speedup vs 1.0 |
|-------|----------------------|-----------------|
| 1.0 | 33.15 | 1.00x |
| 0.75 | 19.41 | 1.71x |
| 0.625 | 13.92 | 2.38x |
| 0.5 | 9.24 | 3.58x |

### 10K SSIM values (screening data + 30K run checkpoints at iter=10000)

| Scene | Scale 1.0 | Scale 0.75 | Scale 0.625 | Scale 0.5 |
|-------|-----------|------------|-------------|-----------|
| Room | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| Garden | 0.8675 | 0.8505 | 0.8452 | 0.8469 |
| Bicycle | 0.7061 | 0.7021 | 0.7029 | 0.7104 |

### |dSSIM| from scale 1.0 at 10K

| Scene | 0.75 | 0.625 | 0.5 |
|-------|------|-------|-----|
| Garden | 0.0170 | 0.0223 | 0.0206 |
| Bicycle | 0.0040 | 0.0032 | 0.0043 |

**Key observation**: At 10K, all Bicycle scales are feasible even at tau=0.005 (max |dSSIM| = 0.0043). At 30K, Bicycle requires scale=1.0 for tau <= 0.020. This is strong stage dependence.

---

## 3. Tau Sweep Results (30K)

### Per-scene feasible sets

| tau | Room feasible | Garden feasible | Bicycle feasible | Global feasible |
|-----|---------------|-----------------|-------------------|-----------------|
| 0.005 | {1.0} | {1.0} | {1.0} | {1.0} |
| 0.010 | {1.0, 0.5} | {1.0} | {1.0} | {1.0} |
| 0.015 | {1.0, 0.5} | {1.0} | {1.0} | {1.0} |
| 0.020 | {1.0, 0.5} | {1.0, 0.75} | {1.0} | {1.0} |
| 0.025 | {1.0, 0.5} | {1.0, 0.75, 0.5} | {1.0, 0.5} | {1.0, 0.5} |
| 0.030 | {1.0, 0.5} | {1.0, 0.75, 0.5} | {1.0, 0.625, 0.5} | {1.0, 0.5} |

### Oracle vs best fixed feasible

| tau | Oracle (Room, Garden, Bicycle) | Oracle avg cost | Best fixed feasible | Fixed cost | Saving | % |
|-----|-------------------------------|-----------------|---------------------|------------|--------|---|
| 0.005 | (1.0, 1.0, 1.0) | 33.15 ms | 1.0 | 33.15 ms | 0.00 ms | 0.0% |
| 0.010 | (0.5, 1.0, 1.0) | 25.18 ms | 1.0 | 33.15 ms | **7.97 ms** | **24.0%** |
| 0.015 | (0.5, 1.0, 1.0) | 25.18 ms | 1.0 | 33.15 ms | **7.97 ms** | **24.0%** |
| 0.020 | (0.5, 0.75, 1.0) | 20.60 ms | 1.0 | 33.15 ms | **12.55 ms** | **37.9%** |
| 0.025 | (0.5, 0.5, 0.5) | 9.24 ms | 0.5 | 9.24 ms | 0.00 ms | 0.0% |
| 0.030 | (0.5, 0.5, 0.5) | 9.24 ms | 0.5 | 9.24 ms | 0.00 ms | 0.0% |

**Maximum oracle opportunity**: tau=0.020, saving=12.55 ms (37.9%).

The oracle opportunity is **REAL and MATERIALLY SIGNIFICANT** at tau=0.010–0.020. The key insight is:
- At tau=0.010–0.015, only Room can downsample to 0.5 (Garden and Bicycle need 1.0). The best fixed feasible is 1.0 (33.15ms). The adaptive oracle saves 7.97ms by using 0.5 for Room only.
- At tau=0.020, Room uses 0.5, Garden uses 0.75, Bicycle needs 1.0. The best fixed feasible is still 1.0. The oracle saves 12.55ms.
- At tau >= 0.025, all scenes can use 0.5, so the oracle coincides with fixed-0.5. No adaptive advantage.

### Why the previous analysis was wrong

The previous analysis compared:
- Oracle average cost = 25.18 ms (at tau=0.010)
- Fixed-0.5 cost = 9.24 ms
- Claimed saving = -15.93 ms (oracle MORE expensive)

But fixed-0.5 **violates** the tau=0.010 constraint on Garden (|dSSIM|=0.0224 > 0.010) and Bicycle (|dSSIM|=0.0213 > 0.010). The correct fixed comparator at tau=0.010 is scale=1.0 (the only globally feasible scale), costing 33.15 ms. The oracle saves 7.97 ms (24.0%), not loses 15.93 ms.

---

## 4. Stage Dependence (10K vs 30K)

| Scene | tau | 10K oracle | 30K oracle | Direction |
|-------|-----|-----------|-----------|-----------|
| Bicycle | 0.005 | 0.5 | 1.0 | More conservative at 30K |
| Bicycle | 0.010 | 0.5 | 1.0 | More conservative at 30K |
| Bicycle | 0.015 | 0.5 | 1.0 | More conservative at 30K |
| Bicycle | 0.020 | 0.5 | 1.0 | More conservative at 30K |

At 10K, Bicycle's |dSSIM| at scale 0.5 is only 0.0043 — all scales are feasible even at tau=0.005. At 30K, the |dSSIM| grows to 0.0213, requiring scale=1.0 for tau <= 0.020.

**Implication**: A stage-dependent oracle that uses scale=0.5 early and switches to 1.0 later for Bicycle would satisfy the constraint at all stages while saving compute in the early phase. However, this is from independently trained trajectories — the counterfactual limitation applies.

Garden's oracle is stable: 1.0 at tau <= 0.015, 0.75 at tau=0.020, 0.5 at tau >= 0.025, at both 10K and 30K.

Room's 10K data is UNKNOWN (no multi-scale screening for Room).

---

## 5. Two Problem Formulations

### Mode A — Speed-first

**Objective**: Maximize training throughput, accept measured quality tradeoff.

**Best current solution**: Fixed scale=0.5.

| Scene | dPSNR | dSSIM | Throughput | Interpretation |
|-------|-------|-------|------------|----------------|
| Room | +0.24 | -0.0078 | 1.90x | Strictly better (higher PSNR, fewer Gaussians) |
| Garden | -0.40 | -0.0224 | 1.68x | Quality degradation accepted for speed |
| Bicycle | -0.04 | -0.0213 | 1.60x | Negligible PSNR loss, SSIM degradation |

Fixed-0.5 is the fastest tested common Pareto operating point with scene-dependent quality degradation. It is **NOT** a quality-constrained policy when tau < 0.025 — it violates the constraint on 2/3 scenes.

### Mode B — Quality-constrained

**Objective**: Minimize supervision cost subject to |dSSIM| <= tau.

This is where adaptive resolution has value. The oracle opportunity exists at tau=0.010–0.020:

- At tau=0.010: oracle saves 7.97ms (24.0%) by using 0.5 for Room, 1.0 for Garden/Bicycle
- At tau=0.020: oracle saves 12.55ms (37.9%) by using 0.5 for Room, 0.75 for Garden, 1.0 for Bicycle
- At tau >= 0.025: no opportunity (fixed-0.5 is feasible and optimal)

**These two modes must not be mixed.** The previous analysis conflated them by treating fixed-0.5 as a quality-constrained competitor.

---

## 6. Predictor Evaluation (Corrected Metrics)

### Deleted metric: "Oracle scale-label accuracy"

The previous analysis evaluated predictors by whether they predicted the exact oracle scale label. This is invalid because:
1. Different scales can satisfy the same constraint at different costs.
2. A predictor that selects a feasible but non-oracle scale is not "wrong" — it has nonzero regret but zero violation.
3. The correct metrics are constraint violation rate and feasible cost regret.

### Constraint violation rate

| Policy | tau=0.005 | tau=0.010 | tau=0.015 | tau=0.020 | tau=0.025 | tau=0.030 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| HF predictor | 3/3 (100%) | 2/3 (67%) | 2/3 (67%) | 1/3 (33%) | 1/3 (33%) | 0/3 (0%) |
| Fixed-0.5 | 3/3 (100%) | 2/3 (67%) | 2/3 (67%) | 2/3 (67%) | 0/3 (0%) | 0/3 (0%) |
| Fixed-0.75 | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 2/3 (67%) | 2/3 (67%) | 2/3 (67%) |

**HF predictor violations at max-opportunity tau=0.020**: 1/3 (33.3%) — Bicycle predicted as 0.625 but |dSSIM|=0.0299 > 0.020. **FAIL.**

### Feasible cost regret (at tau=0.020)

| Policy | Room | Garden | Bicycle | Avg regret (non-violating) |
|--------|------|--------|---------|---------------------------|
| HF predictor | 0.5 (regret=0) | 0.75 (regret=0) | 0.625 (VIOLATION) | 0.00 ms |
| Fixed-0.5 | 0.5 (regret=0) | 0.5 (VIOLATION) | 0.5 (VIOLATION) | 0.00 ms |
| Fixed-0.75 | 0.75 (regret=10.17) | 0.75 (regret=0) | 0.75 (VIOLATION) | 5.08 ms |

The HF predictor has zero regret on non-violating scenes, but the Bicycle violation means it cannot be trusted under the quality constraint at tau=0.020.

### Why the HF predictor fails

The HF predictor maps: Garden HF=0.766 -> 0.75, Bicycle HF=0.561 -> 0.625, Room (no data) -> 0.5.

At tau=0.020, Bicycle's oracle is 1.0 (|dSSIM| at 0.625 = 0.0299 > 0.020, |dSSIM| at 0.5 = 0.0213 > 0.020). The HF predictor predicts 0.625 for Bicycle, which violates the constraint. The problem is that Bicycle's low HF fraction (0.561) suggests it should tolerate aggressive downsampling, but its SSIM degradation at 30K is actually large — the HF fraction does not predict 30K SSIM degradation.

This is a **predictor failure**, not an oracle failure. The oracle opportunity is real; the current signal just cannot capture it.

---

## 7. Leave-One-Scene-Out (Corrected)

| tau | HF violation rate | Fixed-0.5 violation rate |
|-----|-------------------|--------------------------|
| 0.005 | 100% (3/3) | 100% (3/3) |
| 0.010 | 67% (2/3) | 67% (2/3) |
| 0.015 | 67% (2/3) | 67% (2/3) |
| 0.020 | 67% (2/3) | 67% (2/3) |
| 0.025 | 0% (0/3) | 0% (0/3) |
| 0.030 | 0% (0/3) | 0% (0/3) |

At the max-opportunity tau=0.020, both HF predictor and fixed-0.5 have 67% violation rate. Neither satisfies the quality constraint on 2/3 scenes.

**Key limitation**: Only 2 scenes (Garden, Bicycle) have HF data. LOSO with 2 training scenes is statistically weak. Room has no HF probe — any HF-based rule defaults to 0.5 for Room.

---

## 8. Classification and Decision

| Criterion | Status | Evidence |
|-----------|--------|----------|
| ORACLE_OPPORTUNITY | **PASS** | Max saving 12.55ms (37.9%) at tau=0.020. Oracle materially better than best fixed feasible (1.0) at tau=0.010–0.020. |
| CURRENT_PREDICTOR | **FAIL** | HF predictor has 33% constraint violation rate at max-opportunity tau. Cannot reliably predict which scenes need scale=1.0. |
| ADAPTIVE_METHOD | **MODIFY** | Oracle opportunity justifies continued investigation. A better cheap signal is needed. |

### Decision: MODIFY

**Rationale**: The oracle opportunity is REAL — at tau=0.010–0.020, the adaptive oracle saves 7.97–12.55ms (24–38%) over the best globally-feasible fixed policy (scale=1.0). This is because only Room can downsample at tight quality constraints; Garden and Bicycle need full resolution, making the globally fixed policy expensive. The adaptive oracle exploits scene-specific feasibility to save compute where possible.

However, the current HF predictor FAILS — it predicts 0.625 for Bicycle (violating the constraint at tau <= 0.030) because Bicycle's low HF fraction (0.561) does not predict its large 30K SSIM degradation. A better cheap signal is needed to distinguish scenes that require full resolution from those that can downsample.

### Decision rules applied

```
oracle no better than best feasible fixed policy  -> DROP
oracle materially better, predictor fails          -> MODIFY  <== THIS CASE
oracle materially better, cheap predictor works    -> KEEP
```

---

## 9. Mechanism Wording Correction

The previous report claimed the Bicycle scale=0.75 gradient cosine anomaly (cosine=0.506, rel_l2=8.19) "explains" the 0.625-vs-0.5 Pareto domination at 30K. This causal claim was too strong.

**Corrected wording**: The anomalous multi-resolution gradient alignment is consistent with non-monotonic scale-dependent optimization dynamics, but causality has not been established. The anomaly and the Pareto outcome may share a common cause (e.g., Bicycle's high-frequency outdoor content interacting with intermediate downsampling) without one causing the other.

---

## 10. Counterfactual Limitation

**The offline oracle is constructed from independently trained fixed-scale trajectories. It is not yet evidence that switching scales online would follow the same trajectory.**

Each fixed-scale run was trained independently for 30K iterations at a constant scale. An adaptive system that switches scales mid-training would follow a different optimization path — the Gaussian population, densification history, and loss landscape would differ. The oracle saving computed here is an upper bound on what online adaptive scheduling could achieve.

**Therefore**: No full 30K adaptive training should be launched yet. The next step is to find a cheap signal that can predict scene-specific feasibility, then validate with a small-scale online experiment (not a full 30K run).

---

## 11. UNKNOWN Scale Impact

| Scene | Missing scales | Impact if feasible |
|-------|---------------|-------------------|
| Room | 0.75, 0.625 | Room oracle could be cheaper than 0.5 at tau >= 0.010. But 0.5 already costs only 9.24ms, so saving is bounded. |
| Garden | 0.625 (30K) | Garden oracle could use 0.625 (13.92ms) instead of 0.75 (19.41ms) at tau=0.020, saving 5.49ms. |
| Bicycle | 0.75 (30K) | Bicycle could have more options, but 0.75 at 10K had |dSSIM|=0.004, so it might be feasible at tight tau. |

The computed oracle saving is a **lower bound** — true opportunity could be larger if UNKNOWN scales are feasible. The computed best-fixed-feasible may be conservative — more scales might be globally feasible if measurements existed.

---

## 12. Next Actions

1. **Search for better cheap signals**: The HF fraction fails because it doesn't predict 30K SSIM degradation. Candidate signals:
   - Stage-aware signals: measure |dSSIM| at current scale vs full resolution on a few cameras (moderate overhead, but directly measures the constraint)
   - Scene complexity proxies: texture frequency, depth variance, Gaussian scale distribution
   - Training dynamics: PSNR trajectory slope, densification rate

2. **Fill UNKNOWN scale measurements** (small GPU probes, not full 30K):
   - Room at 0.75 and 0.625 (would complete the Room oracle)
   - Garden 0.625 at 30K (would test if 0.625 is feasible, potentially increasing oracle saving)
   - Bicycle 0.75 at 30K (would test if 0.75 is feasible)

3. **Do NOT launch full 30K adaptive training** until a cheap signal is validated and the counterfactual limitation is addressed with a small-scale online experiment.

---

## Deliverables

| File | Description |
|------|-------------|
| `results/c42_adaptive/provenance.json` | Analysis provenance (unchanged) |
| `results/c42_adaptive/existing_evidence.json` | B0: Evidence inventory (unchanged) |
| `results/c42_adaptive/scene_signal_analysis.json` | B1: Scene dependence analysis (unchanged) |
| `results/c42_adaptive/oracle_schedule.json` | **B2 corrected**: Constrained oracle with tau sweep |
| `results/c42_adaptive/predictor_results.json` | **B3 corrected**: Constraint violation rate + cost regret |
| `results/c42_adaptive/leave_one_scene_out.json` | **B4 corrected**: LOSO with constraint-based metrics |
| `results/c42_adaptive/opportunity_estimate.json` | **B5 corrected**: Oracle vs best fixed feasible |
| `results/c42_adaptive/final_decision.json` | **Corrected**: MODIFY decision |
| `reports/c42-adaptive-feasibility.md` | This report |
| `c42_adaptive_corrected.py` | Corrected analysis script |

---

## Constraints Honored

- No GPU used. No training run. No new experiments.
- No Candidate C files touched. No N2 files touched.
- No gsplat CUDA source modified. No Reference V1 modified.
- All analysis used existing JSON data from S2.2 and S2.3.
- No historical C46 data mixed into canonical C42 evidence.
- Counterfactual limitation explicitly stated.
- Mechanism causality claim corrected.
- Two problem formulations (speed-first vs quality-constrained) clearly separated.

---

# B6 — Oracle Robustness Gate

**Date**: 2026-09-15
**Decision**: `C42_ORACLE_ROBUSTNESS = PASS`
**Predictor search**: JUSTIFIED

The corrected adaptive oracle (B2-B5) showed a 7.97-12.55ms (24-38%) opportunity at tau=0.010-0.020. B6 tests whether this opportunity survives four robustness challenges: UNKNOWN-scale sensitivity, multi-metric constraints, cost-model validity, and counterfactual limitations.

---

## B6-A — UNKNOWN-Scale Sensitivity

### Method

Four 30K SSIM measurements are UNKNOWN: Room 0.75, Room 0.625, Garden 0.625, Bicycle 0.75. For each tau, we enumerate all 2^4 = 16 feasible/infeasible assignments and recompute the adaptive advantage DC = C(best-fixed-feasible) - C(adaptive-oracle).

**Key insight**: Adding feasible UNKNOWN scales can change BOTH the adaptive oracle (more options per scene) AND the global fixed feasible set (more options for the fixed policy). The advantage can increase or decrease.

### Results

| tau | Min DC (ms) | Nominal DC (ms) | Max DC (ms) | Range (ms) | Robust (min > 0.5ms) |
|-----|-------------|-----------------|-------------|------------|----------------------|
| 0.005 | 0.00 | 0.00 | 17.40 | 17.40 | No |
| 0.010 | 7.97 | 7.97 | 18.96 | 10.99 | **Yes** |
| 0.015 | 7.97 | 7.97 | 18.96 | 10.99 | **Yes** |
| 0.020 | 3.39 | 12.55 | 18.96 | 15.57 | **Yes** |
| 0.025 | 0.00 | 0.00 | 0.00 | 0.00 | No |
| 0.030 | 0.00 | 0.00 | 0.00 | 0.00 | No |

**Nominal** = all UNKNOWN scales assumed infeasible (conservative current estimate).

### Min-advantage case at tau=0.020 (DC=3.39ms)

When bicycle_0.75 becomes feasible, scale 0.75 becomes globally feasible (Room 0.75 also feasible), making the fixed policy cheaper (19.41ms instead of 33.15ms). The oracle also gets cheaper (Bicycle uses 0.75 instead of 1.0), but the fixed policy benefits MORE. Both policies improve, shrinking the advantage from 12.55ms to 3.39ms.

### Max-advantage case at tau=0.020 (DC=18.96ms)

When garden_0.625 and bicycle_0.75 become feasible but NOT globally feasible (Room 0.75 infeasible), the oracle gets cheaper (Garden uses 0.625, Bicycle uses 0.75) while the fixed policy stays at 1.0. Advantage grows to 18.96ms.

### Conclusion

The opportunity is **ROBUST** at tau=0.010-0.020: the minimum advantage over all 16 UNKNOWN assignments is 3.39-7.97ms, exceeding the 0.5ms materiality threshold. The opportunity does not depend on any specific UNKNOWN-scale outcome.

---

## B6-B — Value of Information for Missing Scales

| Rank | Missing Scale | Max Swing in DC (ms) | Best tau |
|------|--------------|---------------------|----------|
| 1 | **Garden 0.625** | 6.41 | 0.010 |
| 2 | Room 0.75 | 4.58 | 0.020 |
| 3 | Bicycle 0.75 | 4.58 | 0.010 |
| 4 | Room 0.625 | 1.83 | 0.005 |

**Garden 0.625** has the highest value of information: knowing whether it's feasible at 30K could change the adaptive advantage by up to 6.41ms. This is because Garden 0.625 (13.92ms) is cheaper than Garden 0.75 (19.41ms), so if feasible, it reduces the oracle cost without helping the fixed policy (0.625 is unlikely to be globally feasible since Bicycle 0.625 has |dSSIM|=0.0299).

**Room 0.75 and 0.625** have lower value (4.58ms and 1.83ms) because Room 0.5 is already the cheapest feasible scale at tau >= 0.010, so Room intermediate scales cannot improve the Room oracle.

---

## B6-C — Corrected Room Statement

**For tau >= 0.010**: Room scale=0.5 is feasible (|dSSIM|=0.0078 <= tau) and is the cheapest tested candidate (9.24ms). Room 0.75 (19.41ms) and 0.625 (13.92ms) are MORE expensive than 0.5. Therefore, even if Room 0.75/0.625 are feasible at tau >= 0.010, the Room oracle CANNOT improve below 9.24ms. **These measurements have ZERO value for the Room adaptive oracle at tau >= 0.010.**

**For tau=0.005**: Room 0.5 is INFEASIBLE (|dSSIM|=0.0078 > 0.005). If Room 0.75 or 0.625 were feasible, the Room oracle could be 19.41ms or 13.92ms instead of 33.15ms. However, 0.75 and 0.625 are more aggressive downsampling than 0.5, so their |dSSIM| is likely >= 0.0078, making feasibility at tau=0.005 unlikely.

**Global feasibility**: Room 0.75/0.625 do NOT affect global fixed-policy feasibility because global feasibility requires ALL scenes to pass. Room's scale feasibility is irrelevant if Garden or Bicycle fail the same scale.

**Conclusion**: Room 0.75 and 0.625 measurements should NOT be prioritized for GPU measurement.

---

## B6-D — Multi-Metric Constrained Oracle

### Available 30K quality data

| Scene | Metric | 1.0 | 0.75 | 0.625 | 0.5 |
|-------|--------|-----|------|-------|-----|
| Room | PSNR | 32.30 | UNKNOWN | UNKNOWN | 32.54 |
| Room | SSIM | 0.9263 | UNKNOWN | UNKNOWN | 0.9185 |
| Room | LPIPS | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| Garden | PSNR | 29.63 | 29.28 | UNKNOWN | 29.23 |
| Garden | SSIM | 0.8994 | 0.8811 | UNKNOWN | 0.8770 |
| Garden | LPIPS | 0.1400 | 0.1613 | UNKNOWN | 0.1655 |
| Bicycle | PSNR | 26.47 | UNKNOWN | 26.14 | 26.43 |
| Bicycle | SSIM | 0.8383 | UNKNOWN | 0.8084 | 0.8170 |
| Bicycle | LPIPS | 0.2436 | UNKNOWN | 0.2750 | 0.2633 |

### Oracle opportunity under different constraint families

| Constraint Family | Oracle (Room, Garden, Bicycle) | Oracle avg (ms) | Fixed feasible | Advantage (ms) |
|-------------------|-------------------------------|-----------------|----------------|----------------|
| SSIM only, tau=0.010 | (0.5, 1.0, 1.0) | 25.18 | 1.0 | **7.97** |
| SSIM only, tau=0.020 | (0.5, 0.75, 1.0) | 20.60 | 1.0 | **12.55** |
| PSNR>=-0.20, SSIM<=0.010 | (0.5, 1.0, 1.0) | 25.18 | 1.0 | **7.97** |
| PSNR>=-0.50, SSIM<=0.010 | (0.5, 1.0, 1.0) | 25.18 | 1.0 | **7.97** |
| PSNR>=-0.20, SSIM<=0.020 | (0.5, 1.0, 1.0) | 25.18 | 1.0 | **7.97** |
| PSNR>=-0.50, SSIM<=0.020 | (0.5, 0.75, 1.0) | 20.60 | 1.0 | **12.55** |
| PSNR>=-0.50, SSIM<=0.010, LPIPS<=0.03 | (1.0, 1.0, 1.0) | 33.15 | 1.0 | 0.00 |
| PSNR>=-0.50, SSIM<=0.020, LPIPS<=0.03 | (1.0, 0.75, 1.0) | 28.57 | 1.0 | **4.58** |

### Findings

1. **PSNR constraint does not bind**: At eps_P=0.50, all tested scales pass PSNR. At eps_P=0.20, Garden 0.75/0.5 fail (-0.35/-0.40 < -0.20), but these scales already fail SSIM at tau=0.010. So PSNR adds no additional restriction beyond SSIM at these thresholds.

2. **SSIM+PSNR opportunity = SSIM-only opportunity**: The advantage is identical under SSIM-only and PSNR+SSIM at matching tau_S. The opportunity does NOT depend on choosing SSIM as the only quality constraint.

3. **LPIPS constraint reduces but does not eliminate opportunity at tau=0.020**: Adding LPIPS<=0.03 reduces the advantage from 12.55ms to 4.58ms because Room has NO LPIPS data at any scale, forcing Room oracle to 1.0 (UNKNOWN = conservative exclusion). This is a DATA LIMITATION, not a quality limitation. The 4.58ms advantage is still material.

4. **LPIPS eliminates opportunity at tau=0.010**: With LPIPS, Room goes to 1.0, and Garden/Bicycle also need 1.0 at tau=0.010 (SSIM fails), so oracle = fixed = 33.15ms, advantage = 0.0ms.

### Conclusion

The adaptive opportunity does NOT depend on choosing SSIM as the only quality constraint. It survives PSNR+SSIM constraints with identical advantage. Under full PSNR+SSIM+LPIPS, the advantage reduces to 4.58ms at tau=0.020 (due to missing Room LPIPS data) but remains material.

---

## B6-E — Cost-Model Robustness

### Fixed-tensor cost model (current oracle)

| Scale | Forward+Backward (ms) | Source |
|-------|----------------------|--------|
| 1.0 | 33.15 | fixed_tensor_loss_benchmark.json, A100, 1080p frozen tensors |
| 0.75 | 19.41 | same |
| 0.625 | 13.92 | same |
| 0.5 | 9.24 | same |

These are scene-independent (same frozen tensors, same image size). Protocol: 50 warmup, 300 timed, CUDA events, explicit sync.

### Realized scene-specific loss costs

| Scene | Scale 1.0 (ms) | Scale 0.5 (ms) | Scale 0.75 | Scale 0.625 | Source |
|-------|---------------|----------------|------------|-------------|--------|
| Garden | 24.90 (fwd) | 7.12 (fwd) | N/A | N/A | phase_timing.json |
| Bicycle | 24.98 (fwd) | 6.91 (fwd) | N/A | N/A | phase_timing.json |
| Room | N/A | N/A | N/A | N/A | No phase timing |

**Critical distinction**: phase_timing.loss_ms measures the loss FORWARD pass during actual training. The fixed-tensor benchmark measures forward+backward of isolated SSIM. These are NOT directly comparable. The phase_timing also has a separate raster_bwd_ms that includes the loss backward.

### Total iteration times (scene-specific, includes rasterization)

| Scene | 1.0 (ms) | 0.75 (ms) | 0.625 (ms) | 0.5 (ms) |
|-------|----------|-----------|------------|----------|
| Garden | 67.1 | 50.4 | N/A | 39.9 |
| Bicycle | 75.3 | N/A | 50.0 | 47.2 |
| Room | 55.4 | N/A | N/A | 29.1 |

Total iteration times include rasterization (forward+backward), which depends on N_gaussians (different at each scale). The oracle models LOSS cost only, but total training speed also changes due to different Gaussian population trajectories.

### Conclusion

The 24-38% oracle opportunity is a **NORMALIZED_FIXED_TENSOR_OPPORTUNITY** estimate. **REALIZED_SCENE_COST_OPPORTUNITY** cannot be computed from existing data because:
1. No realized loss costs at intermediate scales (0.75, 0.625)
2. No phase timing for Room
3. The fixed-tensor benchmark is scene-independent while actual loss costs vary by scene

The realized opportunity may be larger or smaller than the fixed-tensor estimate.

---

## B6-F — Counterfactual Wording Correction

**Previous (incorrect)**: "The offline oracle is an upper bound on online adaptive scheduling."

**Corrected**: "The oracle is a fixed-trajectory counterfactual opportunity proxy. Because switching supervision scales changes the subsequent optimization trajectory, it is neither a formal upper nor lower bound on online adaptive training performance."

**Rationale**: The previous wording claimed the oracle is an "upper bound", implying online adaptive training can only do worse. This is incorrect: (1) online scale-switching may discover better optimization paths than any fixed scale, making it potentially better than the oracle; (2) it may also discover worse paths. Without running an actual adaptive experiment, neither direction can be established. The oracle is a PROXY for opportunity, not a BOUND.

---

## B6 Final Gate

```
C42_ORACLE_ROBUSTNESS = PASS

SSIM_ONLY:
  min_advantage  = 3.39 ms (at tau=0.020, worst UNKNOWN assignment)
  nominal_advantage = 12.55 ms (at tau=0.020, all UNKNOWN infeasible)
  max_advantage  = 18.96 ms (at tau=0.020, best UNKNOWN assignment)
  robust_range   = tau=0.010 to tau=0.020

MULTI_METRIC:
  opportunity = PASS (survives PSNR+SSIM with identical advantage)
  strongest_supported_constraint = PSNR(eps_P=0.50) + SSIM(tau_S=0.020)
  advantage = 12.55 ms (PSNR+SSIM) / 4.58 ms (PSNR+SSIM+LPIPS, Room LPIPS UNKNOWN)

UNKNOWN_SENSITIVITY:
  highest_value_missing_measurement = Garden 0.625
  max_effect_on_advantage = 6.41 ms

COST_MODEL:
  fixed_tensor_estimate = 12.55 ms (37.9%) at tau=0.020
  realized_scene_cost_available = PARTIAL (only Garden/Bicycle at 1.0 and 0.5)

FIXED_TRAJECTORY_LIMITATION:
  The oracle is a fixed-trajectory counterfactual opportunity proxy.
  Because switching supervision scales changes the subsequent optimization
  trajectory, it is neither a formal upper nor lower bound on online
  adaptive training performance.

PREDICTOR_SEARCH_JUSTIFIED = YES

GPU_MEASUREMENT_JUSTIFIED = YES

IF_GPU_MEASUREMENT_JUSTIFIED:
  single_highest_value_measurement = Garden 0.625 at 30K
  (could change advantage by up to 6.41 ms)

FINAL_DECISION = PASS
  The adaptive oracle opportunity is robust under UNKNOWN-scale sensitivity
  (min 3.39ms at tau=0.020), survives multi-metric constraints (PSNR+SSIM
  identical to SSIM-only; LPIPS reduces to 4.58ms but remains material),
  and does not depend on SSIM-only constraint selection. The cost-model
  estimate (12.55ms/37.9%) is a fixed-tensor proxy, not a realized saving.
  The counterfactual limitation is acknowledged. Predictor search is
  justified. If GPU measurement later becomes justified, Garden 0.625 at
  30K has the highest value of information.
```

---

## B6 Deliverables

| File | Description |
|------|-------------|
| `results/c42_adaptive/b6a_unknown_sensitivity.json` | B6-A: 16-assignment enumeration per tau |
| `results/c42_adaptive/b6b_value_of_information.json` | B6-B: Missing scale value ranking |
| `results/c42_adaptive/b6c_room_correction.json` | B6-C: Room oracle correction |
| `results/c42_adaptive/b6d_multi_metric.json` | B6-D: Multi-metric oracle (PSNR/SSIM/LPIPS) |
| `results/c42_adaptive/b6e_cost_model.json` | B6-E: Cost-model robustness |
| `results/c42_adaptive/b6f_counterfactual.json` | B6-F: Counterfactual wording |
| `results/c42_adaptive/b6_final_gate.json` | B6 final gate decision |
| `c42_b6_robustness.py` | B6 analysis script |

---

## B9 — Strong Fixed-Scale Boundary Search

**Date**: 2026-09-16
**Status**: COMPLETE
**Decision**: `MODIFY` (adaptive advantage = 3.11ms, 2.0 ≤ 3.11 < 5.0ms)

Full report: [`reports/c42-b9-strong-fixed-boundary.md`](c42-b9-strong-fixed-boundary.md)

### Summary

The B9 phase challenged the adaptive-oracle claim by searching for the cheapest globally fixed structural-supervision scale satisfying the primary quality constraint (abs(dSSIM) ≤ 0.020) across Room, Garden, and Bicycle. Four new scales were trained for Bicycle (0.80, 0.85, 0.90, 0.95) at 30K iterations with the canonical Reference V1 codebase, and a new loss-cost benchmark was conducted using the SepSSIM implementation on A100 at 1080p.

**Key finding:** Among the tested scales {0.5, 0.625, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0}, only 1.0 was globally feasible at tau=0.020. Bicycle is the binding constraint — dSSIM at 0.90 (0.0206) and 0.95 (0.0205) are marginal failures. The adaptive oracle exploits Room's tolerance (→0.50, 2.42ms) and Garden's tolerance (→0.75, 4.92ms) but must use 1.0 for Bicycle (8.34ms), yielding an average oracle cost of 5.23ms vs the strong fixed baseline of 8.34ms — a 3.11ms (37.3%) advantage.

All oracle-selected configurations at the primary tau=0.020 operating point satisfy the LPIPS +0.03 constraint.

### B9 Loss-Cost Curve (SepSSIM, A100, 1080p)

| Scale | Cost (ms) | Speedup |
|-------|-----------|---------|
| 0.50  | 2.42      | 3.44×   |
| 0.75  | 4.92      | 1.69×   |
| 0.90  | 7.00      | 1.19×   |
| 1.00  | 8.34      | 1.00×   |

### B9 Deliverables

| File | Description |
|------|-------------|
| `results/c42_adaptive/b9/provenance.json` | Provenance manifest |
| `results/c42_adaptive/b9/loss_cost_curve.json` | Loss-cost benchmark (scales 0.75–1.00) |
| `results/c42_adaptive/b9/loss_cost_supplementary.json` | Supplementary loss-cost (0.50, 0.625) |
| `results/c42_adaptive/b9/bicycle_080.json` | Bicycle 0.80 training (30K) |
| `results/c42_adaptive/b9/bicycle_085.json` | Bicycle 0.85 training (30K) |
| `results/c42_adaptive/b9/bicycle_090.json` | Bicycle 0.90 training (30K) |
| `results/c42_adaptive/b9/bicycle_095.json` | Bicycle 0.95 training (30K) |
| `results/c42_adaptive/b9/bicycle_refinement.json` | B9-D boundary refinement skip decision |
| `results/c42_adaptive/b9/room_candidate.json` | B9-E Room at candidate scale |
| `results/c42_adaptive/b9/garden_candidate.json` | B9-E Garden at candidate scale |
| `results/c42_adaptive/b9/fixed_scale_pareto.json` | B9-I fixed-scale Pareto table |
| `results/c42_adaptive/b9/strong_fixed_baseline.json` | B9-F strong fixed baseline |
| `results/c42_adaptive/b9/adaptive_vs_strong_fixed.json` | B9-G adaptive vs fixed |
| `results/c42_adaptive/b9/final_gate.json` | B9-H decision gate |
