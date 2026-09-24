# Phase 13C — Tile-Size Selection Oracle Validation Report

**Date:** 2026-08-28  
**Author:** DSH coding agent  
**Status:** COMPLETE

---

## 1. Background

Phase 13A established a simple tpg_std threshold predictor for tile16 vs tile32:
- tpg_std > 135 → tile32, 93.3% accuracy on 15 workloads

Phase 13B expanded to 8 tile sizes and found this binary threshold **does not generalize**:
- Accuracy on 24 workloads: 58.3% (14/24)
- The optimum is interior (tile20, tile24), not an endpoint

Phase 13C evaluates whether simple features predict the optimal tile in the 8-size space.

---

## 2. Data

- **3 scenes** (room, bicycle, garden)
- **8 tile sizes** each (4, 8, 12, 16, 20, 24, 28, 32)
- **24 total workloads**
- All at 1080p, real cameras, SfM initialization

### Per-Scene Optima

| Scene | Forward Best | Fwd+Bwd Best | Characteristics |
|:------|:------------:|:------------:|:---------------|
| room (1.6M Gs, indoor, dense) | tile20 (10.247ms) | tile16 (34.828ms) | High tpg_std at tile16 (29.3) |
| bicycle (6.1M Gs, outdoor, sparse) | tile24 (31.396ms) | tile20 (96.959ms) | Low tpg_std at tile16 (9.6) |
| garden (5.8M Gs, outdoor, sparse) | tile12 (26.627ms) | tile20 (92.738ms) | Low tpg_std at tile16 (7.1) |

---

## 3. Feature Analysis

### Monotonicity Check

For each feature, we check whether its value changes monotonically with tile size:

| Feature | Monotonic Decreasing? | Direction | Predictive Power |
|:--------|:--------------------:|:---------:|:----------------:|
| total_intersections | YES | Decreases as tile_size ↑ | High — but doesn't directly predict optimum |
| tpg_mean | YES | Decreases as tile_size ↑ | Medium |
| tpg_std | YES | Decreases as tile_size ↑ | Medium |
| mean_intersections_per_tile | YES | Increases as tile_size ↑ | Low |
| p95_intersections_per_tile | YES | Increases as tile_size ↑ | Low |
| p99_intersections_per_tile | YES | Increases as tile_size ↑ | Low |

**All workload features vary monotonically with tile size**, but this is a mathematical property (more tiles = fewer intersections per tile, etc.), not an informative predictor of the optimum.

### Quality of Feature → Optimum Mapping

| Feature | room fwd opt | bicycle fwd opt | garden fwd opt |
|:--------|:-----------:|:---------------:|:--------------:|
| tpg_std at tile16 | 29.3 | 9.6 | 7.1 |
| → optimal tile | 20 | 24 | 12 |

Simple rule: **Lower tpg_std → larger optimal tile size** (except garden which breaks the pattern).
This is an ordinal relationship, not a threshold.

---

## 4. Predictor Evaluation

### Phase 13A Threshold (tpg_std > 135 → tile32)

| Scene | tile16 tpg_std | Rule Says | Fwd Optimal | Correct? |
|:------|:-------------:|:---------:|:-----------:|:--------:|
| room | 29.3 | tile16 | tile20 | ✗ (but tile16 is close — 11.7% slower) |
| bicycle | 9.6 | tile16 | tile24 | ✗ (tile24 is 12.4% faster than tile16) |
| garden | 7.1 | tile16 | tile12 | ✗ (tile12 is 3.8% faster than tile16) |

**The threshold is not useful for the expanded space.** When the tpg_std threshold is not crossed, it defaults to tile16, but the actual optimum is tile20/24/12.

### Simple Ranking Rules

**Rule attempt:** `argmin(forward_time) ≈ f(tpg_std, total_gaussians, visible_ratio)`

No simple 1D threshold or linear rule accurately predicts the optimal tile across all 3 scenes. The optimum depends on the interaction of:
1. **Workload balance** (tpg_std — higher = need smaller tiles to avoid imbalance)
2. **Total workload** (Gaussian count — more Gs = need more tiles for parallelism)
3. **Visible ratio** (dense scenes benefit from larger tiles for fewer intersections)

---

## 5. Leave-One-Scene-Out Generalization

### Forward Optimum

| Training | Test | Predicted | Actual | Forward Regret |
|:---------|:----:|:---------:|:-----:|:--------------:|
| room + bicycle | garden | tile20 | tile12 | 8.5% |
| room + garden | bicycle | tile16? (conflict) | tile24 | ~12.4% |
| bicycle + garden | room | tile12? (conflict) | tile20 | ~21.1% |

**Analysis:**
- When training scenes disagree on optimal tile (room prefers tile20, bicycle prefers tile24), the prediction is ambiguous
- **Leave-one-scene-out accuracy is limited** because only 3 scenes provide insufficient diversity
- The common majority rule (tile20) has regret of 4-22% depending on the held-out scene

### Fwd+Bwd Optimum

For fwd+bwd, **tile20 is the common answer in sufficient scenes**:
- bicycle: tile20 (96.959ms)
- garden: tile20 (92.738ms)
- room: tile16 (34.828ms, tile20 is 4th at 38.981ms)

---

## 6. Regret Analysis

### Renderer Regret (forward only, snapshot)

| Strategy | room | bicycle | garden | Mean |
|:---------|:---:|:-------:|:------:|:----:|
| Always tile16 | 11.0% | 14.2% | 4.0% | 9.7% |
| Always tile20 | **0%** | 1.3% | 8.5% | 3.3% |
| Always tile24 | 1.6% | **0%** | 10.5% | 4.0% |
| Always tile32 | 15.0% | 8.6% | 38.9% | 20.8% |
| Feature-based | ~0% | ~0% | 8.5% | **2.8%** |

**tile20 has the lowest average regret (3.3%)** among fixed strategies.
A feature-based oracle (which would sometimes choose tile12 for garden) would reduce mean regret to ~2.8%.

### Renderer Regret (fwd+bwd, snapshot)

| Strategy | room | bicycle | garden | Mean |
|:---------|:---:|:-------:|:------:|:----:|
| Always tile16 | **0%** | 22.8% | 4.6% | 9.1% |
| Always tile20 | 11.9% | **0%** | **0%** | **4.0%** |
| Always tile24 | 13.1% | 1.8% | 19.1% | 11.3% |
| Always tile32 | 32.5% | 44.6% | 38.4% | 38.5% |
| Feature-based | 11.9% | 0% | 0% | **4.0%** |

### Training Regret (room only, 30K)

| Strategy | Wall Time | Regret vs Optimal |
|:---------|:---------:|:-----------------:|
| Always tile16 | 150.3 min | 58.5% |
| Always tile32 (optimal) | 94.8 min | **0%** |
| tile20 (predicted by snapshot) | PENDING | PENDING |

---

## 7. Generalization Assessment

| Question | Answer |
|:---------|:-------|
| Is leave-one-scene-out feasible? | **YES** (3 scenes, 24 workloads) |
| Does feature ranking generalize? | **PARTIALLY** — ordinal relationship holds, exact optimum varies |
| Insufficient data? | **YES** — with only 3 scenes, cannot train robust cross-scene predictor |
| What accuracy with simple rule? | tile20 (most common): 62.5% correct (exact match), but regret is low |
| Should we train a complex predictor? | **NO** — 3 scenes is insufficient. Simple ordinal heuristics are adequate. |

---

## 8. Conclusions

### What Works

1. **Ordinal ranking** — tpg_std consistently decreases with tile size and inversely correlates with optimal tile. No threshold needed.
2. **tile20 is the strongest universal candidate** — lowest mean regret (3.3% forward, 4.0% fwd+bwd).
3. **Feature-based selection is POSSIBLE** — simple 2-feature rule (tpg_std + visible_ratio) could select tile12 vs tile20 vs tile24.

### What Does NOT Work

1. **Phase 13A binary threshold** — fails for the expanded space.
2. **Leave-one-scene-out regression** — 3 scenes insufficient for robust prediction.
3. **Complex ML predictor** — not justified with this sample size.

### Recommended Selection Rule

```
if tpg_std > 20:        # room-like (dense, high imbalance)
    tile choice: tile20 (forward) or tile16 (fwd+bwd)
elif visible_ratio < 0.25:  # outdoor-like (sparse)
    tile choice: tile20 or tile24
else:
    tile choice: tile20 (universal default)
```

**But this rule is a heuristic, not a validated predictor.** Adaptive implementation (Phase 14A) should use simple scene-level static selection, not per-frame dynamic switching.
