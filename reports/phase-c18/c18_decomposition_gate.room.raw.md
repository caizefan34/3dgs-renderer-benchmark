# C18 — Quantitative Decomposition Gate
**Scene:** `room`  
**Steps:** 500  
**Tile size:** 16  
**GPU:** NVIDIA A100-PCIE-40GB  
**gsplat:** 1.5.3  
**Date:** 2026-09-07T16:38:35.497748+00:00  
**Fixed camera:** camera 0 (same viewpoint every step)
---
## Executive Decision: **GO**
Strong: changed_membership_ratio P50=6.80% < 10% AND reusable_intersection_ratio P50=98.54% > 70%

## 1. Consecutive-Iteration Gaussian Stability
### Intersection-level overlap (no-topology pairs, n=495)
| Metric | P10 | P25 | P50 | P75 | P90 | Mean |
|:-------|:--:|:--:|:--:|:--:|:--:|:----:|
| Reusable intersection ratio | 0.9753 | 0.9794 | **0.9854** | 0.9898 | 0.9924 | 0.9840 |
| Changed membership ratio | 0.0337 | 0.0483 | **0.0680** | 0.0955 | 0.1027 | 0.0685 |
| Stable membership ratio | 0.8973 | 0.9045 | 0.9320 | 0.9517 | 0.9663 | 0.9315 |
| Intersection overlap (Jaccard) | 0.9521 | 0.9599 | 0.9711 | 0.9798 | 0.9850 | 0.9688 |
| New intersection ratio | 0.0076 | 0.0102 | 0.0146 | 0.0206 | 0.0247 | 0.0160 |

### Temporal window
| Lag | P50 reusable | P50 overlap | n_pairs |
|:---:|:-----------:|:----------:|:-------:|
| t→t+1 | 0.9854 | 0.9711 | 495 |
| t→t+2 | 0.9720 | 0.9454 | 494 |
| t→t+4 | 0.9469 | 0.8987 | 484 |

## 2. Phase Analysis
- **early**: reusable P50=0.9842, changed P50=0.0955, n=99
- **middle**: reusable P50=0.9843, changed P50=0.0763, n=297
- **late**: reusable P50=0.9860, changed P50=0.0423, n=99

## 3. Topology-Change Impact
- **Events:** 4 topology changes across 500 steps
- **Fraction:** 0.80% of steps
- **Reuse during topo steps:** P50=0.0187 (vs no-topo P50=0.9854)
- **Note:** Per-Gaussian membership comparison is INVALID for topology-change pairs because pruning shifts Gaussian parameter indices. Only intersection-level pair comparison is reported for those.

## 4. Potential Pass2 Reduction
- **Theoretical max Pass2 reduction:** 98.54% of intersection records (P50)
- Range: P10=97.53% to P90=99.24%
- Baseline records per step: ~3,250,000
- Candidates for reuse: ~3,202,459 records (P50)
- Records requiring rebuild: ~47,540
**Caveat:** Theoretical upper bound. Actual speedup depends on invalidation scan overhead, incremental merge cost, and memory management.

## 5. Decision Criteria Check
| Criterion | Threshold | Measured | Met? |
|:----------|:---------:|:--------:|:----:|
| Strong GO: changed membership | <0.1 | 0.0680 | ✅ |
| Strong GO: reusable intersect | >0.7 | 0.9854 | ✅ |
| Conditional: changed membership | 10-30% | 0.0680 | ✅ |
| Conditional: reusable intersect | 40-70% | 0.9854 | ✅ |
| **STRONG GO SATISFIED** | both | ✅ | ✅ |

### Verdict
> **GO** — Strong: changed_membership_ratio P50=6.80% < 10% AND reusable_intersection_ratio P50=98.54% > 70%
> **Action:** Proceed to CUDA prototype + forward correctness phase.

---
**Note:** Per-Gaussian membership stability measured only on no-topology pairs. During densification/pruning, state must be fully rebuilt anyway.
