# Phase 7C — Track A: Cross-Scene Full-Training Validation

**Date:** 2026-09-02  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8.5 GB VRAM, Compute 12.0)  
**Experiment Server (EPIC-05):** UNREACHABLE (connection timed out)

---

## 1. Server Status

**EPIC-05 (8.130.30.251:1024): BLOCKED — Connection timed out**

The experiment server configured for this project is unreachable. All experiments requiring GPU compute cannot be executed at this time.

**Impact:**
- A1: Controlled 3000-step rerun → BLOCKED
- A2: Bicycle 30K → BLOCKED
- A3: Garden 30K → BLOCKED
- M2-M5 training gate experiments → BLOCKED

**This is a hard blocker.** No experiments can be launched until the server becomes reachable.

---

## 2. Completed Work Summary (from Phase 7 / 7B)

### Room 30K (COMPLETED)

| Metric | tile16 | tile32 | Ratio |
|:-------|:------:|:------:|:-----:|
| Wall time | 150.3 min | 94.8 min | **tile32 1.58× faster** |
| Best PSNR | 29.27 dB | 29.39 dB | Δ=+0.12 dB |
| Final Gaussians | 1,193,480 | 1,146,273 | -4.0% |
| Peak VRAM | ~2.0 GB | ~1.8 GB | -10% |
| NaN/Inf | None | None | — |

### Room 3000-step Anomaly (RESOLVED — run variance)

The 3000-step "anomaly" (mid-run showing tile16 faster) has been diagnosed as run-to-run variance:
- Mid-run tile32: 446ms/iter (2.38× slower than full-run tile32 at same config)
- Full-run (v2) tile32: 188ms/iter
- Within the v2 full-run, tile32 is faster at EVERY iteration
- Full Phase 7B report: `reports/epic05/phase7b_training_mechanism_analysis.md`

### Remaining: Controlled single-session rerun

A controlled rerun within a single process/session would be the strongest evidence but is BLOCKED by server.

---

## 3. Cross-Scene Status

| Scene | Initial Gs | tile16 30K | tile32 30K | Comparison |
|:------|:----------:|:----------:|:----------:|:----------:|
| room | ~1.6M | ✅ COMPLETED | ✅ COMPLETED | **tile32 1.58× faster**, PSNR equal |
| bicycle | ~6.1M | ❌ BLOCKED | ❌ BLOCKED | █ Server unreachable |
| garden | ~5.8M | ❌ BLOCKED | ❌ BLOCKED | █ Server unreachable |

---

## 4. Cross-Scene Conclusion

**Cannot be determined.** The finding from room cannot be generalized without bicycle and garden data.

### Prior What-If Analysis (from Phase 6 data)

Phase 6 inference-only benchmarks showed tile16 is **1.03×–1.53× faster** than tile32 on RTX 5070 for single-pass rendering at these scene sizes. The full-training results on room contradict this (tile32 wins 1.58× during training).

Possible explanations for the contradiction:
1. **Training workload differs from inference**: Backward pass amplifies tile32's advantage (4× pixel re-use per Gaussian in backward)
2. **Gaussian count evolution**: Training changes the count and distribution in ways that favor tile32
3. **Phase 6 used synthetic/scene Gaussians without densification effects**

This further motivates the need for cross-scene training data.

---

## 5. Summary Table (with server BLOCKED)

| Scene | tile16 (30K) | tile32 (30K) | Winner | Speedup | PSNR Δ | Final Gs Δ |
|:------|:-----------:|:-----------:|:------:|:-------:|:------:|:----------:|
| room (v2) | 150.3 min | 94.8 min | tile32 | 1.58× | +0.12 dB | -4.0% |
| bicycle | BLOCKED | BLOCKED | — | — | — | — |
| garden | BLOCKED | BLOCKED | — | — | — | — |

**Current cross-scene conclusion for tile32 on RTX 5070:** INCONCLUSIVE (insufficient data). Only room scene validated. Cannot claim scene-independence without bicycle and garden data.
