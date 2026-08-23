# Phase 8 — Evidence Chain Status

**Date:** 2026-09-15  
**Project:** 3DGS Renderer Benchmark — Complete Evidence Chain Validation  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8.5 GB VRAM, Compute 12.0)  
**Experiment Server (EPIC-05):** BLOCKED (connection timeout)  
**Phase 8 Directive:** No new optimizations. Complete evidence chain for all existing modules.

---

## 1. Evidence Chain Pipeline

```
Optimization Module
    ↓
Forward Correctness    →  SUPPORTED for M1 (tile16/tile32)
    ↓
Gradient Correctness   →  SUPPORTED for M1, M2, M3, M4, M5
    ↓
GT Quality Preserved   →  SUPPORTED for M1 (3 scenes, pixel-identical)
    ↓
Full Training Correct  →  SUPPORTED for M1 room only; M2-M5 BLOCKED
    ↓
Composability          →  BLOCKED (need >=2 fully eligible modules beyond baseline)
    ↓
End-to-End Benefit     →  BLOCKED (need composability first)
```

## 2. Module Eligibility Status

| Module | Forward | Gradient | GT Quality | Training | Eligible? |
|:-------|:-------:|:--------:|:----------:|:--------:|:---------:|
| M0 baseline | ✅ | ✅ | ✅ | ✅ | ✅ ELIGIBLE |
| M1 tile16 | ✅ | ✅ | ✅ | ✅ room | ✅ ELIGIBLE |
| M1 tile32 | ✅ | ✅ | ✅ | ✅ room | ⏳ PENDING_CROSS_SCENE |
| M2 packed/dense | ❌ INCONCLUSIVE | ✅ | ❌ NOT_TESTED | ❌ NOT_TESTED | ❌ |
| M3 SH degree | ❌ PARTIAL | ✅ | ❌ NOT_TESTED | ❌ NOT_TESTED | ❌ |
| M4 radius_clip | ❌ INCONCLUSIVE | ✅ | ❌ NOT_TESTED | ❌ NOT_TESTED | ❌ |
| M5 eps2d | ❌ INCONCLUSIVE | ✅ | ❌ NOT_TESTED | ❌ NOT_TESTED | ❌ |

**Key change from Phase 7C:** M1_tile32 moved to `PENDING_CROSS_SCENE` (not automatically ELIGIBLE). A single-scene result (room) cannot be treated as universal.

## 3. Phase 8 Priority Tracker

| Track | Priority | Description | Status |
|:------|:--------:|:------------|:-------|
| A | P0 | M1 cross-scene validation (bicycle 30K, garden 30K) | **BLOCKED** 🔴 |
| B | P1 | M1 real-scene snapshot microbenchmark | **BLOCKED** 🔴 |
| C | P0 | M2 packed/dense training gate | **BLOCKED** 🔴 |
| D | P0 | M3 SH degree training gate | **BLOCKED** 🔴 |
| E | P0 | M4 radius_clip training gate | **BLOCKED** 🔴 |
| F | P0 | M5 eps2d training gate | **BLOCKED** 🔴 |
| G | P2 | Composability | **BLOCKED** 🔴 |
| H | P3 | New optimization invention | **FORBIDDEN** ⛔ |

## 4. Server Blocker

**EPIC-05 (8.130.30.251:1024):** TCP connection timeout (5s).  
SSH key exists at `C:\Users\36570\.ssh\epic-node-new`.  
Port: 1024, User: root, Auth: key-based.

Last known good: approximately 2026-08-27.  
Current date: 2026-09-15.

**Blocker duration:** ~19 days.

## 5. What Can Be Done Offline

| Task | Description | Status |
|:-----|:------------|:-------|
| Evidence matrix update | Update `results/epic05/` JSON files with Phase 8 status | ✅ COMPLETED |
| Phase 8 reports | Create evidence chain, cross-scene, microbench, training gate reports | ✅ COMPLETED |
| Real-scene microbenchmark | Run on room checkpoints (12 checkpoints, forward timing) | ✅ COMPLETED |
| - Key finding | tile32 **3.4×–13.5× faster** forward on real checkpoints (vs 1.01× synthetic) | ✅ COMPLETED |
| - Mechanism insight | tiles_per_gauss ≈ total tiles for real scenes (100% tile occupancy) | ✅ CONFIRMED |
| Ready-to-run scripts | Verify all experiment scripts for server recovery | ✅ CHECKED |
| SSH test | Periodically test server connectivity | 🔴 STILL BLOCKED |

## 6. Critical Next Experiment (When Server Recovers)

The single highest-value experiment after server recovery is:

> **bicycle 30K full training with tile16 and tile32**

Bicycle has ~6.1M initial Gaussians (vs room's ~1.6M) — a fundamentally different workload regime. If tile32 maintains its speedup on bicycle, it strengthens the generalization claim significantly. If tile32 loses its advantage on bicycle, it reveals important scene-dependent behavior.

**Priority order after server recovery:**

1. `bicycle` tile16 30K (6-8 hours of GPU)
2. `bicycle` tile32 30K (parallelizable if GPU permits)
3. `garden` tile16 30K
4. `garden` tile32 30K
5. Controlled 3000-step rerun (single session, tile16+tile32)
6. M2 short sanity (500 steps)
7. M3 short sanity (500 steps)
8. M4 short sanity (500 steps)
9. M5 short sanity (500 steps)
