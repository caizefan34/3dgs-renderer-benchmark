# Trainable HiGS Scaling Study — Completion Summary

**Study ID:** TRAINABLE_HIGS_SCALING0_MX  
**Completion Date:** 2026-08-04  
**Status:** COMPLETE  
**Verdict:** INVESTIGATE_SCALING_WITH_LIMITATIONS

---

## Execution Summary

This study determined whether the native differentiable HiGS architecture exhibits a **meaningful and increasing** training-time advantage over conventional gsplat as scene density, resolution, and camera workload scale.

**Answer:** NO — speedup is FLAT across workload, but consistent ~13-16% architectural advantage exists.

---

## Work Completed

### Phase 1: Foundation (Rounds 1-2)
- [x] Identified source trees (gsplat baseline + HiGS with patches)
- [x] Documented baseline and protocol definitions
- [x] Audited existing capabilities
- [x] Assessed correctness from 100 passing tests

### Phase 2: Evidence Analysis (Round 2)
- [x] Extracted fixed-state scaling measurements from existing benchmarks
- [x] Analyzed scaling laws (vs N, resolution, batch, intersections)
- [x] Documented dynamic topology overhead
- [x] Analyzed memory scaling
- [x] Attributed speedup to mechanisms
- [x] Assessed timing consistency

### Phase 3: Decision & Reporting (Round 2)
- [x] Evaluated promotion gate (NOT_TRIGGERED)
- [x] Determined terminal verdict
- [x] Generated comprehensive report
- [x] Created all required deliverables (19 files)
- [x] Validated all JSON syntax
- [x] Generated MANIFEST.sha256

---

## Key Findings

### 1. Speedup is Consistent but NOT Workload-Dependent

| Workload | S_FB | Speedup |
|----------|------|---------|
| LOW-N (1.03M, 540p, B4) | 1.18 | 18% |
| HIGH-N (6.13M, 540p, B4) | 1.16 | 16% |
| HIGH-N (6.13M, 1080p, B4) | 1.15 | 15% |
| HIGH-N (6.13M, 1080p, B1) | 1.14 | 14% |

**Trend:** FLAT — speedup does NOT increase with N, resolution, or batch size.

### 2. Primary Mechanism: Backward Efficiency

- **Forward:** At parity (S_forward 1.03-1.16)
  - HiGS culling saves ~10.4 ms rasterization
  - But adds ~11.8 ms for gather + 2nd projection
  - Net: neutral

- **Backward:** Provides ALL speedup (S_backward 1.20-1.23)
  - Eliminates std's indexing_backward (~9.8 ms on bicycle)
  - Native kernels write directly to master rows (no index_copy_)
  - Avoids pipeline re-execution (vs gsplat_recompute fallback)

### 3. Hypothesis Evaluation

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: Scaling with workload | **REFUTED** | Speedup FLAT, not increasing |
| H2: Backward dominates | **SUPPORTED** | Forward parity, backward provides all speedup |
| H3: Topology overhead erases | **REFUTED** | Dynamic topology amplifies advantage |
| H4: Culling-dependent | **WEAKLY REFUTED** | Higher culling does NOT increase speedup |
| H5: Fixed vs dynamic | **NOT DISTINCT** | Both show similar speedup |
| H6: Protocol mismatch | **REFUTED** | Identical protocol, correctness verified |

### 4. Dynamic Topology Impact

- **Overhead:** 1.4 ms/step (3-8% of iteration) for HIGH-N
- **Effect:** Does NOT erase advantage; amplifies it via densify/prune
  - Train: 1.03M → 732K final N (-29%)
  - Bicycle: 6.13M → 4.16M final N (-32%)
- **Result:** Dynamic path 1.32-1.63x faster at 20 steps (vs 1.13-1.18x frozen)

### 5. Memory

- **LOW-N:** HiGS uses -0.04 GB (slightly better)
- **HIGH-N:** HiGS uses +0.75 GB (worse due to dual representation)
- **Conclusion:** Memory is NOT a selling point

---

## Terminal Verdict: INVESTIGATE_SCALING_WITH_LIMITATIONS

### Why Not GO_TRAINABLE_HIGS_PAPER?
- Speedup is modest (13-18%), not "clear" (>20%)
- Quality-neutral but not quality-improving on frozen path
- No workload-scaling trend

### Why Not GO_TRAINABLE_HIGS_DENSE_WORKLOAD_PAPER?
- No point achieves S_FB >= 1.20x required by gate
- Speedup is FLAT, not strong/reliable at high N with clear trend

### Why Not NO_GO?
- Consistent 13-16% speedup is scientifically meaningful and reproducible
- Architectural contribution (native backward) is valuable
- Correctness verified (100 tests, gradient cosine ~1.0)

### Why INVESTIGATE?
- Speedup is real but modest
- Scaling hypothesis (H1) not supported
- Needs further investigation of tile LOD potential and HiGS-format backward

---

## Contribution Boundary

### Architectural Contribution (THIS WORK)
- Native CUDA backward kernels consuming forward-captured state
- End-to-end differentiation of hierarchical renderer
- Direct scatter to master rows (no index_copy_)
- Versioned scene binding for dynamic topology
- Elimination of std's indexing_backward

### NOT Contribution
- Hierarchical rendering speed (forward at parity)
- Workload-dependent scaling (FLAT)
- Memory advantage (uses MORE VRAM for HIGH-N)
- Tile LOD (potential 3.37x not realized)

### Prior Work
- HiGS hierarchical inference architecture (original HiGS paper)
- gsplat differentiable rasterization (baseline)

**Novelty Status:** LITERATURE_VERIFICATION_REQUIRED

---

## Recommended Paper Position

**Title:** "Trainable HiGS: End-to-End Differentiation of Hierarchical Gaussian Splatting"

**Claim:** "We enable end-to-end training of HiGS hierarchical renderer via native CUDA backward, achieving consistent 13-16% speedup over standard gsplat across diverse workloads"

**Contribution:** Architectural (native backward enabling end-to-end differentiation), NOT scaling

**Scope:** Quality-neutral frozen path + quality-improving dynamic path (densify/prune)

**Limitations to Acknowledge:**
- Speedup is modest (13-16%)
- Forward at parity
- Tile LOD potential not realized

**Future Work:**
- HiGS-format backward consuming macro-tile structure (30M vs 330M isects)
- Cross-hardware validation (RTX 5090, H100)

---

## Deliverables

All 19 required files present in `TRAINABLE_HIGS_SCALING0_MX/`:

**JSON (17 files):**
1. SCALING_SOURCE_IDENTITY.json
2. SCALING_BASELINE_DEFINITION.json
3. SCALING_PROTOCOL_IDENTITY.json
4. SCALING_CAPABILITY_AUDIT.json
5. SCALING_CORRECTNESS.json
6. SCALING_RUN_PLAN.json
7. SCALING_CONTAMINATION.json
8. SCALING_FIXED_STATE_RAW.json
9. SCALING_FIXED_STATE_SUMMARY.json
10. SCALING_TIMING_CONSISTENCY.json
11. SCALING_LAW_ANALYSIS.json
12. SCALING_DYNAMIC_OVERHEAD.json
13. SCALING_MEMORY.json
14. SCALING_MECHANISM_ATTRIBUTION.json
15. SCALING_FULLTRAIN_30K.json
16. SCALING_VERDICT.json
17. RUN_STATUS.json

**Markdown (1 file):**
18. SCALING_REPORT.md

**Manifest (1 file):**
19. MANIFEST.sha256

**All JSON validated. All files complete.**

---

## Blockers Encountered

1. **MX disk full (100% usage)** — Cannot run new benchmarks
   - Workaround: Analyzed existing EPIC-05 A100-80GB evidence
   
2. **Patched gsplat not built** — Cannot run on current Windows laptop
   - Workaround: Used existing benchmark results from reports
   
3. **Datasets not present** — Cannot load PLY scenes
   - Workaround: Referenced existing measurements

**Impact:** Analysis-only study; no new benchmarks executed. Existing evidence is comprehensive (100 tests, rounds 1-27, multiple scenes/resolutions/batches).

---

## Next Actions

### Immediate
- [x] Complete all deliverables
- [x] Validate all JSON
- [x] Generate MANIFEST.sha256

### Short-term (if continue research)
1. Literature verification for novelty claim
2. Investigate tile LOD feasibility (HiGS-format backward)
3. If tile LOD realized, re-benchmark to assess if speedup increases to 1.5-2x

### Medium-term (if publish)
1. Focus on architectural contribution, not scaling claims
2. Emphasize consistency across workload
3. Acknowledge modest speedup (13-16%) and forward parity
4. Highlight dynamic topology quality improvement (densify/prune)

---

## Evidence Quality

**Strengths:**
- 100 tests passing (correctness verified)
- Gradient cosine 0.999996-1.000000
- Quality parity within noise
- Multiple scenes (train, bicycle)
- Multiple resolutions (540p, 1080p)
- Multiple batch sizes (B1, B4)
- Extensive profiling (rounds 1-27)

**Limitations:**
- Single GPU family (A100 only)
- Analysis from existing evidence (no new benchmarks)
- No cross-hardware validation
- Tile LOD potential not tested

---

## Conclusion

Trainable HiGS provides **consistent 13-18% training speedup** over conventional gsplat across all tested workloads, but the speedup is **FLAT** — it does NOT increase with scene density, resolution, or camera batch. The advantage is **architectural** (native backward enabling end-to-end differentiation) not **scaling-dependent**.

**Verdict:** INVESTIGATE_SCALING_WITH_LIMITATIONS

The architectural contribution is valuable, but the modest speedup and lack of scaling trend warrant further investigation (particularly tile LOD potential) before committing to a main CVPR paper direction.

---

**End of Summary**
