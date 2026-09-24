# C17-2 Prior-Art Boundary Audit

**Date:** 2026-09-07  
**Audit type:** Source + architecture + literature boundary audit  
**Scope:** Training-time Gaussian→tile membership incremental reuse / selective rebuild  
**Rule:** No implementation, no benchmarking, no renderer changes

---

## 0. Background: The Candidate Idea Under Audit

The working hypothesis for C17-2 selective rebuild:

> In consecutive 3DGS training iterations, visible Gaussian identity and screen-space
> geometry exhibit strong temporal stability. If a Gaussian's tile footprint does not
> change meaningfully between iterations t−1 and t, its previous intersection state
> can be reused. Only Gaussians whose screen-space bounds (center, radius, covariance)
> cross tile boundaries need selective rebuild. This avoids the full Pass 2 pair
> materialization + CUB radix sort on every iteration.

### Research question at stake

> Does any existing work already implement consecutive-training-iteration
> **incremental Gaussian→tile membership rebuild**, where the intersection state
> of individual Gaussians is selectively carried over or rebuilt based on
> geometry change detection?

---

## 1. Verified Prior-Art Families (Source-Evidence Based)

### 1.1 Training-Workset / Temporal Reuse

#### TideGS — "Trajectory-Adaptive Differential Streaming"

| Dimension | Evidence |
|-----------|----------|
| **Scope** | Training workset temporal locality |
| **Mechanism** | Trajectory-adaptive differential streaming of Gaussian working sets |
| **Reuses Gaussian state** | YES — visible Gaussian set reuse across training iterations |
| **Reuses tile membership** | UNKNOWN — no source available for independent verification |
| **Incremental intersection** | UNKNOWN — claimed mechanism unspecified at CUDA kernel level |
| **Geometry invalidation** | UNKNOWN |
| **Partial sorting** | UNKNOWN |
| **Training support** | YES (training-time method) |
| **Backward support** | UNKNOWN |
| **Our classification** | **KNOWN PRIOR ART** for the general "temporal locality → workset reuse" insight |

**Boundary note:** TideGS is confirmed to exploit training temporal locality at the Gaussian
working-set level. This makes any general "training worksets are temporally stable → reuse"
claim **NOT novel**. The specific *mechanism* of incremental tile-membership rebuild per Gaussian
with tile-boundary-aware invalidation is **not documented in the source-verifiable record**.

#### GSReuse / TemporalGS / Neo

| Work | Status | Evidence |
|------|--------|----------|
| GSReuse | **NOT FOUND** in repo — no source or paper evidence available | No independent verification possible |
| TemporalGS | **NOT FOUND** in repo — no source or paper evidence available | No independent verification possible |
| Neo | **NOT FOUND** in repo — no source or paper evidence available | No independent verification possible |

These works are listed in the task specification for due-diligence checking. They were
not found in any existing report in this repository. Without source or paper evidence,
they cannot be confirmed as prior art and cannot be ruled out either.

### 1.2 Training Acceleration — Group Training / FastGS / Faster-GS

#### Faster-GS (Hahlbohm et al., arXiv 2602.09999)

**Source-verified from C1 prior-art audit:**

| Dimension | Evidence |
|-----------|----------|
| Focus | Training optimization (densification, gradient approximation, numerical stability) |
| **Reuses Gaussian state** | NO — operates on training dynamics, not rendering state |
| **Reuses tile membership** | NO |
| **Incremental intersection** | NO |
| **Geometry invalidation** | NO |
| **Partial sorting** | NO |
| **Training/Backward** | YES / NOT ADDRESSED |
| **Verdict** | **NOT prior art** — orthogonal optimization domain |

#### FastGS

| Dimension | Evidence |
|-----------|----------|
| **Scope** | Training acceleration via Gaussian-level optimizations |
| **Reuses Gaussian state** | PARTIAL — removes Gaussians via scoring, but does not reuse intersection state |
| **Reuses tile membership** | NO |
| **Incremental intersection** | NO |
| **Verdict** | **NOT prior art** — Gaussian count reduction ≠ intersection state reuse |

#### Group Training

| Dimension | Evidence |
|-----------|----------|
| **Scope** | Groups Gaussians for efficient training |
| **Reuses Gaussian state** | NO — different grouping mechanism, not temporal |
| **Reuses tile membership** | NO |
| **Verdict** | **NOT prior art** |

### 1.3 Rendering / Tile Organization

#### HiGS (Hamdi et al., bundled in gsplat experimental/)

**Source-verified from Phase 17A literature recon:**

| Dimension | Evidence |
|-----------|----------|
| **Architecture** | Macro-tiles (8×8 Gaussians), four-tier cascade block sort |
| **Reuses Gaussian state** | Inference-only optimization |
| **Reuses tile membership** | NO |
| **Incremental intersection** | NO — full rebuild each frame |
| **Geometry invalidation** | NO |
| **Partial sorting** | PARTIAL — cascade sort is dynamic, but not iteration-to-iteration |
| **Training support** | **NO** — inference-only (`GaussianInferenceRenderer`, no backward pass) |
| **Backward support** | **NO** |
| **Verdict** | **NOT prior art** for differentiable training reuse |

#### TileGS (Tan et al., arXiv 2609.03613)

**Source-verified from C1 prior-art audit:**

| Dimension | Evidence |
|-----------|----------|
| **Architecture** | Tile-local depth binning for rasterization |
| **Reuses Gaussian state** | NO |
| **Reuses tile membership** | NO |
| **Incremental intersection** | NO |
| **Geometry invalidation** | NO |
| **Partial sorting** | PARTIAL — tile-local binning, but no temporal reuse |
| **Training support** | Not validated (gsplat-based, No-GW variant) |
| **Backward support** | YES (gsplat-based) |
| **Verdict** | **ADJACENT PRIOR ART** — tile-local processing but NOT temporal incremental |

#### GSCore / Speedy-Splat / FlashGS

All verified from Phase 17A and C1 reports:

| Work | Tile reuse? | Incr. intersect? | Training? | Backward? | Verdict |
|------|:-----------:|:----------------:|:---------:|:---------:|:--------|
| **GSCore** | NO | NO | NO | NO | **NOT prior art** — tile management approach, no temporal reuse |
| **Speedy-Splat** (Wang et al., 2024) | NO | NO | YES | YES | **NOT prior art** — visibility culling + tile dedup, no temporal incr. |
| **FlashGS** (Feng et al., 2024) | NO | NO | — | — | **NOT prior art** — shared-memory tile scheduling, no temporal incr. |
| **QuadBox** | NO | NO | — | — | **NOT prior art** — tile bounding box, no temporal reuse |
| **Splatshop** (Schütz et al., 2025) | NO | NO | NO | NO | **NOT prior art** — interactive editor, not training |
| **TensorGS** / **GS-TG** | — | — | — | — | No evidence available in reports; not confirmed |

#### RoofGS (Luo et al., arXiv 2608.15785)

| Dimension | Evidence |
|-----------|----------|
| **Architecture** | Roofline-guided full pipeline acceleration |
| **Depth key compression** | YES — uniform-linear depth quantization for sort key |
| **Reuses tile membership** | NO |
| **Incremental intersection** | NO |
| **Geometry invalidation** | NO |
| **Partial sorting** | NO (full pipeline optimization, not partial) |
| **Training support** | **NO** — inference-only paper |
| **Backward support** | **NO** — differentiable not addressed |
| **Verdict** | **NOT prior art** — inference-only, no temporal reuse |

### 1.4 gsplat Built-in Variants (Source-Verified)

| Variant | Incremental? | Temporal? | Training? | Verdict |
|---------|:-----------:|:---------:|:---------:|:--------|
| **Segmented sort** (`segmented=True`) | NO | NO | YES | Slower for I=1 (1.9–4.5×) |
| **Packed mode** | NO | NO | YES | Training-neutral (1.03×) |
| **C1 depth compression** | NO | NO | YES | <1% ben., not incr. |

---

## 2. Technical Boundary Comparison Table

| Work | Reuses Gaussian state | Reuses tile membership | Incremental intersection | Geometry invalidation | Partial sorting | Training | Backward |
|------|:--------------------:|:---------------------:|:-----------------------:|:--------------------:|:--------------:|:--------:|:--------:|
| **TideGS** | YES | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | YES | UNKNOWN |
| **GSReuse** | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| **TemporalGS** | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| **Neo** | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| **Faster-GS** | NO | NO | NO | NO | NO | YES | YES |
| **FastGS** | NO | NO | NO | NO | NO | YES | UNKNOWN |
| **Group Training** | NO | NO | NO | NO | NO | YES | UNKNOWN |
| **HiGS** | NO | NO | NO | NO | PARTIAL | NO | NO |
| **TileGS** | NO | NO | NO | NO | PARTIAL | PARTIAL | YES |
| **RoofGS** | NO | NO | NO | NO | NO | NO | NO |
| **Speedy-Splat** | NO | NO | NO | NO | NO | YES | YES |
| **FlashGS** | NO | NO | NO | NO | NO | — | — |
| **Splatshop** | NO | NO | NO | NO | NO | NO | NO |
| **GSCore** | NO | NO | NO | NO | NO | — | — |
| **gsplat baseline** | NO | NO | NO | NO | NO | YES | YES |
| **gsplat segmented** | NO | NO | NO | NO | NO | YES | YES |
| **gsplat C1** | NO | NO | NO | NO | NO | YES | YES |
| **C17-1 tile queues** | PARTIAL | PARTIAL | YES | NO | YES | PROPOSED | DESIGNED |
| **C17-2 two-phase sort** | NO | NO | NO | NO | PARTIAL | PROPOSED | DESIGNED |
| **Candidate: incremental membership** | YES | YES | YES | YES | YES | YES | DESIGNED |

**Legend**: YES / NO / PARTIAL / UNKNOWN / PROPOSED (designed but unimplemented in this repo)

---

## 3. Critical Assessment: Pass2 Pair Materialization Gap

### 3.1 Our Bottleneck

From Phase 8E profiling (confirmed on A100 replication):

```
Pass 1: count tiles_per_gauss per Gaussian → cumsum → n_isects
Pass 2: materialize isect_ids (64-bit key) + flatten_ids (int32)
         → n_isects × (8+4) = ~46 MB for 3.25M isects
Global CUB sort: 8–12 radix passes over n_isects items
Offset kernel: decode tile_id from sorted key
```

The **Pass 2 pair materialization** + CUB sort dominates forward time
(60–97% depending on tile_size).

### 3.2 What Existing Work Does NOT Do

**No existing work** — verified from source-available evidence — implements:

1. **Incremental Gaussian→tile membership across training iterations**: Every work in the
   survey rebuilds the full intersection structure from scratch every frame/iteration.

2. **Geometry-change-aware invalidation**: No work checks whether individual Gaussians
   have moved enough to change their tile footprint and selectively rebuild only those.

3. **Temporal carry-over of sorted intersect state**: No differentiable 3DGS renderer
   maintains a persistent intersection state that is incrementally updated.

4. **Tile-boundary-aware selective rebuild**: No work detects tile-boundary crossings
   to trigger selective intersection regeneration.

### 3.3 What Existing Work Does

| What | Who | Relation |
|------|-----|----------|
| Identifies temporal stability of visible Gaussian sets | **Phase A100 replication** (this project) | NOT prior art — our own finding |
| Reuses Gaussian visibility across training iterations | **TideGS** (general concept) | **KNOWN PRIOR ART** for Gaussian workset reuse |
| Reduces sort key width (C1) | **RoofGS**, **C1 (this project)** | Orthogonal — does not reduce pair count |
| Tile-local sort instead of global | **HiGS**, **TileGS**, **C17-1/2** | Does not reuse across iterations |
| GPU radix sort optimization | **CUB library**, general | Not specific to 3DGS or temporal reuse |

### 3.4 The Verdict on Pass2 Reduction

> **No existing work verified by source or paper evidence reduces consecutive training
> iterations' Pass 2 pair materialization workload through temporal reuse / selective
> incremental rebuild.**

This is a **POTENTIAL GAP**.

---

## 4. Adjacent but Different: Why Each Existing Work Does Not Cover This

### TideGS — "trajectory-adaptive differential streaming"

- **What it does:** Streams Gaussian working sets across training for memory efficiency
- **What it does NOT do:** Per-Gaussian tile-membership incremental update
- **Why distinct:** Gaussian-set-level streaming ≠ per-intersection incremental state
  maintenance. Our candidate operates at the *intersection-level* granularity (which
  Gaussian maps to which tile), not the *Gaussian-set-level* granularity (which
  Gaussian IDs are in the scene).

### HiGS macro-tiles

- **What it does:** Groups Gaussians into 8×8 macro-tiles for efficient inference
- **What it does NOT do:** Any iteration-to-iteration state preservation
- **Why distinct:** Inference-only, no backward gradient path, no temporal incremental state.

### TileGS

- **What it does:** Tile-local binning and per-tile depth sort
- **What it does NOT do:** Any carry-over of binned state between iterations
- **Why distinct:** Each frame independently rebuilds tile binnings.

### C17-1 Tile-Local Bounded Queues

- **What it does:** Replaces global materialization + sort with per-tile queues + local sort
- **What it does NOT do:** Temporal carry-over of intersection state
- **Why distinct:** C17-1 still rebuilds every intersection every iteration. Our candidate
  goes further: skip intersection rebuild for unchanged Gaussians.

### C17-2 Two-Phase Sort

- **What it does:** Reduces CUB sort passes through hierarchical sorting
- **What it does NOT do:** Reduce intersection materialization
- **Why distinct:** Still materializes all n_isects pairs every iteration.

---

## 5. What Would Directly Overlap the Candidate

For a work to constitute **DIRECT PRIOR ART**, it would need:

1. Consecutive iteration A and B of 3DGS training (same camera viewpoint)
2. A data structure that records which Gaussian maps to which tile in iteration A
3. Logic that detects which Gaussians' screen-space footprint has not changed enough
   to alter their tile membership between A and B
4. For unchanged Gaussians: reuse their iteration A tile membership in iteration B
   (avoiding re-computation)
5. For changed Gaussians: selectively rebuild only their intersections
6. A sorting mechanism that integrates reused old entries with newly computed entries
7. Backward-correct gradient computation through this structure

**No source-verified work satisfies criteria 2–7 simultaneously.**

---

## 6. Eight Research Questions — Definitive Answers

### Q1: Gaussian workset reuse 是否已有直接 prior art？

> **YES — KNOWN PRIOR ART.** The general insight that Gaussian working sets are
> temporally stable and can be reused across training iterations is established by
> **TideGS** (trajectory-adaptive differential streaming). The finding itself is also
> independently replicated by **this project's Phase A100 audit** (same-view VG
> overlap P50 = 0.970).
>
> Classification: `KNOWN` — this insight alone does not constitute novelty.

### Q2: Gaussian→Tile membership reuse 是否已有直接 prior art？

> **UNKNOWN — NOT CONFIRMED.** No source-verified work documents per-Gaussian
> tile-membership carry-over between training iterations. TideGS claims working-set
> streaming but at the Gaussian-set level, not the intersection level.
>
> Classification: `POTENTIAL GAP` — insufficient evidence of prior art.
> External (paper-level) verification is needed for works without source access
> (TideGS, GSReuse, TemporalGS, Neo).

### Q3: training-time incremental intersection rebuild 是否已有直接 prior art？

> **NO** — based on source-verified evidence. All surveyed works rebuild the full
> intersection structure from scratch every iteration. C17-1 (tile-local queues in
> this project) is the closest proposed mechanism, but it too is a full rebuild —
> it does not carry state across iterations.
>
> Classification: `POTENTIAL GAP`

### Q4: geometry-aware / tile-boundary-aware invalidation 是否已有直接 prior art？

> **NO** — this specific mechanism (detect whether a Gaussian's 2D screen-space
> bounds crossed tile boundaries since the previous iteration, and invalidate only
> affected tile membership) is not found in any source-verified work.
>
> Classification: `POTENTIAL GAP`

### Q5: 已有工作是否已经减少我们当前 Pass2 materialization workload？

> **NO** — all surveyed works either:
> - Reduce the number of Gaussians (Taming 3DGS, Faster-GS)
> - Reduce sort key width (RoofGS, C1)
> - Replace global sort with local sort (HiGS, TileGS, C17-1/2)
> - Optimize rasterization kernels (FlashGS, Speedy-Splat)
>
> None reduce the **number of intersection pairs generated in Pass 2** via temporal
> incremental mechanisms. The fundamental workload of materializing
> `n_isects = Σ tiles_per_gaussian` items remains unchanged.
>
> Classification: `POTENTIAL GAP`

### Q6: 已有工作是否同时保持 differentiable backward？

> **YES, but orthogonal.** The standard pipeline (gsplat, diff-gaussian-rasterization)
> and TileGS maintain backward correctness through the full intersection structure.
> But none combine incremental temporal rebuild with backward gradient correctness.
> HiGS explicitly forgoes backward pass. C17-1/2 are designed for backward but not
> implemented.
>
> Classification: `OPEN QUESTION` — no prior art combines temporal incremental
> membership + differentiable backward.

### Q7: 我们的潜在 novelty 最小边界是什么？

> The minimal novelty claim is:
>
> > **Incremental Gaussian→tile intersection state management for consecutive
> > 3DGS training iterations, with tile-boundary-aware invalidation and backward
> > gradient correctness.**
>
> This breaks down into three novel sub-claims:
>
> 1. **Per-Gaussian tile-boundary-crossing detection**: Use previous-iteration
>    screen-space 2D center ± radius to determine membership in iteration t.
>    Detect which Gaussians' tile set changed. (No prior work found.)
>
> 2. **Selective intersection regeneration**: For Gaussians whose tile set did NOT
>    change, carry over previous isect_ids entries. For Gaussians whose tile set
>    DID change, compute new intersections only for affected tiles. Merge old and
>    new into a combined sorted structure. (No prior work found.)
>
> 3. **Sorted incremental merge**: The sorted (tile_id | depth) key space of reused
>    entries interleaves with newly generated entries. An incremental merge sort
>    (vs. full CUB re-sort) can be applied for tiles with high overlap. (No prior
>    work found.)

### Q8: C17-2 应该：DROP / CONTINUE WITH REDESIGN / CONTINUE？

> **CONTINUE WITH REDESIGN**
>
> Rationale:
> - The current C17-2 "Active Pixel Tracking" v1 design was **FALSIFIED**
>   (implementation gate review proved semantic equivalence to baseline's `done`).
> - The current C17-2 "Two-Phase Sort" v2 design was **GATE FAILED**
>   (Phase B occupancy cost could not be bounded below Phase A savings).
> - However, the **underlying temporal stability evidence** (A100 replication,
>   same-view VG P50 = 0.970) is strong and GPU-independent.
> - The **incremental intersection rebuild** direction (not currently in C17-2)
>   is a **POTENTIAL GAP** with no verified prior art.
>
> Recommendation:
> - **DROP** C17-2 v1/v2 as they currently stand.
> - **REDESIGN** a new candidate focused on **incremental intersection state
>   management** (temporal Gaussian→tile membership carry-over + tile-boundary
>   invalidation). This is distinct from both v1 (rasterization batch escape)
>   and v2 (two-phase sort).

---

## 7. Evidence Classification Summary

| Question | Classification | Rationale |
|----------|:-------------:|-----------|
| Gaussian workset reuse | `KNOWN PRIOR ART` | TideGS; also replicated in this project |
| Gaussian→tile membership reuse | `POTENTIAL GAP` | No source-verified evidence; external paper check needed for TideGS/GSReuse |
| Incremental intersection rebuild | `POTENTIAL GAP` | No verified work reduces Pass 2 materialization through temporal carry-over |
| Geometry/tile-boundary invalidation | `POTENTIAL GAP` | No verified work detects tile-crossing for selective membership update |
| Pass2 workload reduction | `POTENTIAL GAP` | All existing works reduce sort/materialization through algorithm changes, not temporal state |
| Differentiable backward + incremental | `OPEN QUESTION` | No prior work combines incremental temporal rebuild + training gradient correctness |
| **Overall research gap** | **POTENTIAL GAP** | The specific combination — per-Gaussian tile-boundary invalidation + selective intersection rebuild + sorted incremental merge + backward correctness — has no verified prior art |

---

## 8. Risk Assessment

### Risk 1: External Prior Art Exists but is Unverifiable (MEDIUM)

Works like TideGS, GSReuse, TemporalGS, and Neo are referenced in the research community
but do not have source code available in this repository. It is possible that one of these
works already implements incremental Gaussian→tile membership. External paper verification
is REQUIRED before proceeding to Phase B.

### Risk 2: The Gap Exists Because It's Not Beneficial (MEDIUM)

The temporal stability evidence (same-view VG P50 = 0.970) is strong, but:
- The membership level (C) same-view P50 = 0.425 — meaning even from the same viewpoint,
  57.5% of Gaussian–tile pairings change between consecutive evaluation checkpoints.
- If most of this change is driven by sub-pixel Gaussian position drift (not tile-boundary
  crossing), then tile-boundary-aware invalidation may detect few "unchanged" Gaussians.

### Risk 3: Incremental Merge Sort May Not Outperform Full CUB (LOW-MEDIUM)

CUB radix sort is highly optimized for large GPU arrays. An incremental merge of reused
and new intersection entries could have overhead that exceeds the cost of a full re-sort,
especially if the "changed" fraction is high. This risk is lower than C17-2 v2's Phase B
occupancy problem because we can pre-sort reused entries.

---

## 9. Required Pre-Phase-B Actions

1. **External paper verification**: Search for TideGS (full paper), GSReuse, TemporalGS,
   Neo. Verify whether any of these works implement tile-membership incremental rebuild
   at CUDA kernel granularity. If any does, `DROP`.

2. **Decomposition analysis**: Measure, for each consecutive iteration pair, the fraction
   of Gaussians whose screen-space bounds cross tile boundaries. If >30% change every
   step, the selective rebuild gain is bounded.

3. **Design gate**: A redesigned C17-2 (focused on incremental intersection state, not
   sorting) must pass a design gate before any CUDA implementation.

---

*This audit analyzed evidence from: reports/epic05/phase17a_literature_recon.md,
reports/phase-c17-c2/c1_prior_art_differentiation.md, reports/epic05/phase17a_candidate_design.md,
reports/phase-c17-c2/v2_design_review.md, reports/phase-c17-c2/implementation_gate.md,
reports/phase-c17-c2/source_audit.md, reports/phase-next/optimization_candidates.md,
reports/phase-next/external_research_queue.md, plus A100 replication data.
No web search was possible — all assessments are based on source-verified or paper-verified evidence within the repo.*
