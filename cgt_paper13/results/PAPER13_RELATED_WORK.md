# CGT-PAPER13 — Related Work / Literature Comparison

> STATUS: `NOT_APPLES-TO-APPLES` for all published numbers. No published cross-hardware result is
> merged into the authoritative CGT speedup table. A fresh prior-art audit is required before any
> novelty claim; this table is a scaffold for that audit, not the audit itself.

## 1. Mechanism-relevant renderer systems (for the fresh prior-art audit)
The following are candidate directly-comparable renderer systems. For each, the published number
must be recorded with GPU / dataset / metric / baseline / whether it is reproducible on our A100
cohort before any apples-to-apples claim.

| System | Reported mechanism | GPU | Dataset | Metric | Baseline | Comparable? |
|--------|--------------------|-----|---------|--------|----------|-------------|
| Faster-GS | rasterization speedups | NOT_MEASURED | mipnerf360 | NOT_MEASURED | gsplat | AUDIT_PENDING |
| TensorGS | tensor-compressed Gaussian rendering | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | AUDIT_PENDING |
| TileGS | tile-based Gaussian rendering | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | AUDIT_PENDING |
| LocoGS | localized Gaussian rendering | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | AUDIT_PENDING |

## 2. Authoritative CGT numbers (this lane, MX A100, our cohort)
These are the ONLY numbers that go in the authoritative CGT speedup table.

| Metric | Value | Scope |
|--------|-------|-------|
| MIXED_FORWARD S_mixed geomean | 1.2606x | 13-scene, 50-tile samples, kernel-only (build excluded) |
| MIXED_FORWARD S_mixed range | 0.97x – 3.03x | per scene |
| E2E novel-view S | ~0.000x (all 13) | full-frame, build included |
| E2E build geomean | 40.3 min/frame | full-frame per-camera fit+certify |
| E2E render ratio geomean | 6.70x slower than gsplat | full-frame tile-based vs optimized rasterizer |
| Kernel correctness max_abs | 1.85e-6 | vs 2e-5 gate, all 13 pass, R=4 |

## 3. Why the comparison is NOT apples-to-apples
- Published renderer speedups typically amortize representation construction OFFLINE and report
  STEADY-STATE rendering. CGT-PAPER13 E2E charges the per-camera fit+certificate build to every
  frame (required for novel-view synthesis), which is the honest accounting the paper demands.
- The CGT MIXED_FORWARD number (1.26x) is a kernel-level, build-excluded metric on small samples;
  it is NOT an end-to-end rendering claim and must not be headline-numbered as one.

## 4. Required before any novelty claim
- Fresh prior-art audit (dates, venues) for low-rank / certifiable / tile-compressed Gaussian rendering.
- Same-hardware reproduction of at least one external renderer (objective section 29, optional).