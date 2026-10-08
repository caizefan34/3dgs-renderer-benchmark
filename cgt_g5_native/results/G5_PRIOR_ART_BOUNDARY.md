# G5 Prior-Art Boundary (goal §27)

- **Stage**: G5-PREREG. **Date**: 2026-10-04.
- **Purpose**: Establish the boundary for any future novelty claim BEFORE one is made. This document is a *boundary*, not a clearance.
- **Tooling note**: No live literature/web-search tool is available in this environment. A fresh mechanism-level search is therefore **NOT_MEASURED** here and must be completed before any external novelty claim. Nothing below asserts that CGT-native-training is unprecedented.

## What CGT-G5 claims (narrow)
> Directly optimizing the existing per-tile rank-R CGT coefficients (the F0 student representation) from real images, with a native differentiable fused renderer and stop-gradient certificate semantics, eliminating the repeated teacher-to-student fitting step of route A.

## Mandatory audit targets (goal §27) — status
| # | Target | Relevance to G5 | Status |
|---|--------|-----------------|--------|
| 1 | Native tensorized Gaussian representations | direct overlap risk (trainable compressed primitives) | KNOWN_CLASS — needs fresh search |
| 2 | TensorGS / tensorized rasterization | tensor-primitive training precedent | KNOWN_CLASS — needs fresh search |
| 3 | Low-rank Gaussian scene representations | core to CGT representation | KNOWN_CLASS — needs fresh search |
| 4 | Learned compact Gaussian appearance models | appearance-parameter training overlap | KNOWN_CLASS — needs fresh search |
| 5 | Low-rank/tensor scene primitives | direct overlap | KNOWN_CLASS — needs fresh search |
| 6 | Direct optimization of compressed raster representations | closest to G5 claim | KNOWN_CLASS — needs fresh search |
| 7 | Mixture/tensor splatting methods | related | KNOWN_CLASS — needs fresh search |
| 8 | Adaptive low-rank Gaussian representations | rank-adaptation overlap (G5-E relevant) | KNOWN_CLASS — needs fresh search |
| 9 | Neural field compression with native training | differentiable-compression overlap | KNOWN_CLASS — needs fresh search |

## Distinctions G5 must be able to defend AFTER the search
- CGT is **per-tile**, **rank-R**, **coefficient-based**, and carries a **rigorous per-tile interval certificate** with canonical fallback — i.e. a *certified* mixed renderer, not an unconditional low-rank rasterizer.
- G5 target is eliminating the **teacher-to-student refit loop** specifically (route A steady-state), not merely compressing.
- Whether these distinctions survive the fresh search is **HYPOTHESIS until audited**.

## Claims currently ALLOWED
- "**novelty hypothesis**": that a certified per-tile low-rank CGT representation can be trained directly end-to-end. (goal §27 permits this before exhaustive audit.)

## Claims currently FORBIDDEN (until fresh search is done)
- "first native compressed Gaussian representation"
- any "unprecedented / SOTA / first-to-…" external claim
- DROP_PRIOR_ART_COLLISION verdict requires an actual audited collision, not speculation

## Required before final PASS
A fresh mechanism-level search (targets 1–9) recorded in `G5_REPRESENTATION_EVOLUTION.json`/master report. If a strong direct-prior collision is found, the lane closes with `DROP_PRIOR_ART_COLLISION`. This is a hard prerequisite for `PASS_NATIVE_CGT_*`.
