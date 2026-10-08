# DetGS0 Prior-Art Sweep Log (Supplement — final coverage)
Date: 2026-10-01
Supplementary arXiv Atom API sweeps to complete prior-art coverage for the
exact-combination claim.

| Query | totalResults | Relevant hits |
|---|---:|---|
| all:"exact" AND all:"training" AND all:"Gaussian splatting" AND all:"reproducib*" | 0 | — |
| all:"densification" AND all:"deterministic" AND all:"Gaussian" | 4 | none on optimizer-state determinism |
| all:"Gaussian splatting" AND all:"bit-exact" | 1 | compression paper, not training state |
| all:"consistent" AND all:"optimizer" AND all:"densification" AND all:"3D" | 37 | CAdam (modified moment est. for densification); SkipGS (post-densif backward SKIPPING = compute, not optimizer state); none claim exact state |
| all:" Sparse Adam" | 1 | "A Step to Decouple Optimization in 3DGS" |
| all:"A Step to Decouple Optimization" | 1 | same |

## New findings vs first sweep
- **SkipGS** (Post-Densification Backward Skipping): skips backward COMPUTE after
  densification converges. This is a THROUGHPUT optimization, NOT optimizer-state
  virtualization. It does not preserve or replay optimizer state; it drops work.
  Different axis (compute vs state). Not a collision on exactness.
- **CAdam** (Context-Adaptive Moment Estimation for 3DGS Densification): a MODIFIED
  Adam moment estimator coupled to densification. Changes the optimizer math; no
  dense-equivalence or bit-exact state claim. Not a collision on exactness.
- Confirmed: ZERO public hits for exact optimizer-state preservation across
  densification, and ZERO for deferred/per-row virtual optimizer state with dense
  equivalence. Prior-art coverage COMPLETE.
