# VRIR0 Frozen Research Specification

## 1. Research question
For a fixed training view v sampled again at iteration t after being seen at t0<t, can the rasterizer's discrete structure be reused **exactly** despite the evolving Gaussian state Theta_t, provided a cheap certificate proves the structure has not changed?

This is not generic temporal rendering cache. The axis is deliberately inverted relative to most rendering work:
- camera: same training view on revisit;
- scene representation: evolving because optimization updates Gaussian parameters;
- reuse object: discrete tile membership + tile-local ordering;
- correctness goal: exact/fail-closed structural equivalence, not approximate image reuse.

## 2. Structural state
For fixed view v and Gaussian state Theta_t, define

S_v(Theta_t) = (A_t, B_t, L_t)

where:
- A_t(i) is the active/valid status of Gaussian i for the view;
- B_t(i) = [x_min,x_max) x [y_min,y_max) is its tile AABB/membership descriptor;
- L_t(tau) = (i_1,...,i_m) is the exact ordered list of Gaussian IDs consumed by tile tau.

If S_v(Theta_t) == S_v(Theta_t0), then current continuous projected attributes can be rasterized with the cached `flatten_ids/isect_offsets` from t0 and produce the same traversal structure as a fresh intersection/sort/offset build.

## 3. What must still be recomputed
At every revisit, even on a hypothetical future reuse hit:
- current 3D→2D projection;
- current means2d/radii/depths/conics;
- current opacity and SH/color evaluation;
- current rasterization and autograd state.

VRIR0 therefore tests structural memoization, not stale-attribute memoization.

## 4. Topology epochs
Define a monotonically increasing `topology_epoch`.
- It changes whenever clone/split/prune/reindex can alter Gaussian identity or array order.
- Any cache entry with a different epoch is invalid.
- VRIR0-v0 may conservatively increment on every densification/pruning event even if a scheduled event is a no-op. False rejects are allowed.

The primary 15k–30k window is chosen because canonical densification ends at 15k, removing the dominant topology invalidator.

## 5. Why the oracle precedes implementation
The oracle uses the baseline's *fresh* `flatten_ids` and `isect_offsets` as ground truth on every observed revisit. It never skips baseline work. It asks two independent questions:
1. Would our certificate have accepted reuse?
2. Was the old cached structure in fact bit/exact-structurally equal to the freshly rebuilt current structure?

Only the joint statistics establish feasibility. CUDA work is forbidden until the oracle passes all frozen gates.

## 6. Non-claims
VRIR0 must not claim:
- first temporal caching in 3DGS;
- first sorting reuse in 3DGS;
- first training cache in 3DGS;
- first use of certificates to maintain a sorted structure;
- standalone multi-x training speedup.

Potential novelty, if evidence supports it, is narrower: exact/fail-closed reuse of 3DGS raster **structure** across repeated training-view visits while Gaussian parameters evolve, with training/topology-aware certificates.
