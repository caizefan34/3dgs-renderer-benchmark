# Study memo: conservative row-span support is NOT itself an original algorithm

## Exact pixel-support derivation
For projected Gaussian center mu=(mu_x,mu_y), positive-definite precision P=[[a,b],[b,c]], opacity o, at pixel (x,y):

sigma = 0.5 * (a dx^2 + 2b dx dy + c dy^2)
alpha = min(.999, o exp(-sigma)).

For canonical cutoff alpha >= 1/255, when o>=1/255 and alpha is finite,
  a dx^2 + 2b dx dy + c dy^2 <= tau = 2 ln(255 o).
At fixed y, solve the quadratic in dx. Since a>0, valid x coordinates form one interval:
  dx_± = (-b dy ± sqrt(a tau - (ac-b²) dy²)) / a.
No supported pixel can be excluded if bounds are conservatively expanded and exact native alpha is recomputed in the inner loop. The row support membership proof relies on SPD, no generalized EWA footprint, same cutoff, and matching pixel coordinate conventions.

Sorting Gaussian IDs by native depth produces an ordered stream. Removing invalid-alpha terms leaves a stable subsequence. Since alpha compositing is order dependent, the stable subsequence is equivalent for RGB/alpha under exact skipping of no-ops; gradient order equivalence additionally assumes the same branch semantics and saved state. No promise of bitwise identity on GPU from CPU fp64 oracle.

## Complexity and cost (proposal, NOT measured)
Let N be Gaussian count, P image pixels, T tile/Gaussian candidate pairs, A valid pixel/Gaussian pairs, R nonempty Gaussian/scanline spans, and V candidate pixel visits before q-cutoff. Existing tile traversal work is O(V). A materialized run list costs O(R) interval records and O(A) pixel references, plus building/reordering overhead. A naive implementation may have O(NP) prepass or O(A) global writes; either may erase all arithmetic savings. O(R) alone is NOT an algorithm claim without a demonstrably cheap builder and consumer.

One possible novel implementation target (not implemented): generate row spans during AccuTile binning, retain rank-stable warp-block segments, and consume segments as a compressed stream in both forward and backward WITHOUT per-pixel O(A) global list materialization. This requires new CUDA dependency scheduling / memory arrangement and may fail on depth order, divergence or memory traffic. `rasterize_to_indices_in_range` already produces sparse per-pixel indices; any proposed paper must beat it and specify an algorithmic difference.

## Amdahl criterion
With fraction f of complete process time addressable, and local speedup s including preprocessing, E2E speedup = 1/(1-f+f/s). At f=.8, S>=3 requires s>=6. At f=.7, it requires s>=21. If f<=2/3, 3x cannot be attained by this lever even when s→∞. Use SAME-MX Faster-GS fused f, never historical EPIC-05 or HiGS R26 f.

## Why this is not another C51/P17/R26
This proposal does not select gradients to drop, does not replace full-res loss with coarse gradients, does not count atomics, and does not reorder particle-centric backward. It seeks to eliminate q-test/staging on non-supported pixel pairs from BOTH exact compositing and its reverse differentiation. Its chief novelty hazard is existing precise rasterization and indexed sparse rendering.

## Alternate future scientific problem
If this pipeline fails G0 or G5, the other worthwhile hypothesis concerns *overlap-aware counterfactual topology*, not simple per-Gaussian frequency splitting. SAD-GS (SIGGRAPH 2026) admits it ignores joint primitive support. Yet 3DGS² (SIGGRAPH 2025), LeGS, SteepGS, and SAD-GS itself make second-order/group optimization and densification competitive prior art. No GPU arm is authorized for that idea without a distinct operation and CPU oracle.
