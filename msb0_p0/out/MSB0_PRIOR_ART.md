# MSB0-P0 Prior-Art Audit (Gate 2, refreshed on execution day 2026-10-01)

## Method
Fresh direct-collision audit per `PRIOR_ART_BOUNDARY.md`. A **direct collision**
requires a PUBLIC static-3DGS training method that simultaneously:
1. uses the same per-Gaussian/bucket backward family;
2. removes one or more saved checkpoint state components by ALGEBRAIC
   reconstruction;
3. preserves the same gradient semantics;
4. demonstrates training/runtime or VRAM benefit.

A generic reverse-compositing identity alone is NOT a novelty claim.

## Known adjacent mechanisms (from PRIOR_ART_BOUNDARY.md)
- **Original / pixel-centric 3DGS backward** (Kerbl et al. 2023): reconstructs
  transmittance while traversing the Gaussians backward, one pixel at a time.
  No bucket-boundary state is stored.
- **Taming / Faster-GS-style per-Gaussian backward**: stores bucket-boundary
  blending state (`bucket_color_transmittance`) to enable parallel per-bucket
  reverse-mode. This is exactly the FasterGSFused design under study.
- **Recent workload-balanced rasterizers**: bucketed / associative compositing.

## Audit findings

### M3 (forward-carry) == original pixel-centric 3DGS backward — ADJACENCY, not novelty
M3 reconstructs each bucket's entering state `(C_b, T_b)` by carrying the forward
recurrence from the known constant `(C_0=0, T_0=1)`. This is algebraically
identical to the original pixel-centric backward that rebuilds transmittance (and
accumulated color) while walking the Gaussians. It removes `bucket_color_transmittance`
by recomputation. However:
- It is a **second full traversal** + **global cross-bucket sync** (Gate 5), and
- it demonstrates **no benefit** — it is strictly the pre-bucket mechanism that
  the bucket scheme was designed to replace.

So M3 is a **known adjacent mechanism**, not a novel contribution. It does not
constitute a direct-collision *novelty* claim (it is the baseline), but it is
explicitly **not** a state *elimination* win — it re-trades memory for the exact
work the bucket state exists to avoid.

### M1 / M2 (drop one component) — NOT in prior art
No public method drops only `C_b` or only `T_b` while preserving gradient
semantics, because (as Gate 4 shows) each is independently necessary. These
candidates are eliminated by reconstruction error, not by prior art.

### Direct-collision verdict
**No direct collision.** No public static-3DGS training method both (a) removes
a saved bucket-boundary state component by algebraic reconstruction AND (b)
preserves gradient semantics AND (c) shows a benefit. The only reconstruction
that preserves semantics (M3) is the original pixel-centric backward (known
adjacency, no benefit, Gate-5 work amplification). Therefore Gate 2 **PASSES**:
the MSB0 question (are these components redundant?) is not pre-answered by prior
art, and the answer (no) is established by our own Gate 4/5 replay, not by
citation.

## Distinction from BSC0
- **BSC0**: same `bucket_color_transmittance` state, compressed representation
  (quantization / bit-reduction).
- **MSB0 (this task)**: fewer state variables by deriving/reconstructing them.
  Result: the state is non-redundant; the viable next step is to compose BSC0
  compression on the *retained* state, not to eliminate it.
