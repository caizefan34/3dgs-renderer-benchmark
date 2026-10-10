# TFC P0 stop rules (fixed before further work)

## Gate N0 — novelty first

- `TFC_FOOTPRINT_ONLY_NO_NOVELTY`: all concrete TFC contributions are just exact ellipse/tile mapping, pre-binning culling, tight bounding rectangles or standard AccuTile reimplementation. **STOP all new CUDA and training.** Historical results remain useful as implementation baselines or negative-method evidence.
- `BLOCKED_SOURCE_EVIDENCE`: actual TFC source or claimed new method cannot be inspected, or primary prior-art code/equations are unavailable. No novelty declaration and no implementation.
- `CONDITIONAL_RESEARCH`: there is at least one specific, reproducible mechanism *beyond* AccuTile, supported by source code/algorithm sketch and not shown overlapping in the checked references, but novelty/performance evidence remains incomplete.

## Gate E0 — measurable headroom (only after N0 survives)

- The new delta must target costs that actually remain *after* upstream AccuTile or equivalent B2, not weak stock gsplat v1.5.3 overbinning.
- Derive only with same-cohort production measurements; if absent `HEADROOM_UNMEASURED`. No implied speedups from PBR0 / PUB0 sums.
- If new mechanism has low intrinsic headroom compared to overhead, record `INSUFFICIENT_HEADROOM` without starting CUDA.

## Gate C0 — correctness (only if future GPU explicitly authorized)

- Equal camera, Gaussian subset hashes, inputs, background, SH, eps2d, tile size, sorting and loss.
- Verify forward pixel differences incl worst-case thin/degenerate ellipses and FP32 cancellation; compare all five gradient groups, support and absgrad, plus 3-seed quality.
- Audit unexplained TFC gradient A/B differences beyond A/A self-noise; do not suppress.

## Further permission policy

P0 provides at most a **proposal** to the user for Stage 1. Stage 1 must not be started automatically, even if `CONDITIONAL_RESEARCH`.

Forbidden claims: "38.2% pairs = 38.2% GPU time", "1.3564x = full-training F+B", "1.1988x/1.0317x = measured production", "gradient discrepancy is only atomic noise", "TFC/ellipse culling is new in 2026".
