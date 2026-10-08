# BWD-ARITH1 Prior-Art Audit (§28)

## Scope
Focused audit of forward-intermediate caching / Gaussian-response reuse / replay strategies
in Gaussian-splatting rasterizers (2024-2026), distinguishing:
- transmittance / color-state checkpoints
- bucket checkpoints (Faster-GS / H-FasterGS)
- the Gaussian response G = exp(-Q/2) itself
- generic activation checkpointing

## Findings (knowledge-based, no live web access in this lane)

### 1. Faster-GS / H-FasterGS bucket checkpoints
Faster-GS (and the H-FasterGS lineage this repo builds on) stores, per bucket of 32
Gaussians x tile pixels, a **color + transmittance** checkpoint (`bucket_color_transmittance`,
`float4`) so the backward can reconstruct the compositing state without re-walking the
tile. This is a *compositing-state* checkpoint, NOT the Gaussian response G.
- H-FasterGS already stores: bucket color/transmittance (the state entering each 32-Gaussian
  bucket). See `bucket_color_transmittance` in kernels_forward/backward.
- BWD-ARITH1's R1 cache adds a **new state beyond** the existing checkpoint: the per-(Gaussian,
  pixel) Gaussian response G = exp(-Q/2). It does NOT replace the bucket checkpoint; it is
  additive (extra 4 bytes per interaction).
- Cache indexing piggybacks on the existing bucket layout: `[n_buckets, 32, 256]`, same
  bucket/lane/pixel decomposition as `bucket_color_transmittance`. No new sort/metadata.

### 2. Taming-3DGS / 3DGS training-reduction works
Taming-3DGS and similar focus on reducing Gaussian count / opacity / SH capacity during
training. They do NOT cache the Gaussian response for backward; they reduce work by
approximation. Different mechanism (lossy training-time reduction, not exact reuse).

### 3. gsplat / INRIA 3DGS reference
The reference gsplat backward recomputes the Gaussian (alpha, conic, color) from scratch;
it does not cache G across fwd/bwd. No forward-G reuse.

### 4. Generic activation checkpoint / replay (PyTorch, gradient checkpointing)
`torch.utils.checkpoint` trades compute for memory by recomputing activations in backward.
The standard policy is: recompute when the activation is cheap to store relative to its
recompute cost, OR store when recompute is expensive. BWD-ARITH1 is a *targeted* instance
of this tradeoff at the kernel level for one specific intermediate (G). The measured result
(recompute wins) is consistent with the generic principle: materializing a large,
low-arithmetic-intensity intermediate (G is one expf, but the buffer is 4 bytes x N) costs
more HBM traffic than the recompute saves.

### 5. VkSplat / mobile / WebGL rasterizers
VkSplat and mobile splatters are bandwidth-constrained and often *fuse* fwd/bwd or use
reduced precision. None report exact FP32 G caching as a win; the pressure is the opposite
(minimize memory traffic).

## Novelty label
`RESIDUAL_GAP_PLAUSIBLE` -> refined to `DIRECTLY_SOLVED_NEGATIVE_RESULT`:
- The specific mechanism (cache exact FP32 G from forward, load in backward) does not appear
  as a reported optimization in the audited works.
- However, the *result* is a negative one that confirms the generic recompute-vs-materialize
  principle: on A100-class HBM, recomputing one expf is cheaper than round-tripping 4 bytes
  per interaction through HBM. This is a useful, citable negative result for the
  Gaussian-splatting systems literature.
- We do NOT claim novelty of the *negative outcome* (it follows from bandwidth arithmetic);
  we claim the *exact, measured, production-integration negative result on H-FasterGS
  backward* as a residual data point.

## Distinction from existing checkpoint state (§29)
H-FasterGS already stores bucket color/transmittance checkpoints. R1 adds Gaussian-response
state BEYOND that checkpoint. The traffic cost is therefore *additional* to the existing
checkpoint traffic, not a replacement. This materially worsens the tradeoff: R1 pays
+8N bytes (write+read of G) on top of the existing bucket-checkpoint traffic, to save one
expf per interaction. The measurement shows the additional traffic dominates.

## Method note
Audit is knowledge-based (training corpus through 2026-06). No live web/IP lookup performed
in this lane. Labels are conservative; `NOVELTY_UNCERTAIN` applies to any 2025-2026 systems
paper not in the corpus.
