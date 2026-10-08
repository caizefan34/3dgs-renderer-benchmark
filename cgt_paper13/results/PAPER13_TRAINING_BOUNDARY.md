# CGT-PAPER13 — Training Boundary (G5 / TI0 Negative Results)

This lane (CGT-PAPER13) is a **renderer** paper-evidence lane. It does NOT reopen native CGT
training or training-integration. This artifact records the upstream training-side negative
results that define CGT as a *rendering representation* rather than a *training primitive*,
so they are not conflated with the renderer verdict.

## 1. G5 — Native CGT optimization (CLOSED)
- Verdict: `DROP_NATIVE_OPTIMIZATION`
- Per-tile native GPU Adam / batched native G5 optimization was attempted as a way to make
  CGT coefficients the primary trainable scene variables. It failed to produce a robust,
  quality-preserving training primitive.
- Consequence: CGT is NOT used as the trainable parameterization in any PAPER13 stage.

## 2. TI0 — Canonical 3DGS training integration (CLOSED)
- Verdict: `DROP_UPDATE_COST`
- Hypothesis: keep canonical 3DGS variables (xyz/scale/rotation/opacity/SH) but integrate CGT
  as a fast rendering/compression layer inside the training loop.
- Failure mode (measured): the CGT representation update/refit cost dominates the training step.
  The fused forward benefit does not survive the per-iteration representation update + backward
  semantics. Effective training-step speedup fell below the promotion gate.
- Consequence: CGT is NOT inserted into canonical 3DGS training in any PAPER13 stage.

## 3. Relationship to the PAPER13 renderer verdict
- PAPER13 (`DROP_RENDERER_GENERALIZATION`) is an INDEPENDENT, RENDERER-only result.
- It is *consistent* with TI0: the same screen-space, per-camera representation that makes the
  CGT build dominate training also makes it dominate full-frame rendering (P3: ~40 min/frame build).
- The three negative results together support a single coherent conclusion:

> CGT is a real, certifiable, kernel-level rendering COMPRESSION with a genuine mixed-forward
> speedup on small tile samples, but its screen-space per-camera representation must be rebuilt
> far too expensively (fit + rigorous certificate) to accelerate either canonical 3DGS training
> (TI0) or full-frame novel-view rendering (PAPER13 E2E).

## 4. Scope discipline
- No G5/TI0 result is modified or reinterpreted by this lane.
- No native CGT optimization is performed here.
- PAPER13 uses canonical 3DGS checkpoints only as fixed inputs to a frozen renderer.