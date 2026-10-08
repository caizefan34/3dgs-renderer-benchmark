# PSB0-P0 — Associative Segment-Scan Backward

Status: THEORY / SOURCE / TRACE / CPU-ORACLE ONLY. NO CUDA.

## Core question

Modern per-Gaussian 3DGS backward uses bucket-boundary checkpoint state and
replays splats inside each bucket.

PSB0 asks:

> Can front-to-back alpha compositing over a splat segment be represented by a
> compact associative transform, so that the backward state needed at arbitrary
> segment boundaries can be recovered by prefix/suffix scan rather than fixed
> checkpoint + linear replay?

This is an algorithmic replacement of checkpoint+replay.

## Why it can be a paper

If successful, the contribution is not a tuned stride. It is a new exact
parallel formulation of differentiable Gaussian compositing state recovery.

## Hard constraints

- Preserve exact traversal order.
- Preserve gradient semantics.
- No approximation.
- No per-(tile,Gaussian) materialization beyond native traversal state.
- No CUDA in P0.
