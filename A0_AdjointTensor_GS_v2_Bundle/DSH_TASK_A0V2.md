# DSH TASK — A0-v2 AdjointTensor-GS Execution

You are an execution engine.

Read:
- OVERVIEW.md
- PRIOR_ART_BOUNDARY.md
- A0_V2_PREREG.md

Do NOT implement a production optimization kernel.

## Stage 0 — identity

Verify frozen P12/C0:
- source hashes
- checkpoints
- training-camera split
- original loss
- full effective resolution

## Stage 1 — algebra audit

Read the exact frozen backward CUDA and write:

`A0_EXACT_BACKWARD_FACTORIZATION.md`

It must show the equations used to export exact q_alpha / q_color sufficient
state.

If exact factorization is not available:
write `A0_ALGEBRA_BLOCKER.json` and STOP.

## Stage 2 — isolated instrumentation + parity

Use an isolated diagnostic worktree/build.

Instrument coefficient export only.

Run the frozen parity checks before collecting structural data.

Write:
- `A0_INSTRUMENTATION_PARITY.json`

If parity fails:
STOP.

## Stage 3 — bucket freeze

Freeze top-256 + uniform-256 buckets per scene/checkpoint from accepted-hit
counts only.

Write/hash:
- `A0_BUCKETS.json`

## Stage 4 — coefficient dump

Export exact FP32 sufficient-state tensors for frozen buckets.

Allowed sharding.

Required logical content:
- Q_alpha
- Q_color RGB
- accepted mask
- local Gaussian IDs
- native exact selected-bucket gradient contribution
- any exact local derivative factors needed for offline VJP reconstruction

## Stage 5 — analysis

Run the supplied frozen analyzer on the coefficient shards.

Do not change basis ordering/ranks/gates.

## Required artifacts

- `A0_SOURCE_IDENTITY.json`
- `A0_EXACT_BACKWARD_FACTORIZATION.md`
- `A0_INSTRUMENTATION_PARITY.json`
- `A0_BUCKETS.json`
- coefficient shards
- `A0_STRUCTURE_RESULTS.json`
- `A0_VJP_RESULTS.json`
- `A0_COST_PROXY.json`
- `A0_CONTAMINATION.json`
- `MANIFEST_A0V2.json`
- exact scripts/logs

## Forbidden

- no garden
- no test views
- no full training
- no learned basis
- no PCA basis as production evidence
- no rank/threshold changes
- no production CUDA optimization
- no paper/novelty verdict

Return raw artifacts to ChatGPT and STOP.
